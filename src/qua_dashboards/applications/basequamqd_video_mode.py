from pathlib import Path
import logging
import threading

from typing import Literal

from werkzeug.serving import make_server

from qua_dashboards.core import build_dashboard
from qua_dashboards.video_mode import OPXDataAcquirer, scan_modes, VideoModeComponent
from qua_dashboards.virtual_gates import VirtualLayerEditor
from qua_dashboards.physical_voltage_control import QDAC2VoltageControlComponent, VirtualVoltageControlComponent
from qua_dashboards.video_mode.utils.save_backends import SaveBackend, QCoDeSBackend, DataHandlerBackend, QualibrateBackend
from qualang_tools.results.data_handler import DataHandler
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class BaseQuamQDVideoMode:

    SAVE_BACKENDS = {
        "qcodes": QCoDeSBackend,
        "datahandler": DataHandlerBackend,
        "qualibrate": QualibrateBackend,
    }


    def _create_save_backend(self, save_backend, save_dir=None):
        if isinstance(save_backend, SaveBackend):
            return save_backend

        if save_backend == "qualibrate":
            return QualibrateBackend()

        if save_backend == "qcodes":
            return QCoDeSBackend()

        if save_backend == "datahandler":
            if save_dir is None:
                raise ValueError("save_dir must be provided when save_backend='datahandler'.")

            handler = DataHandler(root_data_folder=Path(save_dir))

            return DataHandlerBackend(
                handler,
                instrument_snapshot= None #({"qdac": self.qdac.snapshot} if self.qdac is not None else None)
            )

        raise ValueError(f"Invalid save_backend: {save_backend}")

    def __init__(
        self,
        *,
        machine,
        quam_dir=None,
        dac_device_mapping=None,
        x_axis_name=None,
        y_axis_name=None,
        save_backend: Literal["qcodes", "datahandler", "qualibrate"] | SaveBackend = "qualibrate",
        save_dir: str | Path | None = None,
        video_port=8050,
        voltage_port=8051,
        editor_port=8052,
    ):
        """
        Quantum Dot Video Mode application for BaseQuamQD.

        Parameters
        ----------
        machine
            Loaded BaseQuamQD machine instance.

        quam_dir
            Optional path to the QuAM state.
            Only required if reload/restart should reload the machine from disk.

        dac_device_mapping
            Optional mapping selecting which QDAC channels are shown in
            the physical voltage-control panel.

            For a single bound QDAC:

                {
                    "P1": 1,
                    "P2": 2,
                    "Source": 10,
                }

            If a selected QDAC channel is already represented by a
            physical VoltageGate in QuAM, the QuAM channel id is used
            automatically as the control name.

        x_axis_name, y_axis_name
            Initial scan axes. If omitted, the first two physical gates are used.

        save_backend
            Video Mode save backend. Defaults to QCoDeSBackend.

        video_port, voltage_port, editor_port
            Dashboard server ports.
        """

        self.machine = machine

        self.quam_dir = (Path(quam_dir) if quam_dir is not None else None)

        self.dac_device_mapping = dac_device_mapping

        # Threading locks to avoid race conditions when multiple threads are accessing the same resources.
        self.voltage_lock = threading.RLock()

        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

        self.video_port = video_port
        self.voltage_port = voltage_port
        self.editor_port = editor_port

        self.save_backend = self._create_save_backend(save_backend, save_dir=save_dir)
        self.save_dir = Path(save_dir) if save_dir is not None else None        
        
        # Hardware / runtime
        self.qmm = None

        # Dashboard components
        self.video = None
        self.data_acquirer = None

        self.physical_control = None
        self.virtual_control = None
        self.editor = None

        # Servers
        self.video_server = None
        self.voltage_server = None
        self.editor_server = None

        # Threads
        self.video_thread = None
        self.voltage_thread = None
        self.editor_thread = None

        self._validate_machine()

    # ==========================================================
    # Machine helpers
    # ==========================================================

    def _validate_machine(self):
        if not hasattr(self.machine, "connect"):
            raise TypeError("BaseQuamQDVideoMode requires machine.connect().")

        # This also validates the gate-set structure.
        self._get_gate_sets()

    def _get_physical_channels(self):
        """Return physical VoltageGate channels from BaseQuamQD."""
        physical_channels = getattr(self.machine, "physical_channels", None)
        if not physical_channels:
            raise ValueError("BaseQuamQD machine does not provide any physical_channels.")
        return physical_channels

    def _get_gate_sets(self):
        """Resolve the single VirtualGateSet and matching VirtualDCSet from BaseQuamQD."""
        gate_sets = self.machine.virtual_gate_sets
        if not gate_sets:
            raise ValueError("No VirtualGateSet is configured.")
        if len(gate_sets) > 1:
            raise ValueError("Multiple VirtualGateSets are configured. Please specify which gate set should be used.")
        gate_set_id, vgs = next(iter(gate_sets.items()))
        vdc = self.machine.virtual_dc_sets.get(gate_set_id)
        return gate_set_id, vgs, vdc

    def _get_bound_qdac(self):
        """Resolve the single QDAC runtime registered in BaseQuamQD.machine.dacs."""
        dacs = getattr(self.machine, "dacs", {})
        qdacs = {name: info for name, info in dacs.items() if info.get("is_qdac", False)}
        if len(qdacs) > 1:
            raise ValueError("Multiple QDACs are not supported yet.")
        if not qdacs:
            return None
        dac_name, dac_info = next(iter(qdacs.items()))
        return dac_name, dac_info["driver"]

    def _get_resonators(self):
        """Return readout resonators registered through BaseQuamQD.sensor_dots."""
        sensor_dots = getattr(self.machine, "sensor_dots", None)
        if not sensor_dots:
            raise ValueError("No sensor dots are configured.")
        resonators = []
        for sensor_dot in sensor_dots.values():
            physical_channel = getattr(sensor_dot, "physical_channel", None)
            if physical_channel is None:
                continue
            readout = getattr(physical_channel, "readout", None)
            if readout is not None:
                resonators.append(readout)
        if not resonators:
            raise ValueError("No readout resonators are configured.")
        return resonators

    def _get_default_axes(self):
        """
        Determine the initial X/Y scan axes.
        """

        _, vgs, _ = self._get_gate_sets()
        
        physical_gate_names = list(vgs.channels.keys())

        if len(physical_gate_names) < 2:
            raise ValueError(
                "At least two physical gates are required "
                "to initialize the default X/Y axes."
            )

        x_axis_name = (
            self.x_axis_name
            if self.x_axis_name is not None
            else physical_gate_names[0]
        )

        y_axis_name = (
            self.y_axis_name
            if self.y_axis_name is not None
            else physical_gate_names[1]
        )

        return x_axis_name, y_axis_name

    def _get_dac_device_mapping(self):
        """
        Normalize user-selected QDAC controls against QuAM wiring.

        If a selected QDAC channel is already represented by a physical
        VoltageGate in QuAM, the QuAM physical-channel id is used as the
        canonical control name.
        """

        if self.dac_device_mapping is None:
            return {}

        user_mapping = dict(self.dac_device_mapping)

        bound_qdac = self._get_bound_qdac()

        if bound_qdac is None:
            raise RuntimeError(
                "dac_device_mapping was provided, but no QDAC "
                "is bound to the machine."
            )

        dac_name, _ = bound_qdac

        physical_channels = self._get_physical_channels()

        quam_channel_by_port = {}

        for channel_name, channel in physical_channels.items():
            dac_spec = getattr(channel,"dac_spec", None)

            if dac_spec is None:
                continue

            channel_dac_name = getattr(dac_spec, "dac_name", "main")

            if channel_dac_name != dac_name:
                continue

            port = dac_spec.output_port

            if port in quam_channel_by_port:
                raise ValueError(
                    f"Multiple QuAM channels are mapped to "
                    f"DAC {dac_name!r} port {port}."
                )

            quam_channel_by_port[port] = channel_name

        normalized = {}

        for user_name, channel_number in user_mapping.items():

            canonical_name = quam_channel_by_port.get(channel_number, user_name)

            if canonical_name != user_name:
                logging.getLogger(__name__).warning(
                    "QDAC channel %s is registered in QuAM as %r; "
                    "replacing control name %r with the QuAM name.",
                    channel_number,
                    canonical_name,
                    user_name,
                )

            normalized[canonical_name] = channel_number

        return normalized

    # ==========================================================
    # Machine reload
    # ==========================================================

    def reload_machine(self):
        """
        Reload the machine from the configured QuAM state directory.

        The machine class is inferred from the current machine instance.
        """

        if self.quam_dir is None:
            raise RuntimeError(
                "Cannot reload the machine because "
                "'quam_dir' was not provided."
            )

        machine_cls = type(self.machine)

        self.machine = machine_cls.load(
            self.quam_dir
        )

        self._validate_machine()

        return self.machine

    # ==========================================================
    # Start
    # ==========================================================

    def start(self):
        if self.is_running:
            print("Dashboards are already running.")
            return

        self._validate_machine()

        # Resolve the gate set before QOP connection.
        gate_set_id, vgs, vdc = self._get_gate_sets()

        # Resolve default scan axes.
        x_axis_name, y_axis_name = self._get_default_axes()

        # Connect only to QOP. External DACs are expected to be
        # already registered/bound on the BaseQuamQD machine.
        self.qmm = self.machine.connect(skip_dacs=True)

        # Refresh references in case the machine state changed.
        gate_set_id, vgs, vdc = self._get_gate_sets()

        # ------------------------------------------------------
        # Readout pulses
        # ------------------------------------------------------

        readout_pulses = [
            resonator.operations["readout"]
            for resonator in self._get_resonators()
            if "readout" in resonator.operations
        ]

        if not readout_pulses:
            raise ValueError(
                "No 'readout' operations were found "
                "on the configured resonators."
            )

        # ======================================================
        # Voltage control
        # ======================================================

        voltage_components = []

        # ------------------------------------------------------
        # Physical QDAC control
        # ------------------------------------------------------

        bound_qdac = self._get_bound_qdac()
        device_mapping = self._get_dac_device_mapping()

        if bound_qdac is not None and device_mapping:

            dac_name, qdac = bound_qdac

            self.physical_control = (
                QDAC2VoltageControlComponent(
                    component_id="physical-voltage-control",
                    qdac=qdac,
                    device_mapping=device_mapping,
                    dc_set=vdc,
                    step_size=1e-3,
                    voltage_lock=self.voltage_lock,
                )
            )

            voltage_components.append(
                self.physical_control
            )

        # ------------------------------------------------------
        # Virtual DC control
        # ------------------------------------------------------

        if vdc is not None:

            self.virtual_control = (
                VirtualVoltageControlComponent(
                    component_id="virtual-voltage-control",
                    gate_set=vgs,
                    dc_set=vdc,
                    step_size=1e-3,
                    voltage_lock=self.voltage_lock
                )
            )

            voltage_components.append(
                self.virtual_control
            )

        if voltage_components:
            voltage_app = build_dashboard(
                components=voltage_components,
                title="Voltage Control",
            )
        else:
            voltage_app = None

        # ======================================================
        # Virtual gate editor
        # ======================================================

        self.editor = VirtualLayerEditor(
            component_id="virtual-gates-ui",
            gateset=vgs,
            dc_set=vdc,
            save_path=(
                str(self.quam_dir)
                if self.quam_dir is not None
                else None
            ),
        )

        editor_app = build_dashboard(
            components=[self.editor],
            title="Virtual Gate Editor",
        )

        # ======================================================
        # Video Mode
        # ======================================================

        scan_mode_dict = {
            "Switch_Raster_Scan":
                scan_modes.SwitchRasterScan(),

            "Raster_Scan":
                scan_modes.RasterScan(),

            "Spiral_Scan":
                scan_modes.SpiralScan(),
        }

        self.data_acquirer = OPXDataAcquirer(
            qmm=self.qmm,
            machine=self.machine,
            gate_set=vgs,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            scan_modes=scan_mode_dict,
            result_type="amplitude",
            available_readout_pulses=(
                readout_pulses
            ),
            acquisition_interval_s=1,
            use_buffered_stream=False,
            apply_compensation_pulse=True,
            voltage_control_component=(
                self.physical_control
            ),
        )

        self.video = VideoModeComponent(
            data_acquirer=self.data_acquirer,
            data_polling_interval_s=0.2,
            save_path=(
                str(self.quam_dir)
                if self.quam_dir is not None
                else None
            ),
            save_backend=self.save_backend,
        )

        video_app = build_dashboard(
            components=[self.video],
            title="QD Video Mode",
            full_viewport=True,
        )

        # ======================================================
        # Create controllable servers
        # ======================================================

        self.video_server = make_server(
            "127.0.0.1",
            self.video_port,
            video_app.server,
            threaded=True,
        )

        if voltage_app is not None:
            self.voltage_server = make_server(
                "127.0.0.1",
                self.voltage_port,
                voltage_app.server,
                threaded=True,
            )

        self.editor_server = make_server(
            "127.0.0.1",
            self.editor_port,
            editor_app.server,
            threaded=True,
        )

        # ======================================================
        # Start threads
        # ======================================================

        self.video_thread = threading.Thread(
            target=(
                self.video_server.serve_forever
            ),
            daemon=True,
            name="VideoModeDashboard",
        )

        if self.voltage_server is not None:
            self.voltage_thread = (
                threading.Thread(
                    target=(
                        self.voltage_server
                        .serve_forever
                    ),
                    daemon=True,
                    name=(
                        "VoltageControlDashboard"
                    ),
                )
            )

        self.editor_thread = (
            threading.Thread(
                target=(
                    self.editor_server
                    .serve_forever
                ),
                daemon=True,
                name=(
                    "VirtualGateEditorDashboard"
                ),
            )
        )

        self.video_thread.start()

        if self.voltage_thread is not None:
            self.voltage_thread.start()

        self.editor_thread.start()

        # ======================================================
        # URLs
        # ======================================================

        print(
            "Video Mode:          "
            f"http://127.0.0.1:{self.video_port}"
        )

        if self.voltage_server is not None:
            print(
                "Voltage Control:     "
                f"http://127.0.0.1:{self.voltage_port}"
            )

        print(
            "Virtual Gate Editor: "
            f"http://127.0.0.1:{self.editor_port}"
        )

    # ==========================================================
    # Stop
    # ==========================================================

    def stop(self):
        if not self.is_running:
            print(
                "Dashboards are not running."
            )
            return

        print("Stopping dashboards...")

        # ------------------------------------------------------
        # Stop acquisition / QM job first
        # ------------------------------------------------------

        if self.data_acquirer is not None:
            try:
                self.data_acquirer.shutdown()
            except Exception as exc:
                logging.getLogger(
                    __name__
                ).warning(
                    "Error shutting down "
                    "Video Mode acquisition: %s",
                    exc,
                )

        # ------------------------------------------------------
        # Stop dashboard servers
        # ------------------------------------------------------

        for server in (
            self.video_server,
            self.voltage_server,
            self.editor_server,
        ):
            if server is not None:
                server.shutdown()

        # ------------------------------------------------------
        # Join threads
        # ------------------------------------------------------

        for thread in (
            self.video_thread,
            self.voltage_thread,
            self.editor_thread,
        ):
            if thread is not None:
                thread.join(timeout=2)

        # ------------------------------------------------------
        # Clear runtime references
        # ------------------------------------------------------

        self.video_server = None
        self.voltage_server = None
        self.editor_server = None

        self.video_thread = None
        self.voltage_thread = None
        self.editor_thread = None

        self.video = None
        self.data_acquirer = None

        self.physical_control = None
        self.virtual_control = None
        self.editor = None

        self.qmm = None

        print("Dashboards stopped.")

    # ==========================================================
    # Recover QOP runtime
    # ==========================================================

    def recover(self, reload_machine=False):
        logger = logging.getLogger(__name__)
        print("Recovering QOP runtime and Video Mode...")

        old_qmm = self.qmm
        old_acquirer = self.data_acquirer

        # 1. Stop local fetching first, but avoid blocking shutdown
        if old_acquirer is not None:
            old_acquirer._fetch_running = False

        # 2. Hard cleanup server-side QMs first
        if old_qmm is not None:
            print("Trying to close all QMs...")

            close_all = getattr(old_qmm, "close_all_qms", None)
            if close_all is None:
                close_all = getattr(old_qmm, "close_all_quantum_machines", None)

            if callable(close_all):
                try:
                    close_all()
                    print("All Quantum Machines closed.")
                except Exception:
                    logger.exception("Failed to close all Quantum Machines.")

        # 3. Now stop dashboards / local threads
        try:
            self.stop()
        except Exception:
            logger.exception("Normal Video Mode stop failed during recovery.")

        # 4. Clear runtime references
        self.qmm = None
        self.data_acquirer = None

        # 5. Optional QuAM reload
        if reload_machine:
            self.reload_machine()

        # 6. Create fresh QMM + components
        print("Starting Video Mode again...")
        self.start()

    # ==========================================================
    # Restart
    # ==========================================================

    def restart(
        self,
        *,
        reload_machine=True,
    ):
        """
        Restart all dashboards.

        Parameters
        ----------
        reload_machine
            If True, reload the QuAM state from `quam_dir`
            before restarting.

            If False, reuse the current in-memory machine.
        """

        print("Restarting dashboards...")

        self.stop()

        if reload_machine:
            self.reload_machine()

        self.start()

    # ==========================================================
    # Status
    # ==========================================================

    @property
    def is_running(self):
        threads = (
            self.video_thread,
            self.voltage_thread,
            self.editor_thread,
        )

        return any(
            thread is not None
            and thread.is_alive()
            for thread in threads
        )