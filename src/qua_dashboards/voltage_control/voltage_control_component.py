"""
Dash component for displaying and controlling a series of voltage parameters.
"""

import logging
from typing import Dict, Sequence

import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
    ALL, 
    MATCH,
    State
)

from qua_dashboards.core import BaseComponent, ParameterProtocol

from .voltage_control_row import VoltageControlRow, format_voltage

from quam_builder.architecture.quantum_dots.components.virtual_dc_set import VirtualDCSet

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CLASS_NAME = ""  # Input fields will always have this class
COMPONENT_MAX_WIDTH = "450px"


class VirtualGateParameter:
    """
    A Parameter to be integrated into the VoltageControlComponent, integrating the layers of virtual gates into the VoltageControlComponent. 
    """
    def __init__(
        self, 
        name: str, 
        label: str, 
        dc_set: VirtualDCSet, 
        unit: str = "V",
    ): 
        self.name = name
        self.dc_set = dc_set
        self.label = label
        self.unit = unit

    def get_latest(self): 
        return self.dc_set.get_voltage(self.name)
    
    def set(self, value) -> None:
        self.dc_set.set_voltages({self.name: value}, requery = False, resync = False)


class VoltageControlComponent(BaseComponent):
    """
    A Dash component to display and set voltages from multiple sources.
    Uses VoltageControlRow to manage individual parameter rows.
    Simplified: No edit highlighting, no auto-defocus on enter.
    """

    def __init__(
        self,
        component_id: str,
        voltage_parameters: Sequence[ParameterProtocol] = None,
        dc_set: VirtualDCSet = None,
        update_interval_ms: int = 1000,
        layout_columns: int = 3,
        step_size: float = 100e-6,
        preselected_gates: Sequence[str] | None = None,
    ):
        super().__init__(component_id=component_id)
        if voltage_parameters is None and dc_set is None: 
            raise ValueError("Please provide either a sequence of VoltageParameter objects, or a VirtualDCSet.")
        if voltage_parameters is None: 
            self.dc_set = dc_set
            _ = self.dc_set.all_current_voltages
            voltage_parameters = []
            for ch in self.dc_set.valid_channel_names: 
                virtual_param = VirtualGateParameter(name = ch, label = ch, dc_set = self.dc_set)
                voltage_parameters.append(virtual_param)
        self.update_interval_ms = update_interval_ms
        self.voltage_parameters = voltage_parameters
        self._initial_values_loaded = False  # To ensure first update populates values

        self._row_components: Dict[str, VoltageControlRow] = {}
        input_id_type = self._get_id_type_str("input")
        for param in self.voltage_parameters:
            row = VoltageControlRow(input_id_type=input_id_type, param=param)
            self._row_components[param.name] = row

        self.layout_columns = layout_columns
        self.step_size = step_size

        # Full list of available parameters is stored in self.voltage_parameters, and selected ones are appended to gates_to_display
        all_names = {p.name for p in self.voltage_parameters}
        self._initial_selected_names = [n for n in (preselected_gates or []) if n in all_names]
        self.gates_to_display = [p for p in self.voltage_parameters if p.name in self._initial_selected_names]

    def _get_id_type_str(self, element_name: str) -> str:
        return f"comp-{self.component_id}-{element_name}"
        
    def _build_dropdown_options(self) -> list:
        """Build dropdown options with physical/virtual grouping."""
        options = []
        physical_names = set(self.dc_set.channels.keys()) if self.dc_set else set()
        virtual_names = [n for n in self.dc_set.valid_channel_names if n not in physical_names]
        
        # Physical gates section
        if physical_names:
            options.append({"label": "── Physical Gates ──", "value": "__physical_header__", "disabled": True})
            for name in sorted(physical_names):
                options.append({"label": name, "value": name})
        
        # Virtual gates section  
        if virtual_names:
            options.append({"label": "── Virtual Gates ──", "value": "__virtual_header__", "disabled": True})
            for name in virtual_names:
                options.append({"label": name, "value": name})
        
        return options
    
    @property
    def voltage_parameters_by_name(self) -> Dict[str, "VirtualGateParameter"]:
        # Initial lazy create the param, which is then REused by the update_displayed_rows
        if self.dc_set is not None: 
            existing_names = {p.name for p in self.voltage_parameters}
            for name in self.dc_set.valid_channel_names: 
                if name not in existing_names: 
                    param = VirtualGateParameter(name = name, label = name, dc_set = self.dc_set)
                    self.voltage_parameters.append(param)

        return {p.name: p for p in self.voltage_parameters}

    def get_layout(self) -> html.Div:
        if not self._row_components:
            return html.Div(
                "No voltage parameters configured.",
                style={"maxWidth": COMPONENT_MAX_WIDTH},
            )
        input_ids = [
            {"type": self._get_id_type_str("input"), "index": param.name} 
            for param in self.gates_to_display
        ]
        return html.Div(
            [
                dcc.Dropdown(
                    id = self._get_id("dc-gate-selector"), 
                    options = self._build_dropdown_options(),
                    value = self._initial_selected_names, 
                    multi = True, 
                    placeholder = "Select gates to display", 
                    style = {"color": "black"}
                ),
                html.Div(
                    id = self._get_id("rows-container"), 
                    children = [self._row_components[n].get_layout() for n in self._initial_selected_names],
                ),
                dcc.Interval(
                    id=self._get_id("update-interval"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                dcc.Store(
                    id=self._get_id("keyboard-config"),
                    data={
                        "input_ids": input_ids,
                        "step_size": self.step_size,
                        "dummy_store_id": self._get_id("keyboard-dummy"),
                    }
                ),
                dcc.Store(id=self._get_id("keyboard-dummy")),  # Changed from html.Div
                # *[row.get_layout() for row in self._row_components.values()],
            ],
            style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto"},
        )
    
    def register_callbacks(self, app: dash.Dash) -> None:
        self._app = app
        if not self.voltage_parameters:
            return

        input_id_type_str = self._get_id_type_str("input")

        @app.callback(
            Output({"type": input_id_type_str, "index": ALL}, "value", allow_duplicate=True), 
            Output({"type": input_id_type_str, "index": ALL}, "className", allow_duplicate=True), 
            Input(self._get_id("update-interval"), "n_intervals"), 
            State({"type": input_id_type_str, "index": ALL}, "id"),
            prevent_initial_call = True, 
        )
        def periodic_update(_n_intervals, input_ids): 
            if not input_ids: 
                return [],[]
            values = []
            classnames = []
            for input_id in input_ids: 

                param_name = input_id["index"]
                control_row = self._row_components[param_name]

                param = next(p for p in self.voltage_parameters if p.name == param_name)

                try:
                    live_value = param.get_latest()
                    live_text = format_voltage(live_value)
                except Exception as e:
                    logger.error(f"Err get_latest for {param_name}: {e}", exc_info=True)
                    values.append(dash.no_update)
                    classnames.append(dash.no_update)
                    continue

                if control_row.current_input_text != control_row.last_committed_text:
                    values.append(dash.no_update)
                elif live_text != control_row.current_input_text:
                    control_row.current_input_text = live_text
                    control_row.last_committed_text = live_text
                    values.append(live_text)
                else: 
                    values.append(dash.no_update)
                classnames.append(DEFAULT_INPUT_CLASS_NAME)
            return values, classnames
        
        @app.callback(
            Output(self._get_id("rows-container"), "children"), 
            Output(self._get_id("keyboard-config"), "data"),
            Input(self._get_id("dc-gate-selector"), "value"),
        )
        def update_displayed_rows(selected_names): 
            if not selected_names: 
                return [], {"input_ids": [], "step_size": self.step_size}
            params_by_name = self.voltage_parameters_by_name
            for name in selected_names:
                if name not in self._row_components:
                    param = params_by_name[name]
                    # param = VirtualGateParameter(name=name, label=name, dc_set=self.dc_set)
                    # self.voltage_parameters.append(param)
                    row = VoltageControlRow(input_id_type=self._get_id_type_str("input"), param=param)
                    self._row_components[name] = row
            self.gates_to_display = [p for p in self.voltage_parameters if p.name in selected_names]
            rows = [self._row_components[name].get_layout() for name in selected_names]
            input_ids = [{"type": self._get_id_type_str("input"), "index": name} for name in selected_names]

            return rows, {"input_ids": input_ids, "step_size": self.step_size, "dummy_store_id": self._get_id("keyboard-dummy")}
        
        @app.callback(
            Output({"type": input_id_type_str, "index": MATCH}, "value", allow_duplicate=True), 
            Input({"type": input_id_type_str, "index": MATCH}, "n_submit"), 
            State({"type": input_id_type_str, "index": MATCH}, "value"), 
            State({"type": input_id_type_str, "index": MATCH}, "id"),
            prevent_initial_call = True, 
        )

        def handle_input_submit(_n_submit, submitted_text_value, input_id): 
            if not _n_submit or submitted_text_value is None:
                raise dash.exceptions.PreventUpdate
            
            param_name = input_id["index"]
            control_row = self._row_components.get(param_name)
            param = next((p for p in self.voltage_parameters if p.name == param_name), None)
            
            if control_row is None or param is None:
                raise dash.exceptions.PreventUpdate
            
            try:
                float_value = float(submitted_text_value)
                param.set(float_value)
                control_row.current_input_text = format_voltage(float_value)
                control_row.last_committed_text = control_row.current_input_text
                return control_row.current_input_text
            except ValueError:
                logger.warning(f"Invalid float: '{submitted_text_value}' for {param_name}")
                return dash.no_update
            except Exception as e:
                logger.error(f"Error during set for {param_name}: {e}", exc_info=True)
                return dash.no_update
            
        @app.callback(
            Output({"type": input_id_type_str, "index": ALL}, "value", allow_duplicate=True),
            Input(self._get_id("keyboard-dummy"), "data"),
            State({"type": input_id_type_str, "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def handle_arrow_key_change(store_data, input_ids):
            if not store_data or "param_name" not in store_data:
                raise dash.exceptions.PreventUpdate
            
            param_name = store_data["param_name"]
            new_value_str = store_data["value"]
            
            param = next((p for p in self.voltage_parameters if p.name == param_name), None)
            control_row = self._row_components.get(param_name)
            
            if param is None or control_row is None:
                raise dash.exceptions.PreventUpdate
            
            try:
                float_value = float(new_value_str)
                param.set(float_value)
                formatted = format_voltage(float_value)
                control_row.current_input_text = formatted
                control_row.last_committed_text = formatted
            except (ValueError, Exception) as e:
                logger.error(f"Arrow key set error for {param_name}: {e}")
                raise dash.exceptions.PreventUpdate
            
            # Return updated value for the matching input, no_update for the rest
            return [
                formatted if iid["index"] == param_name else dash.no_update
                for iid in input_ids
            ]

        @app.callback(
            Output({"type": input_id_type_str, "index": MATCH}, "value", allow_duplicate=True),
            Input({"type": input_id_type_str, "index": MATCH}, "n_blur"),
            State({"type": input_id_type_str, "index": MATCH}, "id"),
            prevent_initial_call=True,
        )
        def handle_input_blur(_n_blur, input_id):
            if not _n_blur:
                raise dash.exceptions.PreventUpdate
            
            param_name = input_id["index"]
            control_row = self._row_components.get(param_name)
            param = next((p for p in self.voltage_parameters if p.name == param_name), None)
            
            if control_row is None or param is None:
                raise dash.exceptions.PreventUpdate
            
            control_row.current_input_text = format_voltage(param.get_latest())
            control_row.last_committed_text = control_row.current_input_text
            return control_row.current_input_text

        app.clientside_callback(
            r"""
            function(config) {
                if (!config || !config.input_ids) {
                    return null;
                }

                if (window._voltageKeyboardListenerAttached) {
                    return null;
                }
                window._voltageKeyboardListenerAttached = true;

                var inputIds = config.input_ids || [];
                var stepSize = config.step_size || 0.00001;

                function findVoltageInput(index) {
                    if (index < 0 || index >= inputIds.length) {
                        return null;
                    }
                    var id = inputIds[index];
                    var allInputs = document.querySelectorAll('input');
                    for (var i = 0; i < allInputs.length; i++) {
                        var el = allInputs[i];
                        var rawId = el.id;
                        if (!rawId) continue;
                        try {
                            var parsed = JSON.parse(rawId);
                            if (parsed && parsed.type === id.type && parsed.index === id.index) {
                                return el;
                            }
                        } catch (e) {}
                    }
                    return null;
                }

                function isVoltageInputElement(el) {
                    if (!el || !el.id) return false;
                    try {
                        var parsed = JSON.parse(el.id);
                        for (var i = 0; i < inputIds.length; i++) {
                            var id = inputIds[i];
                            if (id.type === parsed.type && id.index === parsed.index) {
                                return true;
                            }
                        }
                        return false;
                    } catch (e) {
                        return false;
                    }
                }

                document.addEventListener('keydown', function(e) {
                    var activeEl = document.activeElement;
                    var tagName = activeEl ? activeEl.tagName.toLowerCase() : '';
                    var isInInput = (tagName === 'input' || tagName === 'textarea');
                    var isInVoltageInput = isInInput && isVoltageInputElement(activeEl);

                    var shortcuts = [
                        'q','w','e','r','t','y','u','i','o','p',
                        'a','s','d','f','g','h','j','k','l',
                        'z','x','c','v','b','n','m'
                    ];
                    var key = (e.key || '').toLowerCase();
                    var idx = shortcuts.indexOf(key);

                    if (idx !== -1 && (!isInInput || isInVoltageInput)) {
                        var targetEl = findVoltageInput(idx);
                        if (targetEl) {
                            e.preventDefault();
                            if (activeEl && typeof activeEl.blur === 'function') {
                                activeEl.blur();
                            }
                            targetEl.focus();
                            if (typeof targetEl.select === 'function') {
                                targetEl.select();
                            }
                        }
                    }

                    if (isInVoltageInput && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                        e.preventDefault();

                        var currentVal = parseFloat(activeEl.value);
                        if (!isFinite(currentVal)) {
                            currentVal = 0;
                        }

                        var delta = (e.key === 'ArrowUp') ? stepSize : -stepSize;
                        if (e.shiftKey) delta *= 10;
                        if (e.ctrlKey)  delta *= 100;

                        var newVal = currentVal + delta;
                        var newValStr = newVal.toFixed(9).replace(/\.?0+$/, '');

                        activeEl.value = newValStr;
                        var parsedId = JSON.parse(activeEl.id);
                        dash_clientside.set_props(
                            config.dummy_store_id,
                            {data: {param_name: parsedId.index, value: newValStr, ts: Date.now()}}
                        );
                    }
                });

                return null;
            }
            """,
            Output(self._get_id("keyboard-dummy"), "data"),
            Input(self._get_id("keyboard-config"), "data"),
            prevent_initial_call=True,
        )
