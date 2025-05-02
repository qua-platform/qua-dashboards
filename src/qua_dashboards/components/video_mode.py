import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ALL, dcc, html
from dash.exceptions import PreventUpdate

from qua_dashboards.video_mode.data_acquirers import (
    BaseDataAcquirer,
    RandomDataAcquirer,
    OPXDataAcquirer,
    # OPXQuamDataAcquirer,
)
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qua_dashboards.video_mode.scan_modes import ScanMode
from qua_dashboards.video_mode.inner_loop_actions import InnerLoopAction
from qua_dashboards.video_mode.dash_tools import xarray_to_plotly


logger = logging.getLogger(__name__)


class VideoModeComponent:
    """
    A dashboard component for visualizing and controlling data acquisition
    in video mode.

    This component encapsulates the UI layout and callback logic for the
    video mode functionality.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        shared_objects: Dict[str, Any],
        component_id: str = "video-mode",
    ):
        """
        Initializes the VideoModeComponent.

        Args:
            params: Dictionary containing configuration parameters for the
                component and its data acquisition strategy. Expected keys
                depend on 'data_acquirer_type' but generally include:
                'data_acquirer_type' (str: 'random', 'opx', 'opx_quam'),
                'x_axis' (SweepAxis), 'y_axis' (SweepAxis),
                'scan_mode' (ScanMode),
                'data_acquirer_params' (dict),
                'inner_loop_action' (InnerLoopAction, optional),
                'save_path' (str, optional),
                'update_interval' (float, optional, in seconds).
            shared_objects: Dictionary containing objects shared between
                components, such as QMM instances or QUA configurations.
                Expected keys depend on 'data_acquirer_type' (e.g., 'qmm',
                'qua_config').
            component_id: A unique string identifier for this component instance
        """
        self.component_id = component_id
        self.params = params
        self.shared_objects = shared_objects
        logger.info(f"Initializing VideoModeComponent ({self.component_id=})")

        # --- Extract Parameters ---
        self.save_path = Path(params.get("save_path", "./video_mode_output"))
        # Interval for dcc.Interval is in milliseconds
        self.update_interval_ms = int(params.get("update_interval", 0.1) * 1000)
        self.include_update_button = params.get("include_update_button", True)

        # --- Instantiate Core Logic Components ---
        # Assuming these are passed as instantiated objects in params for now
        # Adapt this if they need to be constructed from descriptions in params
        x_axis: SweepAxis = params["x_axis"]
        y_axis: SweepAxis = params["y_axis"]
        scan_mode: ScanMode = params["scan_mode"]

        inner_loop_action: Optional[InnerLoopAction] = params.get("inner_loop_action")

        # --- Instantiate Data Acquirer ---
        acquirer_type = params.get("data_acquirer_type", "random")
        acquirer_params = params.get("data_acquirer_params", {})
        # Ensure the data acquirer gets a unique ID derived from the component ID
        data_acquirer_id = f"{self.component_id}-data-acquirer"

        logger.info(f"Configuring data acquirer of type: {acquirer_type}")
        if acquirer_type == "opx":
            qmm = shared_objects["qmm"]
            qua_config = shared_objects["qua_config"]
            if inner_loop_action is None:
                raise ValueError("inner_loop_action is required for 'opx' acquirer")
            self.data_acquirer: BaseDataAcquirer = OPXDataAcquirer(
                qmm=qmm,
                qua_config=qua_config,
                qua_inner_loop_action=inner_loop_action,
                scan_mode=scan_mode,
                x_axis=x_axis,
                y_axis=y_axis,
                component_id=data_acquirer_id,
                **acquirer_params,
            )
        # Add elif for 'opx_quam' here if needed, retrieving 'machine', etc.
        # elif acquirer_type == "opx_quam":
        #    ...
        elif acquirer_type == "random":
            self.data_acquirer = RandomDataAcquirer(
                x_axis=x_axis,
                y_axis=y_axis,
                component_id=data_acquirer_id,
                **acquirer_params,
            )
        else:
            raise ValueError(f"Unsupported data_acquirer_type: {acquirer_type}")
        logger.info(f"Data acquirer '{type(self.data_acquirer).__name__}' configured.")

        # --- Internal State ---
        self.paused = False
        self._last_save_clicks = 0
        # Initial plot
        self.fig = xarray_to_plotly(self.data_acquirer.data_array)

    def _get_id(self, index: str) -> Dict[str, str]:
        """Helper to generate component IDs with pattern-matching structure."""
        return {"type": self.component_id, "index": index}

    def get_layout(self) -> html.Div:
        """Generates the Dash layout for the video mode component."""
        logger.debug(f"Generating layout for {self.component_id}")
        control_elements = [
            html.H4("Video Mode Controls", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Pause", id=self._get_id("pause-button"), n_clicks=0
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(
                            f"Iteration: {self.data_acquirer.num_acquisitions}",
                            id=self._get_id("iteration-output"),
                            # Vertically align text in the middle
                            className="d-flex align-items-center",
                        ),
                        width="auto",
                    ),
                ],
                className="mb-3",  # Margin bottom
            ),
            # Parameter controls from data acquirer and its sub-components
            html.Div(
                self.data_acquirer.get_dash_components(include_subcomponents=True)
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Save", id=self._get_id("save-button"), n_clicks=0),
                        width="auto",
                    ),
                    # Conditionally add Update button
                    dbc.Col(
                        dbc.Button(
                            "Update Params",
                            id=self._get_id("update-button"),
                            n_clicks=0,
                        ),
                        width="auto",
                    )
                    if self.include_update_button
                    else None,
                ],
                className="mt-3",  # Margin top
            ),
        ]

        layout = html.Div(
            [
                dbc.Row(
                    [
                        # Control panel part
                        dbc.Col(control_elements, width=12, lg=5),
                        # Plot part
                        dbc.Col(
                            dcc.Graph(
                                id=self._get_id("live-heatmap"),
                                figure=self.fig,
                                style={"height": "100%", "min-height": "400px"},
                                # Ensure graph resizes reasonably
                                config={"responsive": True},
                            ),
                            width=12,
                            lg=7,
                            # Ensure column takes height
                            className="d-flex flex-column",
                        ),
                    ],
                    className="g-3",  # Add gutters between columns
                    style={"height": "100%"},  # Try to make row take height
                ),
                dcc.Interval(
                    id=self._get_id("interval-component"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
            ],
            # Main component div - ensure it can take up space if needed
            style={"height": "100%", "min-height": "500px"},
        )
        return layout

    def register_callbacks(self, app: Dash):
        """Registers callbacks for the video mode component."""
        logger.debug(f"Registering callbacks for {self.component_id}")

        # --- Callback for Pause Button ---
        @app.callback(
            Output(self._get_id("pause-button"), "children"),
            Input(self._get_id("pause-button"), "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_pause(n_clicks):
            self.paused = not self.paused
            logger.info(f"Component {self.component_id} paused state: {self.paused}")
            return "Resume" if self.paused else "Pause"

        # --- Callback for Live Update (Interval) ---
        @app.callback(
            Output(self._get_id("live-heatmap"), "figure"),
            Output(self._get_id("iteration-output"), "children"),
            Input(self._get_id("interval-component"), "n_intervals"),
            blocking=True,  # Try blocking to prevent overlapping updates
        )
        def update_heatmap(n_intervals):
            if self.paused:
                logger.debug(f"{self.component_id} update skipped (paused).")
                raise PreventUpdate  # More efficient than returning current fig

            t_start = datetime.now()
            logger.debug(
                f"{self.component_id} update triggered at {t_start.isoformat()}"
            )

            try:
                updated_xarr = self.data_acquirer.update_data()
                # Update the figure object efficiently
                self.fig = xarray_to_plotly(updated_xarr)
                iteration_text = f"Iteration: {self.data_acquirer.num_acquisitions}"
                t_end = datetime.now()
                logger.debug(
                    f"{self.component_id} updated in {(t_end - t_start).total_seconds():.3f}s"
                )
                return self.fig, iteration_text
            except Exception as e:
                logger.error(
                    f"Error during data update for {self.component_id}: {e}",
                    exc_info=True,
                )
                # Return existing figure or handle error state
                iteration_text = (
                    f"Iteration: {self.data_acquirer.num_acquisitions} (Error)"
                )
                return self.fig, iteration_text

        # --- State Inputs for Parameter Update Callback ---
        # Collect all potential inputs from the data acquirer and its children
        all_param_states = []
        # Generate state definitions for all potential inputs
        for da_comp_id in self.data_acquirer.get_component_ids():
            all_param_states.extend(
                [
                    State({"type": da_comp_id, "index": ALL}, "id"),
                    State({"type": da_comp_id, "index": ALL}, "value"),
                ]
            )

        # --- Callback for Parameter Updates (Button) ---
        # Triggered ONLY by the update button if it exists
        update_inputs = []
        if self.include_update_button:
            update_inputs.append(Input(self._get_id("update-button"), "n_clicks"))
        else:
            # If no button, maybe trigger on input changes? More complex.
            # For now, require the button if dynamic updates are needed.
            pass

        if update_inputs:  # Only register if there's an input trigger

            @app.callback(
                # No direct output, modifies internal state and data acquirer
                Output(self._get_id("update-button"), "n_clicks"),
                update_inputs,
                all_param_states,
                prevent_initial_call=True,
            )
            def update_parameters(n_update_clicks, *component_states):
                if n_update_clicks is None or n_update_clicks == 0:
                    raise PreventUpdate

                logger.info(f"Update parameters triggered for {self.component_id}")

                # --- Process component states ---
                params_by_component = {}
                # Iterate through the flattened list of states
                states_iter = iter(component_states)
                for da_comp_id in self.data_acquirer.get_component_ids():
                    ids = next(states_iter, [])
                    values = next(states_iter, [])
                    if not ids:
                        continue  # Skip if component had no inputs captured

                    params_by_component[da_comp_id] = {
                        id_dict["index"]: value
                        for id_dict, value in zip(ids, values)
                        if id_dict and value is not None
                    }

                logger.debug(f"Updating with parameters: {params_by_component}")
                try:
                    # Call the data acquirer's update method
                    self.data_acquirer.update_parameters(params_by_component)
                    # Re-generate initial figure potentially if needed (e.g. axes changed)
                    self.fig = xarray_to_plotly(self.data_acquirer.data_array)
                    logger.info(f"Parameters updated for {self.component_id}")
                except Exception as e:
                    logger.error(
                        f"Error updating parameters for {self.component_id}: {e}",
                        exc_info=True,
                    )

                # Reset button clicks to allow re-triggering if needed, or just return current
                # Returning n_clicks might be simpler if button state isn't critical elsewhere
                return n_update_clicks

        # --- Callback for Save Button ---
        @app.callback(
            Output(self._get_id("save-button"), "children"),  # Update button text
            Input(self._get_id("save-button"), "n_clicks"),
            prevent_initial_call=True,
        )
        def save_data_and_image(n_clicks):
            if n_clicks is None or n_clicks <= self._last_save_clicks:
                raise PreventUpdate

            logger.info(f"Save triggered for {self.component_id}")
            self._last_save_clicks = n_clicks
            try:
                idx = self.save()
                logger.info(
                    f"Data and image saved with index {idx} for {self.component_id}"
                )
                return f"Saved ({idx})"
            except Exception as e:
                logger.error(
                    f"Error saving data/image for {self.component_id}: {e}",
                    exc_info=True,
                )
                return "Save Failed"

    # --- Saving Methods (ported from VideoModeApp) ---

    def save_data(self, idx: Optional[int] = None) -> int:
        """Saves the current data array to a NetCDF file."""
        data_save_path = self.save_path / "data"
        logger.info(f"Attempting to save data to folder: {data_save_path}")
        data_save_path.mkdir(parents=True, exist_ok=True)

        if idx is None:
            idx = 1
            while (data_save_path / f"data_{idx:04d}.nc").exists():
                idx += 1
            if idx > 9999:
                raise ValueError("Max data files (9999) reached.")

        filename = f"data_{idx:04d}.nc"  # Use NetCDF extension
        filepath = data_save_path / filename

        if filepath.exists():
            # Overwrite warning or raise error? Let's warn for now.
            logger.warning(f"Overwriting existing data file: {filepath}")
            # raise FileExistsError(f"File {filepath} already exists.")

        try:
            self.data_acquirer.data_array.to_netcdf(filepath)
            logger.info(f"Data saved successfully: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            raise  # Re-raise the exception
        return idx

    def save_image(self, idx: Optional[int] = None) -> int:
        """Saves the current figure to a PNG file."""
        image_save_path = self.save_path / "images"
        logger.info(f"Attempting to save image to folder: {image_save_path}")
        image_save_path.mkdir(parents=True, exist_ok=True)

        if idx is None:
            idx = 1
            while (image_save_path / f"image_{idx:04d}.png").exists():
                idx += 1
            if idx > 9999:
                raise ValueError("Max image files (9999) reached.")

        filename = f"image_{idx:04d}.png"
        filepath = image_save_path / filename

        if filepath.exists():
            logger.warning(f"Overwriting existing image file: {filepath}")
            # raise FileExistsError(f"File {filepath} already exists.")

        try:
            # Ensure self.fig is up-to-date if needed, though update_heatmap
            # should handle it
            if isinstance(self.fig, go.Figure):
                self.fig.write_image(filepath)
                logger.info(f"Image saved successfully: {filepath}")
            else:
                logger.error("Cannot save image, self.fig is not a Plotly Figure.")
                raise TypeError("Figure object is not a valid Plotly Figure.")
        except Exception as e:
            logger.error(f"Failed to save image to {filepath}: {e}")
            raise  # Re-raise the exception
        return idx

    def save(self) -> int:
        """Saves both the current data and the current image with the same index."""
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        # Determine next available index based on *both* data and image folders
        idx = 1
        data_save_path = self.save_path / "data"
        image_save_path = self.save_path / "images"
        while (data_save_path / f"data_{idx:04d}.nc").exists() or (
            image_save_path / f"image_{idx:04d}.png"
        ).exists():
            idx += 1
        if idx > 9999:
            raise ValueError("Max files (9999) reached.")

        # Save both with the determined index
        saved_data_idx = self.save_data(idx)
        saved_image_idx = self.save_image(idx)

        # Should normally be the same, but good practice to check
        if saved_data_idx != saved_image_idx:
            logger.warning(
                f"Data ({saved_data_idx}) and image ({saved_image_idx})"
                f" saved with different indices!"
            )

        logger.info(f"Save operation completed with index: {idx}")
        return idx
