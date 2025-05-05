import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ALL, dcc, html, ctx
from dash.exceptions import PreventUpdate
import xarray as xr

from qua_dashboards.video_mode.data_acquirers import BaseDataAcquirer
from qua_dashboards.video_mode.dash_tools import xarray_to_plotly
from qua_dashboards.core.base_component import BaseComponent

logger = logging.getLogger(__name__)


__all__ = ["VideoModeComponent"]


class VideoModeComponent(BaseComponent):
    """
    A dashboard component for visualizing and controlling data acquisition
    in video mode.

    This component encapsulates the UI layout and callback logic for the
    video mode functionality, delegating data acquisition and parameter
    handling to a provided DataAcquirer instance.
    """

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        component_id: str = "video-mode",
        save_path: Union[str, Path] = "./video_mode_output",
        update_interval_sec: float = 0.1,
        include_update_button: bool = True,
    ):
        """
        Initializes the VideoModeComponent.

        Args:
            data_acquirer: A pre-configured instance of a BaseDataAcquirer subclass.
            component_id: A unique string identifier for this component instance.
            save_path: Directory path to save data and images.
            update_interval_sec: Interval for polling new data (in seconds).
            include_update_button: Whether to include the 'Update Params' button.
        """
        super().__init__(component_id=component_id)
        self.data_acquirer = data_acquirer
        self.save_path = Path(save_path)
        self.update_interval_ms = int(update_interval_sec * 1000)
        self.include_update_button = include_update_button

        logger.info(f"Initializing VideoModeComponent ({self.component_id=})")
        logger.info(f"Using Data Acquirer: {type(data_acquirer).__name__}")

        # --- Internal State ---
        self.paused: bool = False
        self._last_save_clicks: int = 0
        self._last_update_clicks: int = 0
        # Initial plot based on data acquirer's starting state
        # Ensure data_acquirer has valid initial data_array
        try:
            self.fig = xarray_to_plotly(self.data_acquirer.data_array)
        except Exception as e:
            logger.error(f"Failed to create initial plot: {e}", exc_info=True)
            self.fig = go.Figure()  # Fallback to empty figure

    def _get_id(self, index: str) -> Dict[str, str]:
        """Helper to generate component IDs with pattern-matching structure."""
        # Ensure the index is unique within this component instance
        return {"type": f"vmc-{self.component_id}", "index": index}

    def get_layout(self) -> html.Div:
        """Generates the Dash layout for the video mode component."""
        logger.debug(f"Generating layout for {self.component_id}")

        # --- VMC-Specific Controls ---
        vmc_controls = [
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
        ]

        # --- Data Acquirer Controls ---
        # Get controls directly from the data acquirer instance
        try:
            acquirer_controls = self.data_acquirer.get_dash_components(
                include_subcomponents=True
            )
            logger.debug(
                f"Retrieved {len(acquirer_controls)} control groups from Data Acquirer."
            )
        except Exception as e:
            logger.error(
                f"Error getting controls from Data Acquirer: {e}", exc_info=True
            )
            acquirer_controls = [html.Div("Error loading data acquirer controls.")]

        # --- Action Buttons ---
        action_buttons = dbc.Row(
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
        )

        # --- Assemble Control Panel ---
        control_panel_elements = vmc_controls + acquirer_controls + [action_buttons]

        # --- Final Layout Structure ---
        layout = html.Div(
            [
                dbc.Row(
                    [
                        # Control panel part
                        dbc.Col(
                            control_panel_elements,
                            width=12,
                            lg=5,
                            className="mb-3 mb-lg-0",
                        ),  # Add margin bottom on small screens
                        # Plot part
                        dbc.Col(
                            dcc.Graph(
                                id=self._get_id("live-heatmap"),
                                figure=self.fig,
                                style={"height": "100%", "min-height": "400px"},
                                config={"responsive": True},  # Ensure graph resizes
                            ),
                            width=12,
                            lg=7,
                            className="d-flex flex-column",  # Ensure column takes height
                        ),
                    ],
                    className="g-3",  # Add gutters between columns
                    style={"height": "100%"},  # Try to make row take height
                ),
                dcc.Interval(
                    id=self._get_id("interval-component"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                    # Disable interval when paused - controlled in callback now
                    # disabled=self.paused
                ),
                # Store used to trigger parameter updates cleanly
                dcc.Store(id=self._get_id("update-trigger-store"), data=0),
            ],
            # Main component div - ensure it can take up space
            style={"height": "100%", "min-height": "500px"},
            id=self.component_id,  # Assign the main component ID here
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
            if n_clicks is None or n_clicks == 0:
                raise PreventUpdate
            self.paused = not self.paused
            logger.info(f"Component {self.component_id} paused state: {self.paused}")
            # Disable/enable interval via callback now? No, handle in update_heatmap
            return "Resume" if self.paused else "Pause"

        # --- Callback for Live Update (Interval or Parameter Update Trigger) ---
        @app.callback(
            Output(self._get_id("live-heatmap"), "figure"),
            Output(self._get_id("iteration-output"), "children"),
            # Trigger on interval OR the update trigger store
            Input(self._get_id("interval-component"), "n_intervals"),
            Input(self._get_id("update-trigger-store"), "data"),
            # prevent_initial_call=True # Allow initial load based on store
            # Use blocking=True if updates are slow and overlap causes issues
            # blocking=True,
        )
        def update_heatmap(n_intervals, update_trigger):
            triggered_id = ctx.triggered_id if ctx.triggered_id else "initial load"
            logger.debug(
                f"{self.component_id} update_heatmap triggered by: {triggered_id}"
            )

            if (
                self.paused
                and isinstance(triggered_id, dict)
                and triggered_id.get("index") == "interval-component"
            ):
                logger.debug(f"{self.component_id} update skipped (paused).")
                raise PreventUpdate  # Skip interval updates if paused

            t_start = datetime.now()

            # If triggered by parameter update, force refresh even if paused
            is_param_update = (
                isinstance(triggered_id, dict)
                and triggered_id.get("index") == "update-trigger-store"
            )

            # --- Data Acquisition ---
            # Only acquire new data if not paused or if forced by param update
            # And only acquire if triggered by interval (not just param update)
            if (
                (not self.paused or is_param_update)
                and isinstance(triggered_id, dict)
                and triggered_id.get("index") == "interval-component"
            ):
                try:
                    logger.debug(f"{self.component_id} acquiring new data...")
                    _ = self.data_acquirer.update_data()  # Update internal state
                except Exception as e:
                    logger.error(
                        f"Error during data acquisition for {self.component_id}: {e}",
                        exc_info=True,
                    )
                    # Decide how to handle error - maybe show error message?
                    # For now, just log and proceed to plot existing data

            # --- Plotting ---
            # Always replot after param update or successful interval update
            try:
                # Get the current data array from the acquirer
                current_data_array = self.data_acquirer.data_array
                # Generate the plot
                # Potentially pass self.fig to allow modification instead of recreation
                self.fig = xarray_to_plotly(current_data_array)
                # Apply enhancements if any (future)
                # for enhancer in self.enhancers:
                #    self.fig = enhancer.apply_figure_enhancement(self.fig, current_data_array)

                iteration_text = f"Iteration: {self.data_acquirer.num_acquisitions}"
                t_end = datetime.now()
                logger.debug(
                    f"{self.component_id} plot updated in {(t_end - t_start).total_seconds():.3f}s"
                )
                return self.fig, iteration_text
            except Exception as e:
                logger.error(
                    f"Error generating plot for {self.component_id}: {e}",
                    exc_info=True,
                )
                # Return existing figure or error state
                iteration_text = (
                    f"Iteration: {self.data_acquirer.num_acquisitions} (Plot Error)"
                )
                # Prevent graph update on plot error to avoid broken graph display
                raise PreventUpdate

        # --- State Inputs for Parameter Update Callback ---
        # Collect ALL potential inputs from the data acquirer's components
        all_param_states = []
        try:
            # Use ALL matcher for ids and values for each component type used by data_acquirer
            acquirer_component_ids = self.data_acquirer.get_component_ids() or []
            logger.debug(f"Data Acquirer uses component IDs: {acquirer_component_ids}")
            for da_comp_id_pattern in acquirer_component_ids:
                # We need to match the 'type' field from the IDs the acquirer uses
                all_param_states.extend(
                    [
                        State({"type": da_comp_id_pattern, "index": ALL}, "id"),
                        State({"type": da_comp_id_pattern, "index": ALL}, "value"),
                    ]
                )
        except Exception as e:
            logger.error(
                f"Failed to get component IDs/states from Data Acquirer: {e}",
                exc_info=True,
            )
            # Handle error: perhaps disable update button or show warning

        # --- Callback for Parameter Updates (Button) ---
        # Only register if the update button is included and we have states
        if self.include_update_button and all_param_states:

            @app.callback(
                Output(
                    self._get_id("update-trigger-store"), "data"
                ),  # Trigger figure update
                Input(self._get_id("update-button"), "n_clicks"),
                all_param_states,
                State(
                    self._get_id("update-trigger-store"), "data"
                ),  # Pass current value
                prevent_initial_call=True,
            )
            def update_parameters(n_update_clicks, *component_states_and_trigger):
                # Separate states from the trigger store's current value
                component_states = component_states_and_trigger[:-1]
                current_trigger_val = component_states_and_trigger[-1]

                # Basic check to prevent updates if button hasn't been clicked more
                if (
                    n_update_clicks is None
                    or n_update_clicks <= self._last_update_clicks
                ):
                    raise PreventUpdate
                self._last_update_clicks = n_update_clicks

                logger.info(f"Update parameters triggered for {self.component_id}")

                # --- Process component states ---
                params_by_component_type = {}
                states_iter = iter(component_states)
                for da_comp_id_pattern in self.data_acquirer.get_component_ids():
                    try:
                        ids = next(states_iter, [])
                        values = next(states_iter, [])
                        if not ids:
                            logger.warning(
                                f"No state IDs found for pattern {da_comp_id_pattern}"
                            )
                            continue  # Skip if component type had no inputs captured

                        # Store params keyed by the TYPE (pattern) from the ID dictionary
                        params_by_component_type[da_comp_id_pattern] = {
                            id_dict["index"]: value
                            for id_dict, value in zip(ids, values)
                            if id_dict and "index" in id_dict  # Basic validation
                        }
                    except StopIteration:
                        logger.error(
                            "Mismatch between expected state patterns and provided states."
                        )
                        break  # Exit loop if states run out unexpectedly
                    except Exception as e:
                        logger.error(
                            f"Error processing states for {da_comp_id_pattern}: {e}",
                            exc_info=True,
                        )

                if not params_by_component_type:
                    logger.warning("No valid parameters collected from state inputs.")
                    raise PreventUpdate

                logger.debug(
                    f"Updating Data Acquirer with parameters: {params_by_component_type}"
                )
                try:
                    # Call the data acquirer's update method
                    # This method should handle restarting programs if needed
                    self.data_acquirer.update_parameters(params_by_component_type)
                    logger.info(
                        f"Data Acquirer parameters updated for {self.component_id}"
                    )
                    # Increment the trigger store data to signal update_heatmap
                    return current_trigger_val + 1
                except Exception as e:
                    logger.error(
                        f"Error calling data_acquirer.update_parameters for {self.component_id}: {e}",
                        exc_info=True,
                    )
                    # Do not trigger update if parameters failed
                    raise PreventUpdate

        elif self.include_update_button:
            logger.warning(
                f"Update button included for {self.component_id}, but no parameter states found."
            )

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
                # Pass the current figure object for saving
                idx = self.save(figure_to_save=self.fig)
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

    # --- Saving Methods ---

    def _find_next_save_index(self) -> int:
        """Determines the next available save index based on data and image files."""
        idx = 1
        data_save_path = self.save_path / "data"
        image_save_path = self.save_path / "images"
        # Use NetCDF extension for data
        while (data_save_path / f"data_{idx:04d}.nc").exists() or (
            image_save_path / f"image_{idx:04d}.png"
        ).exists():
            idx += 1
            if idx > 9999:
                raise ValueError("Maximum save index (9999) reached.")
        return idx

    def save_data(self, idx: int) -> None:
        """Saves the current data array to a NetCDF file."""
        data_save_path = self.save_path / "data"
        logger.debug(f"Attempting to save data to folder: {data_save_path}")
        data_save_path.mkdir(parents=True, exist_ok=True)

        filename = f"data_{idx:04d}.nc"
        filepath = data_save_path / filename

        if filepath.exists():
            # This shouldn't happen if _find_next_save_index works correctly
            # but handle defensively.
            logger.warning(f"Data file {filepath} already exists. Overwriting.")

        try:
            # Ensure data is up-to-date before saving
            current_data = self.data_acquirer.data_array
            if isinstance(current_data, xr.DataArray):
                current_data.to_netcdf(filepath)
                logger.info(f"Data saved successfully: {filepath}")
            else:
                logger.error(
                    "Cannot save data, data_acquirer.data_array is not an xarray.DataArray."
                )
                raise TypeError("Invalid data type for saving.")
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}", exc_info=True)
            raise

    def save_image(self, idx: int, figure_to_save: go.Figure) -> None:
        """Saves the provided figure to a PNG file."""
        image_save_path = self.save_path / "images"
        logger.debug(f"Attempting to save image to folder: {image_save_path}")
        image_save_path.mkdir(parents=True, exist_ok=True)

        filename = f"image_{idx:04d}.png"
        filepath = image_save_path / filename

        if filepath.exists():
            logger.warning(f"Image file {filepath} already exists. Overwriting.")

        try:
            if isinstance(figure_to_save, go.Figure):
                figure_to_save.write_image(filepath)
                logger.info(f"Image saved successfully: {filepath}")
            else:
                logger.error(
                    "Cannot save image, provided figure is not a Plotly Figure."
                )
                raise TypeError("Invalid figure object for saving.")
        except Exception as e:
            logger.error(f"Failed to save image to {filepath}: {e}", exc_info=True)
            raise

    def save(self, figure_to_save: go.Figure) -> int:
        """Saves both the current data and the current image with the same index."""
        self.save_path.mkdir(parents=True, exist_ok=True)

        idx = self._find_next_save_index()

        # Save both with the determined index
        self.save_data(idx)
        self.save_image(idx, figure_to_save)

        logger.info(f"Save operation completed with index: {idx}")
        return idx
