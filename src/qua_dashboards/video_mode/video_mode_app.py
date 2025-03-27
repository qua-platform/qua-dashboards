from datetime import datetime
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union
import warnings
import dash
from dash import dcc, html, ALL
from dash_extensions.enrich import (
    DashProxy,
    Output,
    Input,
    State,
    BlockingCallbackTransform,
)
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import logging

from qua_dashboards.video_mode.data_acquirers import BaseDataAcquirer
from qua_dashboards.video_mode.dash_tools import xarray_to_heatmap


__all__ = ["VideoModeApp", "VideoMode"]


class VideoModeApp:
    """
    A class for visualizing and controlling data acquisition in video mode.

    This class provides a dashboard interface for visualizing and controlling data
    acquisition in video mode. It uses Dash for the web interface and Plotly for the
    heatmap visualization.

    Attributes:
        data_acquirer (BaseDataAcquirer): The data acquirer object that provides the
            data to be visualized.
        save_path (Union[str, Path]): The path where data and images will be saved.
        update_interval (float): The interval at which the data is updated in the
            dashboard (in seconds). If the previous update was not finished in the given
            interval, the update will be skipped.
    """

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        save_path: Union[str, Path] = "./video_mode_output",
        update_interval: float = 0.1,
    ):
        self.data_acquirer = data_acquirer
        self.save_path = Path(save_path)
        self.paused = False
        self._last_update_clicks = 0
        self._last_save_clicks = 0
        self.update_interval = update_interval
        self._is_updating = False

        # Using DashProxy so we can use dash_extensions.enrich
        self.app = DashProxy(
            __name__,
            title="Video Mode",
            transforms=[BlockingCallbackTransform(timeout=10)],  # Running callbacks sequentially!!!
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )  # Add Bootstrap theme
        self.create_layout()

    def create_layout(self):
        """
        Create the layout for the video mode dashboard.

        This method sets up the Dash layout for the video mode control panel. It includes:
        - A graph to display the heatmap of acquired data
        - Controls for X and Y parameters (offset, span, points)
        - Buttons for pausing/resuming data acquisition and saving data
        - Display for the current iteration count
        - Input for setting the number of averages

        The layout is designed to be responsive and user-friendly, with aligned input fields
        and clear labeling. It uses a combination of Dash core components and HTML elements
        to create an intuitive interface for controlling and visualizing the data acquisition
        process.

        Returns:
            None: The method sets up the `self.app.layout` attribute but doesn't return anything.
        """
        
        added_points = {'x': [], 'y': [], 'index': []}   # added points
        selected_point_to_move = {'move': False, 'index': None}        # selected point to move
        dict_lines = {'selected_indices': [], 'added_lines': {'start_index':[], 'end_index':[]}}  # selected points for drawing a line
        dict_heatmap = xarray_to_heatmap(self.data_acquirer.data_array)
        #self.heatmap = dict_heatmap['heatmap']
        self.figure = self._generate_figure(dict_heatmap,added_points,dict_lines["added_lines"])
        #logging.debug(f"figure: {self.figure}")
        #logging.debug(f"added points: {added_points}")
        self.radio_options = [
            {"label": "Adding and moving points", "value": "add"},
            {"label": "Deleting points", "value": "delete"},
            {"label": "Adding lines", "value": "line"},
        ]

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(  # Settings
                            [
                                html.H1("Video mode", className="mb-4"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Pause",
                                                id="pause-button",
                                                n_clicks=0,
                                                className="mb-3",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                id="iteration-output",
                                                children="Iteration: 0",
                                                className="mb-3 ml-3 d-flex align-items-center",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                html.Div(
                                    self.data_acquirer.get_dash_components(
                                        include_subcomponents=True
                                    )
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Update",
                                                id="update-button",
                                                n_clicks=0,
                                                className="mt-3 mr-2",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "Save",
                                                id="save-button",
                                                n_clicks=0,
                                                className="mt-3",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Points and lines", className="card-title"),
                                                    dbc.RadioItems(
                                                        id='mode-selector',
                                                        options=self.radio_options,
                                                        value="add",
                                                        #['Adding and moving points', 'Adding lines', 'Deleting points'], 
                                                        #'Adding and moving points',
                                                    ),
                                                ]),
                                            ),
                                            width="auto"
                                        ),
                                        
                                    ],
                                    className="mr-2"
                                ),
                            ],
                            width=5,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="live-heatmap",
                                figure=self.figure,
                                style={"aspect-ratio": "1 / 1"},
                                #config={'scrollZoom': True},
                            ),
                            width=7,
                        ),
                    ]
                ),
                dcc.Interval( # Component  that will fire a callback periodically (for live updates)
                    id="interval-component",
                    interval=self.update_interval * 1000,
                    n_intervals=0,
                    #max_intervals=10,
                ),
                dcc.Store( # Store heatmap that it can be used as an input in a callback
                    id="heatmap-data-store",
                    #data=self.heatmap, # initial assignment, NOT automatically updated
                    data=dict_heatmap['heatmap'], # initial assignment, NOT automatically updated
                ),
                dcc.Store( # Store added points that they can be used as an input in a callback
                    id="added_points-data-store",
                    data={'added_points': added_points, 'selected_point': selected_point_to_move},  # initial assignment, NOT automatically updated
                ),
                dcc.Store( # Store added lines
                    id="added_lines-data-store",
                    data=dict_lines,
                ),
                dcc.Store(id="timestamp-store-heatmap"),  # Hidden store for timing
            ],
            fluid=True,
            style={"height": "100vh"},
        )
        logging.debug(
            f"Dash layout created, update interval: {self.update_interval * 1000} ms"
        )
        self.add_callbacks()

    def add_callbacks(self):
        @self.app.callback(
            [Output("pause-button", "children"),
             Output("interval-component","disabled")],
            [Input("pause-button", "n_clicks")],
        )
        def toggle_pause(n_clicks):
            if n_clicks % 2 == 1:
                self.paused = True
                logging.debug(f"Paused: {self.paused}")
                return "Resume", True
            else:
                self.paused = False
                return "Pause", False
            # if n_clicks>0: # Button is initiated with n_clicks=0, only toggle if button was clicked!
            #     self.paused = not self.paused
            #     logging.debug(f"Paused: {self.paused}")
            # return "Resume" if self.paused else "Pause"

        @self.app.callback(
            Output("added_lines-data-store","data", allow_duplicate=True),
            Input("mode-selector","value"),
            State("added_lines-data-store","data"),
        )
        def update_when_mode_change(mode,dict_lines):
            if mode!='line':
                dict_lines['selected_indices'] = []  # reset selected indices for line in case of mode change
            return dict_lines


        # @self.app.callback(
        #     Output("timestamp-store-heatmap", "data"),
        #     Input("interval-component", "n_intervals"),
        #     prevent_initial_call=True
        #     )
        # def store_timestamp(n_intervals):
        #     return time.perf_counter()  # Store the timestamp before callback execution


        @self.app.callback(
            [
                Output("heatmap-data-store", "data"),  # Change heatmap data -> can use this to fire another callback for updating the figure!
                Output("iteration-output", "children"),
            ],
            [
                Input("interval-component", "n_intervals"),
            ],
            [State("heatmap-data-store", "data")], # new
            #State("timestamp-store-heatmap", "data")],
            blocking=True,
        )
        def update_heatmap(n_intervals,state_heatmap):#,start_time):
            #logging.debug(f"data: {self.figure.data}")
            # logging.debug(
            #     f"*** Dash callback {n_intervals} called at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
            # )
            #logging.debug(f"start_time: {start_time}")
            #processing_start = time.perf_counter()  # Start measuring execution time

            if self.paused or self._is_updating:
                logging.debug(
                    f"Updates paused at iteration {self.data_acquirer.num_acquisitions}"
                )
                return state_heatmap, f"Iteration: {self.data_acquirer.num_acquisitions}"

            # Increment iteration counter and update frontend
            #ts1 = time.time()
            updated_xarr = self.data_acquirer.update_data()
            dict_heatmap = xarray_to_heatmap(updated_xarr)
            #self.heatmap = dict_heatmap['heatmap']
            logging.debug(
                f"Updating heatmap, num_acquisitions: {self.data_acquirer.num_acquisitions}"
            )
            #ts2 = time.time()
            #logging.debug(f"Time to update heatmap: {ts2-ts1}s")

            #processing_end = time.perf_counter()  # End measuring execution time

            #execution_time = processing_end - processing_start  # Callback execution time
            #total_delay = processing_end - start_time  # Full delay from triggering event
            #logging.debug(f"(UPDATE HEATMAP) execution_time (Python processing): {execution_time}s")
            #logging.debug(f"(UPDATE HEATMAP) total_delay (full UI update time): {total_delay}s")

            return dict_heatmap, [f"Iteration: {self.data_acquirer.num_acquisitions}"]

        # Create states for all input components
        component_states = []
        for component_id in self.data_acquirer.get_component_ids():
            component_states += [
                State({"type": component_id, "index": ALL}, "id"),
                State({"type": component_id, "index": ALL}, "value"),
            ]

        @self.app.callback(
            [],
            [Input("update-button", "n_clicks")],
            component_states,
            blocking=True,
        )
        def update_params(n_update_clicks, *component_inputs):
            if n_update_clicks <= self._last_update_clicks:
                return

            params = {}
            component_inputs_iterator = iter(component_inputs)
            for component_id in self.data_acquirer.get_component_ids():
                ids, values = next(component_inputs_iterator), next(
                    component_inputs_iterator
                )
                params[component_id] = {
                    id["index"]: value for id, value in zip(ids, values)
                }

            logging.debug(f"Updating params: {params}")
            self.data_acquirer.update_parameters(params)

        @self.app.callback(
            Output("save-button", "children"),
            [Input("save-button", "n_clicks")],
        )
        def save(n_clicks):
            if n_clicks > self._last_save_clicks:
                self._last_save_clicks = n_clicks
                self.save()
                return "Saved!"
            return "Save"
        
        @self.app.callback(
            [Output("added_points-data-store","data",allow_duplicate=True),
             Output("added_lines-data-store","data",allow_duplicate=True)],
            [Input('live-heatmap','clickData')],
            [State("added_points-data-store","data"),
            State("mode-selector","value"),
            State("added_lines-data-store","data")],
        )
        def handle_clicks(clickData,dict_points,selected_mode,dict_lines):
            #logging.debug(f"added_points: {self.added_points}")
            #logging.debug(f"clickData: {clickData}")  # to check whether clickData is received

            # clickData is not None AND 'points' is not missing AND 'points' is not an empty list
            if clickData and 'points' in clickData and clickData['points']:

                point = clickData['points'][0]
                x_value, y_value = point['x'], point['y']
                added_points = dict_points['added_points']
                selected_point_to_move = dict_points['selected_point']
                added_lines = dict_lines['added_lines']
                selected_points_for_line = dict_lines['selected_indices']
                
                # MODE: Adding and moving points
                if selected_mode=="add":
                    # Select point: Already added point since 'z' is missing & first click --> MARK THIS POINT
                    if 'z' not in point and selected_point_to_move['move']==False:
                        selected_point_to_move['move'] = True
                        selected_point_to_move['index'] = self._find_index(x_value, y_value, added_points)

                    # Second click, i.e. last click was on an existing point --> MOVE selected point to clicked coordinates
                    elif selected_point_to_move['move']==True:
                        added_points['x'][selected_point_to_move['index']] = x_value  # Move point
                        added_points['y'][selected_point_to_move['index']] = y_value
                        selected_point_to_move['move'] = False
                        selected_point_to_move['index'] = None

                    # Add new point
                    elif 'z' in point:
                        # Generate point index
                        point_index = len(added_points['x'])

                        # Append new point
                        added_points['x'].append(x_value)
                        added_points['y'].append(y_value)
                        added_points['index'].append(point_index)
            
                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}
                
                # MODE: Deleting points
                elif selected_mode=="delete":
                    # If click on point
                    if 'z' not in point:
                        # Find index of clicked point
                        index = self._find_index(x_value, y_value, added_points)

                        # Delete lines that have clicked point as start or end point
                        #logging.debug(f"lines: {added_lines}")
                        i_start = [i for i, value in enumerate(added_lines['start_index']) if value==index]
                        i_end = [i for i, value in enumerate(added_lines['end_index']) if value==index]
                        i_to_remove = list(set(i_start) | set(i_end)) # combined list of line indices without duplicates
                        added_lines['start_index'] = [value for i, value in enumerate(added_lines['start_index']) if i not in i_to_remove] # delete lines
                        added_lines['end_index'] = [value for i, value in enumerate(added_lines['end_index']) if i not in i_to_remove]     # delete lines
                        #logging.debug(f"index of point to delete: {index}")
                        #logging.debug(f"i_start: {i_start}")
                        #logging.debug(f"i_end: {i_end}")
                        #logging.debug(f"indices to remove from list: {i_to_remove}")
                        #logging.debug(f"lines: {added_lines}")

                        # Delete point in added_points
                        del added_points['x'][index]
                        del added_points['y'][index]
                        
                        # Recompute indices of the added points
                        added_points['index'] = list(range(len(added_points['x'])))

                        # Recompute the line indices
                        added_lines['start_index'] = [value if value<index else value-1 for i, value in enumerate(added_lines['start_index'])]
                        added_lines['end_index'] = [value if value<index else value-1 for i, value in enumerate(added_lines['end_index'])]
                        #logging.debug(f"lines: {added_lines}")
                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines} # return points even if no point was deleted -> no problems with ordering of traces in figure
                
                elif selected_mode=='line': 
                    if 'z' not in point:
                        # Find index of clicked point
                        index = self._find_index(x_value, y_value, added_points)
                        # If point not yet selected: append index
                        if index not in selected_points_for_line:
                            selected_points_for_line.append(index)
                        # If two indices selected: Save line and delete selected points
                        if len(selected_points_for_line)==2:
                            if not self._duplicate_line(selected_points_for_line,added_lines): # Do not add duplicated line
                                added_lines['start_index'].append(selected_points_for_line[0])
                                added_lines['end_index'].append(selected_points_for_line[1])
                            #else:
                            #    logging.debug(f"Duplicated line!")
                            selected_points_for_line = []
                        #logging.debug(f"selected_points: {selected_points_for_line}")
                        #logging.debug(f"lines: {added_lines}")
                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}
            else:
                return dash.no_update
        
        @self.app.callback(
            [Output("added_points-data-store","data",allow_duplicate=True)],
            [Input('live-heatmap','hoverData')],
            [State("added_points-data-store","data"), 
             State("mode-selector","value")],
        )
        def handle_hovering(hoverData,dict_points,selected_mode):
            #logging.debug(f"hovering: {dict_points}")
            if selected_mode=="add":
                if hoverData and 'points' in hoverData and hoverData['points']:
                    added_points = dict_points['added_points']
                    selected_point_to_move = dict_points['selected_point']
                    if selected_point_to_move['move']==True: # only do something if point was selected
                        point = hoverData['points'][0]

                        # Move selected point
                        added_points['x'][selected_point_to_move['index']] = point['x']
                        added_points['y'][selected_point_to_move['index']] = point['y']

                        return {'added_points': added_points, 'selected_point': selected_point_to_move} 
                    else:
                        return dash.no_update
                else:
                    return dash.no_update       
            else:
                return dash.no_update
        
        @self.app.callback(
            [Output('live-heatmap','figure')],
            [Input("heatmap-data-store", "data"),
             Input("added_points-data-store","data"),
             Input("added_lines-data-store","data")],
            #blocking = True, # Not needed as heatmap data comes from dcc.Store
        )
        def update_figure(dict_heatmap,dict_points,dict_lines):
            ts1 = time.time()
            # OUTCOMMENT THIS PART, SUCH THAT POINTS CAN BE ADDED WHEN PAUSED
            # if self.paused or self._is_updating:
            #     logging.debug(
            #         f"Updates paused at iteration {self.data_acquirer.num_acquisitions}"
            #     )
            #     return self.figure
            added_points = dict_points['added_points']
            added_lines = dict_lines['added_lines']

            self.figure = self._generate_figure(dict_heatmap,added_points,added_lines)
            ts2 = time.time()
            logging.debug(f"Time to update figure: {ts2-ts1}s")
            #logging.debug(f"figure: {figure}")
            #logging.debug(f"type figure: {type(figure)}")
            return self.figure
    

    def _generate_figure(self, dict_heatmap, points, lines):
        # Prepare line_x and line_y lists for a single trace
        line_x = []
        line_y = []

        for start, end in zip(lines["start_index"], lines["end_index"]):
            line_x.extend([points["x"][start], points["x"][end], None])  # None separates line segments
            line_y.extend([points["y"][start], points["y"][end], None])
        #logging.debug(f"line_x: {line_x}")
        #logging.debug(f"line_y: {line_y}")

        # Create Figure
        figure = go.Figure()
        heatmap = dict_heatmap['heatmap']
        figure_titles = dict_heatmap['xy-titles']
        figure.add_trace(heatmap)
        figure.add_trace(go.Scatter(
            x=points['x'], 
            y=points['y'], 
            zorder=2,
            mode='markers + text', 
            marker=dict(color='white', size=10, line=dict(color='black', width=1)),
            text=[str(int(i) + 1) for i in points['index']], # Use indices to label the points
            textposition='top center',#'middle center', 
            textfont=dict(color='white'), #,shadow='1px 1px 10px white'), # offset-x | offset-y | blur-radius | color
            showlegend=False,
            name='',
        ))
        figure.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            zorder=1,
            mode="lines",
            line=dict(color="white"),
            name=""
        ))
        figure.update_layout(xaxis_title=figure_titles['xaxis_title'], 
                             yaxis_title=figure_titles['yaxis_title'], clickmode='event + select')
        return figure

    def _find_index(self, x_value, y_value, added_points):
        """ 
         Find index of a selected point (x_value, y_value) in the dict added_points
        """
        for i in range(len(added_points['x'])):
            if np.isclose(added_points['x'][i], x_value, atol=0.005) and np.isclose(added_points['y'][i], y_value, atol=0.005):
                return i  # Return the index if the point is found
        return None  # Return None if no matching point is found
    
    def _duplicate_line(self,selected_points_for_line, added_lines):
        index1 = selected_points_for_line[0]
        index2 = selected_points_for_line[1]
        found = any((index1 == index_start and index2 == index_end) or 
                    (index1 == index_end and index2 == index_start) for 
                    index_start, index_end in zip(added_lines["start_index"], added_lines["end_index"]))
        return found

    def run(self, debug: bool = True, use_reloader: bool = False):
        logging.debug("Starting Dash server")
        self.app.server.run(debug=debug, use_reloader=use_reloader)

    def save_data(self, idx: Optional[int] = None):
        """
        Save the current data to an HDF5 file.

        This method saves the current data from the data acquirer to an HDF5 file in the specified data save path.
        It automatically generates a unique filename by incrementing an index if not provided.

        Args:
            idx (Optional[int]): The index to use for the filename. If None, an available index is automatically determined.

        Returns:
            int: The index of the saved data file.

        Raises:
            ValueError: If the maximum number of data files (9999) has been reached.
            FileExistsError: If a file with the generated name already exists.

        Note:
            - The data save path is created if it doesn't exist.
            - The filename format is 'data_XXXX.h5', where XXXX is a four-digit index.
        """
        data_save_path = self.save_path / "data"
        logging.info(f"Attempting to save data to folder: {data_save_path}")

        if not data_save_path.exists():
            data_save_path.mkdir(parents=True)
            logging.info(f"Created directory: {data_save_path}")

        if idx is None:
            idx = 1
            while idx <= 9999 and (data_save_path / f"data_{idx}.h5").exists():
                idx += 1

        if idx > 9999:
            raise ValueError(
                "Maximum number of data files (9999) reached. Cannot save more."
            )

        filename = f"data_{idx}.h5"
        filepath = data_save_path / filename

        if filepath.exists():
            raise FileExistsError(f"File {filepath} already exists.")
        self.data_acquirer.data_array.to_netcdf(
            filepath
        )  # , engine="h5netcdf", format="NETCDF4")
        logging.info(f"Data saved successfully: {filepath}")
        logging.info("Data save operation completed.")
        return idx

    def save_image(self):
        """
        Save the current image to a file.

        This method saves the current figure as a PNG image in the specified image save path.
        It automatically generates a unique filename by incrementing an index.

        Returns:
            int: The index of the saved image file.

        Raises:
            ValueError: If the maximum number of screenshots (9999) has been reached.

        Note:
            - The image save path is created if it doesn't exist.
            - The filename format is 'data_image_XXXX.png', where XXXX is a four-digit index.
        """
        image_save_path = self.save_path / "images"
        logging.info(f"Attempting to save image to folder: {image_save_path}")
        if not image_save_path.exists():
            image_save_path.mkdir(parents=True)
            logging.info(f"Created directory: {image_save_path}")

        idx = 1
        while idx <= 9999 and (image_save_path / f"data_image_{idx}.png").exists():
            idx += 1
        if idx <= 9999:
            filename = f"data_image_{idx}.png"
            filepath = image_save_path / filename
            self.fig.write_image(filepath)
            logging.info(f"Image saved successfully: {filepath}")
        else:
            raise ValueError(
                "Maximum number of screenshots (9999) reached. Cannot save more."
            )
        logging.info("Image save operation completed.")

        return idx

    def save(self):
        """
        Save both the current image and data.

        This method saves the current figure as a PNG image and the current data as an HDF5 file.
        It uses the same index for both files to maintain consistency.

        Returns:
            int: The index of the saved files.

        Raises:
            ValueError: If the maximum number of files (9999) has been reached.

        Note:
            - The image is saved first, followed by the data.
            - If data saving fails due to a FileExistsError, a warning is logged instead of raising an exception.
        """
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
            logging.info(f"Created directory: {self.save_path}")

        # Save image first
        idx = self.save_image()

        # Attempt to save data with the same index
        try:
            self.save_data(idx)
        except FileExistsError:
            logging.warning(
                f"Data file with index {idx} already exists. Image saved, but data was not overwritten."
            )

        logging.info(f"Save operation completed with index: {idx}")
        return idx


class VideoMode(VideoModeApp):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VideoMode is deprecated, use VideoModeApp instead", DeprecationWarning
        )
        super().__init__(*args, **kwargs)
