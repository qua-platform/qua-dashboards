from datetime import datetime
import numpy as np
import os
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
import json
import re

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
        load_path: Union[str, Path] = "./video_mode_output/annotations",
        update_interval: float = 0.1,
    ):
        self.data_acquirer = data_acquirer
        self.save_path = Path(save_path)
        self.load_path = Path(load_path)
        self.paused = False
        self.annotation = False
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
        - Buttons for pausing/resuming data acquisition and saving data (images, data, annotations)
        - Display for the current iteration count
        - Input for setting the number of averages
        - Button for changing to the annotation mode (pauses data acquisition)
        - Radio buttons for choosing the annotation mode:
            -- Adding/moving points
            -- Adding lines
            -- Deleting points/lines
        - A button to clear all the annotations
        - A dropdown to select a previously saved annotation file, and a button for loading this file

        The layout is designed to be responsive and user-friendly, with aligned input fields
        and clear labeling. It uses a combination of Dash core components and HTML elements
        to create an intuitive interface for controlling and visualizing the data acquisition
        process.

        Returns:
            None: The method sets up the `self.app.layout` attribute but doesn't return anything.
        """

        dict_points = {'added_points': {'x': [], 'y': [], 'index': []}, 'selected_point': {'move': False, 'index': None}}
        dict_lines = {'selected_indices': [], 'added_lines': {'start_index':[], 'end_index':[]}}  # selected points for drawing a line
        dict_heatmap = xarray_to_heatmap(self.data_acquirer.data_array)
        radio_options = [
            {"label": "Adding and moving points (SHIFT+P)", "value": "point"},
            {"label": "Adding lines (SHIFT+L)", "value": "line"},
            {"label": "Deleting points and lines (SHIFT+D)", "value": "delete"},
            {"label": "Translate all points and lines (SHIFT+T)", "value": "translate-all"},
        ]
        dict_translation = {'translate': False, 'clicked_point': None}
        label_value = ['points']
        self.figure = self.generate_figure(dict_heatmap,dict_points,dict_lines,label_value)

        self.app.layout = dbc.Container(
            [                     # Store to hold key press events
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
                                            dbc.Button(
                                                "Annotation (off)",
                                                id="annotation-button",
                                                n_clicks=0,
                                                className="mt-3 mr-2",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "Clear all",
                                                id="clear-button",
                                                n_clicks=0,
                                                className="mt-3",
                                            ),
                                            width="auto",
                                        ),                                        
                                    ],
                                    className="mb-1",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Points and lines", className="card-title"),
                                                    dbc.RadioItems(
                                                        id='mode-selector',
                                                        options=radio_options,
                                                        value="point",
                                                    ),
                                                ]),
                                            ),
                                            width="auto"
                                        ),
                                        
                                    ],
                                    className="mr-2 mb-4"
                                ),
                                 dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Load annotation",
                                                id="load-button",
                                                n_clicks=0,
                                                className="mt-3",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    className="mb-1",
                                ),                               
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="annotation-file-dropdown",
                                                options=[{"label": f, "value": f} for f in self.list_json_files()],
                                                placeholder="Choose a file...",
                                            ),
                                            width=12,
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Compute compensation",
                                                id="compensation-button",
                                                n_clicks=0,
                                                className="mt-3",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    className="mb-1",
                                ), 
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Pre(
                                                id="compensation-data",
                                                className="bg-light p-3 rounded border",
                                                style={'overflowX': 'scroll'},
                                            ),
                                            width=12,
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                            ],
                            width=5,
                        ),
                        dbc.Col(
                            [   
                                dbc.Checklist(
                                    id="plot-labels-checklist",
                                    options=[
                                        {"label": "point labels", "value": "points"},  # For point labels
                                        #{"label": "line labels", "value": "lines"},   # If we also want to introduce line labels
                                    ],
                                    value=label_value,  # default selection
                                    inline=True,
                                    style={"marginLeft": "5rem"},
                                    className="mt-5 mb-0"
                                ),                               
                                dcc.Graph(
                                    id="live-heatmap",
                                    figure=self.figure,
                                    style={"aspect-ratio": "1 / 1"},
                                    #config={'scrollZoom': True},
                                ),
                            ],
                            width=7,
                        ),
                    ]
                ),
                dcc.Interval( # Component  that will fire a callback periodically (for live updates)
                    id="interval-component",
                    interval=self.update_interval * 1000,
                    n_intervals=0,
                ),
                dcc.Store( # Store heatmap that it can be used as an input in a callback
                    id="heatmap-data-store",
                    data=dict_heatmap, # initial assignment, NOT automatically updated
                ),
                dcc.Store( # Store added points that they can be used as an input in a callback
                    id="added_points-data-store",
                    data=dict_points,
                ),
                dcc.Store( # Store added lines
                    id="added_lines-data-store",
                    data=dict_lines,
                ),
                dcc.Store( # Trigger for updating the heatmap!
                    id="trigger-update-once",
                    data=False,
                ),  
                dcc.Store( # Store the data that is needed to translate all points and lines
                    id="translation-data-store",
                    data=dict_translation,
                )
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
            Output("pause-button", "children"),
            [Input("pause-button", "n_clicks")],
        )
        def toggle_pause(n_clicks):
            #logging.debug(f"Callback toggle_pause triggered!")
            if n_clicks>0: # Button is initiated with n_clicks=0, only toggle if button was clicked!
                self.paused = not self.paused
                logging.debug(f"Paused: {self.paused}")
            return "Resume" if self.paused else "Pause"
        
        @self.app.callback(
            [Output("annotation-button", "children"),
             Output("interval-component","disabled")],
            [Input("annotation-button", "n_clicks")],
        )
        def toggle_annotation(n_clicks):
            '''
            Change between annotation mode (pausing the measurements), where points and lines 
            can be added to the figure, and the measurement mode.
            '''
            #logging.debug(f"Callback toggle_annotation triggered!")
            if n_clicks % 2 == 1:
                self.annotation = True
                logging.debug(f"Annotation mode: {self.annotation}")
                return "Annotation (on)", True
            else:
                self.annotation = False
                return "Annotation (off)", False
            
        @self.app.callback(
            [Output("added_lines-data-store","data"),
            Output("added_points-data-store","data")],
            [Input("clear-button", "n_clicks")],
        )
        def clear_annotation_data(n_clicks):
            '''
            Clear all annotation data, i.e. all points and lines
            '''
            #logging.debug(f"Callback clear_annotation_data triggered!")
            if n_clicks>0: # Button is initiated with n_clicks=0, only clear data if button was clicked!
                dict_points = {'added_points': {'x': [], 'y': [], 'index': []}, 'selected_point':  {'move': False, 'index': None}}
                dict_lines = {'selected_indices': [], 'added_lines': {'start_index':[], 'end_index':[]}}  # selected points for drawing a line
                return dict_lines, dict_points
            else:
                return dash.no_update, dash.no_update

        @self.app.callback(
            Output("added_lines-data-store","data", allow_duplicate=True),
            Input("mode-selector","value"),
            State("added_lines-data-store","data"),
        )
        def update_mode_change(mode,dict_lines):
            '''
            Reset selected indices for a line to None when changing between points and lines (radiobuttons)
            e.g. changing from "Adding lines" to "Deleting points and lines".
            This is to prevent that there is already a first point of a new line selected, when changing
            back to the "Adding lines" mode.
            '''
            if mode!='line':
                dict_lines['selected_indices'] = []  # reset selected indices for line in case of mode change
            return dict_lines
        
        # Use keyboard shortcuts to switch between radio items (modes)
        '''
        Introduce keyboard shortcuts for the point and lines modes (radiobuttons). 
        This is done via Javascript.
        '''
        self.app.clientside_callback(
            """
            function(value) {
                document.addEventListener("keydown", function(event) {
                    if (event.shiftKey) { // Shift key is pressed
                        let key = event.key.toLowerCase();  // Normalize to lowercase
                        let mapping = {"p": "point", "l": "line", "d": "delete", "t": "translate-all"};

                        if (mapping.hasOwnProperty(key)) {
                            event.preventDefault();  // Prevent default browser actions
                            //console.log("Shortcut selected:", mapping[key]);
                            
                            // Update the RadioItems value
                            dash_clientside.set_props("mode-selector", {value: mapping[key]})                         
                        }
                    }
                });
                
                return window.dash_clientside.no_update;
            }
            """,
            Output("mode-selector", "value"),
            Input("mode-selector", "value")
        )

        @self.app.callback(
            [
                Output("heatmap-data-store", "data"),  # Change heatmap data -> can use this to fire another callback for updating the figure!
                Output("iteration-output", "children"),
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("trigger-update-once","data"),  # Parameters changed 
            ],
            [State("heatmap-data-store", "data")], # new
            blocking=True,
        )
        def update_heatmap(n_intervals,trigger_data,state_heatmap):
            #logging.debug(f"Callback update_heatmap triggered!")
            # logging.debug(
            #     f"*** Dash callback {n_intervals} called at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
            # )

            if self.paused or self._is_updating:
                logging.debug(
                    f"Updates paused at iteration {self.data_acquirer.num_acquisitions}"
                )
                return state_heatmap, f"Iteration: {self.data_acquirer.num_acquisitions}"

            updated_xarr = self.data_acquirer.update_data()
            dict_heatmap = xarray_to_heatmap(updated_xarr)

            logging.debug(
                f"Updating heatmap, num_acquisitions: {self.data_acquirer.num_acquisitions}"
            )
            return dict_heatmap, [f"Iteration: {self.data_acquirer.num_acquisitions}"]

        # Create states for all input components
        component_states = []
        for component_id in self.data_acquirer.get_component_ids():
            component_states += [
                State({"type": component_id, "index": ALL}, "id"),
                State({"type": component_id, "index": ALL}, "value"),
            ]

        @self.app.callback(
            [Output("trigger-update-once",'data')], # Ensure that the figure is updated also in annotation mode!
            [Input("update-button", "n_clicks")],
            State("trigger-update-once",'data'),
            component_states,
            blocking=True,
        )
        def update_params(n_update_clicks, trigger_data, *component_inputs):
            #logging.debug(f"Callback update_params triggered!")
            if n_update_clicks <= self._last_update_clicks:
                return dash.no_update

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
            return not trigger_data  # Trigger for updating the figure (flip False -> True -> False etc.)

        @self.app.callback(
            Output("save-button", "children"),
            [Input("save-button", "n_clicks")],
            [State("added_points-data-store","data"),
            State("added_lines-data-store","data")],
        )
        def save(n_clicks,dict_points,dict_lines):
            '''
            Save the current image, data and annotation
            '''
            #logging.debug(f"Callback save triggered!")
            if n_clicks > self._last_save_clicks:
                points = dict_points['added_points']
                lines  = dict_lines['added_lines']
                
                self._last_save_clicks = n_clicks
                self.save(points,lines)
                return "Saved!"
            return "Save"
        
        @self.app.callback(
            [Output("added_points-data-store","data",allow_duplicate=True),
            Output("added_lines-data-store","data",allow_duplicate=True)],
            Input("load-button", "n_clicks"),
            [State("annotation-file-dropdown", "value"),
            State("added_points-data-store","data"),
            State("added_lines-data-store","data")],   
        )
        def load_and_validate_annotation(n_clicks, selected_file, dict_points, dict_lines):
            '''
            Load a previously saved annotation file, which is selected in the dropdown.
            A ValueError is raised if the annotation file is not valid.
            '''
            # logging.debug(f"Callback load_and_validate_annotation triggered!")
            if n_clicks > 0:
                if not selected_file:
                    logging.info("No annotation file selected.")
                    return dash.no_update, dash.no_update
                file_path = os.path.join(self.load_path, selected_file)
                try:
                    data = self.load_and_validate_json(file_path)
                    dict_points['added_points'] = data['points']
                    dict_points['added_points']['index'] = list(range(0, len(data['points']['x'])))
                    dict_points['selected_point'] = {'move': False, 'index': None}
                    dict_lines['added_lines'] = data['lines']
                    dict_lines['selected_indices'] = []
                    logging.info(f"JSON is valid and loaded!")
                    return dict_points, dict_lines
                except ValueError as e:
                    logging.info(f"JSON error: {e}")
                    return dash.no_update, dash.no_update
            else:
                return dash.no_update, dash.no_update
        
        @self.app.callback(
            [Output("added_points-data-store","data",allow_duplicate=True),
             Output("added_lines-data-store","data",allow_duplicate=True),
             Output('live-heatmap','clickData'), # Reset to None, otherwise repeated clicks at the same coordinates in the graph are not possible in Dash!
             Output("translation-data-store","data",allow_duplicate=True)],
            [Input('live-heatmap','clickData')],
            [State("added_points-data-store","data"),
            State("mode-selector","value"),
            State("added_lines-data-store","data"),
            State("translation-data-store","data")],
        )
        def handle_clicks(clickData,dict_points,selected_mode,dict_lines,dict_translation):
            '''
            Handle clicks on the figure dependent on the different points and lines modes
            - MODE Adding and moving points:
              Clicking on the figure adds a new point. An existing point can be moved after having clicked on it.
            - MODE Adding lines:
              Clicking subsequentially on two existing points adds a line between the points.
            - MODE Deleting points and lines:
              Clicking on an existing point or line deletes this point or line.
            - MODE Translate all points and lines:
              After having clicked anywhere on the figure, all points and lines can be translated together.
            '''
            #logging.debug("Callback handle_clicks triggered!")

            # clickData is not None AND 'points' is not missing AND 'points' is not an empty list
            if clickData and 'points' in clickData and clickData['points']:

                point = clickData['points'][0]
                x_value, y_value = point['x'], point['y']
                added_points = dict_points['added_points']
                selected_point_to_move = dict_points['selected_point']
                added_lines = dict_lines['added_lines']
                selected_points_for_line = dict_lines['selected_indices']
                
                # MODE: Adding and moving points (SHIFT + P)
                if selected_mode=="point":
                    # Select point: Already added point since 'z' is missing & first click --> MARK THIS POINT
                    if 'z' not in point and selected_point_to_move['move']==False:
                        selected_point_to_move['move'] = True
                        index = self.find_index(x_value, y_value, added_points)
                        selected_point_to_move['index'] = index

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
            
                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}, None, dash.no_update
                
                # MODE: Deleting points / lines (SHIFT + D)
                elif selected_mode=="delete":
                    # If click on point
                    if 'z' not in point:
                        # Find index of clicked point
                        index = self.find_index(x_value, y_value, added_points)

                        # Delete lines that have clicked point as start or end point
                        i_start = [i for i, value in enumerate(added_lines['start_index']) if value==index]
                        i_end = [i for i, value in enumerate(added_lines['end_index']) if value==index]
                        i_to_remove = list(set(i_start) | set(i_end)) # combined list of line indices without duplicates
                        added_lines['start_index'] = [value for i, value in enumerate(added_lines['start_index']) if i not in i_to_remove] # delete lines
                        added_lines['end_index'] = [value for i, value in enumerate(added_lines['end_index']) if i not in i_to_remove]     # delete lines

                        # Delete point in added_points
                        del added_points['x'][index]
                        del added_points['y'][index]
                        
                        # Recompute indices of the added points
                        added_points['index'] = list(range(len(added_points['x'])))

                        # Recompute the line indices
                        added_lines['start_index'] = [value if value<index else value-1 for i, value in enumerate(added_lines['start_index'])]
                        added_lines['end_index'] = [value if value<index else value-1 for i, value in enumerate(added_lines['end_index'])]

                    else: # click on not existing point
                        # compute distances of the selected coordinates to all lines
                        distances = self.distance_to_lines(x_value, y_value, added_points, added_lines)
                        # if distances not empty
                        if distances:
                            min_index = np.argmin(distances) # index of smallest distance
                            # if min distance smaller than the minimal axis span divided axis points (my tolerance criterium)
                            if distances[min_index] < 2*min(self.data_acquirer.x_axis.span/self.data_acquirer.x_axis.points, self.data_acquirer.y_axis.span/self.data_acquirer.y_axis.points):
                                del added_lines['start_index'][min_index]
                                del added_lines['end_index'][min_index]

                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}, None, dash.no_update # return points even if no point was deleted -> no problems with ordering of traces in figure,
                
                # MODE: Adding lines (SHIFT + L)
                elif selected_mode=='line': 
                    if 'z' not in point:
                        # Find index of clicked point
                        index = self.find_index(x_value, y_value, added_points)
                        #logging.debug(f"Clicked at ({x_value}, {y_value}), found index: {index}")
                        # If point not yet selected: append index
                        if index not in selected_points_for_line:
                            selected_points_for_line.append(index)
                        else:
                            selected_points_for_line.remove(index) # unselect point
                        # If two indices selected: Save line and delete selected points
                        if len(selected_points_for_line)==2:
                            if not self.duplicate_line(selected_points_for_line,added_lines): # Do not add duplicated line
                                added_lines['start_index'].append(selected_points_for_line[0])
                                added_lines['end_index'].append(selected_points_for_line[1])
                            selected_points_for_line.clear()
                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}, None, dash.no_update
                
                # MODE: Translate all points and lines (SHIFT + T)
                elif selected_mode=="translate-all":
                    # After first click: True (turn translation mode on); after second click: False (turn translation mode off)
                    dict_translation['translate'] = not dict_translation['translate']

                    # Add the clicked point when translation mode is turned on
                    if dict_translation['translate'] == True: 
                        dict_translation["clicked_point"] = clickData['points'][0]  
                    # Remove the clicked point when translation mode is turned off
                    else:
                        dict_translation["clicked_point"] = None  

                    return {'added_points': added_points, 'selected_point': selected_point_to_move}, {'selected_indices': selected_points_for_line, 'added_lines': added_lines}, None, dict_translation
            else:
                return dash.no_update
        
        @self.app.callback(
            [Output("added_points-data-store","data",allow_duplicate=True),
             Output("translation-data-store","data",allow_duplicate=True)],
            [Input('live-heatmap','hoverData')],
            [State("added_points-data-store","data"),
             State("mode-selector","value"),
             State("translation-data-store","data")],
        )
        def handle_hovering(hoverData,dict_points,selected_mode,dict_translate):
            '''
            Handle hovering over the figure. 
            '''
            #logging.debug(f"Callback handle_hovering triggered!")
            # MODE: Adding and moving points (SHIFT + P)
            if selected_mode=="point":
                if hoverData and 'points' in hoverData and hoverData['points']:
                    added_points = dict_points['added_points']
                    selected_point_to_move = dict_points['selected_point']
                    if selected_point_to_move['move']==True: # only do something if point was selected
                        point = hoverData['points'][0]

                        # Move selected point
                        added_points['x'][selected_point_to_move['index']] = point['x']
                        added_points['y'][selected_point_to_move['index']] = point['y']

                        return {'added_points': added_points, 'selected_point': selected_point_to_move}, dict_translate
                    else:
                        return dash.no_update, dash.no_update
                else:
                    return dash.no_update, dash.no_update

            # MODE: Translate all points and lines (SHIFT + T)    
            elif selected_mode=="translate-all":
                if dict_translate["translate"]==True:
                    if hoverData and 'points' in hoverData and hoverData['points']:
                        # Compute the direction vector (hover point - clicked point)
                        dx = hoverData['points'][0]['x'] - dict_translate['clicked_point']['x']
                        dy = hoverData['points'][0]['y'] - dict_translate['clicked_point']['y']

                        # Update the added points: added points + direction vector
                        dict_points['added_points']['x'] = [x + dx for x in dict_points['added_points']['x']]
                        dict_points['added_points']['y'] = [y + dy for y in dict_points['added_points']['y']]

                        # Update the clicked point
                        dict_translate['clicked_point']['x'] = dict_translate['clicked_point']['x'] + dx
                        dict_translate['clicked_point']['y'] = dict_translate['clicked_point']['y'] + dy                     

                        return dict_points, dict_translate
                    else:
                        dash.no_update, dash.no_update
                else:
                    return dash.no_update, dash.no_update
            else:
                return dash.no_update, dash.no_update
        
        @self.app.callback(
            [Output('live-heatmap','figure')],
            [Input("heatmap-data-store", "data"),
             Input("added_points-data-store","data"),
             Input("added_lines-data-store","data"), 
             Input("plot-labels-checklist","value"), ]
        )
        def update_figure(dict_heatmap,dict_points,dict_lines,labels_list):
            '''
            Update the figure whenever the heatmap or annotation data changes
            '''
            #logging.debug(f"Callback update_figure triggered!")
            self.figure = self.generate_figure(dict_heatmap,dict_points,dict_lines,labels_list)
            return self.figure

        @self.app.callback(
            Output("compensation-data","children"),
            Input("compensation-button","n_clicks"),
            [State("added_points-data-store","data"),
             State("added_lines-data-store","data"),]
        )
        def compute_compensation(n_clicks,dict_points,dict_lines):
            if n_clicks>0:
                logging.debug(f"Callback compute_compensation triggered!")
                direction,slope,A_inv,Transformation = self.compensation(dict_points['added_points'],dict_lines['added_lines'])
                logging.debug(f"direction: {direction}")
                logging.debug(f"slopes: {slope}")
                logging.debug(f"Inverse of transformation matrix: {A_inv}")
                logging.debug(f"Test (A*A_inv): {np.dot(A_inv, Transformation)}")
                #logging.debug(f"slopes: {direction['dy'][0]/direction['dx'][0]} and {direction['dy'][1]/direction['dx'][1]}")
                return json.dumps(Transformation.tolist(), indent=2)#dash.no_update  # TO DO: output matrix A
            else:
                return dash.no_update

    def compensation(self,added_points,added_lines):
        # TO DO: ensure added_lines has exactly two lines added
        # Compute direction for each line
        direction = {'dx' : [], 'dy' : []}
        for start_index, end_index in zip(added_lines['start_index'], added_lines['end_index']): # loop through lines
            # Line start point (x1,y1) and end point (x2,y2)
            x1, y1 = added_points['x'][start_index], added_points['y'][start_index]
            x2, y2 = added_points['x'][end_index], added_points['y'][end_index]
            
            # Direction vector from start to end point
            direction["dx"].append(x2-x1)
            direction["dy"].append(y2-y1)

        # Compute slope for each line
        slope = [dy / dx if dx != 0 else None for dx, dy in zip(direction['dx'], direction['dy'])]

        # Compute transformation matrix A
        A_inv = np.zeros((2,2))
        if abs(slope[0]) < abs(slope[1]):  # First line more aligned with x-axis, second line more aligned with y-axis
            A_inv[0,0] = - direction['dy'][0]
            A_inv[0,1] =   direction["dx"][0]
            A_inv[1,0] = - direction['dy'][1]
            A_inv[1,1] =   direction["dx"][1]
        else: # First line more aligned with y-axis, second line more aligned with x-axis
            A_inv[0,0] = - direction['dy'][1]
            A_inv[0,1] =   direction["dx"][1]
            A_inv[1,0] = - direction['dy'][0]
            A_inv[1,1] =   direction["dx"][0]

        Transformation = np.linalg.inv(A_inv)            

        return direction,slope,A_inv,Transformation
        # TO DO: Figure out whether first or second line has larger slope dy/dx. 
        #        Flatter line has parameters a, steeper line has parameters b
        # TO DO: Compute A

    def distance_to_lines(self, x, y, added_points, added_lines):
        '''
        Compute the distance of a clicked point to all added lines. The distance is defined as the Euclidean distance
        of the clicked point to the closest point on a given line - this is either the perpendicular distance to the
        line segment or the distance to the start/end point of a given line.
        Input: Clicked point with coordinates (x,y), all added points and all added lines
        Output: List of distances
        '''
        distances = []
        if added_lines['start_index']: # added lines not empty!
            for start_index, end_index in zip(added_lines['start_index'], added_lines['end_index']): # loop through lines
                # Line start point (x1,y1) and end point (x2,y2)
                x1, y1 = added_points['x'][start_index], added_points['y'][start_index]
                x2, y2 = added_points['x'][end_index], added_points['y'][end_index]
                
                # Direction vector from start to end point
                dx, dy = x2-x1, y2-y1
                d_squared = dx**2 + dy**2 # squared segment/line length

                # Compute the projection of (x,y) onto the (infinite) line
                t = ((x-x1)*dx + (y-y1)*dy)/d_squared        
                
                # Check whether projection is between start and end point
                if 0<=t<=1: # On the segment: Use perpendicular distance
                    x_proj, y_proj = x1 + t*dx, y1 + t*dy
                    distances.append(np.sqrt((x-x_proj)**2 + (y-y_proj)**2))
                else: # Outside segment: Use distance to closest end point
                    dist_start = np.sqrt((x-x1)**2 + (y-y1)**2)
                    dist_end   = np.sqrt((x-x2)**2 + (y-y2)**2)
                    distances.append(min(dist_start,dist_end))
        return distances

    def duplicate_line(self,selected_points_for_line, added_lines):
        '''
        Check whether a line was added before.
        Input: selected points to form a line, already added lines
        Output: True (line already added), False (otherwise)
        '''
        index1 = selected_points_for_line[0]
        index2 = selected_points_for_line[1]
        found = any((index1 == index_start and index2 == index_end) or 
                    (index1 == index_end and index2 == index_start) for 
                    index_start, index_end in zip(added_lines["start_index"], added_lines["end_index"]))
        return found

    def find_index(self, x_value, y_value, added_points):
        """ 
         Find the index of a selected point (x_value, y_value) from the figure in the dict added_points.
         An index can only be found if the selected point is "close enough" (measure by some tolerance) 
         to a previously added point. The tolerance is defined as axis.span / axis.points.
        """
        for i in range(len(added_points['x'])):
            if (np.isclose(added_points['x'][i], x_value, atol=self.data_acquirer.x_axis.span/self.data_acquirer.x_axis.points) 
                and np.isclose(added_points['y'][i], y_value, atol=self.data_acquirer.y_axis.span/self.data_acquirer.y_axis.points)):
                return i  # Return the index if the point is found
        return None  # Return None if no matching point is found

    def generate_figure(self, dict_heatmap, dict_points, dict_lines, labels_list):
        points = dict_points['added_points']
        lines = dict_lines['added_lines']
        selected_point_to_move = dict_points['selected_point']
        selected_points_for_line = set(dict_lines['selected_indices'])

        # Prepare line_x and line_y lists for a single trace
        line_x = []
        line_y = []

        for start, end in zip(lines["start_index"], lines["end_index"]):
            line_x.extend([points["x"][start], points["x"][end], None])  # None separates line segments
            line_y.extend([points["y"][start], points["y"][end], None])

        # Generate size list: Points selected to move or for lines appear larger
        marked_index = selected_point_to_move.get('index') 
        sizes = [13 if i==marked_index or i in selected_points_for_line else 10 for i in range(len(points['x']))]

        # Create Figure
        figure = go.Figure()
        heatmap = dict_heatmap['heatmap']
        figure_titles = dict_heatmap['xy-titles']
        figure.add_trace(heatmap)
        figure.add_trace(go.Scatter(
            x=points['x'], 
            y=points['y'], 
            zorder=2, # top layer, heatmap has zorder=0 (background)
            mode='markers + text', 
            marker=dict(color="white", size=list(sizes), line=dict(color='black', width=1), opacity=1.0),
            text=[str(int(i) + 1) for i in points['index']] if 'points' in labels_list else None, # Use indices to label the points
            textposition='top center',
            textfont=dict(color='white'),
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
                             yaxis_title=figure_titles['yaxis_title'], 
                             clickmode='event + select',
                             margin=dict(t=30,b=20,l=20,r=20) # top margin such that tool bar can be shown, but gap to checklist smaller
                             )
        return figure    

    def list_json_files(self):
        """ 
        List all JSON files in the annotations data folder 
        """
        return [f for f in os.listdir(self.load_path) if f.endswith(".json")]
    
    def load_and_validate_json(self, file_path):
        '''
        Open a json-file and validate it. ValueErrors are raised if
        - the file has in invalid json format
        - the file cannot be open or read
        - the data is not a dictionary
        - the data does not contain the required keys
        The points are validated. ValueErrors are raised if
        - the point data is not a dictionary
        - the point data does not contain the required keys 'x' and 'y' (coordinates), and these are not lists
        - the point data coordinates are not of the same length or are not numbers
        The lines are validated. ValueErrors are raised if
        - the line data is not a dictionary
        - the line data does not contain the required keys 'start_index' and 'end_index' (coordinates), and these are not lists
        - the line data indices are not of the same length or are not integers

        Input: file_path
        Output: dictionary, which contains the points and the lines of a previously saved annotation file
        '''
        try:
            with open(file_path,"r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to open or read the file: {e}")
        
        # Top-level check
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON structure must be a dictionary.")

        required_top_keys = {"points", "lines"}
        if not required_top_keys.issubset(data):
            raise ValueError(f"Missing top-level keys: {required_top_keys - data.keys()}")

        # Validate 'points'
        points = data["points"]
        if not isinstance(points, dict):
            raise ValueError("'points' must be a dictionary.")

        if "x" not in points or "y" not in points:
            raise ValueError("Missing 'x' or 'y' in 'points'.")

        if not isinstance(points["x"], list) or not isinstance(points["y"], list):
            raise ValueError("'x' and 'y' must be lists.")

        if len(points["x"]) != len(points["y"]):
            raise ValueError("Length of 'x' and 'y' must be equal.")

        if not all(isinstance(num, (int, float)) for num in points["x"] + points["y"]):
            raise ValueError("All elements in 'x' and 'y' must be numbers.")

        # Validate 'lines'
        lines = data["lines"]
        if not isinstance(lines, dict):
            raise ValueError("'lines' must be a dictionary.")

        if "start_index" not in lines or "end_index" not in lines:
            raise ValueError("Missing 'start_index' or 'end_index' in 'lines'.")

        if not isinstance(lines["start_index"], list) or not isinstance(lines["end_index"], list):
            raise ValueError("'start_index' and 'end_index' must be lists.")

        if len(lines["start_index"]) != len(lines["end_index"]):
            raise ValueError("Length of 'start_index' and 'end_index' must be equal.")

        if not all(isinstance(i, int) for i in lines["start_index"] + lines["end_index"]):
            raise ValueError("All elements in 'start_index' and 'end_index' must be integers.")

        return data

    def run(self, debug: bool = True, use_reloader: bool = False):
        logging.debug("Starting Dash server")
        self.app.server.run(debug=debug, use_reloader=use_reloader)

    def save(self,points,lines):
        """
        Save the current image, data as well as the annotation data.

        This method saves the current figure as a PNG image, the current data as an HDF5 file and the 
        current annotation data (points and lines) as a json file.
        It uses the same index for all files to maintain consistency.

        Returns:
            int: The index of the saved files.

        Raises:
            ValueError: If the maximum number of files (9999) has been reached.           

        Note:
            - The index is computed from the max existing index over all folders plus one
            - The image is saved first, followed by the data. Finally, the annotation data is saved.
        """
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
            logging.info(f"Created directory: {self.save_path}")

        # Information on all data saving folders and file extensions
        dict_saving = {'images': {'folder_name': 'images', 'ending': '.png'}, 
                       'data': {'folder_name': 'data', 'ending': '.h5'}, 
                       'annotations': {'folder_name': 'annotations', 'ending': '.json'}}

        # Find index (max index from all folders + 1)
        max_idx = []
        for key in dict_saving: # iterate over all saving folders
            save_path = self.save_path / dict_saving[key]['folder_name']
            file_extension = dict_saving[key]['ending']

            if not save_path.exists():
                save_path.mkdir(parents=True)
                logging.info(f"Created directory: {save_path}")

            pattern = re.compile(rf"(\d+){re.escape(file_extension)}$")  # regular expression pattern: filenname ends with number (one or more digits) followed by file extension
            max_i = 0  # variable to track max index
            for file in save_path.iterdir(): # iterate over all files in the folder
                if file.is_file():  # skip subdirectories
                    match = pattern.search(file.name)
                    if match:
                        i = int(match.group(1))  # converts the numeric part of the match to an integer
                        max_i = max(max_i,i)
            max_idx.append(max_i)
        idx = max(max_idx) + 1

        # Save all data
        if idx <= 9999:
            self.save_image(idx,dict_saving["images"])
            self.save_data(idx,dict_saving["data"])
            self.save_annotation(points,lines,idx,dict_saving["annotations"])
        else:
            raise ValueError(
                "Maximum number of screenshots (9999) reached. Cannot save more."
            )

        logging.info(f"Save operation completed with index: {idx}")
        return idx        

    def save_annotation(self,points,lines,idx,dict_saving_annotations):
        """
        Save the current annotation data (points and lines) to a json file.

        This method saves the current annotation data drawn in the graph to a json file in the specified data save path.

        Args:
            points:                  Added points
            lines:                   Added lines
            idx:                     The index to use for the filename.
            dict_saving_annotations: Dictionary containing the saving folder and file extension

        Notes:
            - The filename format is 'annotation_XXXX.json', where XXXX is a four-digit index.
            - The annotation data is not saved if there are no points and lines.
        """        
        annotation_save_path = self.save_path / dict_saving_annotations['folder_name']
        filename = f"annotation_{idx}{dict_saving_annotations['ending']}"
        filepath = annotation_save_path / filename    

        # if filepath.exists():
        #     raise FileExistsError(f"File {filepath} already exists.")
        
        if (not points.get("x")) and (not lines.get("start_index")):
            logging.info("There is no annotation data to save.")
        else:            
            data = {
                "points": {k: points[k] for k in ["x", "y"]},
                "lines": lines,
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

            logging.info(f"Annotation data saved successfully: {filepath}")
        logging.info("Annotation data save operation completed.")

    def save_data(self, idx, dict_saving_data):
        """
        Save the current data to an HDF5 file.

        This method saves the current data from the data acquirer to an HDF5 file in the specified data save path.

        Args:
            idx:                     The index to use for the filename.
            dict_saving_annotations: Dictionary containing the saving folder and file extension            

        Notes:
            - The filename format is 'data_XXXX.h5', where XXXX is a four-digit index.
        """
        data_save_path = self.save_path / dict_saving_data['folder_name']
        filename = f"data_{idx}{dict_saving_data['ending']}"
        filepath = data_save_path / filename

        # if filepath.exists():
        #     raise FileExistsError(f"File {filepath} already exists.")
 
        self.data_acquirer.data_array.to_netcdf(
            filepath
        )  # , engine="h5netcdf", format="NETCDF4")

        logging.info(f"Data saved successfully: {filepath}")
        logging.info("Data save operation completed.")

    def save_image(self,idx,dict_saving_images):
        """
        Save the current image to a file.

        This method saves the current figure as a PNG image in the specified image save path.

        Note:
            - The filename format is 'data_image_XXXX.png', where XXXX is a four-digit index.
        """
        image_save_path = self.save_path / dict_saving_images['folder_name']
        filename = f"data_image_{idx}{dict_saving_images['ending']}"
        filepath = image_save_path / filename

        self.figure.write_image(filepath)
        
        logging.info(f"Image saved successfully: {filepath}")
        logging.info("Image save operation completed.")


class VideoMode(VideoModeApp):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VideoMode is deprecated, use VideoModeApp instead", DeprecationWarning
        )
        super().__init__(*args, **kwargs)
