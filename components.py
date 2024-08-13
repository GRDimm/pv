import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid the need for a display server
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances
import shap

from utils import *

class InputTextComponent:
    def __init__(self, full_data, dataset_config):
        self.data = full_data
        self.dataset_config = dataset_config 

    def set(self, full_data, dataset_config):
        self.data = full_data
        self.dataset_config = dataset_config

    def check_active(self):
        return not self.data is None

    def layout(self):
        """Defines the layout of the input text component."""
        return html.Div([
            html.Div([
                html.H4("ID Search"),
                html.Hr(className='HR'),
                dcc.Input(id='input-id', type='text', placeholder="Filter IDs", value="", className="dark-mode-search-input"),
                dcc.Dropdown(
                    id='operation_id',
                    #{'label': rid, 'value': rid} for rid in self.data[self.dataset_config["columns_config"]["id"]]][:int(self.dataset_config["max_dropdown_elements"])
                    options=[],
                    placeholder="Enter or select ID from filtered",
                    searchable=True,
                    className='dark-mode-dropdown'  # Apply the dark mode class
                )
            ]),
            html.Div(
                id='operation_details',
                style={'overflowY': 'auto', 'marginTop': '5px'}
            )
        ], className='cell')

    def component_callbacks(self, app):
        """Defines callbacks for updating the operation details based on the selected ID."""
        @app.callback(
            Output('operation_details', 'children'),
            [Input('operation_id', 'value')]
        )
        def update_operation_details(operation_id):
            if self.check_active():
                if operation_id is None:
                    return "Error: Please enter a valid operation ID."

                # Search for the row data based on the operation_id
                row = self.data[self.data[self.dataset_config["columns_config"]["id"]] == operation_id]
                
                if row.empty:
                    return f"Error: Operation ID {operation_id} not found."
                
                # Convert row data to dictionary and format it for display
                row_data = row.iloc[0].drop(self.dataset_config["columns_config"]["id"]).to_dict()
                details = [html.P(f"{key}: {value}") for key, value in row_data.items() if key in self.dataset_config["display_fields"]]
                
                return html.Div(details)
            else:
                return None

        def fetch_options(search_term):
            if not search_term is None:
                # Assuming 'id_column' is the column in the DataFrame you want to search
                id_column = self.dataset_config["columns_config"]["id"]
                
                # Filter the DataFrame based on the search_term
                filtered_df = self.data[self.data[id_column].str.contains(search_term, case=False, na=False)]
                
                # Extract the relevant column as a list of options
                options = filtered_df[id_column].head(int(self.dataset_config["max_dropdown_elements"])).tolist()
                
                return options
            return []

        @app.callback(
            Output('operation_id', 'options'),
            [Input('input-id', 'value')]
        )
        def update_dropdown(search_term):
            if self.check_active():
                options = fetch_options(search_term)
                return [{'label': option, 'value': option} for option in options]
            else:
                return []

class FeatureImportanceComponent:
    def __init__(self, model, dataset_config):
        self.model = model
        self.dataset_config= dataset_config

    def set(self, model, dataset_config):
        self.model = model
        self.dataset_config = dataset_config

    def check_active(self):
        return not self.model is None

    def layout(self):
        """Defines the layout of the feature importance component."""
        return html.Div([
            html.Div([
                html.H4("Model Feature Importance"),
                dcc.Input(
                    id='num_features_input',
                    type='number',
                    value=10,  # Default value for the number of features to display
                    min=1,
                    step=1,
                    className='small-input dark-mode-input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0'
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="feature_importance_graph", className="graph-scale", config={'responsive': True}),
                className="auto-size-graph-container"
            )
        ], className='cell')

    def scale_importances(self, feature_importances):
        """Scales feature importances to a range of 0 to 1, maintaining original signs."""
        abs_importances = np.abs(feature_importances)
        min_importance = np.min(abs_importances)
        max_importance = np.max(abs_importances)
        
        if max_importance != min_importance:
            scaled_importances = (abs_importances - min_importance) / (max_importance - min_importance)
        else:
            scaled_importances = np.ones_like(abs_importances)  # All importances are equal, scale to 1

        # Restore the original signs of the scaled importances
        scaled_importances = np.sign(feature_importances) * scaled_importances
        
        return scaled_importances

    def component_callbacks(self, app):
        """Defines callbacks for updating the feature importance graph based on the input number of features."""
        @app.callback(
            Output("feature_importance_graph", "figure"),
            [Input('num_features_input', 'value')]
        )
        def update_graph(num_features):
            print("UPDQTE")
            if self.check_active():
                if num_features is None:
                    return dash.no_update

                if self.model.feature_importances_ is None:
                    return dash.no_update

                # Scale feature importances
                feature_importances = self.scale_importances(self.model.feature_importances_)

                # Create a DataFrame and sort by absolute value of importance
                importance_df = pd.DataFrame({
                    'Feature': self.dataset_config["columns_config"]["features"],
                    'Importance': feature_importances
                })
                
                importance_df = importance_df.reindex(importance_df['Importance'].abs().sort_values(ascending=False).index)

                # Limit to top `num_features` and reverse order for plotting
                importance_df = importance_df.head(num_features).iloc[::-1]

                # Create hover text
                importance_df['hover_text'] = importance_df.apply(
                    lambda row: f'Feature: {row["Feature"]}<br>Importance: {row["Importance"]*100:.2f}%<br>Description: {self.dataset_config["features_description"][row["Feature"]] if row["Feature"] in self.dataset_config["features_description"] else "No description for."}', axis=1)

                # Create Plotly bar chart
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    text='Importance',
                    custom_data=['hover_text'],
                    template="plotly_dark"
                )

                # Update layout and appearance
                fig.update_traces(
                    marker=dict(
                        color=color_positive,
                        line=dict(color=color_positive_darker, width=2)
                    ),
                    textfont=dict(color="rgba(255, 255, 255, 1)"),
                    hovertemplate='%{customdata[0]}',
                    texttemplate='%{text:.2f}',
                    textposition='inside'
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis_title=None,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=None,
                    width=None,
                    autosize=True
                )

                return fig
            else:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=None,
                    width=None,
                    autosize=True
                )
                return fig

class ShapExplanationComponent:
    def __init__(self, model, full_data, dataset_config):
        self.model = model
        self.full_data = full_data
        self.dataset_config = dataset_config
        self.src = None

    def set(self, model, full_data, dataset_config):
        self.model = model
        self.full_data = full_data
        self.dataset_config = dataset_config

    def check_active(self):
        print("Check : SE")
        return not self.model is None and not self.full_data is None and not self.dataset_config is None

    def layout(self):
        print("SC layout")
        return html.Div([
            html.Div([
                html.H4("SHAP Waterfall Plot", id="waterfall-title"),
                dcc.Input(
                    id='num_features_input_shap',
                    type='number',
                    value=10,
                    min=1,
                    step=1,
                    className='small-input dark-mode-input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between'
            }),

            html.Hr(className='HR'),
            html.Div([
                html.Img(id="shap_waterfall_image", className="img-scale clickable-img")
            ], className="auto-size-image-container")
        ], className='cell')

    def component_callbacks(self, app):
        @app.callback(
            Output("shap_waterfall_image", "src"),
            Output("waterfall-title", "children"),
            [Input('operation_id', 'value'),
            Input('num_features_input_shap', 'value')]
        )
        def update_shap_waterfall_image(operation_id, num_features):
            if self.check_active():
                self.src = shap_waterfall_image(operation_id, num_features, self.model, self.full_data, self.dataset_config)
                return self.src
            return None, "No data selected"

        @app.callback(
            [Output("shap-modal", "is_open"),
            Output("shap-enlarged-image", "src")],
            [Input("shap_waterfall_image", "n_clicks"),
            Input("modal-close-button", "n_clicks")],
            [dash.dependencies.State("shap-modal", "is_open"),
            dash.dependencies.State("shap_waterfall_image", "src")]
        )
        def toggle_modal_image_1(n1, n2, is_open, img_src):
            if self.check_active():
                if n1:
                    return not is_open, img_src
                if n2:
                    return False, img_src
                return is_open, img_src
            else:
                return False, None


class SimilarShapExplanationComponent:
    def __init__(self, model, full_data, dataset_config):
        self.model = model
        self.full_data = full_data
        self.dataset_config = dataset_config

        self.last_clicked_data = None
        self.history = []

    def check_active(self):
        return not self.model is None and not self.full_data is None and not self.dataset_config is None

    def layout(self):
        """Defines the layout of the SHAP explanation component."""
        return html.Div([
            html.Div([
                html.H4("SHAP Waterfall Plot", id="similar-plot-title"),
                html.Div([
                    dbc.Button("H", id="open-modal-button",
                        className="history-button"),
                    dcc.Input(
                        id='num_features_input_similar_shap',
                        type='number',
                        value=10,
                        min=1,
                        step=1,
                        className='small-input dark-mode-input'
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end'})
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between'
            }),
            html.Hr(className='HR'),
            html.Div([
                html.Img(id="shap_waterfall_image_similar", className="img-scale clickable-img")
            ], className="auto-size-image-container")
        ], className='cell')

    def component_callbacks(self, app):
        """Defines callbacks to update the SHAP waterfall image based on user interaction."""
        @app.callback(
            [Output("shap_waterfall_image_similar", "src"),
             Output("similar-plot-title", "children"),
             Output("history-content", "children")],
            [Input('similar_instances_3d_graph', 'clickData'),
             Input('operation_timeline_graph', 'clickData'),
             Input('num_features_input_similar_shap', 'value')]
        )
        def update_shap_waterfall_image_similar(data_3d, data_similar, num_features):
            if self.check_active():
                ctx = callback_context

                if not ctx.triggered or len(ctx.triggered) == 0:
                    # If no input triggered the callback, return default values
                    return None, "No data selected", None

                triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

                if triggered_id == "similar_instances_3d_graph":
                    data = data_3d
                elif triggered_id == "operation_timeline_graph":
                    data = data_similar
                elif triggered_id == "num_features_input_similar_shap":
                    if self.last_clicked_data is None:
                        return None, "No data selected", None
                    else:
                        data = self.last_clicked_data

                self.last_clicked_data = data

                if data is None or not "points" in data or len(data["points"]) == 0 or not "customdata" in data["points"][0]:
                    return None, "No data selected", None

                operation_id = data["points"][0]["customdata"]

                if operation_id is None:
                    return None, "No data selected", None  # Return None if no ID is selected
                
                if not operation_id in self.history:
                    self.history.append(operation_id)

                if len(self.history) > 10:
                    self.history.pop(0)
                # Generate and return the SHAP waterfall image and updated plot title
                return *shap_waterfall_image(operation_id, num_features, self.model, self.full_data, self.dataset_config), html.Div([html.P(el) for el in self.history])
            else:
                return None, None, None
    
        @app.callback(
            [Output("shap-modal", "is_open", allow_duplicate=True),
            Output("shap-enlarged-image", "src", allow_duplicate=True)],
            [Input("shap_waterfall_image_similar", "n_clicks"),
            Input("modal-close-button", "n_clicks")],
            [dash.dependencies.State("shap-modal", "is_open"),
            dash.dependencies.State("shap_waterfall_image_similar", "src")],
            prevent_initial_call=True
        )
        def toggle_modal_image_2(n1, n2, is_open, img_src):
            if self.check_active():
                if n1:
                    return not is_open, img_src
                if n2:
                    return False, img_src
                return is_open, img_src
            else:
                return False, None


        @app.callback(
            Output("history-modal", "is_open"),  # Output should not be in a list
            [Input("open-modal-button", "n_clicks"), Input("modal-close-button-2", "n_clicks")],
            [State("history-modal", "is_open")]
        )
        def toggle_modal(n1, n2, is_open):
            if self.check_active():
                ctx = dash.callback_context

                if not ctx.triggered:
                    return is_open  # Return the current state unchanged if nothing triggered the callback
                else:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                    if button_id == "open-modal-button" or button_id == "modal-close-button-2":
                        return not is_open  # Toggle the modal's visibility

                return is_open  # Fallback to returning the current state
            else:
                return False


class OperationTimelineComponent:
    def __init__(self, full_data, dataset_config):
        self.full_data = full_data
        self.dataset_config = dataset_config

    def check_active(self):
        return not self.full_data is None and not self.dataset_config is None

    def layout(self):
        """Defines the layout of the operation timeline component."""
        return html.Div([
            html.Div([
                html.H4("Operation Timeline"),
                dcc.Input(
                    id='timeline_num_operations',
                    type='number',
                    value=5,  # Default value for the number of operations to display
                    min=1,
                    step=1,
                    className='small-input dark-mode-input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0'
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="operation_timeline_graph", config={'responsive': True}, className="graph-scale"),
                className="auto-size-graph-container"
            )
        ], className='cell', style={'height': '100%'})

    def component_callbacks(self, app):
        """Defines callbacks to update the operation timeline graph based on selected operation ID and number of operations."""
        @app.callback(
            Output("operation_timeline_graph", "figure"),
            [Input('operation_id', 'value'),
             Input('timeline_num_operations', 'value')]
        )
        def update_timeline(operation_id, timeline_num_operations):
            print("eertet")
            if self.check_active():
                if operation_id is None or timeline_num_operations is None:
                    # Return an empty figure if inputs are not provided
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    return fig

                # Retrieve the history of operations for the given ID
                history_df = find_history(self.full_data, operation_id, maxlength=timeline_num_operations)

                if history_df.empty:
                    # Return an empty figure if no operation history is found
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    return fig

                fig = go.Figure()

                # Sort operations by time, with the most recent at the top
                history_df = history_df.sort_values(by=self.dataset_config["columns_config"]["time"], ascending=False).reset_index(drop=True)

                max_amount = history_df[self.dataset_config["columns_config"]["quantity"]].max()
                min_amount = history_df[self.dataset_config["columns_config"]["quantity"]].min()
                amount_range = max_amount - min_amount if max_amount != min_amount else 1  # Avoid division by zero

                max_line_length = 0.8  # Maximum length of horizontal line
                min_line_length = 0.2  # Minimum length to ensure visibility

                # Dynamic y positions from 0 to 1
                y_positions = np.linspace(0.9, 0.1, len(history_df))

                for idx, (i, row) in enumerate(history_df.iterrows()):
                    # Determine side: Left (-1) or Right (1)
                    side = -1 if idx % 2 == 0 else 1

                    # Calculate line length based on amount
                    normalized_amount = (row[self.dataset_config["columns_config"]["quantity"]] - min_amount) / amount_range
                    line_length = min_line_length + normalized_amount * (max_line_length - min_line_length)

                    x_start = -0.05 if side == 1 else 0.05
                    x_end = side * line_length
                    y = y_positions[idx]

                    # Determine marker shape based on class
                    shape = 'square' if row['Class'] == 1 else 'circle'

                    # Determine marker color based on score
                    color = calculate_gradient_color(row[self.dataset_config["columns_config"]["prediction_score"]], color_positive, color_negative)

                    # Add horizontal line
                    fig.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[y, y],
                        mode='lines',
                        line=dict(color='white', width=2),
                        showlegend=False,
                        hoverinfo='none'
                    ))

                    # Add marker at the end of the horizontal line
                    fig.add_trace(go.Scatter(
                        x=[x_end],
                        y=[y],
                        mode='markers',
                        marker=dict(
                            size=10 + 20*(row[self.dataset_config["columns_config"]["quantity"]]/max_amount),
                            color=color,
                            symbol=shape,
                            line=dict(color='white', width=1)
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Operation ID:</b> {row[self.dataset_config["columns_config"]["id"]]}<br>'
                            f'<b>Amount:</b> {row[self.dataset_config["columns_config"]["quantity"]]}<br>'
                            f'<b>Time:</b> {row[self.dataset_config["columns_config"]["time"]]}<br>'
                            f'<b>Score:</b> {row[self.dataset_config["columns_config"]["prediction_score"]]}<br>'
                            f'<b>Class:</b> {row[self.dataset_config["columns_config"]["class"]]}<extra></extra>'
                        ),
                        customdata=[row[self.dataset_config["columns_config"]["id"]]]
                    ))

                    # Add date text to the side, offset from the vertical bar
                    text_x = -0.2 if side == 1 else 0.2  # Offset to move text away from the line
                    text_align = 'left' if side == 1 else 'right'

                    fig.add_trace(go.Scatter(
                        x=[text_x],
                        y=[y],
                        mode='text',
                        text=[f"{row['Time']}"],
                        textposition='middle left' if side == 1 else 'middle right',
                        textfont=dict(color='white'),
                        showlegend=False,
                        hoverinfo='none'
                    ))

                # Add the vertical timeline
                fig.add_trace(go.Scatter(
                    x=[0, 0],
                    y=[min(y_positions) - 0.05, max(y_positions) + 0.05],
                    mode='lines',
                    line=dict(color='white', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))

                fig.update_layout(
                    xaxis=dict(
                        visible=False,
                        range=[-1, 1],  # Ensure enough space on both sides
                    ),
                    yaxis=dict(
                        visible=False,
                        range=[min(y_positions) - 0.05, max(y_positions) + 0.05],
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=None,  # Use available height
                    hovermode='closest',
                    dragmode=False,
                    autosize=True
                )

                return fig
            else:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                return fig

class SimilarInstances3DComponent:
    def __init__(self, model, full_data, dataset_config):
        self.full_data = full_data
        self.model = model
        self.dataset_config = dataset_config

    def check_active(self):
        return not self.model is None

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("3D Plot of Similar Instances"),
                dcc.Input(
                    id='num_similar',
                    type='number',
                    value=100,
                    min=1,
                    step=1,
                    className='small-input dark-mode-input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between'
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="similar_instances_3d_graph", className="graph-scale"),
                className="auto-size-graph-container"
            )
        ], className='cell')

    def component_callbacks(self, app):
        @app.callback(
            Output("similar_instances_3d_graph", "figure"),
            [Input('operation_id', 'value'),
             Input('num_similar', 'value')]
        )
        def update_3d_graph(operation_id, num_similar):
            if self.check_active():
                if operation_id is None or num_similar is None:
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),  # Hide x-axis
                        yaxis=dict(visible=False),  # Hide y-axis
                        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                    )
                    return fig
                
                # Find similar rows based on the input ID
                similar_df = find_similar_rows(self.full_data, operation_id, self.dataset_config, num_similar)

                if similar_df.empty:
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),  # Hide x-axis
                        yaxis=dict(visible=False),  # Hide y-axis
                        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                    )
                    return fig

                # Assume the model has feature_importances_ attribute and X has feature names
                feature_importances = self.model.feature_importances_

                top_features_indices = np.argsort(feature_importances)[-3:][::-1]
                top_features = self.full_data[self.dataset_config["columns_config"]["features"]].columns[top_features_indices]

                # Extract the data for the 3D plot
                x_data = similar_df[top_features[0]]
                y_data = similar_df[top_features[1]]
                z_data = similar_df[top_features[2]]

                # Create the 3D scatter plot
                # Convert the colors to a color scale
                colorscale = [[0, color_negative], [1, color_positive]]

                # Create the figure with the custom color scale
                fig = go.Figure(data=[go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=similar_df['Score'],  # Color based on Score
                        colorscale=colorscale,  # Custom color scale from color_negative to color_positive
                        opacity=0.7,  # Set alpha to 0.7 for all markers
                        symbol=['square' if int(cl) == 1 else 'circle' for cl in similar_df['Class']],  # Square for Class 1, circle otherwise
                        colorbar=dict(title="Score")  # Adds a color bar for the gradient
                    ),
                    text=[f"ID: {id}<br>Prediction/Score: {cl}/{sc}<br>Time: {t}<br>Amount: {amount}<br>{top_features[0]}: {x}<br>{top_features[1]}: {y}<br>{top_features[2]}: {z}"
                        for id, cl, sc, t, amount, x, y, z in zip(similar_df[self.dataset_config["columns_config"]["id"]], similar_df[self.dataset_config["columns_config"]["class"]], similar_df[self.dataset_config["columns_config"]["prediction_score"]], similar_df[self.dataset_config["columns_config"]["time"]], similar_df[self.dataset_config["columns_config"]["quantity"]], x_data, y_data, z_data)],
                    hoverinfo='text',
                    customdata=similar_df[self.dataset_config["columns_config"]["id"]]
                )])

                # Set the layout for the 3D plot
                fig.update_layout(
                    scene=dict(
                        xaxis_title=top_features[0],
                        yaxis_title=top_features[1],
                        zaxis_title=top_features[2],
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),  # Transparent x-axis background
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),  # Transparent y-axis background
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)')   # Transparent z-axis background
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                    paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                    template="plotly_dark"
                )

                return fig
            else:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),  # Hide x-axis
                    yaxis=dict(visible=False),  # Hide y-axis
                    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                    paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                )
                return fig



