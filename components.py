import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import matplotlib
matplotlib.use('Agg') 
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
                    placeholder="Enter or select ID from filtered",
                    searchable=True,
                    className='dark-mode-dropdown'
                )
            ]),
            html.Div(
                id='operation_details',
                style={'overflowY': 'auto', 'marginTop': '5px'}
            ),
            dcc.Interval(id='interval', interval=3000, n_intervals=0, max_intervals=1)
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

                row = self.data[self.data[self.dataset_config["columns_config"]["id"]] == operation_id]
                
                if row.empty:
                    return f"Error: Operation ID {operation_id} not found."
                
                row_data = row.iloc[0].drop(self.dataset_config["columns_config"]["id"]).to_dict()
                details = [html.P(f"{key}: {value}") for key, value in row_data.items() if key in self.dataset_config["display_fields"]]
                
                return html.Div(details)
            else:
                return None

        def fetch_options(search_term):
            if not search_term is None:
                id_column = self.dataset_config["columns_config"]["id"]
                
                filtered_df = self.data[self.data[id_column].str.contains(search_term, case=False, na=False)]
                
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
                    value=10,
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
            # All importances are equal, scale to 1
            scaled_importances = np.ones_like(abs_importances) 

        # Restore the original signs of the scaled importances
        scaled_importances = np.sign(feature_importances) * scaled_importances
        
        return scaled_importances

    def component_callbacks(self, app):
        """Defines callbacks for updating the feature importance graph based on the input number of features."""
        @app.callback(
            Output("feature_importance_graph", "figure"),
            [Input('num_features_input', 'value'), Input('interval', 'n_intervals')]
        )
        def update_graph(num_features, _):
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

                importance_df['hover_text'] = importance_df.apply(
                    lambda row: f'Feature: {row["Feature"]}<br>Importance: {row["Importance"]*100:.2f}%<br>Description: {self.dataset_config["features_description"][row["Feature"]] if row["Feature"] in self.dataset_config["features_description"] else "No description for."}', axis=1)

                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    text='Importance',
                    custom_data=['hover_text'],
                    template="plotly_dark"
                )

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
        return not self.model is None and not self.full_data is None and not self.dataset_config is None

    def layout(self):
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
    
    def set(self, model, full_data, dataset_config):
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
                    return None, "No data selected", None 
                
                if not operation_id in self.history:
                    self.history.append(operation_id)

                if len(self.history) > 10:
                    self.history.pop(0)

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
            Output("history-modal", "is_open"), 
            [Input("open-modal-button", "n_clicks"), Input("modal-close-button-2", "n_clicks")],
            [State("history-modal", "is_open")]
        )
        def toggle_modal(n1, n2, is_open):
            if self.check_active():
                ctx = dash.callback_context

                if not ctx.triggered:
                    return is_open 
                else:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                    if button_id == "open-modal-button" or button_id == "modal-close-button-2":
                        return not is_open 

                return is_open
            else:
                return False

class OperationTimelineComponent:
    def __init__(self, full_data, dataset_config):
        self.full_data = full_data
        self.dataset_config = dataset_config
    
    def set(self, full_data, dataset_config):
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
                    value=5,
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
            if self.check_active():
                if operation_id is None or timeline_num_operations is None:
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    return fig

                history_df = find_history(self.full_data, operation_id, maxlength=timeline_num_operations)

                if history_df.empty:
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    return fig

                fig = go.Figure()
                
                color_at_10_percent_rgb = calculate_gradient_color(0.1, color_positive, color_negative)
                colorscale = [[0, rgba_to_rgb(color_negative)], [0.85, rgba_to_rgb(color_at_10_percent_rgb)], [1, rgba_to_rgb(color_positive)]]

                history_df = history_df.sort_values(by=self.dataset_config["columns_config"]["time"], ascending=False).reset_index(drop=True)

                max_amount = history_df[self.dataset_config["columns_config"]["quantity"]].max()
                min_amount = history_df[self.dataset_config["columns_config"]["quantity"]].min()
                amount_range = max_amount - min_amount if max_amount != min_amount else 1 

                max_line_length = 0.8 
                min_line_length = 0.2

                y_positions = np.linspace(0.9, 0.1, len(history_df))

                marker_x = []
                marker_y = []
                marker_sizes = []
                marker_colors = []
                marker_shapes = []
                marker_hovertext = []
                marker_customdata = []

                line_x = []
                line_y = []

                text_x = []
                text_y = []
                text_labels = []

                hover_display_columns = self.dataset_config["columns_config"]["hover_display"]

                available_columns = history_df.columns

                for idx, (i, row) in enumerate(history_df.iterrows()):
                    side = -1 if idx % 2 == 0 else 1

                    normalized_amount = (row[self.dataset_config["columns_config"]["quantity"]] - min_amount) / amount_range
                    line_length = min_line_length + normalized_amount * (max_line_length - min_line_length)

                    x_start = -0.05 if side == 1 else 0.05
                    x_end = side * line_length
                    y = y_positions[idx]

                    shape = 'square' if row['Class'] == 1 else 'circle'

                    line_x.extend([x_start, x_end, None]) 
                    line_y.extend([y, y, None])

                    marker_x.append(x_end)
                    marker_y.append(y)
                    marker_sizes.append(10 + 20 * (row[self.dataset_config["columns_config"]["quantity"]] / max_amount))
                    marker_colors.append(row[self.dataset_config["columns_config"]["prediction_score"]])
                    marker_shapes.append(shape)
                    marker_hovertext.append(
                        f'<b>Operation ID:</b> {row[self.dataset_config["columns_config"]["id"]]}<br>'
                        f'<b>Amount:</b> {row[self.dataset_config["columns_config"]["quantity"]]}<br>'
                        f'<b>Time:</b> {row[self.dataset_config["columns_config"]["time"]]}<br>'
                        f'<b>Score:</b> {row[self.dataset_config["columns_config"]["prediction_score"]]}<br>'
                        f'<b>Class:</b> {row[self.dataset_config["columns_config"]["class"]]}<br>' +
                        ''.join([f"<b>{col}</b>: {row[col]}<br>" for col in hover_display_columns if col in available_columns]) + 
                        '<extra></extra>'
                    )
                    marker_customdata.append(row[self.dataset_config["columns_config"]["id"]])

                    text_x.append(-0.2 if side == 1 else 0.2)
                    text_y.append(y)
                    text_labels.append(f"{row['Time']}")

                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    line=dict(color='white', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))

                fig.add_trace(go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode='markers',
                    marker=dict(
                        size=marker_sizes,
                        color=marker_colors,
                        colorscale=colorscale,
                        symbol=marker_shapes,
                        line=dict(color='white', width=1),
                        opacity=1,
                        cmin=0,
                        cmax=1 
                    ),
                    showlegend=False,
                    hovertemplate=marker_hovertext,
                    customdata=marker_customdata
                ))

                fig.add_trace(go.Scatter(
                    x=text_x,
                    y=text_y,
                    mode='text',
                    text=text_labels,
                    textposition=['middle left' if x < 0 else 'middle right' for x in text_x],
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
                        range=[-1, 1], 
                    ),
                    yaxis=dict(
                        visible=False,
                        range=[min(y_positions) - 0.05, max(y_positions) + 0.05],
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=None,
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

    def set(self, model, full_data, dataset_config):
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
                dcc.Graph(id="similar_instances_3d_graph", className="graph-scale", style={"height":"100%"}),
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
                        xaxis=dict(visible=False), 
                        yaxis=dict(visible=False), 
                        plot_bgcolor='rgba(0,0,0,0)',  
                        paper_bgcolor='rgba(0,0,0,0)',  
                    )
                    return fig
                
                similar_df = find_similar_rows(self.full_data, operation_id, self.dataset_config, num_similar)

                if similar_df.empty:
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis=dict(visible=False),  
                        yaxis=dict(visible=False),  
                        plot_bgcolor='rgba(0,0,0,0)',  
                        paper_bgcolor='rgba(0,0,0,0)',  
                    )
                    return fig

                feature_importances = self.model.feature_importances_

                top_features_indices = np.argsort(feature_importances)[-3:][::-1]
                top_features = self.full_data[self.dataset_config["columns_config"]["features"]].columns[top_features_indices]

                x_data = similar_df[top_features[0]]
                y_data = similar_df[top_features[1]]
                z_data = similar_df[top_features[2]]

                color_at_10_percent_rgb = calculate_gradient_color(0.1, color_positive, color_negative)
                colorscale = [[0, color_negative], [0.85, color_at_10_percent_rgb], [1, color_positive]]

                hover_display_columns = self.dataset_config["columns_config"]["hover_display"]

                fig = go.Figure(data=[go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=similar_df['Score'],  
                        colorscale=colorscale,
                        opacity=0.85,  
                        symbol=['square' if int(cl) == 1 else 'circle' for cl in similar_df['Class']], 
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title='Score',  
                            titleside='bottom',
                        ),
                    ),
                    text = [
                        "<br>".join(
                            [
                                f"ID: {id}",
                                f"Prediction/Score: {cl}/{sc}",
                                f"Time: {t}",
                                f"Amount: {amount}",
                                f"{top_features[0]}: {x}",
                                f"{top_features[1]}: {y}",
                                f"{top_features[2]}: {z}"
                            ] + [f"{col}: {similar_df[col].iloc[i]}" for col in hover_display_columns if col in similar_df.columns]
                        )
                        for i, (id, cl, sc, t, amount, x, y, z) in enumerate(
                            zip(
                                similar_df[self.dataset_config["columns_config"]["id"]],
                                similar_df[self.dataset_config["columns_config"]["class"]],
                                similar_df[self.dataset_config["columns_config"]["prediction_score"]],
                                similar_df[self.dataset_config["columns_config"]["time"]],
                                similar_df[self.dataset_config["columns_config"]["quantity"]],
                                x_data,
                                y_data,
                                z_data
                            )
                        )
                    ],
                    hoverinfo='text',
                    customdata=similar_df[self.dataset_config["columns_config"]["id"]]
                )])

                fig.update_layout(
                    scene=dict(
                        xaxis_title=top_features[0],
                        yaxis_title=top_features[1],
                        zaxis_title=top_features[2],
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)')  
                    ),
                    margin=dict(l=0, r=0, b=5, t=0),
                    plot_bgcolor='rgba(0,0,0,0)',  
                    paper_bgcolor='rgba(0,0,0,0)',  
                    template="plotly_dark",
                    height=None,
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



