import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
from sklearn.metrics import pairwise_distances
import json
import pickle
from components import SimilarShapExplanationComponent, ShapExplanationComponent, OperationTimelineComponent, SimilarInstances3DComponent, InputTextComponent, FeatureImportanceComponent
import dash_bootstrap_components as dbc

class DashboardApp:
    def __init__(self, config=None, model=None, dataset=None):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, '/assets/style.css'])
        self.setup_main_page(None, None, None, dummy_setup=True)
        self.start()

    def get_dataset(self, name):
        return pd.read_csv(f'./datasets/{name}/data.csv')
    
    def get_dataset_config(self, name):
        with open(f'./datasets/{name}/config.json', 'r') as file:
            return json.load(file)

    def get_config_ids(self):
        return os.listdir('./configs/')
    
    def get_config(self, config_id):
        with open(f'./configs/{config_id}/config.json', 'r') as file:
            return json.load(file)

    def create_main_menu(self):
        return html.Div([
            html.Div([
                dbc.Col([
                    html.H4("CONFIG SELECTION", id="main-menu-title"),
                    dcc.Dropdown(
                        id='config-dropdown', 
                        placeholder="Config", 
                        className='dark-mode-dropdown',
                        style={'marginBottom': '15px'}
                    ),
                    dcc.Dropdown(
                        id='model-dropdown', 
                        placeholder="Model", 
                        className='dark-mode-dropdown',
                        style={'marginBottom': '15px'}
                    ),
                    dcc.Dropdown(
                        id='dataset-dropdown', 
                        placeholder="Dataset", 
                        className='dark-mode-dropdown',
                        style={'marginBottom': '15px'}
                    ),
                    dbc.Button(
                        'Start Dashboard', 
                        id='start-button', 
                        className="w-100"
                    )
                ], width=12)
            ], style={
                'width': '400px', 
                'margin': 'auto', 
                'padding': '20px', 
                'display': 'flex', 
                'flexDirection': 'column', 
                'justifyContent': 'center', 
                'height': '100vh'
            })
        ], id='main-menu')




    def create_layout(self):
        return html.Div([
            self.create_main_menu(),
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                self.input_component.layout()
                            ], width=3, className="col-style", style={"padding":"0 4px 4px 0"}),

                            dbc.Col([
                                self.feature_importance.layout()
                            ], width=4, className="col-style", style={"padding":"0 4px 4px 4px"}),

                            dbc.Col([
                                self.shap_component.layout()
                            ], width=5, className="col-style", style={"padding":"0 0 4px 4px"}),
                        ], className='row', style={'height': '100%', 'boxSizing': 'border-box'})
                    ], className="card-body",style={'border': 'none', 'margin': '0', 'padding': '0', 'boxSizing': 'border-box'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                self.operation_timeline.layout()
                            ], width=4, className="col-style", style={"padding":"4px 4px 0 0"}),
                            dbc.Col([
                                self.same_instances.layout()
                            ], width=4, className="col-style", style={"padding":"4px 4px 0 4px"}),
                            dbc.Col([
                                self.similar_shap.layout()
                            ], width=4, className="col-style", style={"padding":"4px 0 0 4px"})
                        ], className='row', style={'height': '100%', 'boxSizing': 'border-box'})
                    ], className="card-body", style={'border': 'none', 'margin': '0', 'padding': '0', 'boxSizing': 'border-box'})
                ], style={'borderRadius':'0', 'border': 'none', 'margin': '0', 'padding': '8px', 'height': '100%', 'boxSizing': 'border-box'}),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.Button("Close", id="modal-close-button", className="ml-auto"),
                        style={"padding": "10px"}
                    ),
                    dbc.ModalBody(
                        html.Img(
                            id="shap-enlarged-image",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "100%",
                                "objectFit": "contain",
                                "display": "block",
                                "margin": "auto"
                            }
                        ),
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "height": "100%",
                            "padding": "0",
                            "overflow": "hidden"
                        }
                    ),
                ],
                id="shap-modal",
                centered=True,
                is_open=False,
                style={
                    "position": "fixed",
                    "top": "50%",
                    "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "width": "80vw",
                    "height": "80vh",
                    "maxWidth": "80vw",
                    "maxHeight": "80vh",
                    "zIndex": "1050",
                    "backgroundColor": "transparent",
                    "padding": "0",
                    "margin": "0"
                },
                backdrop=True,
                scrollable=False
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.Button("Close", id="modal-close-button-2"),
                        style={"padding": "10px"}
                    ),
                    dbc.ModalBody(
                        html.Div(
                            id="history-content",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "100%",
                                "objectFit": "contain",
                                "display": "block",
                                "margin": "auto"
                            }
                        ),
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "height": "100%",
                            "padding": "0",
                            "overflow": "hidden"
                        }
                    ),
                ],
                id="history-modal",
                centered=True,
                is_open=False,
                style={
                    "position": "fixed",
                    "top": "50%",
                    "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "width": "80vw",
                    "height": "80vh",
                    "maxWidth": "80vw",
                    "maxHeight": "80vh",
                    "zIndex": "1050",
                    "backgroundColor": "transparent",
                    "padding": "0",
                    "margin": "0"
                },
                backdrop=True,
                scrollable=False
            ),
            dcc.Store(id='page-load-trigger', data=True)

            ], id='dashboard-layout', style={'display': 'none'})
        
        ], style={'height': '100vh', 'padding': '0', 'margin': '0', 'borderRadius': '0'})

    def get_model(self, model_name):
        with open(f'./models/{model_name}', 'rb') as file:
            return pickle.load(file)

    def get_setup_data(self, config_name, model_name, dataset_name):
        self.model = self.get_model(model_name)
        self.dataset = self.get_dataset(dataset_name)
        self.dataset_config = self.get_dataset_config(dataset_name)
        return self.dataset_config, self.model, self.dataset

    def setup_main_page(self, config_name, model_name, dataset_name, dummy_setup=False):

        if dummy_setup:
            dataset_config, model, dataset = None, None, None

            self.input_component = InputTextComponent(dataset, dataset_config)
            self.feature_importance = FeatureImportanceComponent(model, dataset_config)
            self.shap_component = ShapExplanationComponent(model, dataset, dataset_config)
            self.operation_timeline = OperationTimelineComponent(dataset, dataset_config)
            self.same_instances = SimilarInstances3DComponent(model, dataset, dataset_config)
            self.similar_shap = SimilarShapExplanationComponent(model, dataset, dataset_config)

            self.app.layout = self.create_layout()
            self.setup_callbacks()

            # Set up callbacks for all components
            self.input_component.component_callbacks(self.app)
            self.feature_importance.component_callbacks(self.app)
            self.shap_component.component_callbacks(self.app)
            self.operation_timeline.component_callbacks(self.app)
            self.same_instances.component_callbacks(self.app)
            self.similar_shap.component_callbacks(self.app)
        else:
            dataset_config, model, dataset = self.get_setup_data(config_name, model_name, dataset_name)
            self.input_component.set(dataset, dataset_config)
            self.feature_importance.set(model, dataset_config)
            self.shap_component.set(model, dataset, dataset_config)
            self.operation_timeline.set(dataset, dataset_config)
            self.same_instances.set(model, dataset, dataset_config)
            self.similar_shap.set(model, dataset, dataset_config)

    def setup_callbacks(self):
        @self.app.callback(
            [Output('model-dropdown', 'options'), Output('dataset-dropdown', 'options')],
            [Input('config-dropdown', 'value')]
        )
        def update_dropdowns(config_id):
            if not config_id is None:
                config = self.get_config(config_id)
                return config["models"], config["datasets"]
            else:
                return [], []

        @self.app.callback(
            [Output('main-menu', 'style'), Output('dashboard-layout', 'style')],
            [Input('start-button', 'n_clicks')],
            [State('config-dropdown', 'value'), State('model-dropdown', 'value'), State('dataset-dropdown', 'value')]
        )
        def display_dashboard(n_clicks, config_name, model_name, dataset_name):
            if n_clicks and n_clicks > 0:
                self.setup_main_page(config_name, model_name, dataset_name, dummy_setup=False)
                return {'display': 'none'}, {'display': 'block'}
            return {'display': 'block'}, {'display': 'none'}

        @self.app.callback(
            Output('config-dropdown', 'options'),
            Input('page-load-trigger', 'data')
        )
        def set_dropdown1_options_on_load(page_load_trigger):
            options =  self.get_config_ids()
            return options        


        @self.app.callback(
            Output('interval', 'n_intervals'),
            Input('start-button', 'n_clicks')
        )
        def start_interval(n_clicks):
            if n_clicks:
                return 0  # Reset the interval to trigger
            return dash.no_update

    def start(self):
        # Run the Dash app
        self.app.run_server(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    DashboardApp()
