import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import ExplainerComponent, ShapContributionsGraphComponent
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import random
import shap
import string

# Setting the container style for maximum height and overflow handling
chart_container_style = {
    'maxHeight': '50vh',  # Set max height to 50% of the viewport height
    'overflowY': 'auto'   # Enable scrolling if content exceeds max height
}

features_columns = [f'V{i}' for i in range(1, 29)]
non_features_columns = ["Time", "Class", "row_id"]

# Charger les données et entraîner le modèle
df = pd.read_csv("creditcard.csv").head(1000)

# Function to generate a random alphanumeric ID
def generate_random_id(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Add the row_id column with unique random alphanumeric IDs
df['row_id'] = [generate_random_id() for _ in range(len(df))]

# Ensure the generated IDs are unique
while len(df['row_id']) != len(set(df['row_id'])):
    duplicates = df[df.duplicated('row_id', keep=False)]
    for index in duplicates.index:
        df.at[index, 'row_id'] = generate_random_id()

X = df.drop(columns=["Time", "Class", "row_id"])
y = df["Class"]

model = RandomForestClassifier()
model.fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Initialiser Explainer
explainer = ClassifierExplainer(model, X, y)

class InputTextComponent(ExplainerComponent):
    def __init__(self, explainer, full_data, name=None):
        super().__init__(explainer, title="Input and Text")
        self.data = full_data

    def layout(self):
        return html.Div([
            html.H4("Transaction search", style={'textAlign': 'center'}),
            html.Hr(style={'margin': '10px 0'}),
            dcc.Dropdown(
                id='transaction_id', 
                options=[{'label': rid, 'value': rid} for rid in self.data['row_id']], 
                placeholder="Enter or select Transaction ID",
                searchable=True
            ),
            html.Div(
                id='transaction_details', 
                style={'overflowY': 'auto', 'maxHeight': 'calc(50vh - 120px)'}
            )
        ], style={
            'border': '1px solid #ccc',
            'boxShadow': '2px 2px 12px #aaa',
            'padding': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'height': '50vh',  # Set max height to 50% of the viewport height
            'overflowY': 'hidden',
            'width':'100%'
        })
    
    def component_callbacks(self, app):
        @app.callback(
            Output('transaction_details', 'children'),
            [Input('transaction_id', 'value')]
        )
        def update_transaction_details(transaction_id):
            if transaction_id is None:
                return "Please enter a valid transaction ID."
            
            # Search for the row data based on the transaction_id
            row = self.data[self.data['row_id'] == transaction_id]
            
            if row.empty:
                return "Transaction ID not found."
            
            # Convert row data to dictionary and format it for display
            row_data = row.iloc[0].drop('row_id').to_dict()
            details = [html.P(f"{key}: {value}") for key, value in row_data.items()]
            
            return html.Div(details)

class FeatureImportanceComponent(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Feature Importance")

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("Model Feature Importance", style={'textAlign': 'center', 'flex': '1'}),
                dcc.Input(
                    id='num_features_input',
                    type='number',
                    value=10,  # Default value
                    min=1,
                    step=1,
                    style={'marginLeft': 'auto', 'width': '60px'}
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0',
                'marginBottom': '10px'  # Space between header and graph
            }),
            html.Hr(style={'margin': '0', 'flexShrink': '0'}),
            html.Div(
                dcc.Graph(id="feature_importance_graph"),
                style={'flex': '1 1 auto', 'height': '0', 'overflow': 'hidden'}
            )
        ], style={
            'border': '1px solid #ccc',
            'boxShadow': '2px 2px 12px #aaa',
            'padding': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'height': '50vh',  # Total height of the component
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%'
        })

    def component_callbacks(self, app):
        @app.callback(
            Output("feature_importance_graph", "figure"),
            [Input('transaction_id', 'value'),
             Input('num_features_input', 'value')]
        )
        def update_graph(transaction_id, num_features):
            if num_features is None:
                num_features = 10  # Default number of features if input is empty

            # Get feature importances and column names
            feature_importances = self.explainer.model.feature_importances_
            features = self.explainer.columns

            # Create a DataFrame and select the specified number of top features
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False).head(num_features).iloc[::-1]

            # Create hover text
            importance_df['hover_text'] = importance_df.apply(
                lambda row: f"Feature: {row['Feature']}<br>Importance: {row['Importance']:.4f}", axis=1)
            
            # Plotly express bar chart
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                text='Importance',
                custom_data=['hover_text']
            )

            # Update layout to reduce padding, use hover data, and make bars thinner and closer
            fig.update_traces(
                hovertemplate='%{customdata[0]}',  # Display custom hover text
                texttemplate='%{text:.2f}',        # Format displayed importance values
                textposition='inside',             # Position of the text inside the bars
            )

            fig.update_layout(
                margin=dict(l=50, r=0, t=0, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title=None,
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True)
            )

            return fig

import shap
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid the need for a display server
import matplotlib.pyplot as plt
import base64
import io
from dash import dcc, html
from dash.dependencies import Input, Output
from explainerdashboard.custom import ExplainerComponent
import numpy as np

class ShapExplanationComponent(ExplainerComponent):
    def __init__(self, explainer, full_data, shap_values, model, name=None):
        super().__init__(explainer, title="SHAP Waterfall")
        self.explainer = explainer
        self.shap_values = shap_values
        self.full_data = full_data
        self.model = model

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("SHAP Waterfall Plot", style={'textAlign': 'center', 'flex': '1'}),
                dcc.Input(
                    id='num_features_input_shap',
                    type='number',
                    value=10,
                    min=1,
                    step=1,
                    style={'marginLeft': 'auto', 'width': '60px'}
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between'
            }),

            html.Hr(style={'margin': '10px 0'}),
            html.Div([
                html.Img(id="shap_waterfall_image", style={'width': '100%', 'height': 'auto'})
            ], style={'width': '100%', 'height': '100%'})
        ], style={
            'border': '1px solid #ccc',
            'boxShadow': '2px 2px 12px #aaa',
            'padding': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'overflowY': 'auto',
            'width':'100%',
            'height':'50vh'
        })

    def component_callbacks(self, app):
        @app.callback(
            Output("shap_waterfall_image", "src"),
            [Input('transaction_id', 'value'),
             Input('num_features_input_shap', 'value')]
        )
        def update_shap_waterfall_image(transaction_id, num_features):
            if transaction_id is None:
                return None  # Return None if no ID is selected

            # Find the row in the dataset corresponding to the transaction_id
            row = self.full_data[self.full_data['row_id'] == transaction_id]
            if row.empty:
                return None  # If no matching row is found, return None

            row_index = row.index[0]  # Get the index of the row
            feature_values = row.drop(columns=['row_id', 'Class']).iloc[0]

            # Extract SHAP values for the specific instance
            shap_values_instance = self.shap_values.values[row_index, :, 0]  # SHAP values for the first class/output
            base_value = np.sum(self.shap_values.base_values[:, 0])  # Calculate the sum of base values

            # Use SHAP's default waterfall plot design
            shap_exp = shap.Explanation(
                values=shap_values_instance,
                base_values=base_value,
                data=feature_values.values,
                feature_names=feature_values.index
            )

            # Generate the waterfall plot and save to buffer
            plt.figure()
            shap.waterfall_plot(shap_exp, show=False, max_display=num_features)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Encode the PNG image to base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"


import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from explainerdashboard.custom import ExplainerComponent

# Dummy function to be completed
def find_history(df, id, length=5):
    # Placeholder function: return the last `length` transactions for a given card `id`
    # This is a placeholder; you should implement the actual logic based on your dataset
    return df.sample(n=length)

class TransactionTimelineComponent(ExplainerComponent):
    def __init__(self, explainer, full_data, name=None):
        super().__init__(explainer, title="Transaction Timeline")
        self.full_data = full_data

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("Transaction Timeline", style={'textAlign': 'center', 'flex': '1'}),
                dcc.Input(
                    id='timeline_num_transactions',
                    type='number',
                    value=5,  # Default number of transactions
                    min=1,
                    step=1,
                    style={'marginLeft': 'auto', 'width': '60px'}
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0',
                'marginBottom': '10px'  # Space between header and graph
            }),
            html.Hr(style={'margin': '0', 'flexShrink': '0'}),
            html.Div(
                dcc.Graph(id="transaction_timeline_graph", config={'responsive': True}),
                style={'flex': '1', 'height': '100%', 'overflow': 'hidden'}
            )
        ], style={
            'border': '1px solid #ccc',
            'boxShadow': '2px 2px 12px #aaa',
            'padding': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'height': '50vh',  # Total height of the component
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%'
        })

    def component_callbacks(self, app):
        @app.callback(
            Output("transaction_timeline_graph", "figure"),
            [Input('transaction_id', 'value'),
            Input('timeline_num_transactions', 'value')]
        )
        def update_timeline(transaction_id, timeline_num_transactions):
            if transaction_id is None or timeline_num_transactions is None:
                return go.Figure()  # Return an empty figure if no ID or transaction number is provided

            # Retrieve the card history data
            history_df = find_history(self.full_data, transaction_id, length=timeline_num_transactions)

            if history_df.empty:
                return go.Figure()  # Return an empty figure if no transactions are found

            # Plot the timeline with alternating top/bottom lines and varying dot sizes
            fig = go.Figure()

            # Add the horizontal line for time axis
            fig.add_trace(go.Scatter(
                x=history_df['Time'],
                y=[0] * len(history_df),  # Horizontal line at y=0
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

            max_amount = max(history_df["Amount"].values)
            max_height = 0.5  # Arbitrary scale, increase if needed
            min_height = 0.05  # Ensure no zero-height lines

            for i, row in history_df.iterrows():
                # Calculate the proportional height of the line
                y_value = (row["Amount"] / max_amount) * max_height
                y_value = max(y_value, min_height)  # Ensure a minimum height

                # Add a line for each transaction
                fig.add_trace(go.Scatter(
                    x=[row['Time'], row['Time']],
                    y=[0, y_value],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

                # Add a dot at the end of the line, size proportional to Amount
                fig.add_trace(go.Scatter(
                    x=[row['Time']],
                    y=[y_value],
                    mode='markers+text',
                    marker=dict(size=(row['Amount'] / max_amount) * 20 + 5, color='red'),  # Size scaled and offset for visibility
                    text=[f"Amount: {row['Amount']}<br>Time: {row['Time']}"],
                    textposition="bottom center" if y_value < 0 else "top center",
                    showlegend=False,
                    hoverinfo='text'
                ))

            fig.update_layout(
                xaxis_title='Time',
                yaxis=dict(range=[-max_height*1.5, max_height * 1.4], fixedrange=True, showticklabels=False),  # Extended y-axis range
                xaxis=dict(
                    tickangle=-45,  # Rotate x-tick labels by 45 degrees
                    showgrid=False,
                    showticklabels=True,
                    tickmode='array',
                    tickvals=history_df['Time'],
                    ticktext=pd.to_datetime(history_df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                ),
                margin=dict(l=5, r=5, t=60, b=20),  # Increase bottom margin for horizontal line and rotated labels
                height=None,  # Let the graph auto-resize
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                autosize=True,
                hovermode='closest'
            )

            return fig
    

import numpy as np
from sklearn.metrics import pairwise_distances
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from explainerdashboard.custom import ExplainerComponent

# Dummy function to be completed
def find_similar_rows(df, input_id, explainer, n=10):
    # Placeholder function: should return rows with similar predicted class and close feature values
    # This is a placeholder; you should implement the actual logic based on your dataset and model.
    # Here, we use a distance metric to find similar rows in the dataset based on features.
    
    row = df.loc[df['row_id'] == input_id]
    # Get the feature vector of the input row
    input_features = row[features_columns]
    
    if input_features.empty:
        return pd.DataFrame()
  
    # Predict the class for the input row
    input_prediction = row["Class"].iloc[0]
    # Find all rows with the same predicted class
    same_class_df = df[df["Class"] == input_prediction]
    # Calculate the distance from the input features to all other rows
    distances = pairwise_distances(input_features, same_class_df[features_columns]).flatten()
    
    # Get the indices of the closest rows
    closest_indices = np.argsort(distances)[:n]
    
    return same_class_df.iloc[closest_indices]

class SimilarInstances3DComponent(ExplainerComponent):
    def __init__(self, explainer, full_data, name=None):
        super().__init__(explainer, title="Similar Instances 3D Plot")
        self.full_data = full_data

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("3D Plot of Similar Instances", style={'textAlign': 'center'}),
                dcc.Input(
                    id='num_similar',
                    type='number',
                    value=100,  # Default number of transactions
                    min=1,
                    step=1,
                    style={'marginLeft': 'auto', 'width': '60px'}
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0',
                'marginBottom': '10px'  # Space between header and graph
            }),
            html.Hr(style={'margin': '0', 'flexShrink': '0'}),
            html.Div(
                dcc.Graph(id="similar_instances_3d_graph", config={'responsive': True}),
                style={'flex': '1', 'height': '100%', 'overflow': 'hidden'}
            )
        ], style={
            'border': '1px solid #ccc',
            'boxShadow': '2px 2px 12px #aaa',
            'padding': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'height': '50vh',  # Total height of the component
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%'
        })

    def component_callbacks(self, app):
        @app.callback(
            Output("similar_instances_3d_graph", "figure"),
            [Input('transaction_id', 'value'),
            Input('num_similar', 'value')]
        )
        def update_3d_graph(transaction_id, num_similar):
            if transaction_id is None:
                return go.Figure()  # Return an empty figure if no ID is selected

            # Find similar rows based on the input ID
            similar_df = find_similar_rows(self.full_data, transaction_id, self.explainer, num_similar)

            if similar_df.empty:
                return go.Figure()  # Return an empty figure if no similar rows are found

            # Get the three most important features
            feature_importances = self.explainer.model.feature_importances_
            top_features_indices = np.argsort(feature_importances)[-3:][::-1]
            top_features = self.explainer.columns[top_features_indices]

            # Extract the data for the 3D plot
            x_data = similar_df[top_features[0]]
            y_data = similar_df[top_features[1]]
            z_data = similar_df[top_features[2]]

            # Create the 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                text=[f"ID: {rid}<br>{top_features[0]}: {x}<br>{top_features[1]}: {y}<br>{top_features[2]}: {z}" 
                      for rid, x, y, z in zip(similar_df['row_id'], x_data, y_data, z_data)],
                hoverinfo='text'
            )])

            # Set the layout for the 3D plot
            fig.update_layout(
                scene=dict(
                    xaxis_title=top_features[0],
                    yaxis_title=top_features[1],
                    zaxis_title=top_features[2],
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=400,
            )

            return fig


class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, full_data, model, shap_values, name=None, **kwargs):
        super().__init__(explainer, title="Custom Dashboard")
        self.input_text = InputTextComponent(explainer, full_data)
        self.feature_importance = FeatureImportanceComponent(explainer)
        self.shap_waterfall = ShapExplanationComponent(explainer, full_data, shap_values, model)
        self.transaction_history = TransactionTimelineComponent(explainer, full_data)
        self.similar = SimilarInstances3DComponent(explainer, full_data)

    def layout(self):
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            self.input_text.layout(),
                        ], width=2, style={'padding':'0px 5px 0px 10px'}),  # 20% width

                        dbc.Col([
                            self.feature_importance.layout(),
                        ], width=5, style={'padding':'0px 5px 0px 5px'}),  # 30% width

                        dbc.Col([
                            self.shap_waterfall .layout(),
                        ], width=5, style={'padding':'0px 10px 0px 5px'}),  # 50% width
                    ])
                ], style={'border':'none', 'margin':'0', 'padding':'0'}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            self.transaction_history.layout(),
                        ], width=4, style={'padding':'0px 5px 0px 10px'}),
                        dbc.Col([
                            self.similar.layout(),
                        ], width=4, style={'padding':'0px 5px 0px 10px'})

                    ])
                ], style={'border':'none', 'margin':'0', 'padding':'0'})
            ], style={'border':'none', 'margin':'0', 'padding':'0'})
        ])

if __name__ == "__main__":
    db = ExplainerDashboard(explainer, CustomDashboard, full_data=df, model=model, shap_values=shap_values, hide_header=True, mode='inline', hide_poweredby=True)
    db.run()
