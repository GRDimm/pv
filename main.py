import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import random
import dash
import shap
import string
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, '/assets/style.css'])


# Setting the container style for maximum height and overflow handling
chart_container_style = {
    'maxHeight': '50vh',  # Set max height to 50% of the viewport height
    'overflowY': 'auto'   # Enable scrolling if content exceeds max height
}

features_columns = [f'V{i}' for i in range(1, 29)] + ["Amount"]
non_features_columns = ["Time", "Class", "row_id"]

# Charger les données et entraîner le modèle
df = pd.read_csv("datasets/creditcard.csv").head(1000)

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

cell_box_shadow = '2px 1px 14px 2px rgba(0, 0, 0, 0.65)'
cell_border = '1px solid black'
cell_padding = '5px'

small_input_style = {'marginLeft': 'auto', 'width': '50px', 'height':'1.5em', 'fontSize':'0.8em'}

class InputTextComponent:
    def __init__(self, full_data):
        self.data = full_data

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("Transaction search"),
                html.Hr(className='HR'),
                dcc.Dropdown(
                    id='transaction_id',
                    options=[{'label': rid, 'value': rid} for rid in self.data['row_id']],
                    placeholder="Enter or select Transaction ID",
                    searchable=True
                )
            ], style={'marginBottom': '5px'}),
            html.Div(
                id='transaction_details',
                style={'overflowY': 'auto', 'maxHeight': 'calc(50vh - 120px)'}
            )
        ], className='cell', style={'padding':'10px'})

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

class FeatureImportanceComponent:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("Model Feature Importance"),
                dcc.Input(
                    id='num_features_input',
                    type='number',
                    value=10,  # Default value
                    min=1,
                    step=1,
                    className='small_input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0',
                'marginBottom': '5px'  # Space between header and graph
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="feature_importance_graph"),
                style={'flex': '1 1 auto', 'overflow': 'hidden'}
            )
        ], className='cell')

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
            feature_importances = self.model.feature_importances_

            # Create a DataFrame and select the specified number of top features
            importance_df = pd.DataFrame({
                'Feature': self.features,
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
                custom_data=['hover_text'],
                template="plotly_dark"
            )

            # Update layout to reduce padding, use hover data, and make bars thinner and closer
            fig.update_traces(
                hovertemplate='%{customdata[0]}',  # Display custom hover text
                texttemplate='%{text:.2f}',        # Format displayed importance values
                textposition='inside'            # Position of the text inside the bars
            )

            fig.update_layout(
                margin=dict(l=50, r=0, t=0, b=40),
                yaxis_title=None,
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True), 
                plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                paper_bgcolor='rgba(0,0,0,0)'
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
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import shap
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np


class ShapExplanationComponent:
    def __init__(self, model, full_data):
        self.model = model
        self.full_data = full_data
        self.features_columns = features_columns

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("SHAP Waterfall Plot"),
                dcc.Input(
                    id='num_features_input_shap',
                    type='number',
                    value=10,
                    min=1,
                    step=1,
                    className='small_input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between'
            }),

            html.Hr(className='HR'),
            html.Div([
                html.Img(id="shap_waterfall_image", style={'width': '100%', 'height': 'auto'})
            ], style={'width': '100%', 'height': '100%'})
        ], className='cell')

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

            row_index = row.index[0]
            feature_values = row[features_columns].iloc[0]

            explainer = shap.Explainer(self.model, self.full_data[self.features_columns])
            shap_values = explainer(row[self.features_columns])

            # Extract SHAP values for the specific instance
            shap_values_instance = shap_values.values[0, :, int(row["Class"].values[0])]  # SHAP values for the first class/output
            base_value = np.sum(shap_values.base_values[:, int(row["Class"].values[0])])  # Calculate the sum of base values

            # Use SHAP's default waterfall plot design
            shap_exp = shap.Explanation(
                values=shap_values_instance,
                base_values=base_value,
                data=feature_values.values,
                feature_names=feature_values.index
            )

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_exp, max_display=num_features, show=False)

            # Customize for dark theme
            fig.patch.set_facecolor('#282828')
            ax.patch.set_facecolor('#282828')

            # Customize text and other elements
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')

            # Customize SHAP values bars and text
            for text in ax.texts:
                text.set_color('white')
            for artist in ax.artists:
                artist.set_edgecolor('white')
            for line in ax.lines:
                line.set_color('white')
            
            plt.savefig(buf := io.BytesIO(), format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return f"data:image/png;base64,{img_base64}"


import plotly.express as px
from dash import dcc, html
import dash
from dash.dependencies import Input, Output

# Dummy function to be completed
def find_history(df, id, length=5):
    # Placeholder function: return the last `length` transactions for a given card `id`
    # This is a placeholder; you should implement the actual logic based on your dataset
    return df.sample(n=length)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, '/assets/style.css'])


class TransactionTimelineComponent:
    def __init__(self, full_data):
        self.full_data = full_data

    def layout(self):
        return html.Div([
            html.Div([
                html.H4("Transaction Timeline"),
                dcc.Input(
                    id='timeline_num_transactions',
                    type='number',
                    value=5,
                    min=1,
                    step=1,
                    className='small_input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexShrink': '0',
                'marginBottom': '5px'
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="transaction_timeline_graph", config={'responsive': True}),
                style={'flex': '1', 'overflow': 'hidden'}
            )
        ], className='cell')

    def component_callbacks(self, app):
        @app.callback(
            Output("transaction_timeline_graph", "figure"),
            [Input('transaction_id', 'value'),
            Input('timeline_num_transactions', 'value')]
        )
        def update_timeline(transaction_id, timeline_num_transactions):
            if transaction_id is None or timeline_num_transactions is None:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),  # Hide x-axis
                    yaxis=dict(visible=False),  # Hide y-axis
                    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                    paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                )
                return fig

            history_df = find_history(self.full_data, transaction_id, length=timeline_num_transactions)

            if history_df.empty:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),  # Hide x-axis
                    yaxis=dict(visible=False),  # Hide y-axis
                    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                    paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                )
                return fig

            fig = go.Figure()

            # Add the white horizontal line for the x-axis
            fig.add_trace(go.Scatter(
                x=history_df['Time'],
                y=[0] * len(history_df),
                mode='lines',
                line=dict(color='white', width=2),  # White horizontal line
                showlegend=False
            ))

            max_amount = history_df["Amount"].max()
            max_height = 0.5
            min_height = 0.05

            for i, row in history_df.iterrows():
                y_value = max((row["Amount"] / max_amount) * max_height, min_height)
                # Add white vertical lines
                fig.add_trace(go.Scatter(
                    x=[row['Time'], row['Time']],
                    y=[0, y_value],
                    mode='lines',
                    line=dict(color='white', width=2),  # White vertical lines
                    showlegend=False
                ))
                # Add blue dots
                fig.add_trace(go.Scatter(
                    x=[row['Time']],
                    y=[y_value],
                    mode='markers+text',
                    marker=dict(size=(row['Amount'] / max_amount) * 20 + 5, color='blue'),  # Blue dots
                    text=[f"Amount: {row['Amount']}<br>Time: {row['Time']}"],
                    textposition="bottom center" if y_value < 0 else "top center",
                    showlegend=False,
                    hoverinfo='text'
                ))

            fig.update_layout(
                xaxis_title='Time',
                yaxis=dict(visible=False),  # Hide y-axis
                xaxis=dict(
                    color='white',  # White x-axis line
                    tickangle=-45,
                    showgrid=False,  # No grid
                    showticklabels=True,
                    tickmode='array',
                    tickvals=history_df['Time'],
                    ticktext=pd.to_datetime(history_df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                ),
                margin=dict(l=5, r=5, t=60, b=20),
                height=None,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                autosize=True,
                hovermode='closest'
            )

            return fig

    

import numpy as np
from sklearn.metrics import pairwise_distances
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Dummy function to be completed
def find_similar_rows(df, input_id, n=10):
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

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np



# Initialize the Dash app


# Define the SimilarInstances3DComponent as a Dash component
class SimilarInstances3DComponent:
    def __init__(self, model, full_data):
        self.full_data = full_data
        self.model = model

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
                    className='small_input'
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'marginBottom': '5px'
            }),
            html.Hr(className='HR'),
            html.Div(
                dcc.Graph(id="similar_instances_3d_graph", config={'responsive': True}),
                style={'flex': '1', 'overflow': 'hidden'}
            )
        ], className='cell')

    def component_callbacks(self, app):
        @app.callback(
            Output("similar_instances_3d_graph", "figure"),
            [Input('transaction_id', 'value'),
             Input('num_similar', 'value')]
        )
        def update_3d_graph(transaction_id, num_similar):
            if transaction_id is None or num_similar is None:
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(visible=False),  # Hide x-axis
                    yaxis=dict(visible=False),  # Hide y-axis
                    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                    paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                )
                return fig
            
            # Find similar rows based on the input ID
            similar_df = find_similar_rows(self.full_data, transaction_id, num_similar)

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
            feature_importances = np.random.rand(self.full_data.shape[1] - 1)  # Dummy feature importances
            top_features_indices = np.argsort(feature_importances)[-3:][::-1]
            top_features = self.full_data.columns[top_features_indices]

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
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),  # Transparent x-axis background
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),  # Transparent y-axis background
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)')   # Transparent z-axis background
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to transparent
                paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
                template="plotly_dark"
            )

            return fig


transaction_timeline = TransactionTimelineComponent(df)
same_instances = SimilarInstances3DComponent( model, df)
input_component = InputTextComponent(df)
feature_importance = FeatureImportanceComponent(model, features_columns)
shap_component = ShapExplanationComponent(model, df)


# Define the layout
app.layout = html.Div([
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Replace with InputTextComponent layout
                    input_component.layout(),
                ], width=2, style={'padding': '0px 5px 0px 10px'}),  # 20% width

                dbc.Col([
                    # Replace with FeatureImportanceComponent layout
                    feature_importance.layout(),
                ], width=5, style={'padding': '0px 5px 0px 5px'}),  # 30% width

                dbc.Col([
                    # Replace with ShapExplanationComponent layout
                    shap_component.layout(),
                ], width=5, style={'padding': '0px 10px 0px 5px'}),  # 50% width
            ], className='row')
        ], style={'border': 'none', 'margin': '0', 'padding': '0'}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Replace with TransactionTimelineComponent layout
                    transaction_timeline.layout(),
                ], width=4, style={'padding': '0px 5px 0px 10px'}),
                dbc.Col([
                    same_instances.layout(),
                ], width=4, style={'padding': '0px 5px 0px 10px'})
            ], className='row')
        ], style={'border': 'none', 'margin': '0', 'padding': '0'})
    ], style={'border': 'none', 'margin': '0', 'padding': '0'})
])

transaction_timeline.component_callbacks(app)
same_instances.component_callbacks(app)
input_component.component_callbacks(app)
feature_importance.component_callbacks(app)
shap_component.component_callbacks(app)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
