import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import base64
import io
import pandas as pd
import re
from sklearn.metrics import pairwise_distances
import shap


color_positive = "rgba(211, 112, 0, 0.8)"
color_negative = "rgba(5, 0, 97, 0.8)"

color_positive_darker = "rgba(54, 29, 0, 0.8)"
color_negative_darker = "rgba(2, 0, 40, 0.8)"


def rgba_to_rgb(rgba_string):
    rgba_values = rgba_string.replace('rgba(', '').replace(')', '').split(',')
    r, g, b = rgba_values[:3] 
    return f'rgb({r.strip()}, {g.strip()}, {b.strip()})'

# Dummy function to be completed
def find_similar_rows(df, input_id, dataset_config, n=10):
  
    
    columns_config = dataset_config["columns_config"]

    row = df.loc[df[columns_config["id"]] == input_id]
    input_features = row[columns_config["features"]]
    
    if input_features.empty:
        return pd.DataFrame()
  
    distances = pairwise_distances(input_features, df[columns_config["features"]]).flatten()
    
    closest_indices = np.argsort(distances)[:n]
    
    return df.iloc[closest_indices]

# Dummy function to be completed
def find_history(df, id, maxlength=5):
    return df.sample(n=maxlength)

def rgba_to_hex(rgba):
    """Convert an RGBA color string to its hex equivalent."""
    rgba = rgba.replace('rgba(', '').replace(')', '').split(',')
    r, g, b, a = [int(rgba[0]), int(rgba[1]), int(rgba[2]), float(rgba[3])]
    
    hex_color = f'#{r:02x}{g:02x}{b:02x}{int(a * 255):02x}'
    
    return hex_color

def extract_rgb(color_str):
    """Extract RGB components from an RGBA color string."""
    match = re.match(r'rgba\((\d+), (\d+), (\d+), [0-9.]+\)', color_str)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError(f"Invalid color format: {color_str}")

def calculate_gradient_color(score, color_positive, color_negative):
    """Calculate the gradient color based on the score, interpolating between two RGBA colors."""
    r1, g1, b1 = extract_rgb(color_positive)
    r2, g2, b2 = extract_rgb(color_negative)
    
    r = int(r1 * score + r2 * (1 - score))
    g = int(g1 * score + g2 * (1 - score))
    b = int(b1 * score + b2 * (1 - score))
    
    return f'rgba({r}, {g}, {b}, 1)'

def shap_waterfall_image(operation_id, num_features, model, full_data, dataset_config):
    """Generates a SHAP waterfall plot for a specific operation ID."""

    columns_config = dataset_config["columns_config"]

    if operation_id is None:
        return None, "No ID selected"

    if full_data.empty:
        return None, "Error: The dataset is empty. Please provide a non-empty dataset."

    row = full_data[full_data[columns_config["id"]] == operation_id]
    if row.empty:
        return None, f"Error: No matching data found for operation ID {operation_id}."

    if not isinstance(num_features, int) or num_features <= 0:
        return None, "Error: num_features must be a positive integer."

    if not isinstance(columns_config["features"], list) or not columns_config["features"]:
        return None, "Error: features_columns must be a non-empty list of feature names."

    if not hasattr(model, "predict"):
        return None, "Error: The model does not have a predict method. Please provide a valid model."

    feature_values = row[columns_config["features"]].iloc[0]

    explainer = shap.Explainer(model, full_data[columns_config["features"]])
    shap_values = explainer(row[columns_config["features"]])

    predicted_class = int(row[columns_config["class"]].values[0])

    shap_values_instance = shap_values.values[0, :, predicted_class]

    base_value = shap_values.base_values[0, predicted_class]

    shap_exp = shap.Explanation(
        values=shap_values_instance,
        base_values=base_value,
        data=feature_values.values,
        feature_names=feature_values.index
    )

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_exp, max_display=num_features, show=False)

    fig.patch.set_facecolor('#282828')
    ax.patch.set_facecolor('#282828')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    for text in ax.texts:
        text.set_color('white')
    for artist in ax.artists:
        artist.set_edgecolor('white')
    for line in ax.lines:
        line.set_color('white')

    positive_color_hex = rgba_to_hex(color_positive)
    negative_color_hex = rgba_to_hex(color_negative)
    positive_color_darker_hex = rgba_to_hex(color_positive_darker)
    negative_color_darker_hex = rgba_to_hex(color_negative_darker)

    for rect in ax.patches:
        if isinstance(rect, matplotlib.patches.Rectangle):
            if rect.get_width() > 0:
                rect.set_facecolor(positive_color_hex)
                rect.set_edgecolor(positive_color_darker_hex)
            else:
                rect.set_facecolor(negative_color_hex)
                rect.set_edgecolor(negative_color_darker_hex)
        elif isinstance(rect, matplotlib.patches.FancyArrow):
            rect.set_linewidth(2)
            if rect._dx > 0:
                rect.set_facecolor(positive_color_hex)
                rect.set_edgecolor(positive_color_darker_hex)
            else:
                rect.set_facecolor(negative_color_hex)
                rect.set_edgecolor(negative_color_darker_hex)

    ax.figure.canvas.draw()

    plt.savefig(buf := io.BytesIO(), format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}", f"Waterfall plot for: {operation_id}"


