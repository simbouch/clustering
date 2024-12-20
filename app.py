from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
import logging
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the clustering pipeline
pipeline_path = os.getenv("MODEL_PATH", "./models/clustering_pipeline_fixed.pkl")
if not os.path.exists(pipeline_path):
    logger.error(f"Model file not found at {pipeline_path}")
    clustering_pipeline = None
else:
    clustering_pipeline = joblib.load(pipeline_path)
    logger.info(f"Clustering pipeline loaded successfully from {pipeline_path}")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files.get("file", None)
        if not uploaded_file:
            logger.error("No file uploaded.")
            return "Error: Please upload a CSV file."

        try:
            # Read uploaded file
            data = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")
            logger.info(f"Uploaded file columns: {list(data.columns)}")

            # Select numeric columns
            numeric_data = data.select_dtypes(include=["float64", "int64"])
            logger.info(f"Numeric columns detected: {list(numeric_data.columns)}")

            if numeric_data.empty:
                logger.error("No numeric columns found in the uploaded file.")
                return "Error: No numeric data found in the uploaded file."

            # Align features with the model
            if clustering_pipeline:
                expected_features = clustering_pipeline.named_steps["scaler"].feature_names_in_
                logger.info(f"Model expects features: {list(expected_features)}")
                missing_features = [col for col in expected_features if col not in numeric_data.columns]
                if missing_features:
                    logger.error(f"Missing features: {missing_features}")
                    return f"Error: Missing required features: {missing_features}"

                numeric_data = numeric_data[expected_features]
                cluster_labels = clustering_pipeline.predict(numeric_data)
            else:
                logger.error("Clustering pipeline not loaded.")
                return "Error: Clustering pipeline not loaded."

            # Add cluster labels to the data
            data["Cluster"] = cluster_labels

            # Save clustered data
            output_path = "./data/output_with_clusters.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_csv(output_path, index=False)
            logger.info(f"Clustered data saved to {output_path}")

            # Generate clustering summary
            numeric_data["Cluster"] = cluster_labels
            cluster_summary = numeric_data.groupby("Cluster").mean()

            # Generate visualizations
            plots = generate_visualizations(numeric_data, cluster_labels)

            # Render results
            return render_template(
                "results.html",
                tables=[cluster_summary.to_html(classes="table")],
                plots=plots,
                download_path="/download"
            )

        except Exception as e:
            logger.error(f"Error during file processing: {e}")
            return f"An error occurred while processing the file: {str(e)}"

    return render_template("upload.html")

@app.route("/download")
def download():
    file_path = "./data/output_with_clusters.csv"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "Error: Clustered file not found. Please upload and process a dataset first."

def generate_visualizations(numeric_data, cluster_labels):
    plots = []

    # Scatter Plot of Clusters
    plt.figure(figsize=(8, 6))
    for cluster in numeric_data["Cluster"].unique():
        cluster_points = numeric_data[numeric_data["Cluster"] == cluster]
        plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f"Cluster {cluster}")
    plt.title("Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plots.append(figure_to_base64())

    # Distribution of Cluster Counts
    plt.figure(figsize=(8, 6))
    sns.countplot(x=cluster_labels, palette="viridis")
    plt.title("Distribution of Cluster Counts")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plots.append(figure_to_base64())

    # Box Plot for Features
    numeric_data_melted = numeric_data.melt(id_vars=["Cluster"], var_name="Feature", value_name="Value")
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=numeric_data_melted, x="Feature", y="Value", hue="Cluster", palette="viridis")
    plt.title("Box Plot of Features by Cluster")
    plt.xticks(rotation=45)
    plots.append(figure_to_base64())

    return plots

def figure_to_base64():
    """Converts the current Matplotlib figure to a Base64 string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    return base64_string

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
