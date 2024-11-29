from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import glob
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically resolve the clustering pipeline path
pipeline_path = os.getenv("MODEL_PATH", None)
if not pipeline_path:
    try:
        # Dynamic resolution for Azure Linux environments
        dynamic_pipeline_path = glob.glob("/home/site/wwwroot/models/clustering_pipeline.pkl")
        if dynamic_pipeline_path:
            pipeline_path = dynamic_pipeline_path[0]
            logger.info(f"Model found dynamically at: {pipeline_path}")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        # Fall back to a hardcoded path
        pipeline_path = "./models/clustering_pipeline.pkl"
        logger.warning(f"Dynamic resolution failed. Using fallback path: {pipeline_path}")

# Load the clustering pipeline
try:
    clustering_pipeline = joblib.load(pipeline_path)
    logger.info(f"Clustering pipeline loaded successfully from: {pipeline_path}")
except Exception as e:
    logger.error(f"Failed to load clustering pipeline: {e}")
    clustering_pipeline = None

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files.get("file", None)
        if not uploaded_file:
            logger.error("No file uploaded.")
            return "Error: Please upload a CSV file."

        try:
            # Read the uploaded file
            logger.info(f"Uploaded file: {uploaded_file.filename}")
            data = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

            # Select numeric columns for clustering
            numeric_data = data.select_dtypes(include=["float64", "int64"])
            logger.info(f"Numeric columns detected: {numeric_data.columns.tolist()}")

            if numeric_data.empty:
                logger.error("No numeric data found in the uploaded file.")
                return "Error: No numeric data found in the uploaded file."

            # Validate the pipeline and align data
            if clustering_pipeline is not None:
                required_features = clustering_pipeline.named_steps["scaler"].feature_names_in_
                logger.info(f"Required features for the model: {required_features}")

                # Align input data with required features
                missing_features = [f for f in required_features if f not in numeric_data.columns]
                if missing_features:
                    logger.error(f"Missing features: {missing_features}")
                    return f"Error: The following required features are missing: {missing_features}"

                numeric_data = numeric_data[required_features]
                cluster_labels = clustering_pipeline.predict(numeric_data)
            else:
                logger.error("Clustering pipeline not loaded.")
                return "Error: Clustering pipeline not loaded."

            # Add cluster labels to the original data
            data["Cluster"] = cluster_labels

            # Save clustered data to a file
            output_path = "/home/site/wwwroot/data/output_with_clusters.csv"
            os.makedirs("/home/site/wwwroot/data", exist_ok=True)
            data.to_csv(output_path, index=False)
            logger.info(f"Clustered data saved to: {output_path}")

            # Generate clustering summary
            numeric_data["Cluster"] = cluster_labels
            cluster_summary = numeric_data.groupby("Cluster").mean()

            # Render results with a summary
            return render_template(
                "results.html",
                tables=[cluster_summary.to_html(classes="table")],
                download_path="/download"
            )

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return f"An error occurred while processing the file: {str(e)}"

    return render_template("upload.html")


@app.route("/download")
def download():
    file_path = "/home/site/wwwroot/data/output_with_clusters.csv"
    if os.path.exists(file_path):
        logger.info(f"Sending clustered file to user: {file_path}")
        return send_file(file_path, as_attachment=True)
    else:
        logger.error("Clustered file not found.")
        return "Error: Clustered file not found. Please upload and process a dataset first."


if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
