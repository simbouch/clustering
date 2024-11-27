from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

app = Flask(__name__)

# Load the saved clustering pipeline
pipeline_path = "./models/clustering_pipeline_fixed.pkl"
if os.path.exists(pipeline_path):
    clustering_pipeline = joblib.load(pipeline_path)
    print("Clustering pipeline loaded successfully.")
else:
    clustering_pipeline = None
    print("ERROR: Clustering pipeline not found. Ensure the model is saved in the 'models/' directory.")


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files.get("file", None)
        if not uploaded_file:
            return "Error: Please upload a CSV file."

        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

            # Select numeric columns for clustering
            numeric_data = data.select_dtypes(include=["float64", "int64"])
            if numeric_data.empty:
                return "Error: No numeric data found in the uploaded file."

            # Validate the pipeline and align data
            if clustering_pipeline is not None:
                required_features = clustering_pipeline.named_steps["scaler"].feature_names_in_
                numeric_data = numeric_data[required_features]
                cluster_labels = clustering_pipeline.predict(numeric_data)
            else:
                return "Error: Clustering pipeline not found. Ensure the model is saved."

            # Add cluster labels to the original data
            data["Cluster"] = cluster_labels

            # Save the clustered data to a file
            output_path = "./data/output_with_clusters.csv"
            os.makedirs("./data", exist_ok=True)
            data.to_csv(output_path, index=False)

            # Generate clustering summary
            numeric_data["Cluster"] = cluster_labels
            cluster_summary = numeric_data.groupby("Cluster").mean()

            # Generate visualizations
            plots = generate_visualizations(numeric_data, cluster_labels)

            # Render the results page
            return render_template(
                "results.html",
                tables=[cluster_summary.to_html(classes="table")],
                plots=plots,
                download_path="/download"
            )

        except Exception as e:
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
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    print("Starting Flask app...")
    app.run(debug=True)
