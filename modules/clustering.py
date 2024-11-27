import pandas as pd
import os
import joblib


def load_pipeline(pipeline_path='./models/clustering_pipeline.pkl'):
    """Load the clustering pipeline."""
    if os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    else:
        raise FileNotFoundError("Clustering pipeline not found. Please check the models directory.")


def process_data(file, pipeline):
    """Process uploaded file and apply the clustering pipeline."""
    try:
        # Read uploaded file
        data = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")

        # Select numeric columns
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in the uploaded file for clustering.")

        # Align with the pipeline feature requirements
        required_features = pipeline.named_steps['scaler'].feature_names_in_
        missing_features = [col for col in required_features if col not in numeric_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Apply clustering pipeline
        numeric_data = numeric_data[required_features]  # Align columns
        cluster_labels = pipeline.predict(numeric_data)

        # Add cluster labels to the original data
        data["Cluster"] = cluster_labels

        # Save results to CSV
        output_path = "./data/output_with_clusters.csv"
        os.makedirs("./data", exist_ok=True)
        data.to_csv(output_path, index=False)

        # Generate clustering summary
        numeric_data["Cluster"] = cluster_labels
        cluster_summary = numeric_data.groupby("Cluster").mean()

        return cluster_labels, cluster_summary, output_path
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")
