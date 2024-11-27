
 # Clustering Pipeline Web Application

This repository contains a web-based application for clustering data using a pre-trained PCA and K-Means pipeline. The application allows users to upload datasets, process the data, and visualize clustering results interactively. It also supports synthetic data generation for testing purposes.

## Features

- **Dynamic File Upload**: Users can upload CSV files for clustering.
- **Pre-trained Model Integration**: Uses a pre-trained pipeline with PCA and K-Means for clustering.
- **Data Visualization**: Displays clustering results with insightful plots.
- **Synthetic Data Generation**: Generates and downloads synthetic data for testing the application.
- **Cluster Summaries**: Provides statistical summaries of clusters.
- **Easy-to-Use Interface**: Simple upload and visualization workflow powered by Flask.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
Set up a virtual environment:


python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:


pip install -r requirements.txt
Ensure the models/ directory contains the pre-trained pipeline file clustering_pipeline.pkl.

Run the Flask application:


python app.py
Access the application at http://127.0.0.1:5000 in your web browser.

File Structure

.
├── app.py                   # Main Flask application
├── models/
│   └── clustering_pipeline.pkl  # Pre-trained clustering pipeline
├── static/
│   └── plots/               # Generated visualizations
├── templates/
│   ├── upload.html          # Upload page
│   ├── results.html         # Results page with visualizations
├── data/
│   ├── output_with_clusters.csv # Clustered output data
│   └── synthetic_large_data.csv # Synthetic data (if generated)
├── requirements.txt         # Python dependencies
├── README.md                # Project description
└── synthetic_data_generator.py # Script to generate synthetic data
Usage
Upload and Cluster Data
Go to the web interface.
Upload a CSV file containing numeric data.
View the cluster summary and visualization plots on the results page.
Download the clustered dataset with labels.
Generate Synthetic Data
Run the synthetic data generator script:


python synthetic_data_generator.py
The synthetic data will be saved in the data/ directory.

Visualizations
The application provides visualizations for:

Clusters on PCA-reduced data.
Summary statistics for each clu
 
