import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans

# Add the 'scripts' directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join('..', 'scripts')))
from experience_analytics import ExperienceAnalytics
from handset_analysis import HandsetAnalysis
from handset_dashboard import HandsetVisualization

# Load your data
@st.cache_data
def load_data():
    df = pd.read_csv('xdr_cleaned.csv')
    return df

            
# Create a function to perform K-Means clustering and visualize the results
def perform_clustering(analytics, agg, features, n_clusters):
    clustered_df, cluster_centers_ = analytics.k_means_clustering(features, n_clusters)

    st.subheader("Clustered Data")
    st.write(clustered_df)

    fig, ax = plt.subplots()
    sns.scatterplot(data=clustered_df, x=features[0], y=features[1], hue='Cluster', palette='viridis', ax=ax)
    ax.set_title('K-Means Clustering Results')
    st.pyplot(fig)

    st.subheader("Cluster Centers")
    centers = pd.DataFrame(cluster_centers_, columns=features)
    st.write(centers)

# Streamlit app
def main():
    st.title("Telecom User & Handset Analytics Dashboard")

    # Load data
    df = load_data()
   # Initialize the analysis and visualization classes
    # df = load_data()
    handset_analysis = HandsetAnalysis(df)
    handset_visualization = HandsetVisualization()
    analytics = ExperienceAnalytics(df)

   
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["User Analysis", "K-Means Clustering", "Additional Visualizations"])

    # User Analysis Section
    if section == "User Analysis":
        st.subheader("User & Handset Analysis")
        # Sidebar for interaction
        st.sidebar.title("Telecom Data Analysis")
        
        # Top handsets visualization
        top_n = st.slider("Number of top handsets to display", 5, 20, 10)
        top_handsets = handset_analysis.top_handsets(top_n)
        handset_visualization.visualize_top_handsets(top_handsets, top_n)

        # Top manufacturers visualization
        top_n = st.slider("Number of top Manufacturer", 2, 10, 3)
        top_manufacturers = handset_analysis.top_manufacturers(top_n)
        handset_visualization.visualize_top_manufacturers(top_manufacturers, top_n)

        # Display top handsets for each top manufacturer
        manufacturers = st.multiselect(
            "Select manufacturers", 
            handset_analysis.top_manufacturers(top_n).index.tolist())
        if manufacturers:
            # Top handsets per manufacturer visualization
            top_handsets_per_manufacturer = handset_analysis.top_handsets_per_manufacturer(manufacturers)
            handset_visualization.visualize_top_handsets_per_manufacturer(top_handsets_per_manufacturer, manufacturers, top_n)


    # K-Means Clustering Section
    elif section == "K-Means Clustering":
        st.subheader("User Experience Analytics")

        # Load aggregated data for clustering
        agg = analytics.aggregate_user_experience()

        # User inputs for clustering
        st.sidebar.header("Clustering Parameters")
        features = st.sidebar.multiselect(
            "Select features for clustering",
            agg.columns.tolist(),
            default=['TCP Retransmission', 'RTT', 'Throughput']
        )
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

        if len(features) > 0:
            # Perform clustering and visualize the results
            perform_clustering(analytics, agg, features, n_clusters)

    # Additional Visualizations Section
    elif section == "Additional Visualizations":
        st.subheader("Additional Data Visualizations")
        # Add more visualizations here as needed
        st.write("More visualizations coming soon!")

if __name__ == "__main__":
    main()
