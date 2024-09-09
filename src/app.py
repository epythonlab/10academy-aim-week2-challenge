import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
try:
    from scripts.experience_analytics import ExperienceAnalytics
    from scripts.handset_analysis import HandsetAnalysis
    # from scripts.user_engagement_analysis import UserEngagementAnalysis
    from scripts.handset_dashboard import HandsetVisualization
    from scripts.satisfaction_dashboard import satisfaction_dashboard
    from scripts.user_engagement_dashboard import UserEngagementVisualizations

    print("Modules imported successfully.")
except ImportError as e:
    print(f"Error importing modules: {e}")

# Load your data
@st.cache_data
def load_data():
    data_url = url = "https://raw.githubusercontent.com/epythonlab/10academy-aim-week2-challenge/master/src/test_data/xdr_cleaned.csv"
    df = pd.read_csv(data_url)
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
    # Define a custom color palette
    custom_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Teal
    ]

    # Load data
    df = load_data()
   # Initialize the analysis and visualization classes
    try:
        handset_analysis = HandsetAnalysis(df)
        handset_visualization = HandsetVisualization(custom_colors)
        analytics = ExperienceAnalytics(df)
        engagement_vis = UserEngagementVisualizations(df, custom_colors)
    except Exception as e:
        st.error(f"Error initializing classes: {e}")
   
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to", 
        [
            "User Analysis", "User Experience", 
            "Engagement Analysis",
            "User Satisfaction Analytics"
            ]
        )

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
    elif section == "User Experience":
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

    # Engagement Analysis Section
    elif section == "Engagement Analysis":
        st.subheader("User Engagement Analysis")
        # enga_analysis = UserEngagementAnalysis(df)
        # # Aggregate metrics for engagement
        # enga_analysis.aggregate_metrics()
        
        # Show top customers by different metrics
        st.sidebar.subheader("Top Customers Metrics")
        metric_choice = st.sidebar.selectbox(
            "Select Metric for Top Customers",
            ['sessions_frequency', 
             'total_session_duration', 
             'total_download_traffic', 
             'total_upload_traffic']
        )
    
        engagement_vis.plot_top_customers(metric_choice)

        # Elbow Method Visualization
        if st.sidebar.button('Show Elbow Method'):
            enga_analysis.normalize_and_cluster()
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(enga_analysis.normalized_metrics)
                wcss.append(kmeans.inertia_)
            engagement_vis.plot_elbow_method(wcss)

        # Cluster Summary Visualization
        if st.sidebar.button('Show Cluster Summary'):
            enga_analysis.normalize_and_cluster(n_clusters=3)
            cluster_summary = enga_analysis.cluster_summary()
            engagement_vis.plot_cluster_summary(cluster_summary)
            
         # Top 3 apps used by customers
        if st.sidebar.button('Show Top 3 Apps'):
            # Define application traffic columns
            applications = {
                'YouTube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
                'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
                'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
                'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
            }
            app_total_traffic, top_10_engaged_per_app = enga_analysis.aggregate_traffic_per_application(applications=applications)
            top_3_apps = app_total_traffic.nlargest(3, 'total_bytes')
            engagement_vis.plot_top_applications(top_3_apps)
         

    # Satisfaction Dashboard Section
    elif section == "User Satisfaction Analytics":
        satisfaction_dashboard(custom_colors)

if __name__ == "__main__":
    main()
