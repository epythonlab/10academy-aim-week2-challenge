import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

from satisfaction_analytics import UserSatisfactionAnalytics

# Load the data
@st.cache_data
def load_data():
    data_url1 = "https://raw.githubusercontent.com/epythonlab/10academy-aim-week2-challenge/master/src/test_data/engagement_score.csv"
    url2 = "https://raw.githubusercontent.com/epythonlab/10academy-aim-week2-challenge/master/src/test_data/experience_score.csv"
    engagement_scores = pd.read_csv(data_url1)
    experience_scores = pd.read_csv(url2)
    return engagement_scores, experience_scores

# Initialize the UserSatisfactionAnalytics class
analytics = UserSatisfactionAnalytics()

# Load data
engagement_scores, experience_scores = load_data()

# Compute satisfaction scores
satisfaction_df = analytics.compute_satisfaction_score(engagement_scores, experience_scores)

def plot_regression_results(X, y, model):
    st.subheader('Regression Analysis')
    
    # Regression Line Plot for Engagement Score
    st.write('### Regression Line Plot for Engagement Score')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of Engagement Score vs Satisfaction Score
    sns.scatterplot(x=X['Engagement_Score'], y=y, ax=ax, label='Actual Engagement vs Satisfaction')
    
    # Predict using both features
    y_pred = model.predict(X)

    # Plot predicted values against Engagement Score
    sns.lineplot(x=X['Engagement_Score'], y=y_pred, color='red', ax=ax, label='Fitted Line for Engagement')

    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Satisfaction Score')
    ax.set_title('Regression Line Plot for Engagement Score')
    ax.legend()
    st.pyplot(fig)

    # Regression Line Plot for Experience Score
    st.write('### Regression Line Plot for Experience Score')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of Experience Score vs Satisfaction Score
    sns.scatterplot(x=X['Experience_Score'], y=y, ax=ax, label='Actual Experience vs Satisfaction')
    
    # Plot predicted values against Experience Score (Note: This is only a visualization for the second feature)
    sns.lineplot(x=X['Experience_Score'], y=y_pred, color='blue', ax=ax, label='Fitted Line for Experience')

    ax.set_xlabel('Experience Score')
    ax.set_ylabel('Satisfaction Score')
    ax.set_title('Regression Line Plot for Experience Score')
    ax.legend()
    st.pyplot(fig)

    # Residuals Plot
    st.write('### Residuals Plot')
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Plot')
    st.pyplot(fig)


def plot_cluster_results(cluster_df, n_clusters):
    st.subheader('Clustering Analysis')
    
    # Cluster Distribution Plot
    st.write('### Cluster Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cluster_df['Cluster'], bins=n_clusters, edgecolor='black')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Users')
    ax.set_title('Cluster Distribution')
    st.pyplot(fig)
    
    # Cluster Centroids Plot
    st.write('### Cluster Centroids')
    cluster_centers = cluster_df[['Engagement_Score', 'Experience_Score']].groupby(cluster_df['Cluster']).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_centers, annot=True, cmap='viridis', ax=ax)
    ax.set_xlabel('Features')
    ax.set_ylabel('Clusters')
    ax.set_title('Cluster Centroids')
    st.pyplot(fig)
    
    # PCA Plot
    st.write('### PCA Plot')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(cluster_df[['Engagement_Score', 'Experience_Score']])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_df['Cluster'], palette='viridis', ax=ax)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Clusters in PCA Space')
    st.pyplot(fig)

def satisfaction_dashboard(custom_colors):
    st.title('User Satisfaction Analytics Dashboard')

    # Option to display top satisfied customers
    st.header('Top Satisfied Customers')
    top_satisfied_customers = analytics.top_satisfied_customer(engagement_scores, experience_scores)
  
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display the DataFrame in the first column
    with col1:
        st.write(top_satisfied_customers)
    
    # Display the bar chart in the second column
    with col2:
        if not top_satisfied_customers.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=top_satisfied_customers, x='MSISDN/Number', y='Satisfaction_Score', palette='viridis', ax=ax)
            ax.set_xlabel('MSISDN/Number')
            ax.set_ylabel('Satisfaction Score')
            ax.set_title('Top Satisfied Customers')
            plt.xticks(rotation=90)  # Rotate x labels for better readability
            st.pyplot(fig)
        else:
            st.write("No data available for plotting.")

    # Option to build and evaluate the regression model
    st.header('Regression Model Evaluation')
    model_type = st.selectbox('Select model type', ['linear', 'ridge', 'lasso'])
    if st.button('Build and Evaluate Model'):
        model = analytics.build_regression_model(engagement_scores, experience_scores, model_type)
        
        st.write(f"Model Type: {model_type.capitalize()}")
        st.write("Mean Squared Error and R-squared values are printed in the console.")
        plot_regression_results(satisfaction_df[['Engagement_Score', 'Experience_Score']], satisfaction_df['Satisfaction_Score'], model)


    # Option to perform clustering
    st.header('Clustering')
    n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=2)
    if st.button('Perform Clustering'):
        clustering_df = analytics.perform_clustering(engagement_scores, experience_scores, n_clusters)
        st.write(clustering_df)

        # Show clustering results
        plot_cluster_results(clustering_df, n_clusters)

    # # Option to export data to PostgreSQL
    # st.header('Export Data to PostgreSQL')
    # if st.button('Export Data'):
    #     try:
    #         analytics.export_to_postgresql(satisfaction_df)
    #         st.success('Data exported successfully!')
    #     except Exception as e:
    #         st.error(f"Error: {e}")

    # # Option to upload and view a saved model
    # st.header('Model Evaluation and Upload')
    # uploaded_model = st.file_uploader("Upload a model file", type=["pkl"])
    # if uploaded_model is not None:
    #     model = pickle.load(uploaded_model)
    #     st.write("Model uploaded successfully.")
    #     # Display the model type or other information if needed
