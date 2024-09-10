import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scripts.satisfaction_analytics import UserSatisfactionAnalytics

class SatisfactionDashboard:
    def __init__(self, custom_colors):
        """
        Initialize the SatisfactionDashboard with custom colors.
        """
        self.custom_colors = custom_colors
        self.data_url1 = "https://raw.githubusercontent.com/epythonlab/10academy-aim-week2-challenge/master/src/test_data/engagement_score.csv"
        self.url2 = "https://raw.githubusercontent.com/epythonlab/10academy-aim-week2-challenge/master/src/test_data/experience_score.csv"

    def load_data(self):
        """
        Load engagement and experience scores from the URLs.
        """
        engagement_scores = pd.read_csv(self.data_url1)
        experience_scores = pd.read_csv(self.url2)
        return engagement_scores, experience_scores

    def plot_regression_results(self, X, y, model):
        """
        Plot regression results including regression lines and residuals.
        """
        st.subheader('Regression Analysis')

        # Predict using the model
        y_pred = model.predict(X)

        # Create a single plot for both features
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plots for actual values
        sns.scatterplot(x=X['Engagement_Score'], y=y, ax=ax, color='blue', label='Actual vs Satisfaction (Engagement)')
        sns.scatterplot(x=X['Experience_Score'], y=y, ax=ax, color='green', label='Actual vs Satisfaction (Experience)')

        # Regression lines for both features
        sns.lineplot(x=X['Engagement_Score'], y=y_pred, color='red', label='Fitted Line (Engagement)')
        sns.lineplot(x=X['Experience_Score'], y=y_pred, color='orange', label='Fitted Line (Experience)')

        # Plot settings
        ax.set_xlabel('Scores')
        ax.set_ylabel('Satisfaction Score')
        ax.set_title('Regression Line Plot for Engagement and Experience Scores')
        ax.legend()

        st.pyplot(fig)

        st.write('### Residuals Plot')
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot')
        st.pyplot(fig)

    def plot_cluster_results(self, cluster_df, n_clusters):
        """
        Plot clustering results including cluster distribution, centroids, and PCA plot.
        """
        st.subheader('Clustering Analysis')

        st.write('### Cluster Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(cluster_df['Cluster'], bins=n_clusters, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Users')
        ax.set_title('Cluster Distribution')
        st.pyplot(fig)

        st.write('### Cluster Centroids')
        cluster_centers = cluster_df[['Engagement_Score', 'Experience_Score']].groupby(cluster_df['Cluster']).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_centers, annot=True, cmap='viridis', ax=ax)
        ax.set_xlabel('Features')
        ax.set_ylabel('Clusters')
        ax.set_title('Cluster Centroids')
        st.pyplot(fig)

        st.write('### PCA Plot')
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(cluster_df[['Engagement_Score', 'Experience_Score']])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_df['Cluster'], palette='viridis', ax=ax)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Clusters in PCA Space')
        st.pyplot(fig)

    def show_satisfaction(self):
        """
        Display the user satisfaction analytics dashboard.
        """
        st.title('User Satisfaction Analytics Dashboard')

        st.header('Top Satisfied Customers')
        engagement_scores, experience_scores = self.load_data()
        analytics = UserSatisfactionAnalytics()
        top_satisfied_customers = analytics.top_satisfied_customer(engagement_scores, experience_scores)

        col1, col2 = st.columns(2)
        with col1:
            st.write(top_satisfied_customers)

        with col2:
            if not top_satisfied_customers.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=top_satisfied_customers, x='MSISDN/Number', y='Satisfaction_Score', palette=self.custom_colors, ax=ax)
                ax.set_xlabel('MSISDN/Number')
                ax.set_ylabel('Satisfaction Score')
                ax.set_title('Top Satisfied Customers')
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                st.write("No data available for plotting.")

        st.header('Regression Model Evaluation')
        model_type = st.selectbox('Select model type', ['linear', 'ridge', 'lasso'])
        if st.button('Build and Evaluate Model'):
            model = analytics.build_regression_model(engagement_scores, experience_scores, model_type)
            st.write(f"Model Type: {model_type.capitalize()}")
            st.write("Mean Squared Error and R-squared values are printed in the console.")
            satisfaction_df = analytics.compute_satisfaction_score(engagement_scores, experience_scores)
            self.plot_regression_results(satisfaction_df[['Engagement_Score', 'Experience_Score']], 
                                         satisfaction_df['Satisfaction_Score'], model)

        st.header('Clustering')
        n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=2)
        if st.button('Perform Clustering'):
            clustering_df = analytics.perform_clustering(engagement_scores, experience_scores, n_clusters)
            st.write(clustering_df)
            self.plot_cluster_results(clustering_df, n_clusters)
