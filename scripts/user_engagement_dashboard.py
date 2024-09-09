import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class UserEngagementVisualizations:
    
    def __init__(self, data, custom_colors):
        self.data = data
        self.custom_colors = custom_colors

    def plot_top_customers(self, top_customers, metric_name):
        # Plot top customers for a given metric
        plt.figure(figsize=(12, 6))
        sns.barplot(x='MSISDN/Number', y=metric_name, data=top_customers.reset_index(drop=True), palette=self.custom_colors)
        plt.title(f'Top 10 Customers by {metric_name}')
        plt.xlabel('Customer ID (MSISDN/Number)')
        plt.ylabel(metric_name)
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure for the next plot
        
    def plot_top_applications(self, top_3_apps):
        # Plot the top 3 most used applications
        plt.figure(figsize=(12, 6))
        sns.barplot(x='application', y='total_bytes', data=top_3_apps, palette=self.custom_colors)
        plt.title('Top 3 Most Used Applications by Total Traffic')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        # Display the plot
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure for the next plot

    def plot_elbow_method(self, wcss):
        # Plot the elbow method for optimal k
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure for the next plot

    def plot_cluster_summary(self, cluster_summary):
        # Visualizing each metric separately
        metrics = ['sessions_frequency', 'total_session_duration', 'total_download_traffic', 'total_upload_traffic']
      
        # Iterate through each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
    
            # Select rows corresponding to the current metric and transpose
            met = cluster_summary[metric].T.reset_index()
            met.columns = ['Summary Statistic', 'Cluster 0', 'Cluster 1', 'Cluster 2']
                      
            # Melt DataFrame to long format for seaborn
            met_long = met.melt(id_vars='Summary Statistic', var_name='Cluster', value_name='Value')
                  
            # Plot using seaborn
            sns.barplot(x='Summary Statistic', y='Value', hue='Cluster', data=met_long, ax=ax)
            
            ax.set_title(f'Cluster Summary for {metric.replace("_", " ").capitalize()}')
            ax.set_xlabel('Summary Statistic')
            ax.set_ylabel('Value')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Show plot in Streamlit
            st.pyplot(fig)
            plt.clf()  # Clear the figure for the next plot
