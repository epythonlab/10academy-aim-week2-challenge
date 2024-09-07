import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df

    # Task 3.1: Aggregate user experience metrics (handling missing values and outliers)
    def aggregate_user_experience(self):
        # Fill missing values with mean for numerical columns and mode for categorical columns
        self.df['TCP DL Retrans. Vol (Bytes)'] = self.df['TCP DL Retrans. Vol (Bytes)'].fillna(self.df['TCP DL Retrans. Vol (Bytes)'].mean())
        self.df['TCP UL Retrans. Vol (Bytes)'] = self.df['TCP UL Retrans. Vol (Bytes)'].fillna(self.df['TCP UL Retrans. Vol (Bytes)'].mean())
        self.df['Avg RTT DL (ms)'] = self.df['Avg RTT DL (ms)'].fillna(self.df['Avg RTT DL (ms)'].mean())
        self.df['Avg RTT UL (ms)'] = self.df['Avg RTT UL (ms)'].fillna(self.df['Avg RTT UL (ms)'].mean())
        self.df['Avg Bearer TP DL (kbps)'] = self.df['Avg Bearer TP DL (kbps)'].fillna(self.df['Avg Bearer TP DL (kbps)'].mean())
        self.df['Avg Bearer TP UL (kbps)'] = self.df['Avg Bearer TP UL (kbps)'].fillna(self.df['Avg Bearer TP UL (kbps)'].mean())
        self.df['Handset Type'] = self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0])

        # Group by customer (MSISDN/Number) and compute aggregated metrics
        user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Taking the first occurrence for categorical column
        }).reset_index()

        # Combine TCP retransmission, RTT, and throughput metrics
        user_agg['TCP Retransmission'] = user_agg['TCP DL Retrans. Vol (Bytes)'] + user_agg['TCP UL Retrans. Vol (Bytes)']
        user_agg['RTT'] = (user_agg['Avg RTT DL (ms)'] + user_agg['Avg RTT UL (ms)']) / 2
        user_agg['Throughput'] = (user_agg['Avg Bearer TP DL (kbps)'] + user_agg['Avg Bearer TP UL (kbps)']) / 2

        # Drop intermediate columns
        user_agg.drop(columns=['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                               'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                               'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'], inplace=True)

        return user_agg

    # Task 3.2: Compute top, bottom, and most frequent values
    def get_top_bottom_most_frequent(self, column):
        df = self.aggregate_user_experience()
        # Compute top 10 largest values
        top_10 = df[column].nlargest(10)
    
        # Compute top 10 smallest values
        bottom_10 = df[column].nsmallest(10)
        
        most_frequent = df[column].mode()  # Most frequent value

        return top_10,bottom_10, most_frequent
    

    def avg_throughput_per_handset(self):
        
        throughput_cols = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
        self.df['Avg_Throughput'] = self.df[throughput_cols].mean(axis=1)
        avg_throughput_per_handset = self.df.groupby('Handset Type')['Avg_Throughput'].mean().reset_index()
        return avg_throughput_per_handset

    def avg_tcp_rtt_per_handset(self):
        self.df['TCP_Retransmission'] = self.df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].sum(axis=1)
        avg_tcp_retransmission_per_handset = self.df.groupby('Handset Type')['TCP_Retransmission'].mean().reset_index()
        return avg_tcp_retransmission_per_handset

    def plot_distribution(self, metric):
        '''
        Plot the distribution of the specified metric
        for the top 10 handsets by average throughput or
        TCP retransmission
        '''
        # Compute average metric per handset type
        if metric == 'Throughput':
            avg_metric_per_handset = self.avg_throughput_per_handset()
        elif metric == 'TCP Retransmission':
            avg_metric_per_handset = self.avg_tcp_rtt_per_handset()
        else:
            raise ValueError("Metric must be 'Throughput' or 'TCP Retransmission'")
        
        # Get top 10 handsets by the specified metric
        top_10_handsets = avg_metric_per_handset.sort_values(by=avg_metric_per_handset.columns[1], ascending=False).head(10)
        
        # Plot distribution for top 10 handsets
        sns.barplot(x='Handset Type', y=avg_metric_per_handset.columns[1], data=top_10_handsets)
        plt.title(f'Distribution of {metric} for Top 10 Handsets')
        plt.xticks(rotation=90)
        plt.show()
        
    # Task 3.4: K-means clustering to segment users into experience groups
    def k_means_clustering(self, features, k=3):
        df = self.aggregate_user_experience()

        # Standardize the feature data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Cluster centers
        cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
        cluster_centers['Cluster'] = range(k)

        return df, cluster_centers

