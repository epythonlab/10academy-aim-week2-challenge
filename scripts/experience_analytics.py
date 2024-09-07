# experience_analytics.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df

    # Task 3.1: Aggregate per customer
    def aggregate_user_experience(self):
        # Fill missing values with mean/mode
        self.df['TCP Retransmission'].fillna(self.df['TCP Retransmission'].mean(), inplace=True)
        self.df['RTT'].fillna(self.df['RTT'].mean(), inplace=True)
        # self.df['Throughput'].fillna(self.df['Throughput'].mean(), inplace=True)
        self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0], inplace=True)

        # Group by customer (e.g., MSISDN/Number) and compute mean
        user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP Retransmission': 'mean',
            'RTT': 'mean',
            # 'Throughput': 'mean',
            'Handset Type': 'first'
        }).reset_index()

        return user_agg

    # Task 3.2: Top, bottom, and most frequent values
    def get_top_bottom_most_frequent(self, column):
        top_10 = self.df[column].nlargest(10)
        bottom_10 = self.df[column].nsmallest(10)
        most_frequent = self.df[column].mode()[0]

        return top_10, bottom_10, most_frequent

    # Task 3.3: Plot distributions per handset type
    def plot_distribution(self, df, column, group_by):
        sns.boxplot(x=group_by, y=column, data=df)
        plt.title(f'Distribution of {column} per {group_by}')
        plt.xticks(rotation=90)
        plt.show()

    # Task 3.4: K-means clustering
    def k_means_clustering(self, df, features, k=3):
        # Standardize features for clustering
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)

        return df
