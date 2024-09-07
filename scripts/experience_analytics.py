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

    def aggregate_user_experience(self):
        # Fill missing values with mean/mode, avoiding chained assignment
        self.df['TCP DL Retrans. Vol (Bytes)'] = self.df['TCP DL Retrans. Vol (Bytes)'].fillna(self.df['TCP DL Retrans. Vol (Bytes)'].mean())
        self.df['TCP UL Retrans. Vol (Bytes)'] = self.df['TCP UL Retrans. Vol (Bytes)'].fillna(self.df['TCP UL Retrans. Vol (Bytes)'].mean())
        self.df['Avg RTT DL (ms)'] = self.df['Avg RTT DL (ms)'].fillna(self.df['Avg RTT DL (ms)'].mean())
        self.df['Avg RTT UL (ms)'] = self.df['Avg RTT UL (ms)'].fillna(self.df['Avg RTT UL (ms)'].mean())
        self.df['Avg Bearer TP DL (kbps)'] = self.df['Avg Bearer TP DL (kbps)'].fillna(self.df['Avg Bearer TP DL (kbps)'].mean())
        self.df['Avg Bearer TP UL (kbps)'] = self.df['Avg Bearer TP UL (kbps)'].fillna(self.df['Avg Bearer TP UL (kbps)'].mean())
        self.df['Handset Type'] = self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0])

        # Group by customer (e.g., MSISDN/Number) and compute mean
        user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Taking the first as Handset Type is categorical
        }).reset_index()

        # Compute combined fields for retransmission and RTT
        user_agg['TCP Retransmission'] = user_agg['TCP DL Retrans. Vol (Bytes)'] + user_agg['TCP UL Retrans. Vol (Bytes)']
        user_agg['RTT'] = (user_agg['Avg RTT DL (ms)'] + user_agg['Avg RTT UL (ms)']) / 2
        user_agg['Throughput'] = (user_agg['Avg Bearer TP DL (kbps)'] + user_agg['Avg Bearer TP UL (kbps)']) / 2

        # Dropping the intermediate columns
        user_agg.drop(columns=['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                            'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                            'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'], inplace=True)

        return user_agg


    # Task 3.2: Top, bottom, and most frequent values
    def get_top_bottom_most_frequent(self, column):
        df = self.aggregate_user_experience()
        top_10 = df[column].nlargest(10) # nlargest get top 10 
        bottom_10 = df[column].nsmallest(10) # nsmallest get 10 smallest
        most_frequent = df[column].mode()[0] # most frequent
        
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
