# # scripts/satsfaction_analytics.py

# import pandas as pd
# import numpy as np
# from sklearn.metrics import euclidean_distances
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import psycopg2
# from psycopg2 import sql
# import pickle
# import matplotlib.pyplot as plt

# class UserSatisfactionAnalysis:
#     def __init__(self, engagement_data, experience_data):
#         self.engagement_data = engagement_data
#         self.experience_data = experience_data
#         self.satisfaction_data = None

#     def calculate_scores(self, engagement_clusters, experience_clusters):
#         # Assign engagement scores
#         scaler_engagement = StandardScaler()
#         engagement_features = scaler_engagement.fit_transform(self.engagement_data[['total_session_duration', 'total_download_traffic', 'total_upload_traffic']])
#         engagement_data_with_scores = self.engagement_data.copy()
#         engagement_data_with_scores['engagement_score'] = euclidean_distances(
#             engagement_features, engagement_clusters.loc[0, ['total_session_duration', 'total_download_traffic', 'total_upload_traffic']]
#         )

#         # Assign experience scores
#         scaler_experience = StandardScaler()
#         experience_features = scaler_experience.fit_transform(self.experience_data[['TCP Retransmission', 'RTT', 'Throughput']])
#         experience_data_with_scores = self.experience_data.copy()
#         experience_data_with_scores['experience_score'] = euclidean_distances(
#             experience_features, experience_clusters.loc[0, ['TCP Retransmission', 'RTT', 'Throughput']]
#         )

#         # Merge engagement and experience scores
#         self.satisfaction_data = engagement_data_with_scores.merge(experience_data_with_scores, on='MSISDN/Number')
#         self.satisfaction_data['satisfaction_score'] = (self.satisfaction_data['engagement_score'] + self.satisfaction_data['experience_score']) / 2

#     def top_10_satisfied_customers(self):
#         # Report top 10 satisfied customers
#         return self.satisfaction_data.nlargest(10, 'satisfaction_score')

#     def build_regression_model(self):
#         # Prepare data for regression model
#         X = self.satisfaction_data[['engagement_score', 'experience_score']]
#         y = self.satisfaction_data['satisfaction_score']

#         # Fit the regression model
#         model = LinearRegression()
#         model.fit(X, y)

#         # Save the model
#         with open('satisfaction_model.pkl', 'wb') as file:
#             pickle.dump(model, file)

#         return model

#     def k_means_on_scores(self, k=2):
#         # Apply K-means clustering on engagement and experience scores
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         self.satisfaction_data['cluster'] = kmeans.fit_predict(self.satisfaction_data[['engagement_score', 'experience_score']])

#         # Cluster summary
#         cluster_summary = self.satisfaction_data.groupby('cluster').agg({
#             'satisfaction_score': ['mean'],
#             'experience_score': ['mean']
#         }).reset_index()
#         return cluster_summary

#     def export_to_postgresql(self, host, database, user, password):
#         # Export data to PostgreSQL
#         try:
#             connection = psycopg2.connect(
#                 host=host,
#                 database=database,
#                 user=user,
#                 password=password
#             )
#             cursor = connection.cursor()

#             # Drop table if exists and create a new one
#             cursor.execute('DROP TABLE IF EXISTS user_satisfaction_scores')
#             cursor.execute('''
#                 CREATE TABLE user_satisfaction_scores (
#                     MSISDN VARCHAR(20),
#                     engagement_score FLOAT,
#                     experience_score FLOAT,
#                     satisfaction_score FLOAT,
#                     cluster INT
#                 )
#             ''')

#             # Insert data into the table
#             insert_query = '''
#                 INSERT INTO user_satisfaction_scores (MSISDN, engagement_score, experience_score, satisfaction_score, cluster)
#                 VALUES (%s, %s, %s, %s, %s)
#             '''
#             for _, row in self.satisfaction_data.iterrows():
#                 cursor.execute(insert_query, (row['MSISDN/Number'], row['engagement_score'], row['experience_score'], row['satisfaction_score'], row['cluster']))
            
#             connection.commit()
#         except Exception as e:
#             print(f"Error: {e}")
#         finally:
#             if connection:
#                 cursor.close()
#                 connection.close()

#     def model_deployment_tracking(self):
#         # This function would include deployment and monitoring details
#         # For demonstration, print dummy tracking information
#         print("Model Deployment Tracking")
#         print(f"Code Version: 1.0")
#         print(f"Start Time: {pd.Timestamp.now()}")
#         print(f"End Time: {pd.Timestamp.now()}")
#         print(f"Source: satisfaction_model.pkl")
#         print(f"Parameters: None")
#         print(f"Metrics: Model Accuracy (Placeholder)")
#         print(f"Artifacts: satisfaction_model.pkl")
#         # You would need to implement actual Docker or MlOps tracking here

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

class UserSatisfactionAnalytics:
    def __init__(self):
        pass

    def compute_score(self, df, cluster_centers_df, features, score_column_name):
        """
        Generalized method to compute a score based on user metrics and cluster centers.

        :param df: DataFrame containing user metrics.
        :param cluster_centers_df: DataFrame containing cluster centers.
        :param features: List of features to use for score computation.
        :param score_column_name: Name of the score column to be added in the DataFrame.
        :return: DataFrame with user ID and computed score.
        """
        # Extract cluster centers for the specified features
        cluster_centers = cluster_centers_df[features].values
        
        # List to hold computed scores
        scores = []

        # Compute the score for each user by finding the minimum Euclidean distance to the cluster centers
        for index, row in df.iterrows():
            user_metrics = row[features].values
            distances = pairwise_distances([user_metrics], cluster_centers, metric='euclidean')
            min_distance = np.min(distances)
            scores.append(min_distance)
        
        # Add the scores to the DataFrame
        df[score_column_name] = scores
        return df[['MSISDN/Number', score_column_name]]
    
    # # Example DataFrames (replace these with your actual data)
# engagement_df = pd.DataFrame({
#     'MSISDN/Number': ['user1', 'user2', 'user3'],
#     'total_session_duration': [3e10, 2e10, 1.5e10],
#     'total_download_traffic': [1e8, 2e8, 1.5e8],
#     'total_upload_traffic': [1e12, 2e12, 1.2e12],
#     'sessions_frequency': [1e11, 2e11, 1.8e11]
# })

# cluster_centers_df = pd.DataFrame({
#     'cluster': [0, 1, 2],
#     'total_session_duration': [1e10, 2e10, 3e10],
#     'total_download_traffic': [1e8, 2e8, 3e8],
#     'total_upload_traffic': [1e12, 2e12, 3e12],
#     'sessions_frequency': [1e11, 2e11, 3e11]
# })

# # Initialize and compute engagement scores
# calculator = EngagementScoreCalculator(engagement_df, cluster_centers_df)
# engagement_scores_df = calculator.compute_engagement_score()

# # Display results
# print(engagement_scores_df)
