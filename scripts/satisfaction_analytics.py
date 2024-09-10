# # scripts/satsfaction_analytics.py

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import psycopg2
from psycopg2 import sql
import pickle
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.db_connect import conn


class UserSatisfactionAnalytics:
    def __init__(self):
        self.model = None

    def compute_score(self, df, cluster_centers_df, features, score_column_name, target_cluster=None):
        """
        Generalized method to compute a score based on user metrics and a specific cluster center.

        :param df: DataFrame containing user metrics.
        :param cluster_centers_df: DataFrame containing cluster centers.
        :param features: List of features to use for score computation.
        :param score_column_name: Name of the score column to be added in the DataFrame.
        :param target_cluster: The cluster number to compute the distance to (e.g., least engaged or worst experience).
        :return: DataFrame with user ID and computed score.
        """
        # If a target cluster is provided, use its cluster center for the distance calculation
        if target_cluster is not None:
            target_cluster_center = cluster_centers_df[cluster_centers_df['cluster'] == target_cluster][features].values
        else:
            raise ValueError("Target cluster must be specified.")

        # List to hold computed scores
        scores = []

        # Compute the score for each user by finding the Euclidean distance to the specified cluster center
        for index, row in df.iterrows():
            user_metrics = row[features].values
            distance = pairwise_distances([user_metrics], target_cluster_center, metric='euclidean')[0][0]
            scores.append(distance)
        
        # Add the scores to the DataFrame
        df[score_column_name] = scores
        return df[['MSISDN/Number', score_column_name]]

    
    def compute_satisfaction_score(self, engagement_scores, experience_scores):
        # Merge engagement and experience scores
        merged_df = pd.merge(engagement_scores, experience_scores, on='MSISDN/Number')

        # Calculate satisfaction score as the average of engagement and experience scores
        merged_df['Satisfaction_Score'] = (merged_df['Engagement_Score'] + merged_df['Experience_Score']) / 2      

        return merged_df
    
    def top_satisfied_customer(self, engagement_scores, experience_scores,top_n=10):
        # Compute satisfaction scores
        satisfaction_df = self.compute_satisfaction_score(engagement_scores, experience_scores)
       # Sort by satisfaction score in descending order
        sorted_df = satisfaction_df.sort_values(by='Satisfaction_Score', ascending=False)

        # Select top 10 satisfied customers
        top_satisfied_customers = sorted_df.head(top_n)

        return top_satisfied_customers[['MSISDN/Number', 'Satisfaction_Score']]
    
    def build_regression_model(self, engagement_df, experience_df, model_type='linenar'):
        # Compute satisfaction scores
        satisfaction_df = self.compute_satisfaction_score(engagement_df, experience_df)
        
        # Prepare data for regression model
        X = satisfaction_df[['Engagement_Score', 'Experience_Score']]  # Features
        y = satisfaction_df['Satisfaction_Score']  # Target variable
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        
        if model_type == 'rigde':
            self.model = Ridge()
        elif model_type == 'lasso':
            self.model = Lasso()
        else:
            self.model = LinearRegression()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Save the model
        with open('satisfaction_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
            
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        
        return self.model
    
    def perform_clustering(self, engagement_score, experience_score, n_clusters=2):
        """
        Perform K-Means clustering on engagement and experience scores.

        :param engagement_df: DataFrame with engagement scores.
        :param experience_df: DataFrame with experience scores.
        :param n_clusters: Number of clusters for K-Means.
        :return: DataFrame with user IDs and their assigned cluster.
        """
        # Compute satisfaction score by merging together
        cluster_df = self.compute_satisfaction_score(engagement_score, experience_score)

        # Select features for clustering
        features = cluster_df[['Engagement_Score', 'Experience_Score']]

        # Initialize and fit the K-Means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_df['Cluster'] = kmeans.fit_predict(features)
       
        
        return cluster_df
    
    def export_to_postgresql(self, clustering_data):
        try:
            # Get SQLAlchemy engine using the conn function
            engine = conn(db_name='telecom_model')  
            connection = engine.raw_connection()
            cursor = connection.cursor()

            # Drop table if exists and create a new one
            cursor.execute('DROP TABLE IF EXISTS user_satisfaction_scores')
            cursor.execute('''
                CREATE TABLE user_satisfaction_scores (
                    MSISDN FLOAT,
                    engagement_score FLOAT,
                    experience_score FLOAT,
                    satisfaction_score FLOAT
                )
            ''')

            # Prepare the insert query
            insert_query = '''
                INSERT INTO user_satisfaction_scores (MSISDN, engagement_score, experience_score, satisfaction_score)
                VALUES (%s, %s, %s, %s)
            '''

            # Convert numpy types to native Python types before insertion
            data_to_insert = [
                (float(row['MSISDN/Number']), float(row['Engagement_Score']), float(row['Experience_Score']), float(row['Satisfaction_Score']))
                for _, row in clustering_data.iterrows()
            ]
            
            # Insert the data
            cursor.executemany(insert_query, data_to_insert)
            
            # Commit the transaction
            connection.commit()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if connection:
                cursor.close()
                connection.close()
