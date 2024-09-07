# test_experience_analytics.py

import pandas as pd
import numpy as np
import unittest
from scripts.experience_analytics import ExperienceAnalytics

class TestExperienceAnalytics(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for testing
        data = {
            'MSISDN/Number': [1, 2, 3, 4, 5],
            'TCP Retransmission': [0.1, 0.2, 0.3, np.nan, 0.4],
            'RTT': [200, 300, 400, 500, np.nan],
            'Throughput': [1000, 2000, np.nan, 1500, 2500],
            'Handset Type': ['iPhone', 'Samsung', 'Huawei', np.nan, 'Samsung']
        }
        self.df = pd.DataFrame(data)
        self.experience_analytics = ExperienceAnalytics(self.df)

    def test_aggregate_user_experience(self):
        user_agg = self.experience_analytics.aggregate_user_experience()
        # Check if missing values are filled and aggregation is correct
        self.assertEqual(user_agg['TCP Retransmission'].isnull().sum(), 0)
        self.assertEqual(user_agg['RTT'].isnull().sum(), 0)
        self.assertEqual(user_agg['Throughput'].isnull().sum(), 0)
        self.assertEqual(user_agg['Handset Type'].isnull().sum(), 0)

    def test_get_top_bottom_most_frequent(self):
        top_tcp, bottom_tcp, most_freq_tcp = self.experience_analytics.get_top_bottom_most_frequent('TCP Retransmission')
        # Adjusting the test to account for fewer values
        self.assertIn(0.4, top_tcp.values)
        self.assertIn(0.1, bottom_tcp.values)
        self.assertEqual(most_freq_tcp, 0.1)

    def test_k_means_clustering(self):
        user_agg = self.experience_analytics.aggregate_user_experience()
        features = ['TCP Retransmission', 'RTT', 'Throughput']
        user_clusters = self.experience_analytics.k_means_clustering(user_agg, features, k=3)
        # Test if clustering column is added
        self.assertIn('Cluster', user_clusters.columns)
        # Check if there are exactly 3 clusters
        self.assertEqual(len(user_clusters['Cluster'].unique()), 3)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)