import unittest
import pandas as pd
from user_engagement_analysis import UserEngagementAnalysis  # Assuming the class is in user_engagement_analysis.py

class TestUserEngagementAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        cls.data = pd.DataFrame({
            'MSISDN': [1, 1, 2, 2, 3, 3, 4, 4],
            'session_id': [1, 2, 1, 2, 1, 2, 1, 2],
            'session_duration': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
            'download_bytes': [100, 200, 300, 400, 500, 600, 700, 800],
            'upload_bytes': [50, 100, 150, 200, 250, 300, 350, 400],
            'application': ['App1', 'App1', 'App2', 'App2', 'App3', 'App3', 'App4', 'App4']
        })
        cls.analysis = UserEngagementAnalysis(cls.data)
    
    def test_aggregate_metrics(self):
        self.analysis.aggregate_metrics()
        self.assertEqual(self.analysis.metrics.shape[0], 4)  # 4 unique MSISDNs
        self.assertIn('sessions_frequency', self.analysis.metrics.columns)
    
    def test_report_top_customers(self):
        self.analysis.aggregate_metrics()
        top_customers = self.analysis.report_top_customers()
        self.assertEqual(len(top_customers[0]), 1)  # Check for top 10 in session frequency
        self.assertEqual(len(top_customers[1]), 1)  # Check for top 10 in session duration
        self.assertEqual(len(top_customers[2]), 1)  # Check for top 10 in download traffic
        self.assertEqual(len(top_customers[3]), 1)  # Check for top 10 in upload traffic

    def test_normalize_and_cluster(self):
        self.analysis.aggregate_metrics()
        self.analysis.normalize_and_cluster(n_clusters=3)
        self.assertIn('cluster', self.analysis.metrics.columns)
        self.assertEqual(len(set(self.analysis.metrics['cluster'])), 3)  # Ensure 3 clusters
    
    def test_cluster_summary(self):
        self.analysis.aggregate_metrics()
        self.analysis.normalize_and_cluster(n_clusters=3)
        summary = self.analysis.cluster_summary()
        self.assertTrue('sessions_frequency' in summary.columns)
        self.assertTrue('total_session_duration' in summary.columns)
    
    def test_aggregate_traffic_per_application(self):
        app_total_traffic, top_10_engaged_per_app = self.analysis.aggregate_traffic_per_application()
        self.assertEqual(app_total_traffic.shape[0], 4)  # Check number of applications
        self.assertEqual(top_10_engaged_per_app.shape[0], 8)  # Check number of engaged users

    def test_plot_top_applications(self):
        app_total_traffic, _ = self.analysis.aggregate_traffic_per_application()
        top_3_apps = app_total_traffic.nlargest(3, 'download_bytes' + 'upload_bytes')
        try:
            self.analysis.plot_top_applications(top_3_apps)
        except Exception as e:
            self.fail(f"Plotting failed: {e}")
    
    def test_elbow_method(self):
        self.analysis.aggregate_metrics()
        self.analysis.normalize_and_cluster(n_clusters=3)
        try:
            self.analysis.elbow_method()
        except Exception as e:
            self.fail(f"Elbow method plotting failed: {e}")

if __name__ == '__main__':
    unittest.main()
