import unittest
import pandas as pd
from scripts.handset_analysis import HandsetAnalysis  # Replace 'your_module' with the name of your Python file

class TestHandsetAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        data = {
            'Handset Type': ['iPhone 6S', 'iPhone 6', 'Galaxy S8', 'iPhone 6S', 'iPhone 6', 'Galaxy S8', 'B528S-23A'],
            'Handset Manufacturer': ['Apple', 'Apple', 'Samsung', 'Apple', 'Apple', 'Samsung', 'Huawei'],
            'Dur. (ms)': [500, 600, 700, 500, 600, 700, 800],
            'Total DL (Bytes)': [1000, 2000, 3000, 1000, 2000, 3000, 4000],
            'Total UL (Bytes)': [1500, 2500, 3500, 1500, 2500, 3500, 4500]
        }
        cls.df = pd.DataFrame(data)
        cls.analysis = HandsetAnalysis(cls.df)
    
    def test_top_handsets(self):
        top_handsets = self.analysis.top_handsets(self.df, top_n=2)
        expected = pd.Series({
            'iPhone 6S': 2,
            'iPhone 6': 2
        })
        pd.testing.assert_series_equal(top_handsets, expected)

    def test_top_manufacturers(self):
        top_manufacturers = self.analysis.top_manufacturers(self.df, top_n=2)
        expected = pd.Series({
            'Apple': 4,
            'Samsung': 2
        })
        pd.testing.assert_series_equal(top_manufacturers, expected)
    
    def test_top_handsets_per_manufacturer(self):
        manufacturers = ['Apple', 'Samsung']
        top_handsets = self.analysis.top_handsets_per_manufacturer(self.df, manufacturers, top_n_handsets=1)
        
        expected = {
            'Apple': pd.Series({'iPhone 6S': 2}),
            'Samsung': pd.Series({'Galaxy S8': 2})
        }
        
        for manufacturer in manufacturers:
            pd.testing.assert_series_equal(top_handsets[manufacturer], expected[manufacturer])

    

if __name__ == '__main__':
    unittest.main()
