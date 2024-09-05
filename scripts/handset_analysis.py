import pandas as pd
class HandsetAnalysis:
    
    def __init__(self, df):
        self.df = df
        
    def top_handsets(self, top_n=10):
            top_handsets = self.df['Handset Type'].value_counts().head(top_n)
            return top_handsets

    def top_manufacturers(self, top_n=3):
        top_manufacturers = self.df['Handset Manufacturer'].value_counts().head(top_n)
        return top_manufacturers

    def top_handsets_per_manufacturer(self, manufacturers, top_n_handsets=5):
        results = {}
        for manufacturer in manufacturers:
            # Filter by manufacturer
            df_manufacturer = self.df[self.df['Handset Manufacturer'] == manufacturer]
            # Get the top 5 handsets for that manufacturer
            top_handsets = df_manufacturer['Handset Type'].value_counts().head(top_n_handsets)
            results[manufacturer] = top_handsets
        return results