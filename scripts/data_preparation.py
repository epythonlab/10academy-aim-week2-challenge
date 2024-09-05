# Script for data preparation
# scripts/data_preparation
class PrepareData:
    
    def aggregate_user_data(df):
        # Aggregate the required columns
        agg_df = df.groupby('Bearer Id').agg({
            'Start': 'count', # Number of xDR sessions
            'Dur. (ms)': 'sum', # Session duration
            'Total UL (Bytes)': 'sum', # Total Upload Data
            'Total DL (Bytes)': 'sum', # Total Download Data
            'Social Media UL (Bytes)': 'sum',
            'Social Media DL (Bytes)': 'sum',
            'Google UL (Bytes)': 'sum',
            'Google DL (Bytes)': 'sum',
            'Email UL (Bytes)': 'sum',
            'Email DL (Bytes)': 'sum',
            'Youtube UL (Bytes)': 'sum',
            'Youtube DL (Bytes)': 'sum',
            'Netflix UL (Bytes)': 'sum',
            'Netflix DL (Bytes)': 'sum',
            'Gaming UL (Bytes)': 'sum',
            'Gaming DL (Bytes)': 'sum',
            'Other UL (Bytes)': 'sum',
            'Other DL (Bytes)': 'sum'
        }).reset_index()
        
        return agg_df
