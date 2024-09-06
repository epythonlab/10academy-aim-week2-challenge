import pandas as pd

class UserBehavierAnalysis:
    
    def __init__(self, df):
        self.df = df
    
    def aggregate_user_behavior(self):
        """
        Aggregate user behavior from xDR data.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing xDR data.
        
        Returns:
        pd.DataFrame: Aggregated user behavior data.
        """
        # Aggregate per user
        user_behavior = self.df.groupby('Bearer Id').agg(
            num_sessions=('Bearer Id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_dl=('Total DL (Bytes)', 'sum'),
            total_ul=('Total UL (Bytes)', 'sum')
        )

        # Add total data volume (DL + UL)
        user_behavior['total_data_volume'] = user_behavior['total_dl'] + user_behavior['total_ul']

        # Define applications
        applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

        # Add application-specific data
        for app in applications:
            user_behavior[f'{app} Total (Bytes)'] = (
                self.df.groupby('Bearer Id')[f'{app} UL (Bytes)'].sum() + 
                self.df.groupby('Bearer Id')[f'{app} DL (Bytes)'].sum()
            )

        return user_behavior

    def segment_users_by_decile(self):
        """
        Segment users into the top five decile classes based on session duration and compute total data per decile class.
        
        Returns:
        pd.DataFrame: DataFrame with decile class data.
        """
        # Aggregate user behavior
        user_behavior = self.aggregate_user_behavior()
        
        # Ensure 'total_duration' and 'total_data_volume' columns are numeric
        user_behavior['total_duration'] = pd.to_numeric(user_behavior['total_duration'], errors='coerce')
        user_behavior['total_data_volume'] = pd.to_numeric(user_behavior['total_data_volume'], errors='coerce')
        
        # Create deciles based on total duration ('total_duration') with duplicates dropped
        user_behavior['Decile'] = pd.qcut(user_behavior['total_duration'], 10, labels=False, duplicates='drop')

        # Filter the top five deciles
        top_five_deciles = user_behavior[user_behavior['Decile'] >= 5]

        # Compute total data and duration per decile class
        decile_summary = top_five_deciles.groupby('Decile').agg({
            'total_data_volume': 'sum',
            'total_duration': 'sum'
        }).reset_index()

        return decile_summary
# Example usage
# df = pd.read_csv('path_to_data.csv')  # Load your data
# decile_data = segment_by_deciles(df)
# print(decile_data)
