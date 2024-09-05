#scripts/end_pipeline.py
# EDA pipeline implementation


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    
    def __init__(self, df):
        self.df = df

    def compute_basic_metrics(self, variables):
        metrics = {}
        
        # Columns of interest
        # columns_of_interest = ['Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']
        
        for var in variables:
            metrics[var] = {
                'Mean': self.df[var].mean(),
                'Median': self.df[var].median(),
                'Mode': self.df[var].mode().values[0],  # mode() returns a Series
                'Standard Deviation': self.df[var].std(),
                'Variance': self.df[var].var(),
                'Range': self.df[var].max() - self.df[var].min(),
                'IQR': self.df[var].quantile(0.75) - self.df[var].quantile(0.25)
    }

        
        # Convert the metrics into a DataFrame for better presentation
        metrics_df = pd.DataFrame(metrics)
        return metrics_df
    
    def segment_users_by_decile(self):
        # Ensure 'Dur. (ms)', 'Total DL (Bytes)', and 'Total UL (Bytes)' columns are numeric
        self.df['Dur. (ms)'] = pd.to_numeric(self.df['Dur. (ms)'], errors='coerce')
        self.df['Total DL (Bytes)'] = pd.to_numeric(self.df['Total DL (Bytes)'], errors='coerce')
        self.df['Total UL (Bytes)'] = pd.to_numeric(self.df['Total UL (Bytes)'], errors='coerce')
        
        # Calculate total data volume (DL + UL)
        self.df['Total Data (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']

        # Create deciles based on total duration ('Dur. (ms)') with duplicates dropped
        self.df['Decile'] = pd.qcut(self.df['Dur. (ms)'], 10, labels=False, duplicates='drop')

        # Filter the top five deciles
        top_five_deciles = self.df[self.df['Decile'] >= 5]

        # Compute total data per decile class
        decile_summary = top_five_deciles.groupby('Decile').agg({
            'Total Data (Bytes)': 'sum',
            'Dur. (ms)': 'sum'
        }).reset_index()

        return decile_summary
    
    

    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    def plot_correlation_matrix(self):
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
    def univariate_analysis(self):
        # Example: Plot distribution of session duration
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Session Duration'], bins=50, kde=True)
        plt.title('Distribution of Session Duration')
        plt.show()

    def bivariate_analysis(self):
        # Example: Scatter plot of YouTube Data vs. Total Data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df['Youtube DL (Bytes)'] + self.df['Youtube UL (Bytes)'], y=self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)'])
        plt.title('YouTube Data vs. Total Data')
        plt.show()
        
# # basic usage:
# if __name__ == "__main__":
#     df = pd.read_csv("path_to_cleaned_xdr_data.csv")
#     EDA.basic_statistics(df)
#     EDA.univariate_analysis(df)
#     EDA.bivariate_analysis(df)

