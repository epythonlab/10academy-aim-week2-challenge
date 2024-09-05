#scripts/end_pipeline.py
# EDA pipeline implementation


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    
    def handle_missing_values(df):
        # Replace missing values with column means
        return df.fillna(df.mean())

    def handle_outliers(df):
        # Handle outliers by capping them to a certain threshold
        return df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)

    def basic_statistics(df):
        return df.describe()

    def plot_distribution(df, column):
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    def plot_correlation_matrix(df):
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
    def basic_statistics(df):
        print("Basic Statistics:")
        print(df.describe())

    def univariate_analysis(df):
        # Example: Plot distribution of session duration
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Session Duration'], bins=50, kde=True)
        plt.title('Distribution of Session Duration')
        plt.show()

    def bivariate_analysis(df):
        # Example: Scatter plot of YouTube Data vs. Total Data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)'], y=df['Total DL (Bytes)'] + df['Total UL (Bytes)'])
        plt.title('YouTube Data vs. Total Data')
        plt.show()
        
# basic usage:
if __name__ == "__main__":
    df = pd.read_csv("path_to_cleaned_xdr_data.csv")
    EDA.basic_statistics(df)
    EDA.univariate_analysis(df)
    EDA.bivariate_analysis(df)

