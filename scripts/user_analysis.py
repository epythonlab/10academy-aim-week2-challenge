import pandas as pd
class UserAnalysis:
    
    def top_handsets(df, top_n=10):
        top_handsets = df['Handset Type'].value_counts().head(top_n)
        return top_handsets

    def top_manufacturers(df, top_n=3):
        top_manufacturers = df['Handset Manufacturer'].value_counts().head(top_n)
        return top_manufacturers

    # Function to get top 5 handsets per top manufacturers
    def top_handsets_per_manufacturer(df, manufacturers, top_n_handsets=5):
        results = {}
        for manufacturer in manufacturers:  # Loop over the index of the manufacturers (names)
            # Filter by manufacturer
            df_manufacturer = df[df['Handset Manufacturer'] == manufacturer]
            # Get the top 5 handsets for that manufacturer
            top_handsets = df_manufacturer['Handset Type'].value_counts().head(top_n_handsets)
            results[manufacturer] = top_handsets
        return results
    
    def segment_users_by_decile(df):
        # Ensure that 'Dur. (ms)' column is numeric
        df['Dur. (ms)'] = pd.to_numeric(df['Dur. (ms)'], errors='coerce')
        
        # Create deciles
        df['Decile'] = pd.qcut(df['Dur. (ms)'], 10, labels=False)
        
        # Drop duplicate rows if needed
        df = df.drop_duplicates(subset=['Decile', 'Total DL (Bytes)', 'Total UL (Bytes)'])
        
        # Aggregate by decile
        result = df.groupby('Decile').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum',
        }).reset_index()
        
        return result
