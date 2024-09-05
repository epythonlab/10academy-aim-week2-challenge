
class UserAnalysis:
    
    def top_handsets(df, top_n=10):
        top_handsets = df['Handset Type'].value_counts().head(top_n)
        return top_handsets

    def top_manufacturers(df, top_n=3):
        top_manufacturers = df['Handset Manufacturer'].value_counts().head(top_n)
        return top_manufacturers

    def top_handsets_per_manufacturer(df, manufacturer, top_n=5):
        df_manufacturer = df[df['Handset Manufacturer'] == manufacturer]
        top_handsets = df_manufacturer['Handset Type'].value_counts().head(top_n)
        return top_handsets
    
    def segment_users_by_decile(df):
        df['Decile'] = pd.qcut(df['Dur. (ms)'], 10, labels=False)
        return df.groupby('Decile').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()