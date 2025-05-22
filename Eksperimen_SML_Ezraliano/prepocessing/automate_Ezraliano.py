import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(url):
    """Memuat dataset dari URL"""
    df = pd.read_csv(url)
    return df

def handle_missing_values(df):
    """Mengatasi nilai kosong"""
    if df.isnull().values.any():
        df.fillna(df.mean(numeric_only=True), inplace=True)
        for col in df.select_dtypes(include='object').columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def remove_duplicates(df):
    """Menghapus baris duplikat"""
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df):
    """Pisah fitur-target dan lakukan standarisasi"""
    X = df.drop('label', axis=1)
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y

def main():
    url = "https://raw.githubusercontent.com/Ezraliano/Agriculture_Submission_Membangun_Sistem_Machine_Learning/refs/heads/main/Crop_recommendation.csv "
    df = load_data(url)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    X, y = preprocess_data(df)
    
    processed_df = pd.concat([X, y.reset_index(drop=True)], axis=1)
    processed_df.to_csv("preprocessing/dataset_preprocessed.csv", index=False)
    print("Preprocessing selesai. Dataset tersimpan di 'dataset_preprocessed.csv'")

if __name__ == "__main__":
    main()