from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_chem_elements(
    raw_data: str, feature_cols: List[str], idx_col: str='Symbol'
) -> Tuple[pd.DataFrame, pd.Series]:
    ### BEGIN YOUR SOLUTION ###
    df = pd.read_csv(raw_data, sep=';')
    df.dropna(inplace=True)
    if idx_col in df.columns:
        df.set_index(idx_col, inplace=True)
    labels = df['GroupBlock']
    data = df[feature_cols]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
    return df_scaled, labels
    ### END YOUR SOLUTION ###