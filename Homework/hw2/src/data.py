from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def read_gct(gct_fname: str) -> pd.DataFrame:
    ### BEGIN YOUR SOLUTION ###
    df=pd.read_csv(gct_fname, sep='\t', skiprows=2)
    df.set_index('Name', inplace=True)
    df.drop(['Description', 'id'], axis=1, inplace=True)
    df=df.transpose()
    return df
    ### END YOUR SOLUTION ###

def normalize_tpms(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
    ### BEGIN YOUR SOLUTION ###
    if method == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    elif method == 'standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    ### END YOUR SOLUTION ###

def get_gender(annot_fname: str, gtex_ids: List[str], sep: str='\t') -> pd.Series:
    gtex_subjids = ['-'.join(gtex_id.split('-')[:2]) for gtex_id in gtex_ids]
    annot = pd.read_csv(annot_fname, sep=sep).set_index('SUBJID')
    return annot.loc[gtex_subjids, 'SEX'] - 1