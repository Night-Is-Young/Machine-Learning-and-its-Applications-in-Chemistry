from typing import Tuple, Union
import numpy as np
import pandas as pd
import math

R = 8.314    # ideal gas constant
KCAL_TO_KJ = 4.184    # energy unit conversion

def prepare_data(in_df: pd.DataFrame, temperature: float=298.0) -> Tuple[pd.DataFrame, pd.Series]:
    ### BEGIN YOUR SOLUTION ###
    new_data = []
    new_label = []
    for index, row in in_df.iterrows():
        d_pi_d = row["d_pi_d"]
        d_pi_D = row["d_pi_D"]
        e_pi_d = row["e_pi_d"]
        e_pi_D = row["e_pi_D"]
        c_d = math.exp(-e_pi_d / (R * temperature) * 1000 * KCAL_TO_KJ)
        c_D = math.exp(-e_pi_D / (R * temperature) * 1000 * KCAL_TO_KJ)
        d_pi_w = (c_d * d_pi_d + c_D * d_pi_D) / (c_d + c_D)
        e_pi_w = (c_d * e_pi_d + c_D * e_pi_D) / (c_d + c_D)
        new_row = {
            "d_pi_w": d_pi_w,
            "e_pi_w": e_pi_w,
            "de_pi_w": d_pi_w * e_pi_w,
            "L_Alk": row["L_Alk"],
            "B1_Alk": row["B1_Alk"],
            "B5_Alk": row["B5_Alk"],
            "L_Ar": row["L_Ar"],
            "B1_Ar": row["B1_Ar"],
            "B5_Ar": row["B5_Ar"],
        }
        new_data.append(new_row)
        new_label.append(-R * temperature * math.log(row["er(%)"] * 0.01) / KCAL_TO_KJ / 1000)
    return pd.DataFrame(new_data), pd.Series(new_label, name="delta_delta_G")
    ### END YOUR SOLUTION ###

def normalize(
    X_train: Union[pd.DataFrame, np.array], X_test: Union[pd.DataFrame, np.array]
) -> Tuple[np.array, np.array]:
    from sklearn.preprocessing import StandardScaler
    ### BEGIN YOUR SOLUTION ###
    train_data = np.array(X_train)
    test_data = np.array(X_test)
    x_mean = np.mean(train_data, axis=0)
    x_std = np.std(train_data, axis=0)
    train_data = (train_data - x_mean) / x_std
    test_data = (test_data - x_mean) / x_std
    return train_data, test_data
    ### END YOUR SOLUTION ###