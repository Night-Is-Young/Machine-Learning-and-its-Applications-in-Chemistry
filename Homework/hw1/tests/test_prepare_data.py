from ..src.data import prepare_data
import numpy as np
import pandas as pd

in_df = pd.DataFrame({
    'd_pi_d': [4.89, 4.91, 6.48, 7.00],
    'd_pi_D': [7.00, 7.21, 7.22, 7.20],
    'e_pi_d': [-3.48, -2.98, -1.80, -1.45],
    'e_pi_D': [-1.48, -2.00, -1.96, -2.26],
    'L_Alk': [4.36, 4.35, 4.38, 3.08],
    'B1_Alk': [2.92, 2.09, 1.73, 1.70],
    'B5_Alk': [3.35, 3.34, 3.33, 2.20],
    'L_Ar': [6.38, 6.38, 6.38, 6.38],
    'B1_Ar': [1.77, 1.77, 1.77, 1.77],
    'B5_Ar': [3.15, 3.15, 3.15, 3.15],
    'er(%)': [1.71, 3.36, 3.78, 5.03]
})

labels = np.array([2.409283, 2.009314, 1.939568, 1.770392])
weighted_features = np.array([
    [4.959643, -3.413988, -16.932160],
    [5.279007, -2.822771, -14.901427],
    [6.899685, -1.890743, -13.045529],
    [7.159408, -2.095601, -15.003264]
])


def test_prepare_data_names():
    X_, y_true_ = prepare_data(in_df)
    assert set(X_.columns) == {'d_pi_w', 'e_pi_w', 'de_pi_w', 'L_Alk', 'B1_Alk', 'B5_Alk', 'L_Ar', 'B1_Ar', 'B5_Ar'}, "Incorrect column names."
    assert y_true_.name == 'delta_delta_G', "Incorrect label name."

def test_prepare_data_labels():
    _, y_true_ = prepare_data(in_df)
    np.testing.assert_allclose(
        y_true_, labels, atol=1.0e-6,
        err_msg="Incorrect label values."
    )

def test_prepare_data_weighted_features():
    X_, _ = prepare_data(in_df)
    np.testing.assert_allclose(
        X_.loc[:, ['d_pi_w', 'e_pi_w', 'de_pi_w']], weighted_features, atol=1.0e-6,
        err_msg="Incorrect weighted feature values."
    )