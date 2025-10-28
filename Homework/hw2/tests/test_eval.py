import numpy as np
from ..src.train_and_eval import evaluate_model

y_trues = [np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])]
y_preds = [np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1])]
metrics_ans = [0.4666666666666667, 0.5, 0.375, 0.42857142857142855]

def test_evaluate_model_keys():
    metrics = evaluate_model(y_trues, [], y_preds, [], plot=False)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1'}, "Wrong metrics keys."

def test_evaluate_model_values():
    metrics = evaluate_model(y_trues, [], y_preds, [], plot=False)
    np.testing.assert_allclose(
        [metrics['accuracy'][0], metrics['precision'][0], metrics['recall'][0], metrics['f1'][0]], metrics_ans, rtol=1.0e-3,
        err_msg="Wrong metrics values."
    )