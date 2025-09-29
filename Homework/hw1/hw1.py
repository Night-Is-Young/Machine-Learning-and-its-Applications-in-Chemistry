from src.data import prepare_data, normalize
from src.training import train_model, evaluate_model
import argparse
import os
import pickle
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset file (.csv).', default='data/nci_birman.csv')
parser.add_argument('--output-dir', type=str, help='Directory to save results', default='results/')
parser.add_argument('--temperature', type=float, help='Temperature in Kelvin', default=298.0)
parser.add_argument('--test-size', type=int, help='Number of test samples', default=5)

def main(dataset, output_dir, temperature, test_size):
    # data preprocessing
    nci_birman = pd.read_csv(dataset, sep='\t')
    X, y_true = prepare_data(nci_birman, temperature=temperature)
    print(X, y_true)
    
    # train-test split and normalization
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_true_train, y_true_test = train_test_split(X, y_true, test_size=test_size, random_state=0)
    X_train, X_test = normalize(X_train, X_test)
    print(X_train, X_test)
    
    # model training and evaluation
    lr = train_model(X_train, y_true_train)
    rmse_train, r2_train = evaluate_model(lr, X_train, y_true_train, mode='metrics')
    print(f"Train set: RMSE = {rmse_train:.4f}, R^2 = {r2_train:.4f}")
    rmse_test, r2_test = evaluate_model(lr, X_test, y_true_test, mode='metrics')
    print(f"Test set: RMSE = {rmse_test:.4f}, R^2 = {r2_test:.4f}")
    results = pd.DataFrame({
        'Set': ['Train', 'Test'],
        'RMSE': [rmse_train, rmse_test],
        'R^2': [r2_train, r2_test]
    })
    results.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    evaluate_model(lr, X_train, y_true_train, mode='plot', save_path=os.path.join(output_dir, 'pred_train.png'))
    evaluate_model(lr, X_test, y_true_test, mode='plot', save_path=os.path.join(output_dir, 'pred_test.png'))
    
    # model saving
    model_name = 'lr.pkl'
    with open(os.path.join(output_dir, model_name), 'wb') as f:
        pickle.dump(lr, f)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    dataset = args['dataset']
    output_dir = args['output_dir']
    temperature = args['temperature']
    test_size = args['test_size']
    os.makedirs(output_dir, exist_ok=True)
    try:
        main(dataset, output_dir, temperature, test_size)
    except NotImplementedError:
        print("Some functions are not implemented yet.")