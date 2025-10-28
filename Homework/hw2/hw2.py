from src.data import read_gct, normalize_tpms, get_gender
from src.train_and_eval import train_and_eval
import argparse
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, help='Directory containing data files')
parser.add_argument('--output-dir', type=str, help='Directory to save results')
parser.add_argument('--gct-fname', type=str, help='GCT filename.', default='GTEx_gene_amygdala_tpm_v8.gct')
parser.add_argument('--pheno-fname', type=str, help='Subject phenotype filename', default='subject_pheontypes_v8.txt')
parser.add_argument('--test-size', type=float, help='Percentage of test set', default=0.2)
parser.add_argument('--random-state', type=int, help='Random state for train-test split', default=0)

def main(data_dir, output_dir, gct_fname, gender_fname, test_size, random_state):
    # data preprocessing
    gct = read_gct(os.path.join(data_dir, gct_fname))
    gender = get_gender(os.path.join(data_dir, gender_fname), gtex_ids=list(gct.index))
    X_train, X_test, y_true_train, y_true_test = train_test_split(gct, gender, test_size=test_size, random_state=random_state)
    hyperparams = {'C': 0.1, 'class_weight': 'balanced'}
    
    # normalization and model training/evaluation: standard scaling
    X_train_standard, X_test_standard = normalize_tpms(X_train, X_test, method='standard')
    clf_standard, metrics_standard = train_and_eval(
        X_train_standard, X_test_standard, y_true_train, y_true_test, hyperparams, save_path=os.path.join(output_dir, 'roc_standard.png')
    )
    with open(os.path.join(output_dir, 'clf_standard.pkl'), 'wb') as f:
        pickle.dump(clf_standard, f)
    
    # normalization and model training/evaluation: min-max scaling
    X_train_minmax, X_test_minmax = normalize_tpms(X_train, X_test, method='minmax')
    clf_minmax, metrics_minmax = train_and_eval(
        X_train_minmax, X_test_minmax, y_true_train, y_true_test, hyperparams, save_path=os.path.join(output_dir, 'roc_minmax.png')
    )
    with open(os.path.join(output_dir, 'clf_minmax.pkl'), 'wb') as f:
        pickle.dump(clf_minmax, f)
    
    # store metrics as csv
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Train (Standard Scaling)': [metrics_standard['accuracy'][0], metrics_standard['precision'][0], metrics_standard['recall'][0], metrics_standard['f1'][0]],
        'Test (Standard Scaling)': [metrics_standard['accuracy'][1], metrics_standard['precision'][1], metrics_standard['recall'][1], metrics_standard['f1'][1]],
        'Train (Min-Max Scaling)': [metrics_minmax['accuracy'][0], metrics_minmax['precision'][0], metrics_minmax['recall'][0], metrics_minmax['f1'][0]],
        'Test (Min-Max Scaling)': [metrics_minmax['accuracy'][1], metrics_minmax['precision'][1], metrics_minmax['recall'][1], metrics_minmax['f1'][1]],
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False, sep='\t')


if __name__ == '__main__':
    args = vars(parser.parse_args())
    data_dir = args['data_dir']
    output_dir = args['output_dir']
    gct_fname = args['gct_fname']
    gender_fname = args['pheno_fname']
    test_size = args['test_size']
    random_state = args['random_state']
    os.makedirs(output_dir, exist_ok=True)
    try:
        main(data_dir, output_dir, gct_fname, gender_fname, test_size, random_state)
    except NotImplementedError:
        print("Some functions are not implemented yet.")