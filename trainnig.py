import utils as ut
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os


def pre_processing(data, cat_cols):
    data[yaml_inputs['ori_target']] = data[yaml_inputs['ori_target']].apply(lambda x:x.strip())
    data['target'] = data[yaml_inputs['ori_target']].apply(lambda x: 0 if x ==yaml_inputs['target_type'] else 1)
    for c_col in cat_cols:
        data[c_col] = data[c_col].apply(lambda x:x.strip())
    data['location'] = data['location'].apply(lambda x:np.nan if x=='?' else x)
    data_2 = data[yaml_inputs['ml_cols']]
    return data_2


def check_model_performance(data_pipeline, X_test, y_test):
    y_pred = data_pipeline.predict(X_test)
    print(pd.crosstab(y_test, y_pred))
    print(classification_report(y_test, y_pred))



def training_pipeline(ml_df):
    X_train, X_test, y_train, y_test = train_test_split(ml_df.drop('target', axis=1),
                                                        ml_df['target'],
                                                        random_state=42,
                                                        stratify=ml_df['target'],
                                                        shuffle=True,
                                                        test_size=0.25)

    data_pipeline = Pipeline([
        ('num_imputer', MeanMedianImputer(imputation_method='median', variables=yaml_inputs['num_cols'])),
        ('yt_trans', YeoJohnsonTransformer(variables=yaml_inputs['num_cols2'])),
        ('cat_imputer', CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=yaml_inputs['cat_cols'])),
        ('rare1', RareLabelEncoder(n_categories=1, tol=0.02, variables=yaml_inputs['cat_cols1'])),
        ('rare2', RareLabelEncoder(n_categories=1, tol=0.004, variables=yaml_inputs['cat_cols2'])),
        ('mean_enc', MeanEncoder(variables=yaml_inputs['cat_cols'])),
        ('scaler', MinMaxScaler()),
        ('classifier', RandomForestClassifier())
    ])

    data_pipeline.fit(X_train, y_train)

    check_model_performance(data_pipeline, X_test, y_test)
    if not yaml_inputs['model_predict_path']:
        os.makedirs(yaml_inputs['model_predict_path'])

    joblib.dump(data_pipeline, yaml_inputs['model_predict_path']+'/'+'prediction_pipeline.pkl')
    print(f"Successfully Dumped the pickle model in {yaml_inputs['model_predict_path']} folder")
    return None


if __name__ == '__main__':
    yaml_inputs = ut.load_yaml("settings.yml")

    # Set Path
    parent_dir = Path.cwd()
    data_path = parent_dir.joinpath(yaml_inputs['training_path'])
    filename = data_path.joinpath(yaml_inputs['train_filename'])

    # Set logger

    # Ingest and Validate Data
    train_data = pd.read_csv(filename)

    text_response, indicator = ut.column_validation(yaml_inputs['train_req_cols'], train_data, 'my uploaded df')
    if indicator==1:
        sys.exit()
    
    ml_data = pre_processing(train_data, yaml_inputs['cat_cols'])
    print(ml_data.head(2))
    print(ml_data.columns)

    training_pipeline(ml_data)
    # X_train, X_test, y_train, y_testtrain_test_split_pipe(ml_data)





