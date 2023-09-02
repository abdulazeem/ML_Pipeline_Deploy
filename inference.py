import utils as ut
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
import os


def inf_pre_processing(data, cat_cols):
    for c_col in cat_cols:
        data[c_col] = data[c_col].apply(lambda x: x.strip())
    data['location'] = data['location'].apply(lambda x: np.nan if x == '?' else x)
    data_2 = data[yaml_inputs['test_req_cols']]
    return data_2


def inference_pipeline(inf_df):
    pred_model = yaml_inputs['model_predict_path']+'/'+yaml_inputs['model_name']
    data_pipeline = joblib.load(pred_model)

    y_pred = data_pipeline.predict(inf_df)
    inf_df['Prediction'] = y_pred
    inf_df['Prediction'] = inf_df['Prediction'].apply(lambda x:'<=50k' if x==0 else '>50k')
    inf_df.to_csv(yaml_inputs['inference_results_path'])
    return inf_df


if __name__ == '__main__':
    yaml_inputs = ut.load_yaml("settings.yml")

    # Set Path
    parent_dir = Path.cwd()
    data_path = parent_dir.joinpath(yaml_inputs['inference_path'])
    filename = data_path.joinpath(yaml_inputs['inference_filename'])

    # Set logger

    # Ingest and Validate Data
    inf_data = pd.read_csv(filename)

    text_response, indicator = ut.column_validation(yaml_inputs['test_req_cols'], inf_data, 'my inf df')
    if indicator == 1:
        sys.exit()

    ml_inf_data = inf_pre_processing(inf_data, yaml_inputs['cat_cols'])

    print(ml_inf_data.head(2))
    print(ml_inf_data.columns)

    inf_df = inference_pipeline(ml_inf_data)






