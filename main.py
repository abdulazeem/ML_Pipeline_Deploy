import utils as ut
from pathlib import Path
import sys
import pandas as pd

if __name__ == '__main__':
    yaml_inputs = ut.load_yaml("settings.yml")

    # Set Path
    parent_dir = Path.cwd()
    data_path = parent_dir.joinpath(yaml_inputs['datapath'])
    filename = data_path.join(yaml_inputs['train_filepath'])

    # Set logger



    # Ingest and Validate Data
    overall_data = pd.read_csv(filename)
