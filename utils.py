import yaml

def load_yaml(yaml_file_name):
    with open(yaml_file_name) as fp:
        config = yaml.safe_load(fp)
    return config


def column_validation(req_list, df, df_name):
    ingested_list = df.columns
    not_list = []
    for col in req_list:
        if col not in ingested_list:
            not_list.append(col)
    if len(not_list)!=0:
        print(f"Missing columns in the data frame: {df_name}. Here is the list of missing columns:{not_list}")
        x = f"Missing columns in the data frame: {df_name}. Here is the list of missing columns:{not_list}"
        i = 1
        return x,i
    else:
        print(f"Validated {df_name}")
        x= f"Validated {df_name}"
        i = 0
        return x,i