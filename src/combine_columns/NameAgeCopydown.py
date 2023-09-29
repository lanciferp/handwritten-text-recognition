import os
import yaml
import argparse

import pandas as pd


def combine_files(folder_path, csv_name):

    df = pd.DataFrame()
    files = os.listdir(folder_path)

    for file in files:
        path = os.path.join(folder_path, file)

        new_df = pd.read_csv(path, names=["filename", "string", "confidence", "blank"])
        df = pd.concat((df, new_df))

    final_df_path = os.path.join(folder_path, csv_name)
    df.to_csv(final_df_path, index=False)
    return final_df_path


def deduplicate(csv_path):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--job_config', type=str, required=True)
    parser.add_argument('--deduplicate', action="store_true")

    args = parser.parse_args()

    yaml_path = os.path.join(".", "job_config", args.job_config)

    with open(yaml_path, "r") as f:
        job_config = yaml.safe_load(f)

    output_path = job_config['output_path']
    column_model_list = job_config['column_model_list']
    year = job_config['year']
    sub_name = args.job_config.split('.')[0]

    # Run copydown on Name
    for column_dict in column_model_list:
        name_list = column_dict["Name"]

        final_paths = []
        for model in name_list:
            model_output_path = os.path.join(output_path, str(year), sub_name, "Name", model)

            combined_csv_path = combine_files(model_output_path, model + "_final.csv")
            final_paths.append(combined_csv_path)

        print(column_dict)
        relation_list = column_dict["Relationship_To_Head"]

        for model in relation_list:
            model_output_path = os.path.join(output_path, str(year), sub_name, "Relationship_To_Head", model)

            combined_csv_path = combine_files(model_output_path, model + "_final.csv")
            final_paths.append(combined_csv_path)

        copydown_path = os.path.join(output_path, str(year), sub_name, "Name")
        final_paths.append(copydown_path)
        command = "sbatch run_copydown.sh " + " ".join(final_paths)

        print(command)
        os.system(command)

        age_list = column_dict["Age"]

        final_paths = []
        for model in age_list:
            model_output_path = os.path.join(output_path, str(year), sub_name, "Age", model)

            combined_csv_path = combine_files(model_output_path, model + "_final.csv")
            final_paths.append(combined_csv_path)

        relation_list = column_dict["Relationship_To_Head"]

        for model in relation_list:
            model_output_path = os.path.join(output_path, str(year), sub_name, "Relationship_To_Head", model)

            combined_csv_path = combine_files(model_output_path, model + "_final.csv")
            final_paths.append(combined_csv_path)

        copydown_path = os.path.join(output_path, str(year), sub_name, "Name")
        final_paths.append(copydown_path)
        command = "sbatch run_age_months.sh " + " ".join(final_paths)
