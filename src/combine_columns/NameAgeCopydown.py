import os
import yaml
import argparse

import pandas as pd


def combine_files(folder_path, csv_name):
    print(csv_name)

    df = pd.DataFrame()
    files = os.listdir(folder_path)

    for file in files:
        path = os.path.join(folder_path, file)

        new_df = pd.read_csv(path, names=["filename", "image_row_name",  "string", "confidence", "blank"], skiprows=0)
        df = pd.concat((df, new_df))

    csv_path = os.path.sep.join(folder_path.split(os.path.sep)[:-1])
    final_df_path = os.path.join(csv_path, csv_name)
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

    last_final_path = ""
    name_final_path = ""
    relation_final_path = ""
    year_final_path = ""
    month_final_path = ""

    # Run copydown on Name
    for column_dict in column_model_list:
        if "Name" in column_dict:
            name_list = column_dict["Name"]

            model_output_path = os.path.join(output_path, str(year), sub_name, "Name", name_list[0])
            combined_csv_path = combine_files(model_output_path, name_list[0] + "_final.csv")
            name_final_path = combined_csv_path

            model_output_path = os.path.join(output_path, str(year), sub_name, "Name", name_list[1])
            combined_csv_path = combine_files(model_output_path, name_list[1] + "_final.csv")
            last_final_path = combined_csv_path

        elif "Relationship_To_Head" in column_dict:
            relation_list = column_dict["Relationship_To_Head"]

            model_output_path = os.path.join(output_path, str(year), sub_name, "Relationship_To_Head", relation_list[0])
            combined_csv_path = combine_files(model_output_path, relation_list[0] + "_final.csv")
            relation_final_path = combined_csv_path

        elif "Age" in column_dict:
            age_list = column_dict["Age"]

            model_output_path = os.path.join(output_path, str(year), sub_name, "Age", age_list[0])
            combined_csv_path = combine_files(model_output_path, age_list[0] + "_final.csv")
            year_final_path = combined_csv_path

            model_output_path = os.path.join(output_path, str(year), sub_name, "Age", age_list[1])
            combined_csv_path = combine_files(model_output_path, age_list[1] + "_final.csv")
            month_final_path = combined_csv_path
        else:
           for key in column_dict:
               for model in column_dict[key]:
                   model_output_path = os.path.join(output_path, str(year), sub_name, key, model)
                   combine_files(model_output_path, model + "_final.csv")

    copydown_path = os.path.join(output_path, str(year), sub_name, "Name", "Name.csv")
    name_final_paths = [name_final_path, last_final_path, relation_final_path, copydown_path]
    command = "sbatch ./combine_columns/run_copydown.sh " + " ".join(name_final_paths)

    print(command)
    os.system(command)

    age_month_path = os.path.join(output_path, str(year), sub_name, "Age", "Age.csv")
    age_final_paths = [year_final_path, month_final_path, relation_final_path, age_month_path]
    command = "sbatch ./combine_columns/run_age_months.sh " + " ".join(age_final_paths)

    print(command)
    os.system(command)


