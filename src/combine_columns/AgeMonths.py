import os
import argparse
import yaml
import difflib

import pandas as pd
import numpy as np


def main():
    # We need to read in all 3 relevant columns, Name, Last Name, and Relation to Head
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_config", type=str)

    parser.add_argument("--year", type=str)
    parser.add_argument("--month", type=str)
    parser.add_argument("--relation", type=str)

    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    if args.job_config:
        yaml_path = os.path.join(".", "job_config", args.job_config)
        with open(yaml_path, "r") as f:
            job_config = yaml.safe_load(f)

        config_name = args.job_config

        year_path = None
        month_path = None
        relation_path = None

    else:
        year_path = args.year
        month_path = args.month
        relation_path = args.relation
        output_path = args.output

    def makeImageRowName(x):
        filename = str(x['filename'])
        image_name = filename.split("_")[0]
        row_name = "_".join(filename.split("_")[3:])

        image_row_name = image_name + "_" + row_name
        return image_row_name

    year_df = pd.read_csv(year_path, names=["filename", "year_string", "year_confidence", "year_blank"], skiprows=1)
    print(1)
    year_df[["filename", "year_string"]] = year_df[["filename", "year_string"]].astype('string')
    print(2)
    year_df.drop_duplicates(inplace=True)
    print(3)

    month_df = pd.read_csv(month_path, names=["filename", "month_string", "month_confidence", "month_blank"],
                           skiprows=1)
    print(1)
    month_df.drop_duplicates(inplace=True)

    df = pd.merge(year_df, month_df, on=['filename'])
    df['filename'] = df['filename'].astype("string")
    df['image_row_name'] = df.apply(makeImageRowName, axis=1)
    df.drop_duplicates(subset=['filename'], keep='first', inplace=True)

    relation_df = pd.read_csv(relation_path, names=["filename", "relation_string", "relation_confidence", "relation_blank"], skiprows=1)

    relation_df[["filename", "relation_string"]] = relation_df[["filename", "relation_string"]].astype('string')
    print(2)
    relation_df["image_row_name"] = relation_df.apply(makeImageRowName, axis=1)
    print(3)
    relation_df = relation_df[["image_row_name", "relation_string", "relation_confidence", "relation_blank"]]
    print(4)

    df = pd.merge(df, relation_df, on=["image_row_name"])
    df.drop_duplicates(subset=["image_row_name"], keep='first', inplace=True)
    df.dropna(inplace=True)

    df['month_confidence'] = pd.to_numeric(df['month_confidence'], errors='coerce')
    df.dropna(inplace=True)

    df['year_confidence'] = pd.to_numeric(df['year_confidence'], errors='coerce')
    df.dropna(inplace=True)

    possible_month_df = df[df['month_confidence'] > 0.9]
    possible_month_df = possible_month_df[possible_month_df['year_confidence'] < 0.7]
    possible_month_df['replace_month'] = True

    df['age_final'] = np.where(((df.month_confidence > 0.9) & (df.year_confidence < 0.7)), df.month_string,
                               df.year_string)

    selected_df = df[["filename", 'image_row_name', 'age_final', 'year_blank']]
    selected_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
