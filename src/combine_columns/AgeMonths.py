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


    year_df = pd.read_csv(year_path, names=["filename", 'image_row_name', "year_string", "year_confidence", "year_blank"], skiprows=1)
    print("year", year_df.shape[0])
    year_df[["filename", "year_string"]] = year_df[["filename", "year_string"]].astype('string')
    year_df.drop_duplicates(subset=["filename"], keep='first', inplace=True)

    print(year_df.shape[0])

    month_df = pd.read_csv(month_path, names=["filename", "image_row_name", "month_string", "month_confidence", "month_blank"],
                           skiprows=1)
    print("month", month_df.shape[0])
    month_df.drop_duplicates(subset=['image_row_name'], keep='first', inplace=True)
    print(month_df.shape[0])

    df = pd.merge(year_df, month_df, on=['image_row_name'])
    print("year_month", df.shape[0])
    df['image_row_name'] = df['image_row_name'].astype("string")
    df.drop_duplicates(subset=['image_row_name'], keep='first', inplace=True)
    print(df.shape[0])

    relation_df = pd.read_csv(relation_path, names=["filename", 'image_row_name', "relation_string", "relation_confidence", "relation_blank"], skiprows=1)
    print("relation", relation_df.shape[0])

    relation_df[["filename", "relation_string"]] = relation_df[["filename", "relation_string"]].astype('string')
    relation_df.drop_duplicates(subset=["filename"], keep='first', inplace=True)
    print(relation_df.shape[0])

    relation_df = relation_df[["image_row_name", "relation_string", "relation_confidence", "relation_blank"]]

    df = pd.merge(df, relation_df, on=["image_row_name"])
    print("full", df.shape[0])
    df.drop_duplicates(subset=["image_row_name"], keep='first', inplace=True)
    df.dropna(inplace=True)
    print(df.shape[0])

    df['month_confidence'] = pd.to_numeric(df['month_confidence'], errors='coerce')
    df.dropna(inplace=True)

    df['year_confidence'] = pd.to_numeric(df['year_confidence'], errors='coerce')
    df.dropna(inplace=True)

    possible_month_df = df[df['month_confidence'] > 0.9]
    possible_month_df = possible_month_df[possible_month_df['year_confidence'] < 0.7]
    possible_month_df['replace_month'] = True

    df['age_final'] = np.where(((df.month_confidence > 0.9) & (df.year_confidence < 0.7)), df.month_string,
                               df.year_string)

    selected_df = df[['filename', 'image_row_name', 'age_final', 'year_confidence', 'year_blank']]
    selected_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
