import os
import argparse
import yaml
import difflib

import pandas as pd
import numpy as np


def makeImageRowName(x):
    x = x["filename"]
    image_name_parts = x.split("_")

    if len(image_name_parts) == 6:
        img_name = "_".join(image_name_parts[:2])
        row_name = "_".join(x.split("_")[4:])

        img_row_name = img_name + "_" + row_name
    else:
        img_name = image_name_parts[0]
        row_name = "_".join(x.split("_")[3:])

        img_row_name = img_name + "_" + row_name
    return img_row_name


def main():
    # We need to read in all 3 relevant columns, Name, Last Name, and Relation to Head
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_config", type=str)

    parser.add_argument("--name", type=str)
    parser.add_argument("--last_name", type=str)
    parser.add_argument("--relation", type=str)

    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    name_path = args.name
    last_name_path = args.last_name
    relation_path = args.relation
    output_path = args.output

    name_df = pd.read_csv(name_path,
                          names=["filename", "image_row_name", "name_string", "name_confidence", "name_blank"],
                          skiprows=1)
    name_df[["filename", "name_string"]] = name_df[["filename", "name_string"]].astype('string')
    name_df["image_row_name"] = name_df.apply(makeImageRowName, axis=1)
    name_df.drop_duplicates(inplace=True)

    values = ['<nln>', '<sab>']
    last_name_df = pd.read_csv(last_name_path,
                               names=["filename", "image_row_name", "last_string", "last_confidence", "last_blank"],
                               skiprows=1)
    last_name_df[["filename", "last_string"]] = last_name_df[["filename", "last_string"]].astype('string')
    last_name_df['last_token'] = [next(iter(difflib.get_close_matches(name, values)), name) for name in
                                  last_name_df["last_string"]]
    last_name_df["image_row_name"] = last_name_df.apply(makeImageRowName, axis=1)
    last_name_df.drop_duplicates(inplace=True)

    df = pd.merge(name_df, last_name_df, on=["filename", "image_row_name"])
    df.drop_duplicates(subset=['filename'], keep='first', inplace=True)

    relation_df = pd.read_csv(relation_path,
                              names=["filename", "image_row_name", "relation_string", "relation_confidence", "relation_blank"],
                              skiprows=1)
    relation_df[["filename", "relation_string"]] = relation_df[["filename", "relation_string"]].astype('string')
    relation_df["image_row_name"] = relation_df.apply(makeImageRowName, axis=1)
    relation_df = relation_df[["image_row_name", "relation_string", "relation_confidence", "relation_blank"]]
    relation_df.drop_duplicates(inplace=True)

    df = pd.merge(df, relation_df, on=["image_row_name"])
    df.drop_duplicates(subset=["image_row_name"], keep='first', inplace=True)
    df['name_string'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    df['word_count'] = df.apply(lambda x: len(str(x['name_string']).split()), axis=1)

    # now we need more information, first filter out the names that are one word long, these are given names.
    one_word_df = df[df['word_count'] == 1]

    remaining_df = pd.merge(df, one_word_df, indicator=True, how='outer') \
        .query('_merge=="left_only"') \
        .drop('_merge', axis=1)

    one_word_df["has_last"] = False

    # second, filter out names that are two words long, with the second word being one letter, those are also given
    # names
    two_word_df = remaining_df[remaining_df['word_count'] == 2]
    two_word_df["second_name_len"] = two_word_df.apply(lambda x: len(str(x['name_string']).split()[-1]), axis=1)

    name_initial_df = two_word_df[two_word_df["second_name_len"] == 1]

    remaining_df = pd.merge(remaining_df, name_initial_df, indicator=True, how='outer') \
        .query('_merge=="left_only"') \
        .drop('_merge', axis=1)
    name_initial_df["has_last"] = False

    # Now that I've filtered off the easy ones based on just the name, I'm going to filter off those that have
    # both the NLN token and are head of household.

    head_df = remaining_df[remaining_df['relation_string'] == 'head']

    head_df["has_last"] = True

    remaining_df = pd.merge(remaining_df, head_df, indicator=True, how='outer') \
        .query('_merge=="left_only"') \
        .drop('_merge', axis=1)

    remaining_df["has_last"] = [True if x == "<nln>" else False for x in remaining_df["last_token"]]

    final_df = pd.concat([one_word_df, name_initial_df, head_df, remaining_df])

    selected_df = final_df[["filename", "image_row_name", "name_string", "has_last", "relation_string", "last_string",
                            "name_blank"]]
    common_last_prefixes = ['VON', 'VAN', 'LA', 'O', 'MC', 'EL', 'AL', 'LE', 'LA', 'DA', 'DE', 'DI', 'DO', 'DOS', 'DAS',
                            'DEL', 'SAN', 'BIN', 'BEN', 'OF', 'SANTA' 'SANTO', 'SAINT', 'D', 'DES', 'DELLA', 'DELA']

    def split_name(x):
        name = x['name_string']
        name_parts = name.split()

        if not x['has_last'] or len(name_parts) == 1:
            last_name = ""
            first_name = name
        elif len(name_parts) > 1:
            if len(name_parts[0]) == 1 or name_parts[0] in common_last_prefixes or len(name_parts) == 4:
                last_name = " ".join(name_parts[:2])
                first_name = " ".join(name_parts[2:])
            else:
                last_name = name_parts[0]
                first_name = " ".join(name_parts[1:])
        else:
            last_name = ""
            first_name = ""
        return pd.Series([first_name, last_name])

    selected_df[['first_name', 'last_name']] = selected_df.apply(split_name, axis=1)
    selected_df.sort_values("image_row_name", inplace=True, ignore_index=True)
    selected_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
