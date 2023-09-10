import pandas as pd
import numpy as np
import os
import shutil
pd.set_option('display.max_columns', None)


def remove_vacant_last_names():
    previousLast = ""
    for index, row in data.iterrows():
        lastName = row["LastName"]
        if lastName == previousLast:
            row["LastName"] = "<sab>"
        else:
            row["LastName"] = "<nln>"
            previousLast = lastName

        wrongFileName = row["FileName_Updated"]
        snipRow = int(wrongFileName.split("_")[-1].split(".")[0])
        snipRow -= 1
        row["FileName_Updated"] = "_".join(wrongFileName.split("_")[:-1], ) + "_{:02d}".format(snipRow) + ".jpg"


states = {
    "AL": "alabama",
    "AK": "alaska",
    "AZ": "arizona",
    "AR": "arkansas",
    "CA": "california",
    "CO": "colorado",
    "CT": "connecticut",
    "DE": "delaware",
    "FL": "florida",
    "GA": "georgia",
    "HI": "hawaii",
    "ID": "idaho",
    "IL": "illinois",
    "IN": "indiana",
    "IA": "iowa",
    "KS": "kansas",
    "KY": "kentucky",
    "LA": "louisiana",
    "ME": "maine",
    "MD": "maryland",
    "MA": "massachusetts",
    "MI": "michigan",
    "MN": "minnesota",
    "MS": "mississippi",
    "MO": "missouri",
    "MT": "montana",
    "NE": "nebraska",
    "NV": "nevada",
    "NH": "new hampshire",
    "NJ": "new jersey",
    "NM": "new mexico",
    "NY": "new york",
    "NC": "north carolina",
    "ND": "north dakota",
    "OH": "ohio",
    "OK": "oklahoma",
    "OR": "oregon",
    "PA": "pennsylvania",
    "RI": "rhode island",
    "SC": "south carolina",
    "SD": "south dakota",
    "TN": "tennessee",
    "TX": "texas",
    "UT": "utah",
    "VT": "vermont",
    "VA": "virginia",
    "WA": "washington",
    "WV": "west virginia",
    "WI": "wisconsin",
    "WY": "wyoming",
}

tsv_path = "D:/WAData/1940_age_bigger_snippet_list.tsv"
column_name = "Age"

data = pd.read_csv(tsv_path, sep="\t")
data = data[["SnippetName", column_name]]
data = data.rename(columns={"SnippetName": "filename", column_name: "string"})
data = data.sample(n=50000)

tsv_path = "D:\WAData/1940_visit_index_labeled_nonblank.csv"

data1 = pd.read_csv(tsv_path)
data1["string"] = data1["string"].astype(str)
data = pd.concat([data, data1], ignore_index=True)


def fix_string(birthplace):
    if birthplace == "<BLANK>":
        return " "
    elif not birthplace.isnumeric():
        return " "
    else:
        return birthplace.lower()


data = data.dropna()
data['string'] = data['string'].apply(fix_string)
data = data[data["string"] != " "]
#data['string'] = data["string"].astype(int)
# multiple_occurrence = data['string'].value_counts() > 5
# data = data[data["string"].isin(multiple_occurrence[multiple_occurrence].index)]

# data = data[data["string"] != "col"]
# data = data[data["string"] != "ambiguous"]
# data = data[data["string"] != "illegible"]

# non_white = data[data["string"] != "w"]
# white = data[data["string"] == "w"]
# white = white.sample(60000)
# data = pd.concat((white, non_white))


def get_sample(X_test, y_true, **args):
    """Get a sample dataframe from X_test and y_true.

    Args:
        X_test: first input dataframe.
        y_true: second input dataframe.

    Returns:
        Sample dataframe.

    """
    df = pd.concat([X_test, y_true], axis=1)
    sample_df = df.sample(**args).reset_index(drop=True)
    print(y_true.unique())
    target = (
        len(y_true.unique())
        if sample_df.shape[0] > len(y_true.unique())
        else sample_df.shape[0]
    )
    while sample_df.iloc[:, -1].nunique() != target:
        sample_df = df.sample(**args).reset_index(drop=True)
    return sample_df





# data = get_sample(data.iloc[:, :-1], data.iloc[:, -1], n=10000)

csv_path = "D:/WAData/combinedDigit2.csv"
# csv_path = tsv_path.split(".")[0] + ".csv"
print(len(data.index))

data.to_csv(csv_path, index=None)
