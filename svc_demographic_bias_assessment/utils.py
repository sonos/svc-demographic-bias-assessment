import json
from typing import Dict, List

import pandas as pd

COLS_TO_DROP_FOR_STATISTICAL_TESTING = [
    "audioLengthMs",
    "content",
    "text",
    "label.intent",
    "label.slots",
    "age",
    "user_group",
    "slots",
    "intent.intentName",
    "intent.confidence",
    "intent.decodedString",
]


def create_dataframe_for_statistical_description(metadata_dict: Dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metadata_dict, orient="index")
    df.index = df.index.set_names(["audio_id"])
    df = df.reset_index()
    df = df.join(pd.json_normalize(df.pop("speaker")))
    df = df.join(pd.json_normalize(df.pop("transcript")))
    df = df.set_index("audio_id")
    df["dialectal_region"] = df["dialectal_region"].replace("Latino", "LatinX")
    df["dialectal_region"] = df["dialectal_region"].replace("USA ", "", regex=True)
    return df


def create_dataframe_for_bias_assessment(
    metadata_dict: Dict,
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metadata_dict, orient="index")
    df.index = df.index.set_names(["audio_id"])
    df = df.reset_index()
    if "transcript" in df.columns.to_list():
        df = df.join(pd.json_normalize(df.pop("transcript")))
    df = df.join(pd.json_normalize(df.pop("speaker")))
    if "nlu_prediction" in df.columns.to_list():
        df = df.join(pd.json_normalize(df.pop("nlu_prediction")))
    df = df.set_index("audio_id")
    for col in COLS_TO_DROP_FOR_STATISTICAL_TESTING:
        if col in df.columns.to_list():
            df = df.drop(columns=col)
    df["exactlyParsed"] = df["exactlyParsed"].replace({True: 1, False: 0})
    df["dialectal_region"] = df["dialectal_region"].replace("Latino", "LatinX")
    df["dialectal_region"] = df["dialectal_region"].str.replace("USA ", "")
    return df


def save_list_to_txt_file(list_: List[str], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        for element in list_:
            file.write(element + "\n")


def save_metadata_json(metadata_dict: Dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(metadata_dict, file, indent=4, ensure_ascii=False)
