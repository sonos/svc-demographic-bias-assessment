"""
Get statistics needed to make confusion bias tables
"""

import json
import logging
import os
from dataclasses import dataclass, field

from tabulate import tabulate
from transformers import HfArgumentParser

from svc_demographic_bias_assessment import (
    create_dataframe_for_statistical_description,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments needed to run this script.
    """

    set_: str = field(
        metadata={"help": "Either train, dev or test."},
    )

    metadata_filepath: str = field(
        metadata={
            "help": "Filepath pointing towards the metadata of the corresponding set."
        },
    )

    save_tables: bool = field(
        metadata={"help": "Whether to save the generated tables."},
    )

    save_data_directory: str = field(
        metadata={"help": "Directory where all created data will be stored."},
    )

    def __post_init__(self):
        assert self.set_ in ["train", "dev", "test"], ValueError(
            f"set_ should be either train, dev or test; got {self.set_}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataclasses_list = [
        ScriptArguments,
    ]
    parser = HfArgumentParser(dataclasses_list)
    args = parser.parse_args_into_dataclasses()
    datapath_args = args[0]
    os.makedirs(datapath_args.save_data_directory, exist_ok=True)

    logger.info("Running main_confusion_bias.py script with following arguments:")
    logger.info(vars(datapath_args))

    logger.info(f"Loading metadata of set {datapath_args.set_}")
    with open(datapath_args.metadata_filepath, "r", encoding="utf-8") as file:
        metadata_json = json.load(file)

    logger.info("Creating dataset with shape: ")
    df = create_dataframe_for_statistical_description(metadata_json)
    df = df.reset_index()
    logger.info(df.shape)

    ############################################ DIALECTAL REGION ############################################
    logger.info(
        "Compute for each dialectal region, the number of speakers, samples and age and gender repartition to detect "
        "possible empty buckets"
    )

    logger.info("Number of audio samples & speakers per dialectal region:")
    df1 = df.groupby("dialectal_region").agg(
        {"audio_id": "count", "user_id": "nunique"}
    )
    logger.info(tabulate(df1, headers="keys", tablefmt="psql"))

    logger.info("Distribution of age group per dialectal region:")
    df2 = df.groupby(["dialectal_region", "age_group"])["audio_id"].count().to_frame()
    df2["percentage"] = (
        df2.groupby(["dialectal_region"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df2, headers="keys", tablefmt="psql"))

    logger.info("Distribution of gender per dialectal region:")
    df3 = df.groupby(["dialectal_region", "gender"])["audio_id"].count().to_frame()
    df3["percentage"] = (
        df3.groupby(["dialectal_region"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df3, headers="keys", tablefmt="psql"))

    ############################################ AGE ############################################
    logger.info(
        "Compute for each age group, the number of speakers, samples and dialectal region and gender repartition to detect "
        "possible empty buckets"
    )

    logger.info("Number of audio samples & speakers per age group:")
    df1 = df.groupby("age_group").agg({"audio_id": "count", "user_id": "nunique"})
    logger.info(tabulate(df1, headers="keys", tablefmt="psql"))

    logger.info("Distribution of dialectal region per age group:")
    df2 = df.groupby(["age_group", "dialectal_region"])["audio_id"].count().to_frame()
    df2["percentage"] = (
        df2.groupby(["age_group"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df2, headers="keys", tablefmt="psql"))

    logger.info("Distribution of gender per age group:")
    df3 = df.groupby(["age_group", "gender"])["audio_id"].count().to_frame()
    df3["percentage"] = (
        df3.groupby(["age_group"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df3, headers="keys", tablefmt="psql"))

    ############################################ GENDER ############################################
    logger.info(
        "Compute for each gender, the number of speakers, samples and dialectal region and age group repartition to detect "
        "possible empty buckets"
    )

    logger.info("Number of audio samples & speakers per gender:")
    df1 = df.groupby("gender").agg({"audio_id": "count", "user_id": "nunique"})
    logger.info(tabulate(df1, headers="keys", tablefmt="psql"))

    logger.info("Distribution of dialectal region per gender:")
    df2 = df.groupby(["gender", "dialectal_region"])["audio_id"].count().to_frame()
    df2["percentage"] = (
        df2.groupby(["gender"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df2, headers="keys", tablefmt="psql"))

    logger.info("Distribution of gender per age group:")
    df3 = df.groupby(["gender", "age_group"])["audio_id"].count().to_frame()
    df3["percentage"] = (
        df3.groupby(["gender"])["audio_id"]
        .transform(lambda x: (x / x.sum()) * 100)
        .round()
        .to_frame()
    )
    logger.info(tabulate(df3, headers="keys", tablefmt="psql"))

    ############################################ ETHNICITY ############################################
    if datapath_args.set_ == "test":
        logger.info("Focus on the particular case of the partial 'ethnicity' label")
        where = ~df["ethnicity"].isna()
        df_eth = df[where]
        logger.info(df_eth.shape)
        logger.info(f"Number of unique speakers: {df_eth['user_id'].nunique()}")
        logger.info(
            "Compute for each dialectal region, the number of speakers, samples and age and gender repartition to "
            "detect possible empty buckets"
        )
        logger.info("Number of audio samples & speakers per ethnicity:")
        df1 = df_eth.groupby("ethnicity").agg(
            {"audio_id": "count", "user_id": "nunique"}
        )
        logger.info(tabulate(df1, headers="keys", tablefmt="psql"))

        logger.info("Distribution of age group per ethnicity:")
        df2 = df_eth.groupby(["ethnicity", "age_group"])["audio_id"].count().to_frame()
        df2["percentage"] = (
            df2.groupby(["ethnicity"])["audio_id"]
            .transform(lambda x: (x / x.sum()) * 100)
            .round()
            .to_frame()
        )
        logger.info(tabulate(df2, headers="keys", tablefmt="psql"))

        logger.info("Distribution of gender per ethnicity:")
        df3 = df_eth.groupby(["ethnicity", "gender"])["audio_id"].count().to_frame()
        df3["percentage"] = (
            df3.groupby(["ethnicity"])["audio_id"]
            .transform(lambda x: (x / x.sum()) * 100)
            .round()
            .to_frame()
        )
        logger.info(tabulate(df3, headers="keys", tablefmt="psql"))

        logger.info("Distribution of dialectal region per ethnicity:")
        df4 = (
            df_eth.groupby(["ethnicity", "dialectal_region"])["audio_id"]
            .count()
            .to_frame()
        )
        df4["percentage"] = (
            df4.groupby(["ethnicity"])["audio_id"]
            .transform(lambda x: (x / x.sum()) * 100)
            .round()
            .to_frame()
        )
        logger.info(tabulate(df4, headers="keys", tablefmt="psql"))

    logger.info("Done!")
