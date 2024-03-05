"""
Main script dedicated to provide statistical description of SVC Demographic Bias Assessment Dataset

With this script you can visualize statistics about each set separately
"""
import json
import logging
import os
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from svc_demographic_bias_assessment import (
    create_dataframe_for_statistical_description,
    get_distribution_count_statistics,
    VARIABLES_OF_INTEREST,
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

    save_figures: bool = field(
        metadata={"help": "Whether to save the generated plots and tables."},
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

    logger.info("Running main_descriptive_analysis.py script with following arguments:")
    logger.info(vars(datapath_args))

    logger.info(f"Loading metadata of set {datapath_args.set_}")
    with open(datapath_args.metadata_filepath, "r", encoding="utf-8") as file:
        metadata_json = json.load(file)

    logger.info("Creating dataset with shape: ")
    df = create_dataframe_for_statistical_description(metadata_json)
    logger.info(df.shape)

    VARIABLES_OF_INTEREST.remove("ethnicity")

    logger.info(f"Number of unique speakers in this set: {df['user_id'].nunique()}")

    logger.info(
        f"Distribution in terms of number of speakers on {datapath_args.set_} set:"
    )
    fig = get_distribution_count_statistics(
        df,
        VARIABLES_OF_INTEREST,
        set_=datapath_args.set_,
        distribution_type="speakers",
    )
    fig.show()
    if datapath_args.save_figures:
        os.makedirs(
            os.path.join(datapath_args.save_data_directory, "plots"), exist_ok=True
        )
        savepath = os.path.join(
            datapath_args.save_data_directory,
            "plots",
            f"distribution_by_speakers_{datapath_args.set_}_set.png",
        )
        logger.info(f"Saving plot at {savepath}")
        fig.write_image(savepath)

    logger.info(f"Number of audios in this set: {df.index.nunique()}")
    logger.info(
        f"Number of PlayMusic audios among them: {df[df['label.intent'] == 'PlayMusic'].index.nunique()}"
    )

    logger.info(
        f"Distribution in terms of number of samples on {datapath_args.set_} set:"
    )
    fig = get_distribution_count_statistics(
        df,
        VARIABLES_OF_INTEREST,
        set_=datapath_args.set_,
        distribution_type="samples",
    )
    fig.show()
    if datapath_args.save_figures:
        savepath = os.path.join(
            datapath_args.save_data_directory,
            "plots",
            f"distribution_by_samples_{datapath_args.set_}_set.png",
        )
        logger.info(f"Saving plot at {savepath}")
        fig.write_image(savepath)

    if datapath_args.set_ == "test":
        logger.info("Focus on the particular case of the partial 'ethnicity' label")
        where = ~df["ethnicity"].isna()
        df_eth = df[where]
        logger.info(df_eth.shape)
        logger.info(f"Number of unique speakers: {df_eth['user_id'].nunique()}")

        logger.info(
            "Distribution in terms of number of speakers on ethnicity only dataset:"
        )
        fig = get_distribution_count_statistics(
            df_eth,
            VARIABLES_OF_INTEREST,
            set_="ethnicity",
            distribution_type="speakers",
        )
        fig.show()
        if datapath_args.save_figures:
            savepath = os.path.join(
                datapath_args.save_data_directory,
                "plots",
                "distribution_by_speakers_ethnicity_only.png",
            )
            logger.info(f"Saving plot at {savepath}")
            fig.write_image(savepath)

        logger.info(
            f"Number of audios in ethnicity only dataset: {df_eth.index.nunique()}"
        )
        logger.info(
            f"Number of PlayMusic audios among them: {df_eth[df_eth['label.intent'] == 'PlayMusic'].index.nunique()}"
        )

        logger.info(
            "Distribution in terms of number of samples on ethnicity only dataset:"
        )
        fig = get_distribution_count_statistics(
            df_eth,
            VARIABLES_OF_INTEREST,
            set_="ethnicity",
            distribution_type="samples",
        )
        fig.show()
        if datapath_args.save_figures:
            savepath = os.path.join(
                datapath_args.save_data_directory,
                "plots",
                "distribution_by_samples_ethnicity_only.png",
            )
            logger.info(f"Saving plot at {savepath}")
            fig.write_image(savepath)

    logger.info("Done!")
