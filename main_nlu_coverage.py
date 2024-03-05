"""
This script allows to measure the coverage of the train split dataset
"""
import json
import logging
import os
from dataclasses import dataclass, field

import plotly.express as px
from transformers import HfArgumentParser

from svc_demographic_bias_assessment import (
    get_dataframe_nb_values_per_slot,
    get_histogram_of_nb_utterances_per_intent,
    get_histogram_slot_tuples_playmusic,
    get_k_first_slot_tuples_in_playmusic_data,
    get_number_of_unique_utterances_per_intent,
    mapping_nb_of_slots_and_utterances,
    separate_playmusic_and_transportcontrol_utterances,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments needed to run this script.
    """

    train_metadata_filepath: str = field(
        metadata={"help": "Filepath pointing towards the train split metadata."},
    )

    save_figures: bool = field(
        metadata={"help": "Whether to save the generated plots and tables."},
    )

    save_data_directory: str = field(
        metadata={"help": "Directory where all created data will be stored."},
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

    logger.info("Running main_nlu_coverage.py script with following arguments:")
    logger.info(vars(datapath_args))

    logger.info("Loading train metadata")
    with open(datapath_args.train_metadata_filepath, "r", encoding="utf-8") as file:
        train_dataset = json.load(file)

    # separate PlayMusic audios and TransportControl ones
    (
        remapping_train_dict_pm,
        remapping_train_dict_tc,
        transport_control_intents,
    ) = separate_playmusic_and_transportcontrol_utterances(train_metadata=train_dataset)

    logger.info(
        f"There are {len(transport_control_intents) + 1} intents in the dataset"
    )

    logger.info("Start focusing on PlayMusic data")
    logger.info(
        f"Number of unique utterances in train split for PlayMusic intent: {len(remapping_train_dict_pm)}"
    )
    logger.info(
        "Mapping giving the number of utterances containing 1, 2, 3, etc slots:"
    )
    dict_nb_slots_per_utterance = mapping_nb_of_slots_and_utterances(
        remapping_train_dict_pm
    )
    logger.info(dict_nb_slots_per_utterance)
    assert len(remapping_train_dict_pm) == sum(dict_nb_slots_per_utterance.values())

    logger.info(
        "Computing the count of slot_name/entity_name in train split for PlayMusic intent"
    )
    df_pm_slots = get_dataframe_nb_values_per_slot(remapping_train_dict_pm)
    df_plot = df_pm_slots.reset_index()
    df_plot["index"] = df_plot["index"].astype(str)

    df_plot = df_plot.rename(columns={"index": "slot_name/entity_name"})
    fig = px.histogram(df_plot, x="slot_name/entity_name", y="count")
    fig.update_layout(
        title_text="Count of slot_name/entity_name in train split for PlayMusic intent among all unique utterances",
        xaxis_title_text="slot_name/entity_name",
        yaxis_title_text="Count",
        bargap=0.2,
        bargroupgap=0.1,
        height=1000,
        width=1600,
    )
    fig.update_xaxes(categoryorder="total descending")
    fig.show()
    if datapath_args.save_figures:
        os.makedirs(
            os.path.join(datapath_args.save_data_directory, "plots"), exist_ok=True
        )
        savepath = os.path.join(
            datapath_args.save_data_directory,
            "plots",
            "count_slot_name_entity_name_train_split.png",
        )
        print(f"Saving plot at {savepath}")
        fig.write_image(savepath)

    df_top_k_slot_tuples = get_k_first_slot_tuples_in_playmusic_data(
        remapping_train_dict_pm, k=25
    )
    fig = get_histogram_slot_tuples_playmusic(df_top_k_slot_tuples)
    fig.show()
    if datapath_args.save_figures:
        savepath = os.path.join(
            datapath_args.save_data_directory,
            "plots",
            "count_slot_tuples_playmusic_train_split.png",
        )
        print(f"Saving plot at {savepath}")
        fig.write_image(savepath)

    logger.info("Start focusing on TransportControl now:")
    logger.info(
        f"There are {len(transport_control_intents)} intents for TransportControl"
    )

    logger.info(
        f"Number of unique utterances in train split for TransportControl: {len(remapping_train_dict_tc)}"
    )
    logger.info(
        "Mapping giving the number of utterances containing 1, 2, 3, etc slots:"
    )
    dict_nb_slots_per_utterance = mapping_nb_of_slots_and_utterances(
        remapping_train_dict_tc
    )
    logger.info(dict_nb_slots_per_utterance)
    assert len(remapping_train_dict_tc) == sum(dict_nb_slots_per_utterance.values())

    tc_dict_nb_utt_per_intents = get_number_of_unique_utterances_per_intent(
        remapping_train_dict_tc
    )
    logger.info("Number of unique utterances per intent in train split")
    sorted_list = sorted(tc_dict_nb_utt_per_intents.items(), key=lambda x: x[1])
    logger.info(sorted_list)
    fig = get_histogram_of_nb_utterances_per_intent(sorted_list)
    fig.show()
    if datapath_args.save_figures:
        savepath = os.path.join(
            datapath_args.save_data_directory,
            "plots",
            "nb_utt_per_intent_tc_train_split.png",
        )
        print(f"Saving plot at {savepath}")
        fig.write_image(savepath)

    logger.info("Done!")
