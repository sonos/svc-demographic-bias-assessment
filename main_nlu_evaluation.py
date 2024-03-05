import json
import logging
import os
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from svc_demographic_bias_assessment import (
    compute_inference_metrics,
    merge_metadata_and_bert_predictions,
    save_metadata_json,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments needed to run this script, related to data path/directories.
    """

    bert_predictions_filepath: str = field(
        metadata={"help": "Filepath pointing towards the BERT predictions"},
    )

    test_metadata_filepath: str = field(
        metadata={"help": "Filepath pointing towards the test metadata filepath"},
    )

    save_data_directory: str = field(
        metadata={"help": "Directory where all cleaned data will be stored."},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataclasses_list = [
        ScriptArguments,
    ]
    parser = HfArgumentParser(dataclasses_list)
    args = parser.parse_args_into_dataclasses()
    datapath_args = args[0]

    logger.info("Loading BERT predictions")
    with open(datapath_args.bert_predictions_filepath, "r", encoding="utf-8") as file:
        predictions_json = json.load(file)

    logger.info("Loading test metadata")
    with open(datapath_args.test_metadata_filepath, "r", encoding="utf-8") as file:
        test_metadata_json = json.load(file)

    assert len(test_metadata_json.keys()) == len(predictions_json.keys())

    logger.info("Merging BERT predictions and test metadata")
    updated_metadata = merge_metadata_and_bert_predictions(
        test_metadata_json, predictions_json
    )

    logger.info("Start computing metrics")
    metrics, updated_predictions_json = compute_inference_metrics(
        metadata_dict=updated_metadata,
    )

    os.makedirs(datapath_args.save_data_directory, exist_ok=True)
    save_filepath = os.path.join(
        datapath_args.save_data_directory,
        "metadata_test_with_bert_predictions.json",
    )
    logger.info(
        f"Saving test metadata with updated field exactlyParsed at {save_filepath}"
    )
    save_metadata_json(updated_predictions_json, save_filepath)
    logger.info("Done!")
