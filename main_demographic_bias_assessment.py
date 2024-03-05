import json
import logging
import os
import time
from dataclasses import dataclass, field

import pandas as pd
from transformers import HfArgumentParser

from svc_demographic_bias_assessment import (
    compute_emr_boxplot_per_speaker,
    compute_heatmap_for_all_contingency_tables,
    compute_probability_tables,
    compute_splitted_emr_boxplot_per_speaker,
    compute_wer_boxplot_per_speaker,
    compute_wer_on_dataframe,
    create_dataframe_for_bias_assessment,
    CustomFormatter,
    perform_all_adjustment_tests,
    perform_all_anova_one_way_tests,
    perform_all_chi2_tests,
    perform_all_univariate_log_reg_tests,
)

logger = logging.getLogger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


@dataclass
class ScriptArguments:
    """
    Arguments needed to run this script, related to data path/directories.
    """

    asr_predictions_filepath: str = field(
        metadata={
            "help": "Filepath pointing towards the ASR predictions and associated metadata"
        },
    )

    save_figures: bool = field(
        metadata={"help": "Whether to save the generated plots and tables."},
    )

    save_data_directory: str = field(
        metadata={"help": "Directory where all cleaned data will be stored."},
    )

    group_usa_dialectal_regions: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to group all USA dialectal regions into a single one for impact on user analysis."
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataclasses_list = [
        ScriptArguments,
    ]
    parser = HfArgumentParser(dataclasses_list)
    args = parser.parse_args_into_dataclasses()
    datapath_args = args[0]
    save_plots_directory = os.path.join(datapath_args.save_data_directory, "plots")
    os.makedirs(save_plots_directory, exist_ok=True)

    logger.info("Loading ASR predictions and associated metadata")
    with open(datapath_args.asr_predictions_filepath, "r", encoding="utf-8") as file:
        predictions_json = json.load(file)

    df = create_dataframe_for_bias_assessment(predictions_json)

    logger.info("Compute WER per speaker boxplot")
    if "wer" not in df.columns.tolist():
        df = compute_wer_on_dataframe(dataframe=df)
    df_per_speaker_wer = compute_wer_boxplot_per_speaker(
        dataframe=df,
        save_directory=datapath_args.save_data_directory,
        save_figures=datapath_args.save_figures,
    )

    logger.info("Compute EMR per speaker boxplot")
    df_per_speaker_emr = compute_emr_boxplot_per_speaker(
        dataframe=df,
        save_directory=datapath_args.save_data_directory,
        save_figures=datapath_args.save_figures,
    )

    logger.info("Compute splitted EMR per speaker boxplot")
    compute_splitted_emr_boxplot_per_speaker(
        dataframe=df,
        save_directory=datapath_args.save_data_directory,
        save_figures=datapath_args.save_figures,
    )

    df_per_speaker = pd.merge(
        df_per_speaker_emr,
        df_per_speaker_wer,
        how="inner",
        on=["user_id", "gender", "age_group", "dialectal_region", "ethnicity"],
    )
    del df_per_speaker_emr, df_per_speaker_wer

    logger.info("Computing probability tables")
    all_probability_tables = compute_probability_tables(
        df, main_variable="exactlyParsed"
    )

    logger.info("Performing chi2 contingency tests")
    time.sleep(0.5)
    all_chi2_results = perform_all_chi2_tests(
        list_of_contingency_tables=all_probability_tables
    )
    logger.info("Compute and save heatmaps with contribution to non-independence score")
    os.makedirs(datapath_args.save_data_directory, exist_ok=True)
    compute_heatmap_for_all_contingency_tables(
        all_probability_tables,
        save_directory=datapath_args.save_data_directory,
        save_figures=datapath_args.save_figures,
    )

    logger.info("Performing univariate logistic regressions")
    time.sleep(0.5)
    all_fitted_log_reg = perform_all_univariate_log_reg_tests(dataframe=df)

    logger.info("Performing adjustment tests")
    time.sleep(0.5)
    all_fitted_adjustment_tests = perform_all_adjustment_tests(
        dataframe=df, fitted_univariate_log_regs=all_fitted_log_reg
    )

    variable_to_run_anova_on = ["mean_wer", "emr"]
    for variable_to_test in variable_to_run_anova_on:
        logger.info(f"Performing One-way ANOVA test on {variable_to_test}")
        time.sleep(0.5)
        all_anova_tests = perform_all_anova_one_way_tests(
            dataframe=df_per_speaker,
            variable_to_test_against=variable_to_test,
            group_usa_dialectal_regions=datapath_args.group_usa_dialectal_regions,
        )

    logger.info("Done!")
