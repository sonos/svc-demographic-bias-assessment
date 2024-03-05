from .constants import VARIABLES_OF_INTEREST
from .dataset_manipulator import DatasetManipulator, SVCDataset
from .descriptive_analysis import (
    get_distribution_count_statistics,
    plot_histogram_groupby,
)
from .logger_utils import CustomFormatter
from .nlu_evaluation import (
    compute_inference_metrics,
    merge_metadata_and_bert_predictions,
)
from .nlu_utils import (
    get_dataframe_nb_values_per_slot,
    get_histogram_of_nb_utterances_per_intent,
    get_histogram_slot_tuples_playmusic,
    get_k_first_slot_tuples_in_playmusic_data,
    get_number_of_unique_utterances_per_intent,
    mapping_nb_of_slots_and_utterances,
    separate_playmusic_and_transportcontrol_utterances,
)
from .statistical_analysis import (
    compute_emr_boxplot_per_speaker,
    compute_heatmap_for_all_contingency_tables,
    compute_probability_tables,
    compute_splitted_emr_boxplot_per_speaker,
    compute_wer_boxplot_per_speaker,
    compute_wer_on_dataframe,
    perform_all_adjustment_tests,
    perform_all_anova_one_way_tests,
    perform_all_chi2_tests,
    perform_all_univariate_log_reg_tests,
)
from .utils import (
    create_dataframe_for_bias_assessment,
    create_dataframe_for_statistical_description,
    save_metadata_json,
)
