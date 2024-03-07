import itertools
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
from scipy.stats._mstats_basic import f_oneway, F_onewayResult
from scipy.stats.contingency import chi2_contingency, Chi2ContingencyResult
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from .asr_evaluation import WERAccumulator
from .constants import VARIABLES_OF_INTEREST
from .descriptive_analysis import age_sorting
from .logger_utils import Colors

logger = logging.getLogger(__name__)


def compute_wer_on_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    wer_accumulator = WERAccumulator()
    dataframe = dataframe.reset_index()
    dataframe["wer"] = dataframe.apply(
        lambda row: _helper_compute_wer_on_dataframe_raw(
            audio_id=row["audio_id"],
            reference=row["normalized_text"],
            hypothesis=row["asr_prediction"],
            wer_accumulator=wer_accumulator,
        ),
        axis=1,
    )
    print(str(wer_accumulator))
    return dataframe.set_index("audio_id")


def _helper_compute_wer_on_dataframe_raw(
    audio_id: str, reference: str, hypothesis: str, wer_accumulator: WERAccumulator
) -> float:
    wer, _ = wer_accumulator.compute(
        idx=audio_id,
        reference_string=reference,
        hypothesis_string=hypothesis,
    )
    return wer


def compute_wer_boxplot_per_speaker(
    dataframe: pd.DataFrame,
    save_directory: str,
    save_figures: bool,
) -> pd.DataFrame:
    dataframe = dataframe.reset_index()
    dataframe["mean_wer"] = dataframe.groupby("user_id")["wer"].transform("mean")
    df_per_speaker = dataframe[
        ["user_id", "mean_wer", "gender", "age_group", "dialectal_region", "ethnicity"]
    ].drop_duplicates()
    for col in ["gender", "age_group", "dialectal_region", "ethnicity"]:
        df_tmp = df_per_speaker.copy()
        if col == "ethnicity":
            where_ethnicity = ~(df_per_speaker["ethnicity"].isna())
            df_tmp = df_per_speaker[where_ethnicity]
        fig = px.box(df_tmp, x=col, y="mean_wer")
        if col == "age_group":
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=sorted(df_per_speaker[col].unique(), key=age_sorting),
            )
        elif col == "dialectal_region":
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=[
                    "Asian",
                    "LatinX",
                    "Inland-North",
                    "Mid-Atlantic",
                    "Midland",
                    "New England",
                    "Southern",
                    "Western",
                ],
            )

        fig.update_layout(
            font=dict(
                size=30,
            ),
            xaxis_title="",
            yaxis_title="Mean WER",
            height=800,
            width=1000,
        )
        fig.show()
        if save_figures:
            savefilepath = os.path.join(
                save_directory, "plots", f"boxplot_wer_{col}.png"
            )
            logger.info(f"Saving boxplot at {savefilepath}")
            fig.write_image(file=savefilepath, format="png")
    return df_per_speaker


def compute_emr_boxplot_per_speaker(
    dataframe: pd.DataFrame,
    save_directory: str,
    save_figures: bool,
) -> pd.DataFrame:
    os.makedirs(os.path.join(save_directory, "plots"), exist_ok=True)
    dataframe = dataframe.reset_index()
    dataframe["emr"] = dataframe.groupby("user_id")["exactlyParsed"].transform("mean")
    df_per_speaker = dataframe[
        ["user_id", "emr", "gender", "age_group", "dialectal_region", "ethnicity"]
    ].drop_duplicates()
    for col in ["gender", "age_group", "dialectal_region", "ethnicity"]:
        df_tmp = df_per_speaker.copy()
        if col == "ethnicity":
            where_ethnicity = ~(df_per_speaker["ethnicity"].isna())
            df_tmp = df_per_speaker[where_ethnicity]
        fig = px.box(df_tmp, x=col, y="emr")
        if col == "age_group":
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=sorted(df_per_speaker[col].unique(), key=age_sorting),
            )
            fig.layout.xaxis2 = go.layout.XAxis(
                overlaying="x", range=[0, 5], showticklabels=False
            )
            fig.add_scatter(
                x=[0, 1, 2, 3, 4, 5],
                y=[0.91, 0.91, 0.91, 0.91, 0.91, 0.91],
                mode="lines",
                xaxis="x2",
                showlegend=False,
                line=dict(color="green", width=2),
                opacity=0.5,
            )
        elif col == "dialectal_region":
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=[
                    "Asian",
                    "LatinX",
                    "Inland-North",
                    "Mid-Atlantic",
                    "Midland",
                    "New England",
                    "Southern",
                    "Western",
                ],
            )
            fig.layout.xaxis2 = go.layout.XAxis(
                overlaying="x", range=[0, 6], showticklabels=False
            )
            fig.add_scatter(
                x=[
                    "Asian",
                    "LatinX",
                    "Inland-North",
                    "Mid-Atlantic",
                    "Midland",
                    "New England",
                    "Southern",
                    "Western",
                ],
                y=[0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.99],
                mode="lines",
                xaxis="x2",
                showlegend=False,
                line=dict(color="green", width=2),
                opacity=0.5,
            )
        else:
            fig.layout.xaxis2 = go.layout.XAxis(
                overlaying="x", range=[0, 2], showticklabels=False
            )
            fig.add_scatter(
                x=[0, 2],
                y=[0.91, 0.91],
                mode="lines",
                xaxis="x2",
                showlegend=False,
                line=dict(color="green", width=2),
                opacity=0.5,
            )

        fig.update_layout(
            font=dict(
                size=30,
            ),
            xaxis_title="",
            yaxis_title="EMR",
            height=800,
            width=1000,
        )
        fig.show()
        if save_figures:
            savefilepath = os.path.join(
                save_directory, "plots", f"boxplot_emr_{col}.png"
            )
            logger.info(f"Saving boxplot at {savefilepath}")
            fig.write_image(file=savefilepath, format="png")
    return df_per_speaker


def compute_splitted_emr_boxplot_per_speaker(
    dataframe: pd.DataFrame,
    save_directory: str,
    save_figures: bool,
) -> None:
    dataframe = dataframe.reset_index()
    dataframe["emr"] = dataframe.groupby("user_id")["exactlyParsed"].transform("mean")
    df_per_speaker = dataframe[
        ["user_id", "emr", "gender", "age_group", "dialectal_region", "ethnicity"]
    ].drop_duplicates()
    fig = px.box(
        df_per_speaker,
        x="dialectal_region",
        y="emr",
        color="gender",
        color_discrete_sequence=["darkblue", "royalblue"],
    )
    fig.update_layout(
        font=dict(
            size=30,
        ),
        xaxis_title="",
        yaxis_title="EMR",
        boxmode="group",
        height=800,
        width=1000,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=sorted(df_per_speaker["dialectal_region"].unique()),
    )
    fig.show()
    if save_figures:
        savefilepath = os.path.join(
            save_directory,
            "plots",
            "splitted_boxplot_dialectal_region_gender.png",
        )
        logger.info(f"Saving boxplot at {savefilepath}")
        fig.write_image(file=savefilepath, format="png")

    fig = px.box(
        df_per_speaker,
        x="dialectal_region",
        y="emr",
        color="age_group",
        category_orders={"age_group": ["9-16", "17-28", "29-41", "42-54", "55-100"]},
        color_discrete_sequence=[
            px.colors.qualitative.Antique[0],
            px.colors.qualitative.Antique[1],
            px.colors.qualitative.Antique[2],
            px.colors.qualitative.Antique[3],
            px.colors.qualitative.Antique[4],
        ],
    )
    fig.update_layout(
        font=dict(
            size=30,
        ),
        xaxis_title="",
        yaxis_title="EMR",
        boxmode="group",
        height=800,
        width=1200,
        legend_title="age",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=[
            "Asian",
            "LatinX",
            "Inland-North",
            "Mid-Atlantic",
            "Midland",
            "New England",
            "Southern",
            "Western",
        ],
    )
    fig.show()
    if save_figures:
        savefilepath = os.path.join(
            save_directory,
            "plots",
            "splitted_boxplot_dialectal_region_age_group.png",
        )
        logger.info(f"Saving boxplot at {savefilepath}")
        fig.write_image(file=savefilepath, format="png")


def _get_probability_table(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Compute probability of having an exactly parsed utterance based on the given category"""
    df_tmp = df.copy()
    if col2 == "ethnicity":
        where_ethnicity = ~(df["ethnicity"].isna())
        df_tmp = df[where_ethnicity]
    print(pd.crosstab(df_tmp[col1], df_tmp[col2], normalize="columns").T)
    print()
    return pd.crosstab(df_tmp[col1], df_tmp[col2], margins=True, margins_name="Total").T


def compute_probability_tables(
    df: pd.DataFrame,
    main_variable: str,
) -> List[pd.DataFrame]:
    all_probability_tables = []
    for variable in VARIABLES_OF_INTEREST:
        print(
            f"{Colors.OKYELLOW}Probability table: {main_variable} vs. {variable}{Colors.ENDC}"
        )
        proba_table = _get_probability_table(df, main_variable, variable)
        all_probability_tables.append(proba_table)

    return all_probability_tables


def _perform_chi2_in_contingency_table(table: pd.DataFrame) -> Chi2ContingencyResult:
    """
    Chi-square test of independence of variables in a contingency table.
    An often quoted guideline for the validity of this calculation is that the test should be used only if the observed
    and expected frequencies in each cell are at least 5.
    """
    res = chi2_contingency(table.iloc[:-1, :-1], correction=False)
    return res


def _assert_if_application_condition_chi2_is_met(
    chi2_contingency_result: Chi2ContingencyResult,
) -> bool:
    return (chi2_contingency_result.expected_freq > 5).all()


def _is_chi2_statistically_significant(
    chi2_contingency_result: Chi2ContingencyResult,
) -> bool:
    return chi2_contingency_result.pvalue < 0.05


def perform_all_chi2_tests(
    list_of_contingency_tables: List[pd.DataFrame],
) -> Dict[str, Chi2ContingencyResult]:
    all_chi2_results = {}
    for table in list_of_contingency_tables:
        logger.info(f"Performing chi2 test for {table.index.name}")
        time.sleep(0.1)
        res = _perform_chi2_in_contingency_table(table)
        is_chi2_application_condition_met = (
            _assert_if_application_condition_chi2_is_met(res)
        )
        if not is_chi2_application_condition_met:
            print(
                f"{Colors.FAIL}FAIL: chi2 application condition is NOT met for the following contingency table:"
                f"{Colors.ENDC}"
            )
            print(f"{table.columns.name} vs {table.index.name}")
            print(table)
        is_chi2_statistically_significant = _is_chi2_statistically_significant(res)
        if not is_chi2_statistically_significant:
            print(
                f"{Colors.FAIL}Chi2 contingency test for {Colors.UNDERLINE}{table.index.name} is NOT statistically "
                f"significant at the 5% level - pvalue is {res.pvalue}{Colors.ENDC}"
            )
        else:
            print(
                f"{Colors.OKCYAN}Chi2 contingency test for {Colors.UNDERLINE}{table.index.name} is statistically "
                f"significant at the 5% level - pvalue is {res.pvalue}{Colors.ENDC}"
            )
        all_chi2_results[table.index.name] = res
    return all_chi2_results


def _get_heatmap_for_contingency_table(
    table: pd.DataFrame, savepath: str, save_figures: bool
):
    tx = table.loc[:, ["Total"]]
    ty = table.loc[["Total"], :]
    n = table.loc[["Total"], ["Total"]].values.item()
    indep = tx.dot(ty) / n
    c = table.fillna(0)
    measure = (c - indep) ** 2 / indep
    xi_n = measure.sum().sum()
    heatmap = measure / xi_n
    figure = sns.heatmap(heatmap.iloc[:-1, :-1], annot=c.iloc[:-1, :-1], cmap="crest")
    heatmap_fig = figure.get_figure()
    if save_figures:
        logger.info(f"Saving heatmap at {savepath}")
        heatmap_fig.savefig(savepath)
    heatmap_fig.clf()


def compute_heatmap_for_all_contingency_tables(
    list_of_contingency_tables: List[pd.DataFrame],
    save_directory: str,
    save_figures: bool,
) -> None:
    os.makedirs(os.path.join(save_directory, "plots"), exist_ok=True)
    for table in list_of_contingency_tables:
        savepath = os.path.join(
            save_directory,
            "plots",
            f"heatmap_{table.index.name}.png",
        )
        _get_heatmap_for_contingency_table(
            table, savepath=savepath, save_figures=save_figures
        )


def _check_if_log_reg_converged(model: BinaryResultsWrapper) -> bool:
    return model.converged


def _is_log_reg_statistically_significant(
    model: BinaryResultsWrapper,
) -> bool:
    return model.llr_pvalue < 0.05


def perform_all_univariate_log_reg_tests(
    dataframe: pd.DataFrame,
) -> Dict[str, BinaryResultsWrapper]:
    all_fitted_log_reg = {}
    for variable in VARIABLES_OF_INTEREST:
        log_reg = _perform_univariate_log_reg_test_for_given_variable(
            dataframe, variable
        )
        all_fitted_log_reg[variable] = log_reg

    # do the same but for the ethnicity subset
    where_not_nan = ~dataframe["ethnicity"].isna()
    dataframe = dataframe[where_not_nan]
    for variable in VARIABLES_OF_INTEREST:
        if variable != "ethnicity":
            log_reg = _perform_univariate_log_reg_test_for_given_variable(
                dataframe, variable, on_ethnicity_subset=True
            )
            all_fitted_log_reg[f"{variable}-ethnicity"] = log_reg

    return all_fitted_log_reg


def _perform_univariate_log_reg_test_for_given_variable(
    dataframe: pd.DataFrame, variable_name: str, on_ethnicity_subset: bool = False
) -> Optional[BinaryResultsWrapper]:
    dataset_tag = ""
    if on_ethnicity_subset:
        dataset_tag = "[Ethnicity subset]: "
    print(
        f"{Colors.OKYELLOW}{dataset_tag}Univariate test with {variable_name}:{Colors.ENDC}"
    )

    if variable_name == "ethnicity":
        where_ethnicity = ~(dataframe["ethnicity"].isna())
        dataframe = dataframe[where_ethnicity]

    try:
        log_reg = smf.logit(
            formula=f"exactlyParsed ~ C({variable_name})", data=dataframe
        ).fit(disp=0)
    except LinAlgError as e:
        print(
            f"{Colors.FAIL}{dataset_tag}Univariate log reg for {Colors.UNDERLINE}{variable_name} failed{Colors.ENDC}"
        )
        print("LinAlgError:", e)
        return None
    log_reg_converged = _check_if_log_reg_converged(log_reg)
    if not log_reg_converged:
        print(
            f"{Colors.FAIL}{dataset_tag}Univariate logistic regression with {Colors.UNDERLINE}{variable_name} did NOT "
            f"converged"
            f"{Colors.ENDC}"
        )
    is_log_reg_statistically_significant = _is_log_reg_statistically_significant(
        log_reg
    )
    if not is_log_reg_statistically_significant:
        print(
            f"{Colors.FAIL}{dataset_tag}Univariate log reg for {Colors.UNDERLINE}{variable_name} is NOT statistically "
            f"significant "
            f"at "
            f"the 5% level{Colors.ENDC}"
        )
    else:
        print(
            f"{Colors.OKCYAN}{dataset_tag}Univariate log reg for {Colors.UNDERLINE}{variable_name} is statistically "
            f"significant at "
            f"the 5% level{Colors.ENDC}"
        )
        print("Logistic regression pvalues:")
        print(log_reg.pvalues)
    print(log_reg.summary())
    print("------------------------------------------------")
    _print_odd_ratios(model=log_reg)
    print("------------------------------------------------")
    return log_reg


def _print_odd_ratios(model: BinaryResultsWrapper) -> None:
    odds_ratios = pd.DataFrame(
        {
            "OR": model.params,
            "Lower CI": model.conf_int(0.05)[0],
            "Upper CI": model.conf_int(0.05)[1],
        }
    )
    odds_ratios = np.exp(odds_ratios)
    print("Odd ratios:")
    print(odds_ratios)
    print("------------------------------------------------")


def perform_all_adjustment_tests(
    dataframe: pd.DataFrame,
    fitted_univariate_log_regs: Dict[str, BinaryResultsWrapper],
) -> Dict[Tuple[str], BinaryResultsWrapper]:
    all_fitted_adjustment_tests = {}
    for variable1, variable2 in list(itertools.combinations(VARIABLES_OF_INTEREST, 2)):
        log_reg = _perform_adjustment_test_for_given_variable(
            dataframe, variable1, variable2, fitted_univariate_log_regs
        )
        all_fitted_adjustment_tests[(variable1, variable2)] = log_reg
    return all_fitted_adjustment_tests


def _check_if_adjustment_test_converged(model: BinaryResultsWrapper) -> bool:
    return model.converged


def _perform_adjustment_test_for_given_variable(
    dataframe: pd.DataFrame,
    variable_name1: str,
    variable_name2: str,
    fitted_univariate_log_reg: Dict[str, BinaryResultsWrapper],
) -> Optional[BinaryResultsWrapper]:
    print(
        f"{Colors.OKYELLOW}Adjustment test with {variable_name1}, {variable_name2}:{Colors.ENDC}"
    )
    if variable_name2 == "ethnicity" or variable_name1 == "ethnicity":
        where_not_nan = ~dataframe["ethnicity"].isna()
        dataframe = dataframe[where_not_nan]
    try:
        log_reg = smf.logit(
            formula=f"exactlyParsed ~ C({variable_name1}) + C({variable_name2})",
            data=dataframe,
        ).fit(disp=0)
    except LinAlgError as e:
        print(
            f"{Colors.FAIL}Adjustment test with {Colors.UNDERLINE}({variable_name1}, {variable_name2}) failed"
            f"{Colors.ENDC}"
        )
        print("LinAlgError:", e)
        return None
    adjustment_test_converged = _check_if_adjustment_test_converged(log_reg)
    if not adjustment_test_converged:
        print(
            f"{Colors.FAIL}Adjustment test with {Colors.UNDERLINE}({variable_name1}, {variable_name2}) did NOT "
            f"converged{Colors.ENDC}"
        )

    # First, make the adjustment of variable 1 on variable 2 i.e. is variable 2 a confounding factor of variable 1?
    var = (
        f"{variable_name1}-ethnicity"
        if variable_name2 == "ethnicity"
        else variable_name1
    )
    univariate_log_reg_to_compare_to = fitted_univariate_log_reg[var]
    (
        is_variable2_confounding_factor,
        are_conclusions_changed,
    ) = _is_adjustment_test_statistically_significant_and_are_conclusions_changed(
        log_reg, univariate_log_reg_to_compare_to
    )
    if not is_variable2_confounding_factor:
        print(
            f"{Colors.OKCYAN}{Colors.UNDERLINE}{variable_name2} is NOT a confounding factor for {variable_name1} "
            f"at the 5% level{Colors.ENDC}"
        )
    else:
        if are_conclusions_changed:
            print(
                f"{Colors.FAIL}{Colors.UNDERLINE}Test is statistically significant. {variable_name2} is a confounding "
                f"factor for {variable_name1}. "
                f"Conclusions about {variable_name1} are changed at the 5% level{Colors.ENDC}"
            )
        else:
            print(
                f"{Colors.FAIL}{Colors.UNDERLINE}Test is statistically significant. "
                f"But conclusions about {variable_name1} are unchanged at the 5% level therefore "
                f"{variable_name2} is NOT a confounding factor for {variable_name1}.{Colors.ENDC}"
            )

    # Now, make the adjustment of variable 2 on variable 1 i.e. is variable 1 a confounding factor of variable 2?
    var = (
        f"{variable_name2}-ethnicity"
        if variable_name1 == "ethnicity"
        else variable_name2
    )
    univariate_log_reg_to_compare_to = fitted_univariate_log_reg[var]
    (
        is_variable1_confounding_factor,
        are_conclusions_changed,
    ) = _is_adjustment_test_statistically_significant_and_are_conclusions_changed(
        log_reg, univariate_log_reg_to_compare_to
    )
    if not is_variable1_confounding_factor:
        print(
            f"{Colors.OKCYAN}{Colors.UNDERLINE}{variable_name1} is NOT a confounding factor for {variable_name2} "
            f"at the 5% level{Colors.ENDC}"
        )
    else:
        if are_conclusions_changed:
            print(
                f"{Colors.FAIL}{Colors.UNDERLINE}Test is statistically significant. {variable_name1} is a confounding "
                f"factor for {variable_name2}. "
                f"Conclusions about {variable_name2} are changed at the 5% level{Colors.ENDC}"
            )
        else:
            print(
                f"{Colors.FAIL}{Colors.UNDERLINE}Test is statistically significant. "
                f"But conclusions about {variable_name2} are unchanged at the 5% level therefore "
                f"{variable_name1} is NOT a confounding factor for {variable_name2}.{Colors.ENDC}"
            )
    print()
    return log_reg


def _is_adjustment_test_statistically_significant_and_are_conclusions_changed(
    model: BinaryResultsWrapper, univariate_log_reg_to_compare_to: BinaryResultsWrapper
) -> Tuple[bool, bool]:
    # perform likelihood ratio test
    is_likelihood_ratio_test_significant = _likelihood_ratio_test_significance(
        multivariate_model=model, univariate_model=univariate_log_reg_to_compare_to
    )
    if not is_likelihood_ratio_test_significant:
        return False, False

    # there is allegedly a demographic bias regarding the variable in the univariate model
    # is the variable added in th emultivariate model a confounding factor?
    # are the pvalues similar or not?
    confounding_factor, conclusions_changed = _compare_pvalues_of_shared_variables(
        multivariate_model_pvalues=model.pvalues,
        univariate_model_pvalues=univariate_log_reg_to_compare_to.pvalues,
    )
    return confounding_factor, conclusions_changed


def _are_pvalues_statistically_significant(pvalues: pd.Series):
    for pvalue in pvalues:
        if pvalue > 0.05:
            return False
    return True


def _compare_pvalues_of_shared_variables(
    multivariate_model_pvalues: pd.Series, univariate_model_pvalues: pd.Series
) -> Tuple[bool, bool]:
    shared_variables = multivariate_model_pvalues.index.intersection(
        univariate_model_pvalues.index
    )
    shared_variables = shared_variables.drop("Intercept")
    # check if pvalues are close enough
    are_pvalues_close_enough = np.allclose(
        multivariate_model_pvalues[shared_variables],
        univariate_model_pvalues[shared_variables],
        atol=0.001,
    )
    if are_pvalues_close_enough:
        # if close enough, no confounding bias and the conclusions about the first variable are unchanged
        return True, False

    # else, there are several cases to check
    are_univariate_model_pvalues_significant = _are_pvalues_statistically_significant(
        univariate_model_pvalues[shared_variables]
    )
    are_multivariate_model_pvalues_significant = _are_pvalues_statistically_significant(
        multivariate_model_pvalues[shared_variables]
    )

    # either pvalues of univariate model are significant AND pvalues of multivariate model are significant,
    # in that case: we have a confounding variable but the conclusions about the first variable are unchanged
    if (
        are_univariate_model_pvalues_significant
        and are_multivariate_model_pvalues_significant
    ):
        return True, False

    # either pvalues of univariate model are significant BUT pvalues of multivariate model are not,
    # in that case: we have a confounding variable and the conclusions about the first variable changed
    if (
        are_univariate_model_pvalues_significant
        and not are_multivariate_model_pvalues_significant
    ):
        return True, True

    # either pvalues of univariate model are not significant BUT pvalues of multivariate model are significant,
    # in that case: we have a confounding variable and the conclusions about the first variable are changed
    if (
        not are_univariate_model_pvalues_significant
        and are_multivariate_model_pvalues_significant
    ):
        return True, True

    # either pvalues of univariate model are not significant AND pvalues of multivariate model not significant,
    # in that case: we do not have a confounding variable and the conclusions about the first variable are unchanged
    if (
        not are_univariate_model_pvalues_significant
        and not are_multivariate_model_pvalues_significant
    ):
        return True, False


def _likelihood_ratio_test_significance(
    multivariate_model: BinaryResultsWrapper, univariate_model: BinaryResultsWrapper
) -> bool:
    difference_in_nb_parameters = (
        multivariate_model.bse.size - univariate_model.bse.size
    )
    if difference_in_nb_parameters == 1:
        quantile = 3.84
    elif difference_in_nb_parameters == 3:
        quantile = 7.81
    elif difference_in_nb_parameters == 4:
        quantile = 9.49
    elif difference_in_nb_parameters == 5:
        quantile = 11.07
    elif difference_in_nb_parameters == 7:
        quantile = 14.07
    else:
        raise NotImplementedError(
            f"Difference in number of parameters: {difference_in_nb_parameters}"
        )
    test_statistic = 2 * (multivariate_model.llf - univariate_model.llf)
    return test_statistic > quantile


def _is_one_way_anova_statistically_significant(
    anova_test: F_onewayResult,
) -> bool:
    return anova_test.pvalue < 0.05


def _perform_anova_one_way_test_for_given_variable(
    dataframe: pd.DataFrame, variable: str, variable_to_test_against: str
) -> F_onewayResult:
    sub_groups = dataframe[variable].unique().tolist()
    all_populations_to_test = []
    for sub_group in sub_groups:
        sub_group_values = dataframe[dataframe[variable] == sub_group][
            variable_to_test_against
        ].values.tolist()
        all_populations_to_test.append(sub_group_values)
    anova_test_result = f_oneway(*all_populations_to_test)
    is_anova_way_significant = _is_one_way_anova_statistically_significant(
        anova_test_result
    )
    if not is_anova_way_significant:
        print(
            f"{Colors.FAIL}One-way ANOVA test with {Colors.UNDERLINE}{variable} on {variable_to_test_against} is NOT "
            f"statistically significant at the 5% level - pvalue is {anova_test_result.pvalue}{Colors.ENDC}"
        )
    else:
        print(
            f"{Colors.OKCYAN}One-way ANOVA test with {Colors.UNDERLINE}{variable} on {variable_to_test_against} is "
            f"statistically significant at the 5% level - pvalue is {anova_test_result.pvalue}{Colors.ENDC}"
        )
    return anova_test_result


def regroup_usa_dialectal_regions(dialectal_region: str) -> str:
    if dialectal_region in ["Asian", "LatinX"]:
        return dialectal_region
    return "USA"


def perform_all_anova_one_way_tests(
    dataframe: pd.DataFrame,
    variable_to_test_against: str,
    group_usa_dialectal_regions: bool,
) -> Dict[str, F_onewayResult]:
    if group_usa_dialectal_regions:
        dataframe["dialectal_region"] = dataframe["dialectal_region"].apply(
            lambda x: regroup_usa_dialectal_regions(x)
        )
    all_anova_tests = {}
    for variable in VARIABLES_OF_INTEREST:
        df_tmp = dataframe.copy()
        if variable == "ethnicity":
            where_not_nan = ~dataframe["ethnicity"].isna()
            df_tmp = dataframe[where_not_nan]
        anova_test_result = _perform_anova_one_way_test_for_given_variable(
            df_tmp, variable, variable_to_test_against
        )
        all_anova_tests[variable] = anova_test_result
    return all_anova_tests
