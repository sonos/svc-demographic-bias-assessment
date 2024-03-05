from copy import deepcopy
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = px.colors.qualitative.Antique


def get_distribution_count_statistics(
    df: pd.DataFrame,
    cols_of_interest: List[str],
    set_: str,
    distribution_type: str,
) -> go.Figure:
    if distribution_type == "speakers":
        copy_cols_of_interest = deepcopy(cols_of_interest)
        copy_cols_of_interest.append("user_id")
        df_tmp = df[copy_cols_of_interest].drop_duplicates()
        count_variable = "user_id"
    elif distribution_type == "samples":
        df_tmp = df[cols_of_interest]
        df_tmp = df_tmp.reset_index()
        count_variable = "audio_id"
    else:
        raise ValueError(
            "Incorrect distribution_type: can either be `speakers` or `samples`"
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}],
        ],
    )
    i = 1
    j = 1
    for k, col_name in enumerate(cols_of_interest):
        values = df_tmp.groupby(col_name)[count_variable].count().values
        labels = df_tmp.groupby(col_name)[count_variable].count().index.tolist()
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=COLORS,
                textinfo="label+value",
                insidetextorientation="radial",
                textposition="outside",
                legendgroup=str(k),
                showlegend=False,
            ),
            row=i,
            col=j,
        )
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=COLORS,
                textinfo="percent",
                insidetextorientation="radial",
                textposition="inside",
                legendgroup=str(k),
                showlegend=False,
            ),
            row=i,
            col=j,
        )
        j += 1
        if j > 2 and i == 1:
            i = 2
            j = 1
        if j > 2 and i == 2:
            i = 2
            j = 1
    fig.update_layout(
        title=f"Distribution in terms of number of {distribution_type} on {set_} set",
        width=2000,
        height=1000,
        legend_tracegroupgap=40,
        font=dict(
            size=28,
        ),
    )
    return fig


def age_sorting(value):
    first_age = int(value.split("-")[0])
    return first_age


def plot_histogram_groupby(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    variable3: str,
    x_axis_name: str,
    y_axis_name: str,
    title: str,
    width: int = 600,
    height: int = 400,
) -> go.Figure():
    fig = go.Figure()
    if variable1 == "age_group":
        key = age_sorting
    else:
        key = None
    for possible_value in sorted(df[variable1].unique().tolist(), key=key):
        df_tmp = df[df[variable1] == possible_value]
        fig.add_trace(
            go.Histogram(
                x=df_tmp.groupby(variable3)[variable2].count(),
                name=f"{possible_value}",
            ),
        )
    fig.update_layout(
        barmode="group",
        title_text=title,
        xaxis_title_text=x_axis_name,
        yaxis_title_text=y_axis_name,
        width=width,
        height=height,
    )
    return fig
