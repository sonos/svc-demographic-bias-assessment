from collections import Counter, defaultdict
from typing import Dict, Set, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def separate_playmusic_and_transportcontrol_utterances(
    train_metadata: Dict,
) -> Tuple[Dict, Dict, Set]:
    remapping_train_dict_pm = {}
    remapping_train_dict_tc = {}
    transport_control_intents = set()
    for audio_id, audio_dict in train_metadata.items():
        intent = audio_dict["transcript"]["label"]["intent"]
        utterance = audio_dict["transcript"]["text"]
        slots = audio_dict["transcript"]["label"]["slots"]
        current_dict = {"intent": intent, "slots": []}
        if slots:
            for slot in slots:
                slot_dict = {
                    "slot_name": slot["name"],
                    "entity_name": slot["entity"]["name"],
                    "slot_value": slot["value"]["slot_value"],
                }
                current_dict["slots"].append(slot_dict)
        if intent == "PlayMusic":
            remapping_train_dict_pm[utterance] = current_dict
        else:
            transport_control_intents.add(intent)
            remapping_train_dict_tc[utterance] = current_dict
    return remapping_train_dict_pm, remapping_train_dict_tc, transport_control_intents


def mapping_nb_of_slots_and_utterances(
    train_dict: Dict[str, Dict],
) -> Dict[int, int]:
    """
    Returns a dict giving the number of utterances that contain 1, 2, 3, etc slots
    """
    mapping = {k: len(v["slots"]) for k, v in train_dict.items()}
    count = dict(Counter(mapping.values()))
    return dict(sorted(count.items()))


def get_dataframe_nb_values_per_slot(train_dict: Dict[str, Dict]) -> pd.DataFrame:
    slots_count = defaultdict(int)
    for utterance, slots_list in train_dict.items():
        for slot_dict in slots_list["slots"]:
            slot_name = slot_dict["slot_name"]
            entity_name = slot_dict["entity_name"]
            slots_count[(slot_name, entity_name)] += 1
    df = pd.DataFrame.from_dict(
        slots_count, orient="index", columns=["count"]
    ).sort_values(by="count", ascending=False)
    return df


def get_histogram_of_nb_utterances_per_intent(sorted_list) -> go.Figure:
    intents = [sorted_list[i][0] for i in range(len(sorted_list))]
    frequencies = [sorted_list[i][1] for i in range(len(sorted_list))]
    fig = go.Figure()
    fig.add_trace(go.Histogram(histfunc="sum", y=frequencies, x=intents, name="stv1"))
    fig.update_layout(
        title_text="Number of unique utterances per intent in train split for TransportControl domain",
        xaxis_title_text="intent",
        yaxis_title_text="Count",
        bargap=0.2,
        bargroupgap=0.1,
        height=1000,
        width=1600,
    )
    return fig


def get_k_first_slot_tuples_in_playmusic_data(
    train_dict: Dict[str, Dict], k: int
) -> pd.DataFrame:
    all_seen_slot_tuples = []
    for utterance, slots_list in train_dict.items():
        slot_tuple_for_current_utterance = []
        for slot_dict in slots_list["slots"]:
            slot_name = slot_dict["slot_name"]
            slot_tuple_for_current_utterance.append(slot_name)
        all_seen_slot_tuples.append(sorted(slot_tuple_for_current_utterance))
    all_seen_slot_tuples = map(tuple, all_seen_slot_tuples)
    df = Counter(all_seen_slot_tuples)
    df = pd.DataFrame.from_dict(df, orient="index").reset_index()
    df = df.rename(columns={"index": "slot_tuple", 0: "count"})
    df = df.sort_values(by="count", ascending=False)
    df_top_k = df.iloc[:k].reset_index(drop=True)
    df_top_k["slot_tuple"] = df_top_k["slot_tuple"].astype(str)
    return df_top_k


def get_histogram_slot_tuples_playmusic(df_top_k: pd.DataFrame) -> go.Figure:
    figure_data = []
    figure_data.extend(
        [i for i in px.histogram(df_top_k, x="slot_tuple", y="count").to_dict()["data"]]
    )
    fig = go.Figure(figure_data)
    fig.update_layout(
        title_text="Count of slot_tuples in PlayMusic unique train split utterances",
        xaxis_title_text="slot_tuple",
        yaxis_title_text="Count",
        bargap=0.2,
        bargroupgap=0.1,
        height=1000,
        width=1600,
    )
    return fig


def get_number_of_unique_utterances_per_intent(
    train_dict: Dict[str, Dict],
) -> Dict[str, int]:
    tc_dict_nb_utt_per_intents = defaultdict(int)
    for utterance, utterance_dict in train_dict.items():
        intent = utterance_dict["intent"]
        tc_dict_nb_utt_per_intents[intent] += 1
    return tc_dict_nb_utt_per_intents
