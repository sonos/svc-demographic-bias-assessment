import logging
from concurrent.futures import as_completed, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Dict, List, Tuple

from tqdm import tqdm

SluParsingT = Dict
PredictedSlotT = Dict

logger = logging.getLogger(__name__)


def merge_metadata_and_bert_predictions(
    metadata_dict: Dict, predictions_dict: Dict
) -> Dict:
    for audio_id, metadata in metadata_dict.items():
        assert audio_id in predictions_dict.keys()
        metadata["nlu_prediction"] = predictions_dict[audio_id]
        metadata["asr_prediction"] = predictions_dict[audio_id]["intent"][
            "decodedString"
        ]
    return metadata_dict


def _compare_transcript_intent_and_predicted_intent(
    label: Dict, slu_parsing: SluParsingT
) -> bool:
    transcript_intent = label["intent"]
    predicted_intent = slu_parsing["intent"]["intentName"]
    return transcript_intent == predicted_intent


def _all_alternatives(slot: Dict) -> List[Dict]:
    main_alternative = deepcopy(slot)
    main_alternative["alternatives"] = []
    alternatives = []
    for alt in slot["alternatives"]:
        alternative = deepcopy(slot)
        alternative.update(
            {
                "entity": alt["entity"],
                "slot_value": alt["slot_value"]
                if "slot_value" in alt.keys()
                else alt["value"],
                "slotName": alt["slotName"],
                "alternatives": [],
            }
        )
        alternatives.append(alternative)
    return [main_alternative, *alternatives]


def _compare_transcript_slots_and_predicted_slots(
    label: Dict,
    slu_parsing: SluParsingT,
) -> bool:
    label_slots = label["slots"] if label is not None else None
    predicted_slots = slu_parsing["slots"]

    if label_slots is None:
        # there are no slots in the ground truth transcript
        return not predicted_slots

    matched_predictions = set()  # indices of predicted slots which were matched
    matched_expectations = set()  # indices of expected slots which were matched

    for p_idx, p_slot in enumerate(predicted_slots):
        # iterate over alternatives and do same logic as for unique slot
        all_alternatives = _all_alternatives(p_slot)
        for alt_slot in all_alternatives:
            for e_idx, t_slot in enumerate(label_slots):
                if e_idx not in matched_expectations and _slot_matching(
                    t_slot,
                    alt_slot,
                ):
                    matched_predictions.add(p_idx)
                    matched_expectations.add(e_idx)
                    break

    unmatched_pred = set(range(len(predicted_slots))).difference(matched_predictions)
    unmatched_exp = set(range(len(label_slots))).difference(matched_expectations)

    return not (unmatched_pred or unmatched_exp)


def _slot_matching(
    transcript_slot: Dict,
    predicted_slot: PredictedSlotT,
) -> bool:
    if transcript_slot["name"] != predicted_slot["slotName"]:
        return False
    normalized_transcript_text = transcript_slot["value"]["normalized_slot_value"]
    predicted_slot_value = predicted_slot["value"]

    return normalized_transcript_text == predicted_slot_value


def _compute_if_exactly_parsed(
    label: Dict,
    slu_parsing: SluParsingT,
) -> bool:
    # look first at intent prediction
    is_predicted_intent_correct = _compare_transcript_intent_and_predicted_intent(
        label, slu_parsing
    )
    if not is_predicted_intent_correct:
        # predicted intent is not correct
        return False

    # predicted intent is correct, check slots
    are_predicted_slots_correct = _compare_transcript_slots_and_predicted_slots(
        label,
        slu_parsing,
    )
    if are_predicted_slots_correct:
        return True

    return False


def _update_inference_metrics(metrics_dict: Dict, metadata: Dict) -> Dict:
    intent = metadata["transcript"]["label"]["intent"]
    if intent not in metrics_dict:
        metrics_dict[intent] = {"n_exact": 0, "n_samples": 0, "confusion": []}
    metrics_dict[intent]["n_exact"] += int(metadata["nlu_prediction"]["exactlyParsed"])
    metrics_dict[intent]["n_samples"] += 1
    if not metadata["nlu_prediction"]["exactlyParsed"]:
        pred_intent = metadata["nlu_prediction"]["intent"]["intentName"]
        if intent != pred_intent:
            metrics_dict[intent]["confusion"].append(pred_intent)
    return metrics_dict


def _compute_inference_metrics_on_all_metadata(
    metadata_dict: Dict,
) -> Dict:
    pbar = tqdm(total=len(metadata_dict))
    processes = []
    updated_metadata_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for audio_id, metadata in metadata_dict.items():
            processes.append(
                executor.submit(
                    _compute_nlu_inference_metric_for_given_audio,
                    audio_id,
                    metadata,
                    updated_metadata_dict,
                )
            )
        for _ in as_completed(processes):
            pbar.update(n=1)
    wait(processes)
    return updated_metadata_dict


def _compute_nlu_inference_metric_for_given_audio(
    audio_id: str,
    metadata: Dict,
    updated_metadata_dict: Dict,
) -> None:
    # compute if exactly parsed
    labels = metadata["transcript"]["label"]
    predictions = metadata["nlu_prediction"]
    metadata["nlu_prediction"]["exactlyParsed"] = _compute_if_exactly_parsed(
        labels,
        predictions,
    )
    updated_metadata_dict[audio_id] = metadata


def compute_inference_metrics(
    metadata_dict: Dict,
) -> Tuple[Dict, Dict]:
    logger.info("Determine if exactly parsed or not")
    updated_metadata = _compute_inference_metrics_on_all_metadata(
        metadata_dict=metadata_dict,
    )

    logger.info("Compute EPR")
    metrics_dict = {}
    for audio_id, metadata in tqdm(updated_metadata.items()):
        # update metrics_dict
        metrics_dict = _update_inference_metrics(metrics_dict, metadata)

    # get average EPR
    avg_epr = 0
    total_n_exact = 0
    total_nsamples = 0
    for intent in metrics_dict:
        epr = metrics_dict[intent]["n_exact"] / metrics_dict[intent]["n_samples"]
        epr = round(epr, 2)
        metrics_dict[intent]["epr"] = epr
        avg_epr += epr
        total_n_exact += metrics_dict[intent]["n_exact"]
        total_nsamples += metrics_dict[intent]["n_samples"]
        logger.info("{}: {}".format(intent, epr))
    logger.info("Average macro EPR is {}".format(avg_epr / len(metrics_dict)))
    logger.info("Average micro EPR is {}".format(total_n_exact / total_nsamples))
    metrics_dict["average_epr"] = round(avg_epr / len(metrics_dict), 2)
    metrics_dict["average_micro_epr"] = round(total_n_exact / total_nsamples, 2)
    return metrics_dict, updated_metadata
