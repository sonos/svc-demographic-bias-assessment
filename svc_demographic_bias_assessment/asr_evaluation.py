from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torchmetrics.functional import word_error_rate

# Constants helpers for edits and their costs
COR = 0
SUB = 1
DEL = 2
INS = 3
EDIT_COST = [0, 4, 3, 3]
ERR_TYPES = [COR, SUB, DEL, INS]
ERR_NAME = ["cor.", "sub.", "del.", "ins."]


class DatasetException(Exception):
    pass


class UnkMetadataError(DatasetException):
    def __init__(self, idx, key):
        super(UnkMetadataError, self).__init__(
            "Item {} does not have metadata {}".format(idx, key)
        )


def compute_wer_on_each_utterance(prediction: str, reference: str) -> float:
    return word_error_rate(prediction, reference)


def edit_distance(
    reference_list: List[str], hypothesis_list: List[str], edit_cost=EDIT_COST
):
    """
    Compute the Levenshtein edit distance between two sequences.
    This outputs the minimal distance, and the edits of an arbitrarily chosen path of minimal distance,
    which is chosen among those paths of minimal distance which also minimize the number of edits.

    Returns a tuple (edits, distance, alignments):
      - edits is a list of edits (e.g. `[COR COR SUB COR INS COR DEL]`)
      - distance is the total distance
      - alignments is a list of tuple (ref_idx, hyp_idx, edit)
    """
    R = len(reference_list)
    H = len(hypothesis_list)
    # Initialize the distance matrix and back-pointers
    distance_matrix = np.zeros((R + 1, H + 1)).astype(np.uint8)
    edits_matrix = np.zeros((R + 1, H + 1)).astype(np.uint8)
    backpointers = [[None for _ in range(H + 1)] for _ in range(R + 1)]
    # Initialization of the Levenstein method
    for r in range(1, R + 1):
        distance_matrix[r, 0] = r * edit_cost[DEL]
        edits_matrix[r, 0] = r
        backpointers[r][0] = (r - 1, 0, DEL)
    for h in range(1, H + 1):
        distance_matrix[0, h] = h * edit_cost[INS]
        edits_matrix[0, h] = h
        backpointers[0][h] = (0, h - 1, INS)
    # Core alignment (fill the distance and pointer matrices)
    for r, rtoken in enumerate(reference_list):
        for h, htoken in enumerate(hypothesis_list):
            # Indices are 1-based in the matrices
            r_index = r + 1
            h_index = h + 1
            # "Correct" case
            if rtoken == htoken:
                distance_matrix[r_index, h_index] = distance_matrix[r, h]
                edits_matrix[r_index, h_index] = edits_matrix[r, h]
                backpointers[r_index][h_index] = (r, h, COR)
            else:
                # Select the minimum error between a substitution, insertion
                # and deletion, and among those having equal cost, a path which also minimizes the number of edits
                sub_cost = distance_matrix[r, h] + edit_cost[SUB]
                sub_edits = 1 + edits_matrix[r, h]
                del_cost = distance_matrix[r, h_index] + edit_cost[DEL]
                del_edits = 1 + edits_matrix[r, h_index]
                ins_cost = distance_matrix[r_index, h] + edit_cost[INS]
                ins_edits = 1 + edits_matrix[r_index, h]

                local_costs = [
                    ((sub_cost, sub_edits), (r, h, SUB)),
                    ((del_cost, del_edits), (r, h_index, DEL)),
                    ((ins_cost, ins_edits), (r_index, h, INS)),
                ]
                cost, edit = sorted(local_costs)[0]
                distance_matrix[r_index, h_index] = cost[0]
                edits_matrix[r_index, h_index] = cost[1]
                backpointers[r_index][h_index] = edit
    # The total distance is the one stored in the lower-right corner
    distance = int(distance_matrix[R, H])
    # Backtrace
    edits = []
    current_edit = backpointers[R][H]
    pointers = [(R, H)]
    while current_edit is not None:
        r, h, edit = current_edit
        edits.insert(0, edit)
        pointers.insert(0, (r, h))
        current_edit = backpointers[r][h]
    return edits, distance, pointers


def ER(reference_list: List[str], hypothesis_list: List[str]):
    """
    Compute the error rates (incl. subs, dels and ins) between two lists.
    Returns a pair (error rate, dict of alignment data)
    """
    edits, distance, pointers = edit_distance(reference_list, hypothesis_list)
    edit_counts = Counter(edits)
    n_correct = edit_counts[COR]
    n_substitution = edit_counts[SUB]
    n_insertion = edit_counts[INS]
    n_deletion = edit_counts[DEL]
    n_errors = n_substitution + n_deletion + n_insertion
    n_tokens = n_substitution + n_deletion + n_correct
    edit_counts = dict(edit_counts)
    error_rate = 100.0 * n_errors / max(1, n_tokens)
    return error_rate, locals()


def WER(reference_string: str, hypothesis_string: str):
    """
    Compute the WORD error rates (incl. subs, dels and ins) between two lists.
    Returns a pair (error rate, dict of alignment data)
    """
    reference_list = reference_string.strip().split()
    hypothesis_list = hypothesis_string.strip().split()
    return ER(reference_list, hypothesis_list)


def CER(reference_string: str, hypothesis_string: str, with_spaces: bool = False):
    """
    Compute the CHARACTER error rates (incl. subs, dels and ins) between two lists.
    Returns a pair (error rate, dict of alignment data)
    """
    if with_spaces:
        reference_list = list(reference_string)
        hypothesis_list = list(hypothesis_string)
    else:
        reference_list = list(reference_string.replace(" ", "").strip())
        hypothesis_list = list(hypothesis_string.replace(" ", "").strip())
    return ER(reference_list, hypothesis_list)


def format_alignment(data):
    """
    Formats the alignment data for printing
    """
    reference_list = data["reference_list"]
    hypothesis_list = data["hypothesis_list"]
    pointers = data["pointers"]
    edits = data["edits"]
    refline = "REF:"
    hypline = "HYP:"
    editline = "    "
    for edit, p in zip(edits, pointers):
        if edit in [COR, SUB]:
            rword = reference_list[p[0]]
            hword = hypothesis_list[p[1]]
            max_length = max(map(len, [rword, hword]))
            refline += " " + rword + " " * (max_length - len(rword))
            hypline += " " + hword + " " * (max_length - len(hword))
            editline += " "
            editline += "S" if edit == SUB else " "
            editline += " " * (max_length - 1)
        elif edit == INS:
            hword = hypothesis_list[p[1]]
            refline += " " + "*" * (len(hword))
            hypline += " " + hword
            editline += " " + "I" + " " * (len(hword) - 1)
        else:
            rword = reference_list[p[0]]
            refline += " " + rword
            hypline += " " + "*" * (len(rword))
            editline += " " + "D" + " " * (len(rword) - 1)
    return refline + "\n" + hypline + "\n" + editline


class WERAccumulator(object):
    er_name = "WER"
    """
    This object does the accumulation of alignment data to compute a global
    word error rate on a dataset for you

    The basic usage is as follows

        wer_accumulator = WERAccumulator()
        for idx in dataset:
            ref = dataset.get_transcript(idx)
            hyp = decode_wave(dataset.get_audio_file(idx))
            wer_accumulator.compute(idx, ref, hyp)
        wer, _ = wer_accumulator.get()
        print str(wer_accumulator)
    """

    def __init__(self):
        """
        Constructor
        """
        self.edits = Counter()
        self.data = {}
        self.bootstrap = {}

    def compute(self, idx, reference_string, hypothesis_string):
        """
        Computes the word edit distance between `reference_string` and
        `hypothesis_string` and accumulate the alignment data and edits
        Returns the word error rate and alignment data for these.
        """
        wer, data = WER(reference_string, hypothesis_string)
        self.add(idx, data)
        return wer, data

    def add(self, idx, data):
        self.data[idx] = data
        self.edits += Counter(data["edits"])

    def __add__(self, other):
        """
        Add two accumulators
        """
        new = self.__class__()
        new.edits = self.edits + other.edits
        new.data = {}
        new.data.update(self.data)
        new.data.update(other.data)
        return new

    def get(self):
        """
        Get the current aggregated WER and list of rates of
        (cor., subs., dels., ins.)
        """
        n_correct = self.edits[COR]
        n_substitution = self.edits[SUB]
        n_insertion = self.edits[INS]
        n_deletion = self.edits[DEL]
        n_errors = n_substitution + n_deletion + n_insertion
        n_tokens = n_substitution + n_deletion + n_correct
        wer = 100.0 * n_errors / max(1, n_tokens)
        detail = [100.0 * self.edits[etype] / max(1, n_tokens) for etype in ERR_TYPES]
        return wer, detail

    def get_idx(self):
        pass

    def __str__(self):
        """
        Formats the current WER as a string in the form:

            WER%: 18.1 [9.9 cor., 5.3 sub. 1.8 del. 1.1 ins.]
        """
        wer, detail = self.get()
        ans = "{}%: {:.1f} [{:.1f} cor., {:.1f} sub., {:.1f} del., {:.1f} ins.]"
        ans = ans.format(
            self.er_name, wer, detail[COR], detail[SUB], detail[DEL], detail[INS]
        )
        return ans

    def __getitem__(self, key):
        """
        Get the raw alignment data for index `key`
        """
        return self.data.get(key)

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data.keys())

    def __contains__(self, idx):
        return idx in self.data


def run_wer_accumulator_on_metadata(
    metadata_dict: Dict,
    wer_accumulator: WERAccumulator,
    use_list_of_possible_verbalizations: bool,
) -> Tuple[Dict, WERAccumulator]:
    for audio_id, metadata in metadata_dict.items():
        asr_prediction = metadata["asr_prediction"]
        if use_list_of_possible_verbalizations:
            possible_verbalizations = metadata.get("gramophone_verbalizations")
            if possible_verbalizations is None:
                raise ValueError("`gramophone_verbalizations` field is not available")
            # compute wer for each possible verbalizations and select the smallest possible wer
            possible_wers = [
                WER(reference_string=verbalization, hypothesis_string=asr_prediction)[0]
                for verbalization in possible_verbalizations
            ]
            argmin_wer = possible_wers.index(min(possible_wers))
            gt_transcript = possible_verbalizations[argmin_wer]
            metadata["closest_verbalization"] = gt_transcript
        else:
            gt_transcript = metadata["transcript"]["normalized_text"]
        wer, _ = wer_accumulator.compute(
            idx=audio_id,
            reference_string=gt_transcript,
            hypothesis_string=asr_prediction,
        )
        metadata["wer"] = wer
    return metadata_dict, wer_accumulator
