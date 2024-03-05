import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from pydub import AudioSegment
from pydub.playback import play

SPLITS = ["train", "dev", "test"]


@dataclass
class SVCDataset:
    dataset_directory: str
    audio_directory: Optional[str] = None
    metadata_directory: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        assert self.dataset_directory is not None
        if self.audio_directory is None:
            self.audio_directory = os.path.join(self.dataset_directory, "audios")
        if self.metadata_directory is None:
            self.metadata_directory = os.path.join(self.dataset_directory, "metadata")
        self.metadata = {}
        for filename in os.listdir(self.metadata_directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.metadata_directory, filename)
                split = filename.split("_")[0]
                with open(filepath, "r", encoding="utf-8") as file:
                    current_metadata = json.load(file)
                self.metadata[split] = current_metadata
        assert len(self.metadata.keys()) == 3


class DatasetManipulator:
    def __init__(self, dataset: SVCDataset):
        self.dataset = dataset

    def play_audio_file_by_audio_id(self, audio_id: str) -> None:
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            if audio_id in dataset_split.keys():
                wav_filepath = os.path.join(
                    self.dataset.audio_directory, split, f"{audio_id}.wav"
                )
                assert os.path.exists(wav_filepath), ValueError(
                    f"wav filepath for {audio_id} could not be found."
                )
                audio = AudioSegment.from_wav(wav_filepath)
                play(audio)

    @staticmethod
    def _helper_get_audio_ids_of_speaker(dataset: Dict, speaker_id: str) -> Set[str]:
        audio_ids_of_speakers = set()
        for audio_id, metadata in dataset.items():
            current_speaker_id = metadata["speaker"]["user_id"]
            if current_speaker_id == speaker_id:
                audio_ids_of_speakers.add(audio_id)
        return audio_ids_of_speakers

    def play_audio_files_speaker_id(self, speaker_id: str, k: int) -> None:
        # play k files from given speaker_id
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            all_speakers_in_split = self._helper_get_unique_speakers(split)
            if speaker_id in all_speakers_in_split:
                # find all audio ids associated to this speaker
                audio_ids_of_speakers = self._helper_get_audio_ids_of_speaker(
                    dataset=dataset_split, speaker_id=speaker_id
                )
                if k <= len(audio_ids_of_speakers):
                    chosen_audio_ids = random.sample(audio_ids_of_speakers, k)
                else:
                    chosen_audio_ids = audio_ids_of_speakers
                for audio_id in chosen_audio_ids:
                    print(f"Playing audio {audio_id} of speaker {speaker_id}")
                    wav_filepath = os.path.join(
                        self.dataset.audio_directory, split, f"{audio_id}.wav"
                    )
                    assert os.path.exists(wav_filepath), ValueError(
                        f"wav filepath for {audio_id} could not be found."
                    )
                    audio = AudioSegment.from_wav(wav_filepath)
                    play(audio)

    def get_list_of_intents_in_dataset(self) -> Set[str]:
        list_intents = set()
        for split in SPLITS:
            for audio_id, metadata in self.dataset.metadata[split].items():
                intent = metadata["transcript"]["label"]["intent"]
                list_intents.add(intent)
        return list_intents

    def get_all_transcripts_associated_to_given_intent(self, intent: str) -> Set[str]:
        transcripts = set()
        for split in SPLITS:
            for audio_id, metadata in self.dataset.metadata[split].items():
                current_intent = metadata["transcript"]["label"]["intent"]
                if current_intent == intent:
                    transcript = metadata["transcript"]["text"]
                    transcripts.add(transcript)
        return transcripts

    def get_all_audio_ids_associated_to_given_intent(self, intent: str) -> Set[str]:
        audio_ids = set()
        for split in SPLITS:
            for audio_id, metadata in self.dataset.metadata[split].items():
                current_intent = metadata["transcript"]["label"]["intent"]
                if current_intent == intent:
                    audio_ids.add(audio_id)
        return audio_ids

    def _helper_get_unique_speakers(self, split: str) -> Set[str]:
        dataset_split = self.dataset.metadata[split]
        unique_speakers = set()
        for audio_id, metadata in dataset_split.items():
            speaker_id = metadata["speaker"]["user_id"]
            unique_speakers.add(speaker_id)
        return unique_speakers

    def get_number_of_unique_speakers(self, split: Optional[str] = None) -> int:
        # if split is given, restrict to that given dataset split
        # else, give for all dataset
        if split:
            return len(self._helper_get_unique_speakers(split))
        else:
            total_nb = 0
            for split in SPLITS:
                total_nb += len(self._helper_get_unique_speakers(split))
            return total_nb

    def get_number_of_audios(self, split: Optional[str] = None) -> int:
        if split:
            return len(self.dataset.metadata[split].keys())
        else:
            total_nb_audios = 0
            for split in SPLITS:
                total_nb_audios += len(self.dataset.metadata[split].keys())
            return total_nb_audios

    def _helper_get_number_of_audios_per_intent_type(
        self, split: str
    ) -> Tuple[int, int]:
        dataset_split = self.dataset.metadata[split]
        nb_audios_pm = 0
        nb_audios_tc = 0
        for audio_id, metadata in dataset_split.items():
            intent = metadata["transcript"]["label"]["intent"]
            if intent == "PlayMusic":
                nb_audios_pm += 1
            else:
                nb_audios_tc += 1
        return nb_audios_pm, nb_audios_tc

    def get_number_of_audios_per_intent_type(
        self, split: Optional[str] = None
    ) -> Tuple[int, int]:
        if split:
            return self._helper_get_number_of_audios_per_intent_type(split)
        else:
            total_nb_audios_pm = 0
            total_nb_audios_tc = 0
            for split in SPLITS:
                (
                    current_pm,
                    current_tc,
                ) = self._helper_get_number_of_audios_per_intent_type(split)
                total_nb_audios_pm += current_pm
                total_nb_audios_tc += current_tc
            return total_nb_audios_pm, total_nb_audios_tc

    def _helper_get_number_of_unique_transcripts(self, split: str) -> Set[str]:
        dataset_split = self.dataset.metadata[split]
        unique_transcripts = set()
        for audio_id, metadata in dataset_split.items():
            transcript = metadata["transcript"]["text"]
            unique_transcripts.add(transcript)
        return unique_transcripts

    def get_number_of_unique_transcripts(self, split: Optional[str] = None) -> int:
        if split:
            return len(self._helper_get_number_of_unique_transcripts(split))
        else:
            set_unique_transcripts = set()
            for split in SPLITS:
                set_unique_transcripts |= self._helper_get_number_of_unique_transcripts(
                    split
                )
            return len(set_unique_transcripts)

    def _helper_get_number_of_unique_transcripts_per_intent_type(
        self, split: str
    ) -> Tuple[Set[str], Set[str]]:
        dataset_split = self.dataset.metadata[split]
        unique_transcripts_pm = set()
        unique_transcripts_tc = set()
        for audio_id, metadata in dataset_split.items():
            transcript = metadata["transcript"]["text"]
            intent = metadata["transcript"]["label"]["intent"]
            if intent == "PlayMusic":
                unique_transcripts_pm.add(transcript)
            else:
                unique_transcripts_tc.add(transcript)
        return unique_transcripts_pm, unique_transcripts_tc

    def get_number_of_unique_transcripts_per_intent_type(
        self, split: Optional[str] = None
    ) -> Tuple[int, int]:
        if split:
            (
                set_unique_transcripts_pm,
                set_unique_transcripts_tc,
            ) = self._helper_get_number_of_unique_transcripts_per_intent_type(split)
            return len(set_unique_transcripts_pm), len(set_unique_transcripts_tc)
        else:
            set_unique_transcripts_pm = set()
            set_unique_transcripts_tc = set()
            for split in SPLITS:
                (
                    current_pm,
                    current_tc,
                ) = self._helper_get_number_of_unique_transcripts_per_intent_type(split)
                set_unique_transcripts_pm |= current_pm
                set_unique_transcripts_tc |= current_tc
            return len(set_unique_transcripts_pm), len(set_unique_transcripts_tc)

    def _helper_get_number_of_hours(self, split: str) -> float:
        dataset_split = self.dataset.metadata[split]
        length_ms = 0
        for _, metadata in dataset_split.items():
            length_ms += metadata["audioLengthMs"]
        return length_ms / 3.6e6

    def get_number_of_hours(self, split: Optional[str] = None) -> float:
        if split:
            return self._helper_get_number_of_hours(split)
        else:
            total_nb = 0
            for split in SPLITS:
                total_nb += self._helper_get_number_of_hours(split)
            return total_nb

    def _helper_get_number_of_hours_per_intent_type(
        self, split: str
    ) -> Tuple[float, float]:
        dataset_split = self.dataset.metadata[split]
        length_ms_pm = 0
        length_ms_tc = 0
        for _, metadata in dataset_split.items():
            intent = metadata["transcript"]["label"]["intent"]
            if intent == "PlayMusic":
                length_ms_pm += metadata["audioLengthMs"]
            else:
                length_ms_tc += metadata["audioLengthMs"]
        return length_ms_pm / 3.6e6, length_ms_tc / 3.6e6

    def get_number_of_hours_per_intent_type(
        self, split: Optional[str] = None
    ) -> Tuple[float, float]:
        if split:
            return self._helper_get_number_of_hours_per_intent_type(split)
        else:
            total_nb_pm = 0
            total_nb_tc = 0
            for split in SPLITS:
                (
                    current_pm,
                    current_tc,
                ) = self._helper_get_number_of_hours_per_intent_type(split)
                total_nb_pm += current_pm
                total_nb_tc += current_tc
            return total_nb_pm, total_nb_tc

    def get_music_entities_coverage(self) -> Dict[str, int]:
        """
        Number of distinct artists, songs and albums in the release dataset.
        """
        artists = set()
        songs = set()
        albums = set()
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            for audio_id, metadata in dataset_split.items():
                intent = metadata["transcript"]["label"]["intent"]
                if intent == "PlayMusic":
                    list_slots = metadata["transcript"]["label"]["slots"]
                    if list_slots:
                        for slot_dict in list_slots:
                            slot_name = slot_dict["name"]
                            slot_value = (
                                slot_dict["value"]["slot_value"]
                                .lower()
                                .lstrip()
                                .rstrip()
                            )
                            if slot_name == "album_name":
                                albums.add(slot_value)
                            elif slot_name == "song_name":
                                songs.add(slot_value)
                            elif slot_name == "artist_name":
                                artists.add(slot_value)
        return {
            "number_artists": len(artists),
            "number_songs": len(songs),
            "number_albums": len(albums),
        }

    def get_slot_name_coverage(self) -> Dict[str, int]:
        dict_unique_values_per_slot_names = defaultdict(set)
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            for audio_id, metadata in dataset_split.items():
                list_slots = metadata["transcript"]["label"]["slots"]
                if list_slots:
                    for slot_dict in list_slots:
                        slot_name = slot_dict["name"]
                        slot_value = (
                            slot_dict["value"]["slot_value"].lower().lstrip().rstrip()
                        )
                        dict_unique_values_per_slot_names[slot_name].add(slot_value)
        dict_number_unique_values_per_slot_names = {
            k: len(v) for k, v in dict_unique_values_per_slot_names.items()
        }
        return dict(sorted(dict_number_unique_values_per_slot_names.items()))

    def display_transcript_by_audio_id(self, audio_id: str) -> Dict:
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            if audio_id in dataset_split.keys():
                return dataset_split[audio_id]["transcript"]

    def _helper_get_audio_ids(self) -> Set[str]:
        audio_ids = set()
        for split in SPLITS:
            for audio_id, metadata in self.dataset.metadata[split].items():
                audio_ids.add(audio_id)
        return audio_ids

    def get_k_random_examples(self, k: int):
        list_of_audio_ids = self._helper_get_audio_ids()
        # take k random audio ids
        chosen_audio_ids = random.sample(list_of_audio_ids, k)
        to_display = {}
        for split in SPLITS:
            dataset_split = self.dataset.metadata[split]
            intersection = set(dataset_split.keys()).intersection(chosen_audio_ids)
            if len(intersection) != 0:
                for audio_id in intersection:
                    to_display[audio_id] = dataset_split[audio_id]
        return to_display
