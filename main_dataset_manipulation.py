import logging
import os.path
import time
from dataclasses import dataclass, field
from pprint import pprint

from transformers import HfArgumentParser

from svc_demographic_bias_assessment import DatasetManipulator, SVCDataset

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments needed to run this script.
    """

    dataset_directory: str = field(
        metadata={"help": "Directory pointing towards the SVC Dataset."},
    )

    def __post_init__(self):
        assert os.path.exists(self.dataset_directory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataclasses_list = [
        ScriptArguments,
    ]
    parser = HfArgumentParser(dataclasses_list)
    args = parser.parse_args_into_dataclasses()
    datapath_args = args[0]

    svc_dataset = SVCDataset(dataset_directory=datapath_args.dataset_directory)
    dataset_manipulator = DatasetManipulator(dataset=svc_dataset)

    available_methods = [
        method
        for method in dir(dataset_manipulator)
        if method.startswith("__") is False
        and method.startswith("_") is False
        and method != "dataset"
    ]
    logger.info("All available methods are:")
    time.sleep(0.5)
    pprint(available_methods)
