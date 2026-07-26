import unittest
from types import SimpleNamespace
from unittest.mock import patch

from easydict import EasyDict

from src.dataset import data_factory


class RecordingDataset:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def make_args():
    return EasyDict(
        data="ett_all",
        embed="timeF",
        percent=100,
        max_len=-1,
        var_needed=1,
        noisy_ratio=0.0,
        batch_size=1,
        freq="h",
        seq_len=4,
        label_len=0,
        pred_len=2,
        features="M",
        target="OT",
        num_workers=0,
        multiple_dataset_info={
            "train": [SimpleNamespace(data="recording", root_path="train", data_path="train.csv")],
            "test": [SimpleNamespace(data="recording", root_path="test", data_path="test.csv")],
        },
    )


class DataProviderSplitTests(unittest.TestCase):
    def setUp(self):
        RecordingDataset.calls.clear()

    def test_train_uses_train_dataset_info(self):
        with patch.dict(data_factory.data_dict, {"recording": RecordingDataset}):
            data_factory.data_provider(make_args(), "train")
        self.assertEqual(RecordingDataset.calls[0]["root_path"], "train")

    def test_validation_and_test_use_test_dataset_info(self):
        with patch.dict(data_factory.data_dict, {"recording": RecordingDataset}):
            data_factory.data_provider(make_args(), "val")
            data_factory.data_provider(make_args(), "test")
        self.assertEqual(
            [call["root_path"] for call in RecordingDataset.calls],
            ["test", "test"],
        )


if __name__ == "__main__":
    unittest.main()
