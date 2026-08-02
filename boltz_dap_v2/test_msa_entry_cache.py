import os
import sys
import unittest
from unittest import mock

import torch
from torch import nn


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "boltz", "src"))

from dap_trunk import _project_msa_features_chunked, _run_msa_dap


class _FakeMsaModule:
    training = False
    use_paired_feature = False
    subsample_msa = False
    msa_blocks = 0

    def __init__(self, num_tokens):
        self.msa_proj = nn.Linear(num_tokens + 2, 4, bias=False)
        self.s_proj = nn.Linear(3, 4, bias=False)


class MsaEntryCacheTest(unittest.TestCase):
    def test_chunked_projection_matches_full_projection(self):
        torch.manual_seed(7)
        num_tokens = 6
        msa_proj = nn.Linear(num_tokens + 3, 4, bias=False)
        msa = torch.randint(num_tokens, (1, 5, 7))
        has_deletion = torch.rand((1, 5, 7, 1))
        deletion_value = torch.rand((1, 5, 7, 1))
        is_paired = torch.rand((1, 5, 7, 1))
        full_features = torch.cat(
            [
                torch.nn.functional.one_hot(msa, num_classes=num_tokens),
                has_deletion,
                deletion_value,
                is_paired,
            ],
            dim=-1,
        )
        expected = msa_proj(full_features)

        actual = _project_msa_features_chunked(
            msa_proj,
            msa,
            has_deletion,
            deletion_value,
            is_paired,
            use_paired_feature=True,
            num_tokens=num_tokens,
            sequence_chunk=2,
        )

        torch.testing.assert_close(actual, expected)

    def test_cache_is_released_before_one_hot_feature_expansion(self):
        from boltz.data import const

        events = []
        original_one_hot = torch.nn.functional.one_hot

        def record_one_hot(*args, **kwargs):
            events.append("one_hot")
            return original_one_hot(*args, **kwargs)

        feats = {
            "msa": torch.zeros((1, 1, 2), dtype=torch.long),
            "has_deletion": torch.zeros((1, 1, 2)),
            "deletion_value": torch.zeros((1, 1, 2)),
            "msa_paired": torch.zeros((1, 1, 2)),
            "msa_mask": torch.ones((1, 1, 2), dtype=torch.bool),
            "token_pad_mask": torch.ones((1, 2), dtype=torch.bool),
        }

        with (
            mock.patch.object(torch.cuda, "empty_cache", side_effect=lambda: events.append("empty_cache")),
            mock.patch.object(torch.nn.functional, "one_hot", side_effect=record_one_hot),
        ):
            _run_msa_dap(
                _FakeMsaModule(const.num_tokens),
                torch.zeros((1, 1, 2, 4)),
                torch.zeros((1, 2, 3)),
                feats,
                torch.ones((1, 2, 2), dtype=torch.bool),
                use_kernels=False,
            )

        self.assertEqual(events[:2], ["empty_cache", "one_hot"])


if __name__ == "__main__":
    unittest.main()