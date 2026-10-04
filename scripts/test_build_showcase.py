"""Regression for preserving an already collected task when adding another."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_showcase as builder
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class ShowcaseCollectionTest(unittest.TestCase):
    def test_adding_taunt_keeps_existing_steering_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            original, resumed, evaluation = (root / n for n in ("original", "resumed", "evaluation"))
            evaluation.mkdir()
            for run, step, value in ((original, 0, 10.0), (resumed, 1, 20.0)):
                writer = EventFileWriter(str(run))
                writer.add_event(Event(step=step, summary=Summary(value=[
                    Summary.Value(tag="Train/mean_reward", simple_value=value)])))
                writer.close()
            acceptance = {
                "requested_frames": 1190,
                "continuous_frames_before_first_termination": 1190,
                "first_termination": None,
                "failure_resets_disabled": [],
                "start_frame": 0,
                "all_recorded_states_finite": True,
            }
            (evaluation / "taunt_acceptance.json").write_text(json.dumps(acceptance))
            (evaluation / "taunt_policy_states.npz").write_bytes(b"test-placeholder")
            index_path = root / "evidence/artifact_index.json"
            index_path.write_text(json.dumps({
                "closeout_training": {"steering": "previously collected branch"},
                "curves": {}, "videos": {},
            }))
            args = argparse.Namespace(taunt_original_run=original,
                                      taunt_resume_run=[resumed], taunt_eval=evaluation)
            # Video encoding is outside this metadata regression; use no media
            # commands and keep all test output confined to the temp directory.
            with patch.object(builder, "ROOT", root), patch.object(
                    builder, "copy_video", return_value={"frames": 1190}):
                builder.add_completed_taunt(args)
            result = json.loads(index_path.read_text())
            self.assertEqual(result["closeout_training"]["steering"], "previously collected branch")
            self.assertIn("taunt", result["closeout_training"])


if __name__ == "__main__":
    unittest.main()
