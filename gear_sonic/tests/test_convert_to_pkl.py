import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS, MJ_TO_IL
from gear_sonic.data_process.convert_to_pkl import (
    MotionConversionError,
    convert_npz_file,
    main,
)


def _motion_arrays(joint_pos):
    frames = joint_pos.shape[0]
    body_pos_w = np.zeros((frames, 30, 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.1
    body_pos_w[:, 0, 2] = 0.8
    body_quat_w = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    sqrt_half = np.sqrt(0.5).astype(np.float32)
    body_quat_w[-1, 0] = [sqrt_half, 0.0, 0.0, sqrt_half]
    return {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
    }


def _write_motion(path, joint_pos, **overrides):
    arrays = _motion_arrays(joint_pos)
    arrays.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def _load_entry(path):
    payload = joblib.load(path)
    assert list(payload) == [path.stem]
    return payload[path.stem]


class ConvertToPklTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_isaaclab_order_is_converted_to_mujoco(self):
        joint_pos = np.arange(3 * 29, dtype=np.float32).reshape(3, 29) / 100.0
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        arrays = _write_motion(input_path, joint_pos)

        result = convert_npz_file(
            input_path, output_path, joint_order="isaaclab"
        )
        entry = _load_entry(output_path)

        self.assertEqual(result.frames, 3)
        self.assertEqual(result.fps, 50)
        self.assertEqual(entry["fps"], 50)
        self.assertEqual(entry["dof"].shape, (3, 29))
        self.assertEqual(entry["pose_aa"].shape, (3, 30, 3))
        self.assertEqual(entry["smpl_joints"].shape, (3, 24, 3))
        np.testing.assert_allclose(entry["dof"], joint_pos[:, MJ_TO_IL])
        np.testing.assert_allclose(entry["root_trans_offset"], arrays["body_pos_w"][:, 0])
        np.testing.assert_allclose(entry["root_rot"][-1], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(
            entry["pose_aa"][:, 1:], entry["dof"][:, :, None] * DOF_AXIS[None]
        )
        for field in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
            self.assertEqual(entry[field].dtype, np.float32)

    def test_mujoco_order_is_preserved(self):
        joint_pos = np.arange(2 * 29, dtype=np.float32).reshape(2, 29) / 100.0
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        _write_motion(input_path, joint_pos)

        convert_npz_file(input_path, output_path, joint_order="mujoco")

        np.testing.assert_allclose(_load_entry(output_path)["dof"], joint_pos)

    def test_equivalent_orders_produce_equivalent_outputs(self):
        mujoco_joint_pos = np.arange(4 * 29, dtype=np.float32).reshape(4, 29) / 100.0
        isaaclab_joint_pos = np.empty_like(mujoco_joint_pos)
        isaaclab_joint_pos[:, MJ_TO_IL] = mujoco_joint_pos
        mujoco_input = self.root / "mujoco.npz"
        isaaclab_input = self.root / "isaaclab.npz"
        mujoco_output = self.root / "mujoco.pkl"
        isaaclab_output = self.root / "isaaclab.pkl"
        _write_motion(mujoco_input, mujoco_joint_pos)
        _write_motion(isaaclab_input, isaaclab_joint_pos)

        convert_npz_file(mujoco_input, mujoco_output, joint_order="mujoco")
        convert_npz_file(isaaclab_input, isaaclab_output, joint_order="isaaclab")

        mujoco_entry = _load_entry(mujoco_output)
        isaaclab_entry = _load_entry(isaaclab_output)
        for field in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
            np.testing.assert_allclose(mujoco_entry[field], isaaclab_entry[field])

    def test_invalid_quaternion_does_not_leave_partial_output(self):
        joint_pos = np.zeros((2, 29), dtype=np.float32)
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        arrays = _motion_arrays(joint_pos)
        arrays["body_quat_w"][0, 0] = [2.0, 0.0, 0.0, 0.0]
        np.savez(input_path, **arrays)

        with self.assertRaisesRegex(MotionConversionError, "unit wxyz quaternions"):
            convert_npz_file(input_path, output_path, joint_order="mujoco")

        self.assertFalse(output_path.exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_missing_field_is_rejected(self):
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        arrays = _motion_arrays(np.zeros((2, 29), dtype=np.float32))
        arrays.pop("body_pos_w")
        np.savez(input_path, **arrays)

        with self.assertRaisesRegex(MotionConversionError, "body_pos_w"):
            convert_npz_file(input_path, output_path, joint_order="mujoco")

        self.assertFalse(output_path.exists())

    def test_invalid_shape_nonfinite_value_and_fps_are_rejected(self):
        nonfinite_joint_pos = np.zeros((2, 29), dtype=np.float32)
        nonfinite_joint_pos[0, 0] = np.nan
        invalid_cases = [
            (
                "joint_shape",
                {"joint_pos": np.zeros((2, 28), dtype=np.float32)},
                "joint_pos.*shape",
            ),
            ("nonfinite", {"joint_pos": nonfinite_joint_pos}, "NaN or Inf"),
            ("fps", {"fps": np.array([0], dtype=np.int64)}, "positive integer"),
        ]

        for name, overrides, expected_error in invalid_cases:
            with self.subTest(name=name):
                input_path = self.root / f"{name}.npz"
                output_path = self.root / f"{name}.pkl"
                arrays = _motion_arrays(np.zeros((2, 29), dtype=np.float32))
                arrays.update(overrides)
                np.savez(input_path, **arrays)

                with self.assertRaisesRegex(MotionConversionError, expected_error):
                    convert_npz_file(input_path, output_path, joint_order="mujoco")

                self.assertFalse(output_path.exists())

    def test_existing_output_requires_overwrite(self):
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        _write_motion(input_path, np.zeros((2, 29), dtype=np.float32))
        convert_npz_file(input_path, output_path, joint_order="mujoco")

        with self.assertRaisesRegex(MotionConversionError, "--overwrite"):
            convert_npz_file(input_path, output_path, joint_order="mujoco")

        convert_npz_file(input_path, output_path, joint_order="mujoco", overwrite=True)
        self.assertTrue(output_path.is_file())

    def test_batch_names_are_unique_and_match_outer_keys(self):
        input_root = self.root / "dataset"
        output_root = self.root / "converted"
        joint_pos = np.zeros((2, 29), dtype=np.float32)
        _write_motion(input_root / "CMU" / "01" / "poses" / "motion_seg1" / "motion.npz", joint_pos)
        _write_motion(input_root / "CMU" / "02" / "poses" / "motion_seg1" / "motion.npz", joint_pos)

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "--input",
                    str(input_root),
                    "--output",
                    str(output_root),
                    "--joint-order",
                    "mujoco",
                ]
            )

        self.assertEqual(exit_code, 0)
        output_files = sorted(output_root.glob("*.pkl"))
        self.assertEqual(
            [path.name for path in output_files],
            [
                "CMU__01__poses__motion_seg1.pkl",
                "CMU__02__poses__motion_seg1.pkl",
            ],
        )
        for output_path in output_files:
            self.assertEqual(list(joblib.load(output_path)), [output_path.stem])

    def test_joint_order_is_required_and_auto_is_rejected(self):
        input_path = self.root / "input.npz"
        output_path = self.root / "output.pkl"
        _write_motion(input_path, np.zeros((2, 29), dtype=np.float32))

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing_order:
                main(["--input", str(input_path), "--output", str(output_path)])
            with self.assertRaises(SystemExit) as auto_order:
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--joint-order",
                        "auto",
                    ]
                )

        self.assertEqual(missing_order.exception.code, 2)
        self.assertEqual(auto_order.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
