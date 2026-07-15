#!/usr/bin/env python3
"""Convert G1 motion NPZ files to the PKL format consumed by SONIC MotionLib.

The input joint order is deliberately never inferred. Callers must pass either
``--joint-order isaaclab`` or ``--joint-order mujoco``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import joblib
import numpy as np

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    NUM_BODIES,
    NUM_DOF,
    convert_sequence,
)


VALID_JOINT_ORDERS = ("isaaclab", "mujoco")
REQUIRED_FIELDS = ("fps", "joint_pos", "body_pos_w", "body_quat_w")
EXPECTED_NUM_BODIES = 30
QUATERNION_NORM_TOLERANCE = 1e-3


class MotionConversionError(ValueError):
    """Raised when an input cannot be converted without guessing its format."""


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one successfully converted motion."""

    input_path: Path
    output_path: Path
    motion_name: str
    frames: int
    fps: int


def _read_fps(value: np.ndarray, input_path: Path) -> int:
    fps_array = np.asarray(value)
    if fps_array.size != 1 or fps_array.dtype.kind not in "fiu":
        raise MotionConversionError(
            f"{input_path}: 'fps' must be one numeric value, got shape={fps_array.shape} "
            f"dtype={fps_array.dtype}"
        )

    fps_value = float(fps_array.reshape(-1)[0])
    if not np.isfinite(fps_value) or fps_value <= 0 or not fps_value.is_integer():
        raise MotionConversionError(
            f"{input_path}: 'fps' must be a positive integer, got {fps_value}"
        )
    return int(fps_value)


def _read_float_array(
    data: np.lib.npyio.NpzFile,
    field: str,
    input_path: Path,
) -> np.ndarray:
    value = np.asarray(data[field])
    if value.dtype.kind not in "fiu":
        raise MotionConversionError(
            f"{input_path}: '{field}' must be numeric, got dtype={value.dtype}"
        )
    if not np.isfinite(value).all():
        raise MotionConversionError(f"{input_path}: '{field}' contains NaN or Inf")
    return value.astype(np.float32, copy=False)


def _load_and_validate_npz(input_path: Path, joint_order: str) -> tuple[dict, int]:
    if joint_order not in VALID_JOINT_ORDERS:
        raise MotionConversionError(
            f"joint_order must be one of {VALID_JOINT_ORDERS}, got {joint_order!r}"
        )
    if input_path.suffix != ".npz":
        raise MotionConversionError(f"Input must be a .npz file: {input_path}")
    if not input_path.is_file():
        raise MotionConversionError(f"Input file does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise MotionConversionError(
                f"{input_path}: missing required field(s): {', '.join(missing)}"
            )

        fps = _read_fps(data["fps"], input_path)
        joint_pos = _read_float_array(data, "joint_pos", input_path)
        body_pos_w = _read_float_array(data, "body_pos_w", input_path)
        body_quat_w = _read_float_array(data, "body_quat_w", input_path)

    if joint_pos.ndim != 2 or joint_pos.shape[1] != NUM_DOF:
        raise MotionConversionError(
            f"{input_path}: 'joint_pos' must have shape (T, {NUM_DOF}), got {joint_pos.shape}"
        )
    frames = joint_pos.shape[0]
    if frames < 2:
        raise MotionConversionError(f"{input_path}: expected at least 2 frames, got {frames}")
    if body_pos_w.shape != (frames, EXPECTED_NUM_BODIES, 3):
        raise MotionConversionError(
            f"{input_path}: 'body_pos_w' must have shape "
            f"({frames}, {EXPECTED_NUM_BODIES}, 3), got {body_pos_w.shape}"
        )
    if body_quat_w.shape != (frames, EXPECTED_NUM_BODIES, 4):
        raise MotionConversionError(
            f"{input_path}: 'body_quat_w' must have shape "
            f"({frames}, {EXPECTED_NUM_BODIES}, 4), got {body_quat_w.shape}"
        )

    quat_norm = np.linalg.norm(body_quat_w, axis=-1)
    max_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_norm_error > QUATERNION_NORM_TOLERANCE:
        raise MotionConversionError(
            f"{input_path}: 'body_quat_w' must contain unit wxyz quaternions; "
            f"maximum norm error is {max_norm_error:.6g}"
        )

    # Normalize the accepted near-unit quaternions so root_rot and pose_aa encode
    # exactly the same rotation. The source arrays are never modified.
    body_quat_w = body_quat_w / quat_norm[..., None]
    return {
        "joint_pos": joint_pos,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": "il" if joint_order == "isaaclab" else "mj",
    }, fps


def _atomic_joblib_dump(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        joblib.dump(payload, temp_path, compress=True)
        temp_path.chmod(0o644)
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def convert_npz_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    joint_order: str,
    overwrite: bool = False,
) -> ConversionResult:
    """Validate and convert one G1 NPZ motion into a single-entry PKL."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.suffix != ".pkl":
        raise MotionConversionError(f"Output must be a .pkl file: {output_path}")
    if output_path.exists() and not overwrite:
        raise MotionConversionError(
            f"Output already exists: {output_path} (pass --overwrite to replace it)"
        )

    sequence, fps = _load_and_validate_npz(input_path, joint_order)
    entry = convert_sequence(sequence, fps)
    if entry["pose_aa"].shape[1:] != (NUM_BODIES, 3):
        raise MotionConversionError(
            f"Internal conversion produced invalid pose_aa shape: {entry['pose_aa'].shape}"
        )

    motion_name = output_path.stem
    if not motion_name:
        raise MotionConversionError(f"Output file has no usable motion name: {output_path}")
    _atomic_joblib_dump({motion_name: entry}, output_path)
    return ConversionResult(
        input_path=input_path,
        output_path=output_path,
        motion_name=motion_name,
        frames=entry["dof"].shape[0],
        fps=fps,
    )


def _sanitize_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "motion"


def _batch_motion_name(input_root: Path, input_path: Path) -> str:
    relative = input_path.relative_to(input_root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1].lower() == "motion":
        parts.pop()
    if not parts:
        parts = [input_root.name]
    return "__".join(_sanitize_name_part(part) for part in parts)


def _build_batch_plan(input_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    input_files = sorted(input_root.rglob("*.npz"))
    if not input_files:
        raise MotionConversionError(f"No .npz files found under: {input_root}")

    plan = []
    output_names = {}
    for input_path in input_files:
        motion_name = _batch_motion_name(input_root, input_path)
        output_path = output_root / f"{motion_name}.pkl"
        if motion_name in output_names:
            raise MotionConversionError(
                f"Batch output name collision for {input_path} and {output_names[motion_name]}: "
                f"{motion_name}.pkl"
            )
        output_names[motion_name] = input_path
        plan.append((input_path, output_path))
    return plan


def _convert_input(
    input_path: Path,
    output_path: Path,
    *,
    joint_order: str,
    overwrite: bool,
) -> list[ConversionResult]:
    if input_path.is_file():
        return [
            convert_npz_file(
                input_path,
                output_path,
                joint_order=joint_order,
                overwrite=overwrite,
            )
        ]
    if not input_path.is_dir():
        raise MotionConversionError(f"Input path does not exist: {input_path}")
    if output_path.suffix == ".pkl":
        raise MotionConversionError(
            f"Directory input requires an output directory, not a .pkl file: {output_path}"
        )

    plan = _build_batch_plan(input_path, output_path)
    existing = [target for _, target in plan if target.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        suffix = " ..." if len(existing) > 3 else ""
        raise MotionConversionError(
            f"{len(existing)} output file(s) already exist: {preview}{suffix} "
            "(pass --overwrite to replace them)"
        )

    results = []
    for index, (source, target) in enumerate(plan, start=1):
        result = convert_npz_file(
            source,
            target,
            joint_order=joint_order,
            overwrite=overwrite,
        )
        results.append(result)
        print(
            f"[{index}/{len(plan)}] {source} -> {target} "
            f"({result.frames} frames @ {result.fps} fps)"
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert G1 motion NPZ files to SONIC MotionLib PKL files"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .npz file or directory")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .pkl file for one input, or output directory for directory input",
    )
    parser.add_argument(
        "--joint-order",
        required=True,
        choices=VALID_JOINT_ORDERS,
        help="Explicit input DOF order; automatic detection is intentionally unsupported",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        results = _convert_input(
            args.input,
            args.output,
            joint_order=args.joint_order,
            overwrite=args.overwrite,
        )
    except (MotionConversionError, OSError, EOFError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    frames = sum(result.frames for result in results)
    print(f"Converted {len(results)} motion(s), {frames} total frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
