#!/usr/bin/env python3
"""Run the fixed PPB pipeline on EuRoC/ORI with loop closure on or off."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

try:
    from eval_euroc import evaluate_ate, format_clock
except ModuleNotFoundError:
    from scripts.eval_euroc import evaluate_ate, format_clock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EUROC_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]
ORI_SEQUENCES = ["r01", "r02", "r03", "r04", "r05"]


def count_images(sequence_dir):
    path = sequence_dir / "mav0" / "cam0" / "data.csv"
    with path.open(encoding="utf-8") as handle:
        return sum(
            1 for line in handle
            if line.strip() and not line.startswith("#")
        )


def convert_euroc_groundtruth(sequence_dir, output_path):
    source = sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    with source.open(encoding="utf-8") as source_handle, \
            output_path.open("w", encoding="utf-8") as output_handle:
        output_handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for line in source_handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(",")]
            seconds, nanoseconds = divmod(int(fields[0]), 1_000_000_000)
            output_handle.write(
                f"{seconds}.{nanoseconds:09d} {fields[1]} {fields[2]} {fields[3]} "
                f"{fields[5]} {fields[6]} {fields[7]} {fields[4]}\n"
            )


def prepare_config(family, sequence_dir, output_dir, feature_count, loop_enabled):
    base_path = PROJECT_ROOT / "config" / (
        "euroc/euroc.yaml" if family == "euroc" else "ori/ori_cam0.yaml"
    )
    with base_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset_path"] = str(sequence_dir)
    config["max_features_to_detect"] = int(feature_count)
    config["visualize"] = False
    config["enable_viewer"] = False
    config["enable_loop_detection"] = bool(loop_enabled)
    config["enable_loop_optimization"] = bool(loop_enabled)
    config["loop_detector_visualize"] = False
    config["trajectory_output_path"] = str(output_dir / "estimated_trajectory.txt")
    config["loop_corrected_trajectory_output_path"] = str(
        output_dir / "estimated_trajectory_loop_corrected.txt"
    )
    config["log_dir"] = str(output_dir / "state_logs")

    # EuRoC's base config predates loop closure. Reuse only loop-related keys
    # from the ORI config; camera, IMU and dataset calibration remain EuRoC's.
    if family == "euroc":
        with (PROJECT_ROOT / "config/ori/ori_cam0.yaml").open(
                encoding="utf-8") as handle:
            loop_template = yaml.safe_load(handle)
        for key, value in loop_template.items():
            if key.startswith("loop_"):
                config[key] = value
        config["enable_loop_detection"] = bool(loop_enabled)
        config["enable_loop_optimization"] = bool(loop_enabled)
        config["loop_detector_visualize"] = False

    config_path = output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return config_path


def completed_successfully(log_path, trajectory_path):
    if not log_path.exists() or not trajectory_path.exists():
        return False
    if trajectory_path.stat().st_size == 0:
        return False
    tail_size = min(log_path.stat().st_size, 8192)
    with log_path.open("rb") as handle:
        handle.seek(-tail_size, os.SEEK_END)
        tail = handle.read().decode("utf-8", errors="replace")
    return "sent IMU count does not match" not in tail and \
        "SLAM system shut down" in tail


def run_sequence(config_path, log_path, sequence_name, image_count):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    command = [sys.executable, str(PROJECT_ROOT / "main.py"),
               "--config", str(config_path)]
    started = time.perf_counter()
    processed = 0
    last_report = started
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        try:
            for line in process.stdout:
                log_handle.write(line)
                if "【FeatureTracker】Image data:" in line:
                    processed += 1
                now = time.perf_counter()
                if now - last_report >= 20.0:
                    elapsed = now - started
                    print(
                        f"  {sequence_name}: {processed}/{image_count} images "
                        f"({processed / max(image_count, 1):.1%}), "
                        f"{format_clock(elapsed)}",
                        flush=True,
                    )
                    last_report = now
            return_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return return_code, time.perf_counter() - started, processed


def metric_row(label, metric):
    return {
        f"{label}_rmse": metric["rmse"],
        f"{label}_mean": metric["mean"],
        f"{label}_median": metric["median"],
        f"{label}_matched": metric["matched"],
    }


def evaluate(output_dir, groundtruth, label, trajectory):
    evaluation_dir = output_dir / f"{label}_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    se3 = evaluate_ate(groundtruth, trajectory, evaluation_dir, "se3")
    sim3 = evaluate_ate(groundtruth, trajectory, evaluation_dir, "sim3")
    return {
        **metric_row(f"{label}_se3", se3),
        **metric_row(f"{label}_sim3", sim3),
    }


def write_summary(output_root, rows):
    fields = [
        "family", "sequence", "loop", "features", "status", "exit_code",
        "images", "elapsed_sec",
        "vio_se3_rmse", "vio_se3_mean", "vio_se3_median", "vio_se3_matched",
        "vio_sim3_rmse", "vio_sim3_mean", "vio_sim3_median", "vio_sim3_matched",
        "loop_se3_rmse", "loop_se3_mean", "loop_se3_median", "loop_se3_matched",
        "loop_sim3_rmse", "loop_sim3_mean", "loop_sim3_median", "loop_sim3_matched",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    header = (
        f"{'sequence':<18} {'loop':<4} {'status':<7} "
        f"{'VIO SE3':>9} {'VIO Sim3':>9} {'Loop SE3':>9} {'time':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        def value(key):
            item = row.get(key)
            return "n/a" if item is None or item == "" else f"{item:.4f}"
        lines.append(
            f"{row['sequence']:<18} {row['loop']:<4} {row['status']:<7} "
            f"{value('vio_se3_rmse'):>9} {value('vio_sim3_rmse'):>9} "
            f"{value('loop_se3_rmse'):>9} {format_clock(row['elapsed_sec']):>8}"
        )
    text = "\n".join(lines) + "\n"
    (output_root / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("euroc", "ori"), required=True)
    parser.add_argument("--loop", choices=("on", "off"), required=True)
    parser.add_argument("--features", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.features < 20:
        parser.error("--features must be at least 20")
    if shutil.which("evo_ape") is None:
        parser.error("evo_ape is not available")

    loop_enabled = args.loop == "on"
    if args.family == "euroc":
        dataset_root = Path("/home/geneta/dataset/source_dataset/euroc")
        sequences = EUROC_SEQUENCES
    else:
        dataset_root = Path("/home/geneta/dataset/ORI")
        sequences = ORI_SEQUENCES
    if args.sequences:
        unknown = sorted(set(args.sequences) - set(sequences))
        if unknown:
            parser.error(f"unknown sequences: {', '.join(unknown)}")
        sequences = args.sequences

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sequence in enumerate(sequences, 1):
        print(
            f"[{index}/{len(sequences)}] {args.family}/{sequence} "
            f"loop={args.loop} features={args.features} pipeline=PPB",
            flush=True)
        sequence_dir = dataset_root / sequence
        output_dir = output_root / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = prepare_config(
            args.family, sequence_dir, output_dir, args.features, loop_enabled)
        groundtruth = output_dir / "groundtruth.txt"
        if args.family == "euroc":
            convert_euroc_groundtruth(sequence_dir, groundtruth)
        else:
            source = sequence_dir / f"{sequence}_gt" / "poses_gt.txt"
            shutil.copyfile(source, groundtruth)

        trajectory = output_dir / "estimated_trajectory.txt"
        log_path = output_dir / "run.log"
        image_count = count_images(sequence_dir)
        if args.resume and completed_successfully(log_path, trajectory):
            return_code, elapsed, processed = 0, 0.0, image_count
            print("  reusing completed run", flush=True)
        else:
            return_code, elapsed, processed = run_sequence(
                config_path, log_path, sequence, image_count)

        row = {
            "family": args.family, "sequence": sequence, "loop": args.loop,
            "features": args.features, "exit_code": return_code,
            "images": processed, "elapsed_sec": elapsed,
            "status": "ok" if return_code == 0 else "run_fail",
        }
        row.update(evaluate(output_dir, groundtruth, "vio", trajectory))
        if loop_enabled:
            corrected = output_dir / "estimated_trajectory_loop_corrected.txt"
            row.update(evaluate(output_dir, groundtruth, "loop", corrected))
        if row.get("vio_se3_rmse") is None:
            row["status"] = "eval_fail"
        rows.append(row)
        print(
            f"  result status={row['status']} "
            f"VIO={row.get('vio_se3_rmse')} "
            f"loop={row.get('loop_se3_rmse')} time={format_clock(elapsed)}",
            flush=True,
        )
        print(write_summary(output_root, rows), flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
