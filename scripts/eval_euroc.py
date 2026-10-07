#!/usr/bin/env python3
"""Run EuRoC sequences one by one and evaluate ATE.

Usage (from PyVINS-Mono, inside the GTSAM environment):

    python scripts/eval_euroc.py
    python scripts/eval_euroc.py --sequences MH_01_easy MH_03_medium
    python scripts/eval_euroc.py --dataset-root /path/to/euroc

Each sequence writes its trajectory, ground truth, evo report and log under
output/euroc/<sequence>/. The combined table is output/euroc/summary.txt.
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "euroc"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "euroc"

# Folder name on disk -> shared yaml. dataset_path is overridden per sequence.
SEQUENCES = [
    ("MH_01_easy", "euroc_MH01-05.yaml"),
    ("MH_02_easy", "euroc_MH01-05.yaml"),
    ("MH_03_medium", "euroc_MH01-05.yaml"),
    ("MH_04_difficult", "euroc_MH01-05.yaml"),
    ("MH_05_difficult", "euroc_MH01-05.yaml"),
    ("V1_01_easy", "euroc_V101-03.yaml"),
    ("V1_02_medium", "euroc_V101-03.yaml"),
    ("V1_03_difficult", "euroc_V101-03.yaml"),
    ("V2_01_easy", "euroc_V201-03.yaml"),
    ("V2_02_medium", "euroc_V201-03.yaml"),
    ("V2_03_difficult", "euroc_V201-03.yaml"),
]


def default_dataset_root():
    config_path = CONFIG_DIR / "euroc_MH01-05.yaml"
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    return Path(config["dataset_path"]).parent


def count_images(sequence_dir):
    csv_path = sequence_dir / "mav0" / "cam0" / "data.csv"
    if not csv_path.exists():
        csv_path = sequence_dir / "image" / "data.csv"
    count = 0
    with open(csv_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            count += 1
    return count


def convert_euroc_groundtruth(sequence_dir, tum_path):
    """EuRoC state_groundtruth_estimate0 -> TUM (seconds, qx qy qz qw)."""
    csv_path = sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    lines = ["# timestamp tx ty tz qx qy qz qw\n"]
    with open(csv_path, "r") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [item.strip() for item in raw.split(",")]
            timestamp_ns = int(parts[0])
            seconds, nano = divmod(timestamp_ns, 1_000_000_000)
            tx, ty, tz = parts[1], parts[2], parts[3]
            qw, qx, qy, qz = parts[4], parts[5], parts[6], parts[7]
            lines.append(
                f"{seconds}.{nano:09d} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
            )

    tum_path.parent.mkdir(parents=True, exist_ok=True)
    tum_path.write_text("".join(lines))
    return len(lines) - 1


def write_sequence_config(template_path, sequence_dir, output_dir):
    with open(template_path, "r") as handle:
        config = yaml.safe_load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    config["dataset_path"] = str(sequence_dir)
    config["trajectory_output_path"] = str(output_dir / "estimated_trajectory.txt")
    config["log_dir"] = str(output_dir / "log")
    config["visualize"] = False
    config["enable_viewer"] = False

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return config_path


def format_clock(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(seq_index, seq_total, name, done, total, elapsed, phase):
    width = 28
    fraction = 0.0 if total <= 0 else min(done / total, 1.0)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    rate = done / elapsed if elapsed > 0 else 0.0
    line = (
        f"[{seq_index}/{seq_total}] {name:<16} |{bar}| "
        f"{done:>5}/{total:<5} {fraction * 100:5.1f}%  "
        f"{rate:4.1f} img/s  {phase:<10} {format_clock(elapsed)}"
    )
    if sys.stdout.isatty():
        print("\r" + line.ljust(110), end="", flush=True)
    else:
        print(line, flush=True)


def run_sequence(config_path, log_path, seq_index, seq_total, name, image_count):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"

    command = [sys.executable, str(PROJECT_ROOT / "main.py"), "--config", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    done = 0
    phase = "tracking"
    last_draw = 0.0

    def draw(force=False):
        nonlocal last_draw
        now = time.perf_counter()
        interval = 0.0 if (force and sys.stdout.isatty()) else (0.25 if sys.stdout.isatty() else 2.0)
        if now - last_draw < interval:
            return
        render_progress(
            seq_index, seq_total, name, done, image_count, now - started, phase,
        )
        last_draw = now

    with open(log_path, "w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        try:
            for line in process.stdout:
                log_handle.write(line)
                if "【FeatureTracker】Image data:" in line:
                    done += 1
                    phase = "tracking"
                    draw(force=True)
                elif "Visual Feature Tracker has finished" in line:
                    phase = "estimator"
                    draw(force=True)
                elif phase == "estimator":
                    draw()
            return_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            print()
            raise

    elapsed = time.perf_counter() - started
    if sys.stdout.isatty():
        print()
    return return_code, elapsed, done


def parse_ate_stdout(text):
    def grab(name):
        match = re.search(rf"^{name}\s+([0-9.eE+-]+)", text, re.MULTILINE)
        return float(match.group(1)) if match else None

    compared = re.search(r"Compared\s+(\d+)\s+absolute pose pairs", text)
    return {
        "rmse": grab("rmse"),
        "mean": grab("mean"),
        "median": grab("median"),
        "std": grab("std"),
        "matched": int(compared.group(1)) if compared else None,
    }


def evaluate_ate(groundtruth, estimated, output_dir, label):
    """label is 'se3' or 'sim3'."""
    report_path = output_dir / f"ate_{label}.txt"
    if not estimated.exists() or estimated.stat().st_size == 0:
        report_path.write_text("estimated trajectory is missing\n")
        return {"rmse": None, "mean": None, "median": None, "std": None, "matched": None}

    command = [
        "evo_ape", "tum", str(groundtruth), str(estimated),
        "-a", "-r", "trans_part",
        "--t_max_diff", "0.02",
        "-v", "--no_warnings",
        "--save_results", str(output_dir / f"ate_{label}.zip"),
        "--save_plot", str(output_dir / f"ate_{label}.png"),
    ]
    if label == "sim3":
        command.append("-s")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    report_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        return {"rmse": None, "mean": None, "median": None, "std": None, "matched": None}
    return parse_ate_stdout(result.stdout)


def fmt_meter(value):
    return f"{value:.4f}" if value is not None else "   n/a"


def write_summary(output_root, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    header = (
        f"{'sequence':<18} {'status':<8} {'SE3 RMSE':>10} {'Sim3 RMSE':>10} "
        f"{'SE3 mean':>10} {'matched':>8} {'time':>10}"
    )
    lines = [
        "EuRoC ATE  (translation, meters, Umeyama alignment)",
        header,
        "-" * len(header),
    ]
    se3_values = []
    sim3_values = []
    for row in rows:
        status = "ok" if row["se3_rmse"] is not None else "fail"
        matched = row["matched"] if row["matched"] is not None else "-"
        lines.append(
            f"{row['sequence']:<18} {status:<8} {fmt_meter(row['se3_rmse']):>10} "
            f"{fmt_meter(row['sim3_rmse']):>10} {fmt_meter(row['se3_mean']):>10} "
            f"{str(matched):>8} {format_clock(row['elapsed']):>10}"
        )
        if row["se3_rmse"] is not None:
            se3_values.append(row["se3_rmse"])
        if row["sim3_rmse"] is not None:
            sim3_values.append(row["sim3_rmse"])

    lines.append("-" * len(header))
    se3_mean = sum(se3_values) / len(se3_values) if se3_values else None
    sim3_mean = sum(sim3_values) / len(sim3_values) if sim3_values else None
    lines.append(
        f"{'mean':<18} {len(se3_values):>3}/{len(rows):<4} {fmt_meter(se3_mean):>10} "
        f"{fmt_meter(sim3_mean):>10}"
    )
    text = "\n".join(lines) + "\n"
    (output_root / "summary.txt").write_text(text)

    csv_path = output_root / "summary.csv"
    fieldnames = [
        "sequence", "status", "exit_code", "images", "elapsed_sec", "matched",
        "se3_rmse", "se3_mean", "se3_median", "se3_std",
        "sim3_rmse", "sim3_mean", "sim3_median", "sim3_std",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            status = "ok" if row["se3_rmse"] is not None else "fail"
            writer.writerow({
                "sequence": row["sequence"],
                "status": status,
                "exit_code": row["exit_code"],
                "images": row["images"],
                "elapsed_sec": f"{row['elapsed']:.3f}",
                "matched": row["matched"] if row["matched"] is not None else "",
                "se3_rmse": row["se3_rmse"] if row["se3_rmse"] is not None else "",
                "se3_mean": row["se3_mean"] if row["se3_mean"] is not None else "",
                "se3_median": row["se3_median"] if row["se3_median"] is not None else "",
                "se3_std": row["se3_std"] if row["se3_std"] is not None else "",
                "sim3_rmse": row["sim3_rmse"] if row["sim3_rmse"] is not None else "",
                "sim3_mean": row["sim3_mean"] if row["sim3_mean"] is not None else "",
                "sim3_median": row["sim3_median"] if row["sim3_median"] is not None else "",
                "sim3_std": row["sim3_std"] if row["sim3_std"] is not None else "",
            })
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run EuRoC sequences and evaluate ATE.")
    parser.add_argument(
        "--sequences", nargs="+", default=None,
        help="Sequence folder names. Default: all 11 EuRoC sequences.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Directory that contains MH_01_easy, V1_01_easy, ...",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Result directory. Default: output/euroc",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root or default_dataset_root()
    output_root = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    selected = {name: config_name for name, config_name in SEQUENCES}
    if args.sequences:
        unknown = [name for name in args.sequences if name not in selected]
        if unknown:
            print(f"Unknown sequence(s): {', '.join(unknown)}")
            print("Known sequences: " + ", ".join(name for name, _ in SEQUENCES))
            return 1
        order = [(name, selected[name]) for name in args.sequences]
    else:
        order = list(SEQUENCES)

    if shutil.which("evo_ape") is None:
        print("evo_ape was not found on PATH. Install evo before running this evaluation.")
        return 1

    print(f"Dataset root: {dataset_root}")
    print(f"Output:       {output_root}")
    print(f"Sequences:    {len(order)}")

    rows = []
    for index, (name, config_name) in enumerate(order, start=1):
        sequence_dir = dataset_root / name
        sequence_output = output_root / name
        template = CONFIG_DIR / config_name
        print(f"\n[{index}/{len(order)}] {name}")

        if not sequence_dir.is_dir():
            print(f"  missing dataset directory: {sequence_dir}")
            rows.append(_empty_row(name, exit_code=None, images=0, elapsed=0.0))
            continue
        if not template.exists():
            print(f"  missing config: {template}")
            rows.append(_empty_row(name, exit_code=None, images=0, elapsed=0.0))
            continue

        image_count = count_images(sequence_dir)
        config_path = write_sequence_config(template, sequence_dir, sequence_output)
        groundtruth = sequence_output / "groundtruth.txt"
        convert_euroc_groundtruth(sequence_dir, groundtruth)

        exit_code, elapsed, seen = run_sequence(
            config_path, sequence_output / "run.log",
            index, len(order), name, image_count,
        )
        estimated = sequence_output / "estimated_trajectory.txt"
        se3 = evaluate_ate(groundtruth, estimated, sequence_output, "se3")
        sim3 = evaluate_ate(groundtruth, estimated, sequence_output, "sim3")
        row = {
            "sequence": name,
            "exit_code": exit_code,
            "images": seen,
            "elapsed": elapsed,
            "matched": se3["matched"],
            "se3_rmse": se3["rmse"],
            "se3_mean": se3["mean"],
            "se3_median": se3["median"],
            "se3_std": se3["std"],
            "sim3_rmse": sim3["rmse"],
            "sim3_mean": sim3["mean"],
            "sim3_median": sim3["median"],
            "sim3_std": sim3["std"],
        }
        rows.append(row)
        print(
            f"  {name}  exit {exit_code}  "
            f"SE3 RMSE {fmt_meter(se3['rmse'])} m  "
            f"Sim3 RMSE {fmt_meter(sim3['rmse'])} m  "
            f"matched {se3['matched']}  {format_clock(elapsed)}"
        )
        write_summary(output_root, rows)

    print()
    print(write_summary(output_root, rows))
    print(f"Summary written to {output_root / 'summary.txt'}")
    return 0 if rows and all(row["se3_rmse"] is not None for row in rows) else 1


def _empty_row(name, exit_code, images, elapsed):
    return {
        "sequence": name,
        "exit_code": exit_code,
        "images": images,
        "elapsed": elapsed,
        "matched": None,
        "se3_rmse": None,
        "se3_mean": None,
        "se3_median": None,
        "se3_std": None,
        "sim3_rmse": None,
        "sim3_mean": None,
        "sim3_median": None,
        "sim3_std": None,
    }


if __name__ == "__main__":
    sys.exit(main())
