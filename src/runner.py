from pathlib import Path

from .load_yoga_training_data import load_training_date
from .make_pose_dataset_csv import pose_dataset
from .make_angles_dataset_csv import angles_dataset
from .train_model import train as train_model
from .test_model import run_webcam

# Resolve project root (folder that contains "src")
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR    = ROOT / "training-data"
DATA_DIR   = ROOT / "data"
POSE_CSV   = ROOT / "src" / "pose_dataset.csv"
ANGLES_CSV = ROOT / "src" / "pose_angles_dataset.csv"
BUNDLE     = ROOT / "src" / "pose_knn_runtime.pkl"

def prepare():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "src").mkdir(parents=True, exist_ok=True)

    load_training_date(str(RAW_DIR), str(DATA_DIR))
    pose_dataset(str(DATA_DIR), str(POSE_CSV))
    angles_dataset(str(POSE_CSV), str(ANGLES_CSV))
    print("prepare complete")

def train_cmd():
    if not ANGLES_CSV.exists():
        print("Angles CSV missing; running prepare first…")
        prepare()
    # adjust signature if your train() expects different args
    train_model(str(ANGLES_CSV), str(BUNDLE))
    print("training complete")

def test():
    run_webcam()

def main():
    import argparse
    p = argparse.ArgumentParser("Yoga Pose Runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("test")
    args = p.parse_args()

    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "train":
        train_cmd()
    elif args.cmd == "test":
        test()

if __name__ == "__main__":
    main()
