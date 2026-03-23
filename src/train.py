import argparse
import os
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = os.path.join(args.output_dir, f"run_{now}.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Hello from training script!\n")
        f.write("This file was created successfully.\n")

    print(f"saved to: {result_path}")


if __name__ == "__main__":
    main()