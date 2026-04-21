from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch

from src.sasrec_model import SASRec
from src.sasrec_utils import BatchSampler, evaluate, load_sasrec_dataset, recommend_topk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions_path", type=str, default="data/processed/bpi2012_complete_only/sasrec_interactions.txt")
    parser.add_argument("--output_dir", type=str, default="outputs/sasrec_bpi2012")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_users", type=int, default=10000)
    parser.add_argument("--num_negative_samples", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--recommend_user", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(user_num: int, item_num: int, args):
    model = SASRec(user_num, item_num, args).to(args.device)
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    return model


def make_run_dir(output_dir: Path, args) -> Path:
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{timestamp}_hu{args.hidden_units}_b{args.num_blocks}_h{args.num_heads}_"
            f"ml{args.maxlen}_lr{args.lr:g}_do{args.dropout_rate:g}_seed{args.seed}"
        )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args, dataset_stats: dict) -> dict:
    config = vars(args).copy()
    config["device"] = str(config["device"])
    config["dataset_stats"] = dataset_stats
    return config


def write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_metrics(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def update_experiment_index(output_dir: Path, summary: dict):
    index_path = output_dir / "experiment_index.csv"
    row = {
        "run_name": summary["run_name"],
        "run_dir": summary["run_dir"],
        "completed_at": summary["completed_at"],
        "best_epoch": summary.get("best_epoch"),
        "best_valid_ndcg": summary.get("best_valid", {}).get("ndcg"),
        "best_valid_hr": summary.get("best_valid", {}).get("hr"),
        "best_test_ndcg": summary.get("best_test_at_best_valid", {}).get("ndcg"),
        "best_test_hr": summary.get("best_test_at_best_valid", {}).get("hr"),
        "last_valid_ndcg": summary.get("last_valid", {}).get("ndcg"),
        "last_valid_hr": summary.get("last_valid", {}).get("hr"),
        "last_test_ndcg": summary.get("last_test", {}).get("ndcg"),
        "last_test_hr": summary.get("last_test", {}).get("hr"),
        "checkpoint_best": summary.get("checkpoint_best"),
        "checkpoint_last": summary.get("checkpoint_last"),
        "hidden_units": summary["config"]["hidden_units"],
        "num_blocks": summary["config"]["num_blocks"],
        "num_heads": summary["config"]["num_heads"],
        "maxlen": summary["config"]["maxlen"],
        "lr": summary["config"]["lr"],
        "dropout_rate": summary["config"]["dropout_rate"],
        "batch_size": summary["config"]["batch_size"],
        "num_epochs": summary["config"]["num_epochs"],
        "eval_users": summary["config"]["eval_users"],
        "num_negative_samples": summary["config"]["num_negative_samples"],
        "topk": summary["config"]["topk"],
        "seed": summary["config"]["seed"],
    }
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, sampler, optimizer, criterion, args, num_batch: int):
    model.train()
    total_loss = 0.0
    for _ in range(num_batch):
        user, seq, pos, neg = sampler.sample()
        pos_logits, neg_logits = model(user, seq, pos, neg)
        pos_labels = torch.ones(pos_logits.shape, device=args.device)
        neg_labels = torch.zeros(neg_logits.shape, device=args.device)
        indices = np.where(pos != 0)

        optimizer.zero_grad()
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters():
            loss += args.l2_emb * torch.sum(param ** 2)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(num_batch, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(output_dir, args)
    metrics_path = run_dir / "metrics_history.csv"

    dataset = load_sasrec_dataset(args.interactions_path)
    user_train, _, _, user_num, item_num = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    avg_len = sum(len(v) for v in user_train.values()) / max(len(user_train), 1)
    dataset_stats = {
        "users": user_num,
        "items": item_num,
        "train_interactions": int(sum(len(v) for v in user_train.values())),
        "avg_train_len": avg_len,
        "batches_per_epoch": num_batch,
    }
    config = serializable_args(args, dataset_stats)
    write_json(run_dir / "config.json", config)
    print(f"users={user_num}, items={item_num}, avg_train_len={avg_len:.2f}, batches={num_batch}")
    print(f"run_dir={run_dir}")

    model = init_model(user_num, item_num, args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if args.inference_only:
        val = evaluate(model, dataset, args, split="valid")
        test = evaluate(model, dataset, args, split="test")
        print(f"valid NDCG@{args.topk}: {val[0]:.4f}, HR@{args.topk}: {val[1]:.4f}")
        print(f"test  NDCG@{args.topk}: {test[0]:.4f}, HR@{args.topk}: {test[1]:.4f}")
        summary = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "inference_only",
            "config": config,
            "valid": {"ndcg": val[0], "hr": val[1]},
            "test": {"ndcg": test[0], "hr": test[1]},
            "checkpoint": args.checkpoint,
        }
        write_json(run_dir / "metrics_summary.json", summary)
        if args.recommend_user is not None:
            print("recommendations:", recommend_topk(model, args.recommend_user, dataset, args))
        return

    sampler = BatchSampler(user_train, user_num, item_num, args.batch_size, args.maxlen, args.seed)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val = -1.0
    best_row = None
    last_row = None
    best_ckpt = run_dir / "sasrec_best.pth"
    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(model, sampler, optimizer, criterion, args, num_batch)
        print(f"epoch={epoch}, loss={loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            val = evaluate(model, dataset, args, split="valid")
            test = evaluate(model, dataset, args, split="test")
            row = {
                "epoch": epoch,
                "loss": loss,
                "valid_ndcg": val[0],
                "valid_hr": val[1],
                "test_ndcg": test[0],
                "test_hr": test[1],
            }
            append_metrics(metrics_path, row)
            last_row = row
            print(
                f"epoch={epoch}, valid NDCG@{args.topk}: {val[0]:.4f}, HR@{args.topk}: {val[1]:.4f}, "
                f"test NDCG@{args.topk}: {test[0]:.4f}, HR@{args.topk}: {test[1]:.4f}"
            )
            if val[0] > best_val:
                best_val = val[0]
                best_row = row
                torch.save(model.state_dict(), best_ckpt)
                print(f"saved checkpoint: {best_ckpt}")

    final_ckpt = run_dir / "sasrec_last.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"saved checkpoint: {final_ckpt}")
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "train",
        "config": config,
        "best_epoch": best_row["epoch"] if best_row else None,
        "best_valid": {"ndcg": best_row["valid_ndcg"], "hr": best_row["valid_hr"]} if best_row else None,
        "best_test_at_best_valid": {"ndcg": best_row["test_ndcg"], "hr": best_row["test_hr"]} if best_row else None,
        "last_valid": {"ndcg": last_row["valid_ndcg"], "hr": last_row["valid_hr"]} if last_row else None,
        "last_test": {"ndcg": last_row["test_ndcg"], "hr": last_row["test_hr"]} if last_row else None,
        "checkpoint_best": str(best_ckpt) if best_row else None,
        "checkpoint_last": str(final_ckpt),
        "metrics_history": str(metrics_path),
    }
    write_json(run_dir / "metrics_summary.json", summary)
    update_experiment_index(output_dir, summary)


if __name__ == "__main__":
    main()
