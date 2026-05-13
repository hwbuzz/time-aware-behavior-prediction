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
from src.sasrec_utils import (
    BatchSampler,
    evaluate_all,
    load_sasrec_dataset,
    parse_topks,
    print_dataset_split_summary,
    recommend_topk,
    summarize_dataset_splits,
)


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
    parser.add_argument("--topk_list", type=str, default="5,10")
    parser.add_argument("--eval_protocol", type=str, choices=["full", "sampled", "both"], default="both")
    parser.add_argument("--use_time_embedding", action="store_true")
    parser.add_argument("--time_features_path", type=str, default=None)
    parser.add_argument("--time_delta_column", type=str, default="delta_prev_seconds")
    parser.add_argument(
        "--time_encoding",
        type=str,
        choices=["bucket", "continuous"],
        default="bucket",
        help="Time representation for time-aware SASRec. 'bucket' uses embedding lookup, 'continuous' uses log1p(delta) projected by a linear layer.",
    )
    parser.add_argument(
        "--time_bucket_boundaries",
        type=str,
        default="60,600,3600,86400",
        help="Comma-separated upper boundaries in seconds for positive time-delta buckets.",
    )
    parser.add_argument(
        "--time_bucket_first_event_separate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to allocate a dedicated bucket for the first event in each sequence.",
    )
    parser.add_argument(
        "--time_bucket_zero_gap_separate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to allocate a dedicated bucket for non-first events with zero time gap.",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="full_valid_ndcg@10",
        help=(
            "Metric used to select the best epoch. "
            "Examples: full_valid_ndcg@5, full_valid_ndcg@10, sampled_valid_ndcg@10. "
            "Shorthand ndcg@5 / ndcg@10 is also accepted and resolves to full_valid_*."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--recommend_user", type=int, default=None)
    parser.add_argument("--save_every_eval", action="store_true")
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
    if getattr(args, "use_time_embedding", False):
        if getattr(args, "time_encoding", "bucket") == "bucket":
            model.time_emb.weight.data[0, :] = 0
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
    config["topks"] = parse_topks(config["topk_list"])
    config["time_bucket_boundaries_raw"] = config.get("time_bucket_boundaries")
    config["time_bucket_boundaries_parsed"] = dataset_stats.get("time_bucket_boundaries", [])
    config["time_feature_dim"] = dataset_stats.get("time_feature_dim", 0)
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


def flatten_metrics(prefix: str, metrics_by_mode: dict) -> dict:
    row = {}
    for mode, metrics in metrics_by_mode.items():
        for key, value in metrics.items():
            row[f"{mode}_{prefix}_{key}"] = value
    return row


def normalize_selection_metric(selection_metric: str) -> str:
    metric = selection_metric.strip()
    if "_" not in metric:
        return f"full_valid_{metric}"
    if metric.count("_") == 1:
        mode, metric_key = metric.split("_", 1)
        return f"{mode}_valid_{metric_key}"
    return metric


def validate_selection_metric(selection_metric: str, args, topks: list[int]):
    mode, split, metric_key = selection_metric.split("_", 2)
    if mode not in {"full", "sampled"}:
        raise ValueError(
            f"selection_metric must start with 'full_' or 'sampled_'. Got: {selection_metric}"
        )
    if split != "valid":
        raise ValueError(
            f"selection_metric must target validation metrics (e.g. full_valid_ndcg@5). Got: {selection_metric}"
        )
    if mode == "sampled" and args.eval_protocol == "full":
        raise ValueError(
            "selection_metric requests sampled metrics, but eval_protocol=full only computes full metrics."
        )
    if mode == "full" and args.eval_protocol == "sampled":
        raise ValueError(
            "selection_metric requests full metrics, but eval_protocol=sampled only computes sampled metrics."
        )
    valid_metric_keys = {f"ndcg@{k}" for k in topks} | {f"hr@{k}" for k in topks} | {
        "mrr",
        "mean_rank",
        "median_rank",
        "num_eval_users",
    }
    if metric_key not in valid_metric_keys:
        raise ValueError(
            f"selection_metric '{selection_metric}' is not compatible with topk_list={topks}. "
            f"Available metric keys: {sorted(valid_metric_keys)}"
        )


def metric_value(metrics_by_mode: dict, metric_name: str) -> float:
    mode, split, metric_key = metric_name.split("_", 2)
    return float(metrics_by_mode[mode][metric_key])


def summarize_eval_result(metrics_by_mode: dict) -> dict:
    summary = {}
    for mode, metrics in metrics_by_mode.items():
        summary[mode] = metrics
    return summary


def write_latest_run_pointer(output_dir: Path, summary: dict):
    latest_path = output_dir / "latest_run.json"
    latest_payload = {
        "run_name": summary["run_name"],
        "run_dir": summary["run_dir"],
        "completed_at": summary["completed_at"],
        "mode": summary["mode"],
        "checkpoint_best": summary.get("checkpoint_best"),
        "checkpoint_last": summary.get("checkpoint_last"),
        "checkpoint_dir": summary.get("checkpoint_dir"),
        "config_path": str(Path(summary["run_dir"]) / "config.json"),
        "metrics_history": summary.get("metrics_history"),
        "metrics_summary": str(Path(summary["run_dir"]) / "metrics_summary.json"),
        "dataset_users": summary["config"]["dataset_stats"].get("users"),
        "dataset_items": summary["config"]["dataset_stats"].get("items"),
        "train_only_users": summary["config"]["dataset_stats"].get("train_only_users"),
        "selection_metric": summary["config"].get("selection_metric"),
    }
    write_json(latest_path, latest_payload)


def update_experiment_index(output_dir: Path, summary: dict):
    index_path = output_dir / "experiment_index.csv"
    best_valid = summary.get("best_valid", {})
    best_test = summary.get("best_test_at_best_valid", {})
    last_valid = summary.get("last_valid", {})
    last_test = summary.get("last_test", {})

    def pick(metrics_group: dict, mode: str, key: str):
        return metrics_group.get(mode, {}).get(key)

    selection_metric = summary["config"].get("selection_metric")
    primary_mode, _, primary_metric_key = selection_metric.split("_", 2)

    best_valid_primary = pick(best_valid, primary_mode, primary_metric_key)
    best_test_primary = pick(best_test, primary_mode, primary_metric_key)
    last_valid_primary = pick(last_valid, primary_mode, primary_metric_key)
    last_test_primary = pick(last_test, primary_mode, primary_metric_key)

    row = {
        "run_name": summary["run_name"],
        "run_dir": summary["run_dir"],
        "completed_at": summary["completed_at"],
        "mode": summary["mode"],
        "selection_metric": selection_metric,
        "primary_metric_name": selection_metric,
        "best_valid_primary": best_valid_primary,
        "best_test_primary": best_test_primary,
        "last_valid_primary": last_valid_primary,
        "last_test_primary": last_test_primary,
        "best_epoch": summary.get("best_epoch"),
        "best_valid_ndcg": pick(best_valid, "full", "ndcg@10"),
        "best_valid_hr": pick(best_valid, "full", "hr@10"),
        "best_valid_mrr": pick(best_valid, "full", "mrr"),
        "best_test_ndcg": pick(best_test, "full", "ndcg@10"),
        "best_test_hr": pick(best_test, "full", "hr@10"),
        "best_test_mrr": pick(best_test, "full", "mrr"),
        "last_valid_ndcg": pick(last_valid, "full", "ndcg@10"),
        "last_valid_hr": pick(last_valid, "full", "hr@10"),
        "last_valid_mrr": pick(last_valid, "full", "mrr"),
        "last_test_ndcg": pick(last_test, "full", "ndcg@10"),
        "last_test_hr": pick(last_test, "full", "hr@10"),
        "last_test_mrr": pick(last_test, "full", "mrr"),
        "checkpoint_best": summary.get("checkpoint_best"),
        "checkpoint_last": summary.get("checkpoint_last"),
        "checkpoint_dir": summary.get("checkpoint_dir"),
        "config_path": str(Path(summary["run_dir"]) / "config.json"),
        "metrics_history": summary.get("metrics_history"),
        "metrics_summary": str(Path(summary["run_dir"]) / "metrics_summary.json"),
        "dataset_users": summary["config"]["dataset_stats"].get("users"),
        "dataset_items": summary["config"]["dataset_stats"].get("items"),
        "train_only_users": summary["config"]["dataset_stats"].get("train_only_users"),
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
        "eval_protocol": summary["config"]["eval_protocol"],
        "topk": summary["config"]["topk"],
        "topk_list": summary["config"]["topk_list"],
        "seed": summary["config"]["seed"],
        "time_encoding": summary["config"].get("time_encoding"),
        "time_bucket_boundaries_raw": summary["config"].get("time_bucket_boundaries_raw"),
        "time_bucket_boundaries_parsed": summary["config"].get("time_bucket_boundaries_parsed"),
        "time_feature_dim": summary["config"].get("time_feature_dim"),
    }
    for prefix, metrics in [("best_valid", best_valid), ("best_test", best_test), ("last_valid", last_valid), ("last_test", last_test)]:
        for mode, mode_metrics in metrics.items():
            for key, value in mode_metrics.items():
                row[f"{prefix}_{mode}_{key}"] = value
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
        user, seq, pos, neg, time_seq = sampler.sample()
        pos_logits, neg_logits = model(user, seq, pos, neg, time_seq)
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


def print_eval_metrics(split: str, metrics_by_mode: dict, topks: list[int]):
    for mode, metrics in metrics_by_mode.items():
        pieces = [f"{split} [{mode}]"]
        for k in topks:
            pieces.append(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
            pieces.append(f"HR@{k}: {metrics[f'hr@{k}']:.4f}")
        pieces.append(f"MRR: {metrics['mrr']:.4f}")
        print(', '.join(pieces))


def main():
    args = parse_args()
    topks = parse_topks(args.topk_list)
    args.selection_metric = normalize_selection_metric(args.selection_metric)
    validate_selection_metric(args.selection_metric, args, topks)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(output_dir, args)
    metrics_path = run_dir / "metrics_history.csv"

    dataset = load_sasrec_dataset(
        args.interactions_path,
        use_time_embedding=args.use_time_embedding,
        time_features_path=args.time_features_path,
        time_delta_column=args.time_delta_column,
        time_encoding=args.time_encoding,
        time_bucket_boundaries=args.time_bucket_boundaries,
        time_bucket_first_event_separate=args.time_bucket_first_event_separate,
        time_bucket_zero_gap_separate=args.time_bucket_zero_gap_separate,
    )
    user_train, _, _, user_num, item_num, user_train_time, _, _, time_bucket_meta = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    dataset_stats = summarize_dataset_splits(dataset)
    dataset_stats["batches_per_epoch"] = num_batch
    if time_bucket_meta.get("enabled"):
        args.time_bucket_count = time_bucket_meta["time_bucket_count"]
        args.time_feature_dim = time_bucket_meta.get("time_feature_dim", 0)
        args.time_encoding = time_bucket_meta.get("time_encoding", args.time_encoding)
    else:
        args.time_bucket_count = 0
        args.time_feature_dim = 0
    config = serializable_args(args, dataset_stats)
    write_json(run_dir / "config.json", config)
    print_dataset_split_summary(dataset_stats)
    print(f"batches_per_epoch={num_batch}")
    print(f"selection_metric={args.selection_metric}")
    print(f"run_dir={run_dir}")

    model = init_model(user_num, item_num, args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if args.inference_only:
        valid_metrics = evaluate_all(model, dataset, args, split="valid", topks=topks)
        test_metrics = evaluate_all(model, dataset, args, split="test", topks=topks)
        print_eval_metrics("valid", valid_metrics, topks)
        print_eval_metrics("test", test_metrics, topks)
        summary = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "inference_only",
            "config": config,
            "valid": summarize_eval_result(valid_metrics),
            "test": summarize_eval_result(test_metrics),
            "checkpoint": args.checkpoint,
            "checkpoint_best": args.checkpoint,
            "checkpoint_last": args.checkpoint,
            "checkpoint_dir": None,
            "metrics_history": None,
        }
        write_json(run_dir / "metrics_summary.json", summary)
        write_latest_run_pointer(output_dir, summary)
        if args.recommend_user is not None:
            print("recommendations:", recommend_topk(model, args.recommend_user, dataset, args))
        return

    sampler = BatchSampler(
        user_train,
        user_num,
        item_num,
        args.batch_size,
        args.maxlen,
        user_train_time=user_train_time,
        seed=args.seed,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_score = float("-inf")
    best_row = None
    last_row = None
    last_valid_metrics = None
    last_test_metrics = None
    best_ckpt = run_dir / "sasrec_best.pth"
    eval_ckpt_dir = run_dir / "checkpoints"
    if args.save_every_eval:
        eval_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(model, sampler, optimizer, criterion, args, num_batch)
        print(f"epoch={epoch}, loss={loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            valid_metrics = evaluate_all(model, dataset, args, split="valid", topks=topks)
            test_metrics = evaluate_all(model, dataset, args, split="test", topks=topks)
            row = {"epoch": epoch, "loss": loss}
            row.update(flatten_metrics("valid", valid_metrics))
            row.update(flatten_metrics("test", test_metrics))
            append_metrics(metrics_path, row)
            last_row = row
            last_valid_metrics = summarize_eval_result(valid_metrics)
            last_test_metrics = summarize_eval_result(test_metrics)
            print_eval_metrics("valid", valid_metrics, topks)
            print_eval_metrics("test", test_metrics, topks)

            if args.save_every_eval:
                eval_ckpt = eval_ckpt_dir / f"sasrec_epoch_{epoch:03d}.pth"
                torch.save(model.state_dict(), eval_ckpt)
                print(f"saved eval checkpoint: {eval_ckpt}")

            selection_score = metric_value(valid_metrics, args.selection_metric)
            if selection_score > best_score:
                best_score = selection_score
                best_row = {
                    "epoch": epoch,
                    "loss": loss,
                    "valid": summarize_eval_result(valid_metrics),
                    "test": summarize_eval_result(test_metrics),
                }
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
        "best_valid": best_row["valid"] if best_row else None,
        "best_test_at_best_valid": best_row["test"] if best_row else None,
        "last_valid": last_valid_metrics,
        "last_test": last_test_metrics,
        "checkpoint_best": str(best_ckpt) if best_row else None,
        "checkpoint_last": str(final_ckpt),
        "metrics_history": str(metrics_path),
        "checkpoint_dir": str(eval_ckpt_dir) if args.save_every_eval else None,
        "primary_metric_name": args.selection_metric,
        "best_valid_primary": metric_value(best_row["valid"], args.selection_metric) if best_row else None,
        "best_test_primary": metric_value(best_row["test"], args.selection_metric) if best_row else None,
        "last_valid_primary": metric_value(last_valid_metrics, args.selection_metric) if last_valid_metrics else None,
        "last_test_primary": metric_value(last_test_metrics, args.selection_metric) if last_test_metrics else None,
    }
    write_json(run_dir / "metrics_summary.json", summary)
    write_latest_run_pointer(output_dir, summary)
    update_experiment_index(output_dir, summary)


if __name__ == "__main__":
    main()
