from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import random
from statistics import median

import numpy as np
import torch


DEFAULT_TOPKS = (5, 10)


def parse_time_bucket_boundaries(boundary_spec) -> list[float]:
    if boundary_spec is None:
        return []
    if isinstance(boundary_spec, (list, tuple)):
        values = [float(v) for v in boundary_spec]
    else:
        values = [float(part.strip()) for part in str(boundary_spec).split(",") if part.strip()]
    values = sorted(v for v in values if v > 0)
    if len(values) != len(set(values)):
        raise ValueError(f"time bucket boundaries must be unique. Got: {values}")
    return values


def time_bucket_count(boundaries: list[float], separate_first_event: bool = True, separate_zero_gap: bool = True) -> int:
    return 1 + int(separate_first_event) + int(separate_zero_gap) + len(boundaries) + 1


def time_gap_bucket_count(boundaries: list[float], separate_zero_gap: bool = True) -> int:
    return int(separate_zero_gap) + len(boundaries) + 1


def bucketize_time_delta(
    delta_seconds: float,
    event_idx: int,
    boundaries: list[float],
    separate_first_event: bool = True,
    separate_zero_gap: bool = True,
) -> int:
    bucket_idx = 1
    if separate_first_event:
        if event_idx == 0:
            return bucket_idx
        bucket_idx += 1
    if separate_zero_gap and delta_seconds == 0:
        return bucket_idx
    if separate_zero_gap:
        bucket_idx += 1
    for boundary in boundaries:
        if delta_seconds < boundary:
            return bucket_idx
        bucket_idx += 1
    return bucket_idx


def default_time_features_path(interactions_path: str) -> str:
    return str(Path(interactions_path).resolve().with_name("events_encoded_time_features.csv"))


def encode_raw_time_delta(delta_seconds: float) -> float:
    return float(max(delta_seconds, 0.0))


def encode_continuous_time_features(delta_seconds: float, event_idx: int) -> tuple[float, float]:
    # log1p compresses the long-tail time-gap distribution while preserving zero.
    # `is_first_event` distinguishes the first event from non-first zero-gap events.
    return float(np.log1p(max(delta_seconds, 0.0))), float(event_idx == 0)


def transform_time_value(value, transform: str):
    if transform == "none":
        return value
    if transform == "log1p":
        return np.log1p(np.maximum(value, 0.0))
    raise ValueError(f"Unknown time_target_transform: {transform}")


def inverse_time_value(value, transform: str):
    if transform == "none":
        result = value
    elif transform == "log1p":
        result = np.expm1(value)
    else:
        raise ValueError(f"Unknown time_target_transform: {transform}")
    return np.maximum(result, 0.0)


def load_time_feature_sequences(
    time_features_path: str,
    time_delta_column: str,
    time_encoding: str,
    boundaries: list[float],
    separate_first_event: bool = True,
    separate_zero_gap: bool = True,
    target_time_column: str | None = None,
):
    rows_by_user: dict[int, list[tuple[int, int, object, float | None]]] = defaultdict(list)
    with open(time_features_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"user_id", "item_id", "event_idx", time_delta_column}
        if target_time_column is not None:
            required_columns.add(target_time_column)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"time features file is missing required columns: {sorted(missing)}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            event_idx = int(row["event_idx"])
            delta_seconds = float(row[time_delta_column])
            if time_encoding == "bucket":
                time_value = bucketize_time_delta(
                    delta_seconds,
                    event_idx,
                    boundaries,
                    separate_first_event=separate_first_event,
                    separate_zero_gap=separate_zero_gap,
                )
            elif time_encoding == "continuous":
                time_value = encode_continuous_time_features(delta_seconds, event_idx)
            elif time_encoding == "raw":
                time_value = encode_raw_time_delta(delta_seconds)
            else:
                raise ValueError(f"Unknown time_encoding: {time_encoding}")
            target_value = float(row[target_time_column]) if target_time_column is not None else None
            rows_by_user[user_id].append((event_idx, item_id, time_value, target_value))

    item_sequences = {}
    time_sequences = {}
    target_sequences = {}
    for user_id, rows in rows_by_user.items():
        rows.sort(key=lambda x: x[0])
        item_sequences[user_id] = [item_id for _, item_id, _, _ in rows]
        time_sequences[user_id] = [time_value for _, _, time_value, _ in rows]
        if target_time_column is not None:
            target_sequences[user_id] = [float(target_value) for _, _, _, target_value in rows]

    bucket_meta = {
        "time_features_path": str(Path(time_features_path).resolve()),
        "time_delta_column": time_delta_column,
        "time_encoding": time_encoding,
        "time_bucket_boundaries": boundaries,
        "time_bucket_first_event_separate": bool(separate_first_event),
        "time_bucket_zero_gap_separate": bool(separate_zero_gap),
        "time_bucket_count": (
            time_bucket_count(
                boundaries,
                separate_first_event=separate_first_event,
                separate_zero_gap=separate_zero_gap,
            )
            if time_encoding == "bucket"
            else 0
        ),
        "time_feature_dim": 1 if time_encoding in {"bucket", "raw"} else 2,
        "next_time_target_column": target_time_column,
    }
    return item_sequences, time_sequences, bucket_meta, target_sequences


def load_sasrec_dataset(
    interactions_path: str,
    use_time_embedding: bool = False,
    use_time_attention_bias: bool = False,
    time_features_path: str | None = None,
    time_delta_column: str = "delta_prev_seconds",
    time_encoding: str = "bucket",
    time_bucket_boundaries=None,
    time_bucket_first_event_separate: bool = True,
    time_bucket_zero_gap_separate: bool = True,
    enable_time_prediction: bool = False,
    time_prediction_target: str = "delta_next_seconds",
):
    """Load user-item interactions and create leave-one-out train/valid/test splits."""
    user_sequences: dict[int, list[int]] = defaultdict(list)
    user_num = 0
    item_num = 0

    with open(interactions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            user_id, item_id = map(int, line.strip().split()[:2])
            user_sequences[user_id].append(item_id)
            user_num = max(user_num, user_id)
            item_num = max(item_num, item_id)

    train_time, valid_time, test_time = {}, {}, {}
    train_next_time, valid_next_time, test_next_time = {}, {}, {}
    time_bucket_meta = {
        "enabled": False,
        "time_prediction_enabled": bool(enable_time_prediction),
        "time_prediction_target": time_prediction_target,
        "next_time_targets": {"train": train_next_time, "valid": valid_next_time, "test": test_next_time},
        "use_time_embedding": bool(use_time_embedding),
    }
    if use_time_embedding or use_time_attention_bias or enable_time_prediction:
        boundaries = parse_time_bucket_boundaries(time_bucket_boundaries)
        resolved_time_features_path = time_features_path or default_time_features_path(interactions_path)
        effective_time_encoding = "raw" if use_time_attention_bias else time_encoding
        time_item_sequences, time_bucket_sequences, time_bucket_meta_from_file, next_time_sequences = load_time_feature_sequences(
            resolved_time_features_path,
            time_delta_column=time_delta_column,
            time_encoding=effective_time_encoding,
            boundaries=boundaries,
            separate_first_event=time_bucket_first_event_separate,
            separate_zero_gap=time_bucket_zero_gap_separate,
            target_time_column=time_prediction_target if enable_time_prediction else None,
        )
        time_bucket_meta.update(time_bucket_meta_from_file)
        time_bucket_meta["use_time_attention_bias"] = bool(use_time_attention_bias)
        time_bucket_meta["time_attention_bias_bucket_count"] = (
            time_gap_bucket_count(
                boundaries,
                separate_zero_gap=time_bucket_zero_gap_separate,
            )
            if use_time_attention_bias
            else 0
        )
        for user_id, seq in user_sequences.items():
            time_items = time_item_sequences.get(user_id)
            time_seq = time_bucket_sequences.get(user_id)
            if time_items is None or time_seq is None:
                raise ValueError(f"Missing time-feature sequence for user_id={user_id}")
            if time_items != seq:
                raise ValueError(
                    f"Item sequence mismatch between interactions and time features for user_id={user_id}"
                )
            if enable_time_prediction and user_id not in next_time_sequences:
                raise ValueError(f"Missing next-time target sequence for user_id={user_id}")
        time_bucket_meta["enabled"] = True
    else:
        time_bucket_sequences = {}
        next_time_sequences = {}

    train, valid, test = {}, {}, {}
    for user_id, seq in user_sequences.items():
        time_seq = (
            time_bucket_sequences.get(user_id, [0] * len(seq))
            if (use_time_embedding or use_time_attention_bias)
            else [0] * len(seq)
        )
        next_time_seq = next_time_sequences.get(user_id, [0.0] * len(seq)) if enable_time_prediction else [0.0] * len(seq)
        if len(seq) < 4:
            train[user_id], valid[user_id], test[user_id] = seq, [], []
            train_time[user_id], valid_time[user_id], test_time[user_id] = time_seq, [], []
            train_next_time[user_id], valid_next_time[user_id], test_next_time[user_id] = next_time_seq, [], []
        else:
            train[user_id], valid[user_id], test[user_id] = seq[:-2], [seq[-2]], [seq[-1]]
            train_time[user_id], valid_time[user_id], test_time[user_id] = time_seq[:-2], [time_seq[-2]], [time_seq[-1]]
            train_next_time[user_id] = next_time_seq[:-2]
            valid_next_time[user_id] = [next_time_seq[len(train[user_id]) - 1]]
            test_next_time[user_id] = [next_time_seq[len(train[user_id])]]

    return train, valid, test, user_num, item_num, train_time, valid_time, test_time, time_bucket_meta


def summarize_dataset_splits(dataset):
    train, valid, test, user_num, item_num, *_ = dataset
    time_bucket_meta = dataset[8] if len(dataset) >= 9 and isinstance(dataset[8], dict) else {}
    train_users = sum(1 for u in range(1, user_num + 1) if len(train.get(u, [])) > 0)
    valid_users = sum(1 for u in range(1, user_num + 1) if len(valid.get(u, [])) > 0)
    test_users = sum(1 for u in range(1, user_num + 1) if len(test.get(u, [])) > 0)
    train_only_users = sum(
        1
        for u in range(1, user_num + 1)
        if len(train.get(u, [])) > 0 and len(valid.get(u, [])) == 0 and len(test.get(u, [])) == 0
    )
    train_interactions = int(sum(len(v) for v in train.values()))
    valid_interactions = int(sum(len(v) for v in valid.values()))
    test_interactions = int(sum(len(v) for v in test.values()))
    total_interactions = train_interactions + valid_interactions + test_interactions
    avg_train_len = train_interactions / max(train_users, 1)

    summary = {
        "users": int(user_num),
        "items": int(item_num),
        "train_users": int(train_users),
        "valid_users": int(valid_users),
        "test_users": int(test_users),
        "train_only_users": int(train_only_users),
        "users_with_eval_targets": int(valid_users),
        "train_interactions": train_interactions,
        "valid_interactions": valid_interactions,
        "test_interactions": test_interactions,
        "total_interactions": total_interactions,
        "train_ratio": train_interactions / max(total_interactions, 1),
        "valid_ratio": valid_interactions / max(total_interactions, 1),
        "test_ratio": test_interactions / max(total_interactions, 1),
        "avg_train_len": avg_train_len,
        "time_prediction_enabled": bool(time_bucket_meta.get("time_prediction_enabled", False)),
        "time_prediction_target": time_bucket_meta.get("time_prediction_target"),
    }
    if time_bucket_meta.get("enabled"):
        summary["time_embedding_enabled"] = bool(time_bucket_meta.get("use_time_embedding", False))
        summary["time_delta_column"] = time_bucket_meta.get("time_delta_column")
        summary["time_encoding"] = time_bucket_meta.get("time_encoding")
        summary["time_bucket_boundaries"] = time_bucket_meta.get("time_bucket_boundaries", [])
        summary["time_bucket_count"] = time_bucket_meta.get("time_bucket_count")
        summary["time_feature_dim"] = time_bucket_meta.get("time_feature_dim")
        summary["use_time_attention_bias"] = time_bucket_meta.get("use_time_attention_bias", False)
        summary["time_attention_bias_bucket_count"] = time_bucket_meta.get("time_attention_bias_bucket_count", 0)
        summary["time_modeling_mode"] = (
            "attention_bias"
            if time_bucket_meta.get("use_time_attention_bias", False)
            else ("input_embedding" if time_bucket_meta.get("use_time_embedding", False) else "disabled")
        )
    else:
        summary["time_embedding_enabled"] = False
        summary["use_time_attention_bias"] = False
        summary["time_modeling_mode"] = "disabled"
    return summary


def print_dataset_split_summary(stats: dict):
    print(
        "dataset split summary:"
        f" users={stats['users']}, items={stats['items']}, "
        f"train_users={stats['train_users']}, valid_users={stats['valid_users']}, test_users={stats['test_users']}, "
        f"train_only_users={stats['train_only_users']}"
    )
    print(
        "interaction split:"
        f" train={stats['train_interactions']} ({stats['train_ratio']:.2%}), "
        f"valid={stats['valid_interactions']} ({stats['valid_ratio']:.2%}), "
        f"test={stats['test_interactions']} ({stats['test_ratio']:.2%}), "
        f"total={stats['total_interactions']}"
    )
    print(
        f"avg_train_len={stats['avg_train_len']:.2f}, "
        f"users_with_eval_targets={stats['users_with_eval_targets']}"
    )


def random_neq(low: int, high: int, excluded: set[int]) -> int:
    item = np.random.randint(low, high)
    while item in excluded:
        item = np.random.randint(low, high)
    return item


class BatchSampler:
    """Small single-process sampler; simpler than the original multiprocessing sampler."""

    def __init__(
        self,
        user_train,
        user_num,
        item_num,
        batch_size,
        maxlen,
        user_train_time=None,
        user_train_next_time=None,
        enable_time_prediction: bool = False,
        time_target_transform: str = "none",
        seed=42,
    ):
        self.user_train = user_train
        self.user_train_time = user_train_time or {}
        self.user_train_next_time = user_train_next_time or {}
        self.user_num = user_num
        self.item_num = item_num
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.enable_time_prediction = enable_time_prediction
        self.time_target_transform = time_target_transform
        self.rng = np.random.default_rng(seed)
        self.users = [u for u in range(1, user_num + 1) if len(user_train.get(u, [])) > 1]
        if not self.users:
            raise ValueError("No users have enough training interactions for SASRec sampling.")

    def sample(self):
        users, seqs, poss, negs, time_seqs = [], [], [], [], []
        next_time_targets = []
        for _ in range(self.batch_size):
            user = int(self.rng.choice(self.users))
            sample = self._sample_user(user)
            if self.enable_time_prediction:
                seq, pos, neg, time_seq, next_time_target = sample
                next_time_targets.append(next_time_target)
            else:
                seq, pos, neg, time_seq = sample
            users.append(user)
            seqs.append(seq)
            poss.append(pos)
            negs.append(neg)
            time_seqs.append(time_seq)
        if self.enable_time_prediction:
            return (
                np.array(users),
                np.array(seqs),
                np.array(poss),
                np.array(negs),
                np.array(time_seqs),
                np.array(next_time_targets, dtype=np.float32),
            )
        return np.array(users), np.array(seqs), np.array(poss), np.array(negs), np.array(time_seqs)

    def _sample_user(self, user: int):
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros(self.maxlen, dtype=np.int32)
        items = self.user_train[user]
        time_items = self.user_train_time.get(user, [0] * len(items))
        next_time_items = self.user_train_next_time.get(user, [0.0] * len(items))
        sample_time_value = time_items[0] if time_items else 0
        if isinstance(sample_time_value, (tuple, list, np.ndarray)):
            time_seq = np.zeros((self.maxlen, len(sample_time_value)), dtype=np.float32)
        else:
            time_dtype = np.float32 if any(isinstance(v, float) for v in time_items) else np.int32
            time_seq = np.zeros(self.maxlen, dtype=time_dtype)
        next_time_target = np.zeros(self.maxlen, dtype=np.float32)
        nxt = items[-1]
        idx = self.maxlen - 1
        rated = set(items)

        for item, time_bucket, raw_next_time in reversed(list(zip(items[:-1], time_items[:-1], next_time_items[:-1]))):
            seq[idx] = item
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num + 1, rated)
            time_seq[idx] = time_bucket
            next_time_target[idx] = float(transform_time_value(raw_next_time, self.time_target_transform))
            nxt = item
            idx -= 1
            if idx == -1:
                break
        if self.enable_time_prediction:
            return seq, pos, neg, time_seq, next_time_target
        return seq, pos, neg, time_seq


def parse_topks(topk_spec) -> list[int]:
    if isinstance(topk_spec, int):
        return [topk_spec]
    if isinstance(topk_spec, (list, tuple)):
        values = [int(v) for v in topk_spec]
    else:
        values = [int(part.strip()) for part in str(topk_spec).split(",") if part.strip()]
    return sorted(set(v for v in values if v > 0)) or list(DEFAULT_TOPKS)


def _build_eval_sequence(train, valid, train_time, valid_time, user: int, args, split: str):
    seq = np.zeros(args.maxlen, dtype=np.int32)
    idx = args.maxlen - 1
    eval_source = train[user] + (valid[user] if split == "test" else [])
    eval_time_source = train_time.get(user, []) + (valid_time.get(user, []) if split == "test" else [])
    if len(eval_time_source) != len(eval_source):
        eval_time_source = [0] * len(eval_source)
    sample_time_value = eval_time_source[0] if eval_time_source else 0
    if isinstance(sample_time_value, (tuple, list, np.ndarray)):
        time_seq = np.zeros((args.maxlen, len(sample_time_value)), dtype=np.float32)
    else:
        time_dtype = np.float32 if any(isinstance(v, float) for v in eval_time_source) else np.int32
        time_seq = np.zeros(args.maxlen, dtype=time_dtype)
    for item, time_bucket in reversed(list(zip(eval_source, eval_time_source))):
        seq[idx] = item
        time_seq[idx] = time_bucket
        idx -= 1
        if idx == -1:
            break
    return seq, time_seq, eval_source


def _candidate_items(
    eval_source: list[int],
    exclusion_source: list[int],
    target_item: int,
    item_num: int,
    mode: str,
    num_negative_samples: int,
):
    rated = set(exclusion_source)
    rated.add(0)
    if mode == "sampled":
        item_idx = [target_item]
        for _ in range(num_negative_samples):
            item_idx.append(random_neq(1, item_num + 1, rated))
        return item_idx
    if mode == "full":
        negatives = [item for item in range(1, item_num + 1) if item not in rated and item != target_item]
        return [target_item] + negatives
    raise ValueError(f"Unknown evaluation mode: {mode}")


def _macro_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    f1_scores = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores))


def _ranking_metric_summary(ranks: list[int], topks: list[int]) -> dict:
    num_users = len(ranks)
    if num_users == 0:
        summary = {f"ndcg@{k}": 0.0 for k in topks}
        summary.update({f"hr@{k}": 0.0 for k in topks})
        summary.update({"mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0, "num_eval_users": 0})
        return summary

    summary = {}
    for k in topks:
        ndcg = 0.0
        hit = 0.0
        for rank in ranks:
            if rank < k:
                ndcg += 1 / np.log2(rank + 2)
                hit += 1
        summary[f"ndcg@{k}"] = ndcg / num_users
        summary[f"hr@{k}"] = hit / num_users

    reciprocal_ranks = [1.0 / (rank + 1) for rank in ranks]
    one_based_ranks = [rank + 1 for rank in ranks]
    summary["mrr"] = float(sum(reciprocal_ranks) / num_users)
    summary["mean_rank"] = float(sum(one_based_ranks) / num_users)
    summary["median_rank"] = float(median(one_based_ranks))
    summary["num_eval_users"] = int(num_users)
    return summary


def _classification_metric_summary(
    y_true_items: list[int],
    y_pred_items: list[int],
    topk_hits: dict[int, int],
    topks: list[int],
) -> dict:
    num_users = len(y_true_items)
    if num_users == 0:
        summary = {f"top{k}_accuracy": 0.0 for k in topks}
        summary.update({"accuracy": 0.0, "top1_accuracy": 0.0, "macro_f1": 0.0})
        return summary

    accuracy = float(sum(1 for truth, pred in zip(y_true_items, y_pred_items) if truth == pred) / num_users)
    summary = {f"top{k}_accuracy": float(topk_hits.get(k, 0) / num_users) for k in topks}
    summary["accuracy"] = accuracy
    summary["top1_accuracy"] = accuracy
    summary["macro_f1"] = _macro_f1_score(y_true_items, y_pred_items)
    return summary


def _time_metric_summary(y_true_time: list[float], y_pred_time: list[float]) -> dict:
    if not y_true_time:
        return {"time_mae": 0.0, "time_rmse": 0.0, "time_median_ae": 0.0}
    errors = np.abs(np.asarray(y_pred_time, dtype=np.float64) - np.asarray(y_true_time, dtype=np.float64))
    return {
        "time_mae": float(np.mean(errors)),
        "time_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "time_median_ae": float(np.median(errors)),
    }


def _sample_eval_users(user_num: int, args) -> list[int]:
    users = list(range(1, user_num + 1))
    if args.eval_users > 0 and len(users) > args.eval_users:
        users = random.sample(users, args.eval_users)
    return users


def evaluate_ranking(model, dataset, args, split: str = "test", mode: str = "sampled", topks: list[int] | None = None, users: list[int] | None = None):
    train, valid, test, user_num, item_num, train_time, valid_time, _, _ = dataset
    target_dict = test if split == "test" else valid
    topks = topks or parse_topks(getattr(args, "topk_list", getattr(args, "topk", DEFAULT_TOPKS)))
    users = users or _sample_eval_users(user_num, args)

    ranks = []
    model.eval()
    with torch.no_grad():
        for user in users:
            if len(train.get(user, [])) < 1 or len(target_dict.get(user, [])) < 1:
                continue

            seq, time_seq, eval_source = _build_eval_sequence(train, valid, train_time, valid_time, user, args, split)
            target_item = target_dict[user][0]
            exclusion_source = train[user] if mode == "sampled" else eval_source
            item_idx = _candidate_items(
                eval_source,
                exclusion_source,
                target_item,
                item_num,
                mode,
                args.num_negative_samples,
            )
            rank_scores = model.predict(np.array([user]), np.array([seq]), item_idx, np.array([time_seq]))[0]
            rank_scores = rank_scores.detach().cpu().numpy()
            rank = (-rank_scores).argsort().argsort()[0].item()
            ranks.append(rank)

    return _ranking_metric_summary(ranks, topks)


def evaluate_shared(model, dataset, args, split: str = "test", topks: list[int] | None = None, users: list[int] | None = None):
    train, valid, test, user_num, item_num, train_time, valid_time, _, time_meta = dataset
    target_dict = test if split == "test" else valid
    time_target_dict = time_meta.get("next_time_targets", {}).get(split, {})
    topks = topks or parse_topks(getattr(args, "topk_list", getattr(args, "topk", DEFAULT_TOPKS)))
    users = users or _sample_eval_users(user_num, args)
    all_items = np.arange(1, item_num + 1, dtype=np.int32)

    y_true_items = []
    y_pred_items = []
    topk_hits = {k: 0 for k in topks}
    y_true_time = []
    y_pred_time = []
    model.eval()
    with torch.no_grad():
        for user in users:
            if len(train.get(user, [])) < 1 or len(target_dict.get(user, [])) < 1:
                continue

            seq, time_seq, _ = _build_eval_sequence(train, valid, train_time, valid_time, user, args, split)
            target_item = target_dict[user][0]

            class_scores = model.predict(np.array([user]), np.array([seq]), all_items, np.array([time_seq]))[0]
            class_scores = class_scores.detach().cpu().numpy()
            ranked_indices = np.argsort(-class_scores)
            predicted_item = int(all_items[ranked_indices[0]])
            y_true_items.append(target_item)
            y_pred_items.append(predicted_item)
            for k in topks:
                top_items = all_items[ranked_indices[:k]]
                if int(target_item) in top_items:
                    topk_hits[k] += 1

            if getattr(args, "enable_time_prediction", False):
                raw_targets = time_target_dict.get(user, [])
                if raw_targets:
                    pred_time = model.predict_next_time(np.array([user]), np.array([seq]), np.array([time_seq]))
                    pred_time_value = float(pred_time.detach().cpu().reshape(-1)[0].item())
                    pred_time_value = float(inverse_time_value(pred_time_value, getattr(args, "time_target_transform", "none")))
                    y_pred_time.append(pred_time_value)
                    y_true_time.append(float(raw_targets[0]))

    summary = _classification_metric_summary(y_true_items, y_pred_items, topk_hits, topks)
    if getattr(args, "enable_time_prediction", False):
        summary.update(_time_metric_summary(y_true_time, y_pred_time))
    return summary


def evaluate_all(model, dataset, args, split: str = "test", topks: list[int] | None = None):
    topks = topks or parse_topks(getattr(args, "topk_list", getattr(args, "topk", DEFAULT_TOPKS)))
    _, _, _, user_num, *_ = dataset
    users = _sample_eval_users(user_num, args)
    modes = ["full"] if getattr(args, "eval_protocol", "both") == "full" else []
    if getattr(args, "eval_protocol", "both") in {"sampled", "both"}:
        if "sampled" not in modes:
            modes.append("sampled")
    if getattr(args, "eval_protocol", "both") == "both" and "full" not in modes:
        modes.insert(0, "full")

    results = {"shared": evaluate_shared(model, dataset, args, split=split, topks=topks, users=users)}
    for mode in modes:
        results[mode] = evaluate_ranking(model, dataset, args, split=split, mode=mode, topks=topks, users=users)
    return results


def recommend_topk(model, user_id: int, dataset, args, topk: int | None = None):
    train, valid, test, _, item_num, train_time, valid_time, _, _ = dataset
    topk = topk or getattr(args, "topk", max(parse_topks(getattr(args, "topk_list", DEFAULT_TOPKS))))
    history = train.get(user_id, []) + valid.get(user_id, [])
    history_time = train_time.get(user_id, []) + valid_time.get(user_id, [])
    if not history:
        raise ValueError(f"User {user_id} has no history for inference.")
    if len(history_time) != len(history):
        history_time = [0] * len(history)

    seq = np.zeros(args.maxlen, dtype=np.int32)
    sample_time_value = history_time[0] if history_time else 0
    if isinstance(sample_time_value, (tuple, list, np.ndarray)):
        time_seq = np.zeros((args.maxlen, len(sample_time_value)), dtype=np.float32)
    else:
        time_dtype = np.float32 if any(isinstance(v, float) for v in history_time) else np.int32
        time_seq = np.zeros(args.maxlen, dtype=time_dtype)
    idx = args.maxlen - 1
    for item, time_bucket in reversed(list(zip(history, history_time))):
        seq[idx] = item
        time_seq[idx] = time_bucket
        idx -= 1
        if idx == -1:
            break

    seen = set(history)
    candidates = [item for item in range(1, item_num + 1) if item not in seen]
    model.eval()
    with torch.no_grad():
        scores = model.predict(np.array([user_id]), np.array([seq]), candidates, np.array([time_seq]))[0]
    top_indices = torch.topk(scores, k=min(topk, len(candidates))).indices.cpu().numpy()
    return [(candidates[i], float(scores[i].detach().cpu())) for i in top_indices]

