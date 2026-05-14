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


def load_time_feature_sequences(
    time_features_path: str,
    time_delta_column: str,
    time_encoding: str,
    boundaries: list[float],
    separate_first_event: bool = True,
    separate_zero_gap: bool = True,
):
    rows_by_user: dict[int, list[tuple[int, int, object]]] = defaultdict(list)
    with open(time_features_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"user_id", "item_id", "event_idx", time_delta_column}
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
            rows_by_user[user_id].append((event_idx, item_id, time_value))

    item_sequences = {}
    time_sequences = {}
    for user_id, rows in rows_by_user.items():
        rows.sort(key=lambda x: x[0])
        item_sequences[user_id] = [item_id for _, item_id, _ in rows]
        time_sequences[user_id] = [bucket_idx for _, _, bucket_idx in rows]

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
    }
    return item_sequences, time_sequences, bucket_meta


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
    time_bucket_meta = {"enabled": False}
    if use_time_embedding or use_time_attention_bias:
        boundaries = parse_time_bucket_boundaries(time_bucket_boundaries)
        resolved_time_features_path = time_features_path or default_time_features_path(interactions_path)
        effective_time_encoding = "raw" if use_time_attention_bias else time_encoding
        time_item_sequences, time_bucket_sequences, time_bucket_meta = load_time_feature_sequences(
            resolved_time_features_path,
            time_delta_column=time_delta_column,
            time_encoding=effective_time_encoding,
            boundaries=boundaries,
            separate_first_event=time_bucket_first_event_separate,
            separate_zero_gap=time_bucket_zero_gap_separate,
        )
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
        time_bucket_meta["enabled"] = True

    train, valid, test = {}, {}, {}
    for user_id, seq in user_sequences.items():
        time_seq = (
            time_bucket_sequences.get(user_id, [0] * len(seq))
            if (use_time_embedding or use_time_attention_bias)
            else [0] * len(seq)
        )
        if len(seq) < 4:
            train[user_id], valid[user_id], test[user_id] = seq, [], []
            train_time[user_id], valid_time[user_id], test_time[user_id] = time_seq, [], []
        else:
            train[user_id], valid[user_id], test[user_id] = seq[:-2], [seq[-2]], [seq[-1]]
            train_time[user_id], valid_time[user_id], test_time[user_id] = time_seq[:-2], [time_seq[-2]], [time_seq[-1]]

    return train, valid, test, user_num, item_num, train_time, valid_time, test_time, time_bucket_meta


def summarize_dataset_splits(dataset):
    train, valid, test, user_num, item_num, *rest = dataset
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
    }
    if len(rest) >= 4:
        time_bucket_meta = rest[3]
        if time_bucket_meta.get("enabled"):
            summary["time_embedding_enabled"] = not time_bucket_meta.get("use_time_attention_bias", False)
            summary["time_delta_column"] = time_bucket_meta.get("time_delta_column")
            summary["time_encoding"] = time_bucket_meta.get("time_encoding")
            summary["time_bucket_boundaries"] = time_bucket_meta.get("time_bucket_boundaries", [])
            summary["time_bucket_count"] = time_bucket_meta.get("time_bucket_count")
            summary["time_feature_dim"] = time_bucket_meta.get("time_feature_dim")
            summary["use_time_attention_bias"] = time_bucket_meta.get("use_time_attention_bias", False)
            summary["time_attention_bias_bucket_count"] = time_bucket_meta.get("time_attention_bias_bucket_count", 0)
            summary["time_modeling_mode"] = (
                "attention_bias" if time_bucket_meta.get("use_time_attention_bias", False) else "input_embedding"
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

    def __init__(self, user_train, user_num, item_num, batch_size, maxlen, user_train_time=None, seed=42):
        self.user_train = user_train
        self.user_train_time = user_train_time or {}
        self.user_num = user_num
        self.item_num = item_num
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)
        self.users = [u for u in range(1, user_num + 1) if len(user_train.get(u, [])) > 1]
        if not self.users:
            raise ValueError("No users have enough training interactions for SASRec sampling.")

    def sample(self):
        users, seqs, poss, negs, time_seqs = [], [], [], [], []
        for _ in range(self.batch_size):
            user = int(self.rng.choice(self.users))
            seq, pos, neg, time_seq = self._sample_user(user)
            users.append(user)
            seqs.append(seq)
            poss.append(pos)
            negs.append(neg)
            time_seqs.append(time_seq)
        return np.array(users), np.array(seqs), np.array(poss), np.array(negs), np.array(time_seqs)

    def _sample_user(self, user: int):
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros(self.maxlen, dtype=np.int32)
        items = self.user_train[user]
        time_items = self.user_train_time.get(user, [0] * len(items))
        sample_time_value = time_items[0] if time_items else 0
        if isinstance(sample_time_value, (tuple, list, np.ndarray)):
            time_seq = np.zeros((self.maxlen, len(sample_time_value)), dtype=np.float32)
        else:
            time_dtype = np.float32 if any(isinstance(v, float) for v in time_items) else np.int32
            time_seq = np.zeros(self.maxlen, dtype=time_dtype)
        nxt = items[-1]
        idx = self.maxlen - 1
        rated = set(items)

        for item, time_bucket in reversed(list(zip(items[:-1], time_items[:-1]))):
            seq[idx] = item
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num + 1, rated)
            time_seq[idx] = time_bucket
            nxt = item
            idx -= 1
            if idx == -1:
                break
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


def _metric_summary_from_ranks(ranks: list[int], topks: list[int]) -> dict:
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


def evaluate(model, dataset, args, split: str = "test", mode: str = "sampled", topks: list[int] | None = None):
    train, valid, test, user_num, item_num, train_time, valid_time, _, _ = dataset
    target_dict = test if split == "test" else valid
    topks = topks or parse_topks(getattr(args, "topk_list", getattr(args, "topk", DEFAULT_TOPKS)))

    users = list(range(1, user_num + 1))
    if args.eval_users > 0 and len(users) > args.eval_users:
        users = random.sample(users, args.eval_users)

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
            predictions = -model.predict(np.array([user]), np.array([seq]), item_idx, np.array([time_seq]))[0]
            rank = predictions.argsort().argsort()[0].item()
            ranks.append(rank)

    return _metric_summary_from_ranks(ranks, topks)


def evaluate_all(model, dataset, args, split: str = "test", topks: list[int] | None = None):
    topks = topks or parse_topks(getattr(args, "topk_list", getattr(args, "topk", DEFAULT_TOPKS)))
    modes = ["full"] if getattr(args, "eval_protocol", "both") == "full" else []
    if getattr(args, "eval_protocol", "both") in {"sampled", "both"}:
        if "sampled" not in modes:
            modes.append("sampled")
    if getattr(args, "eval_protocol", "both") == "both" and "full" not in modes:
        modes.insert(0, "full")
    return {mode: evaluate(model, dataset, args, split=split, mode=mode, topks=topks) for mode in modes}


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
