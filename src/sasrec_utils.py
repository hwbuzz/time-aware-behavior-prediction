from __future__ import annotations

from collections import defaultdict
import random

import numpy as np
import torch


def load_sasrec_dataset(interactions_path: str):
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

    train, valid, test = {}, {}, {}
    for user_id, seq in user_sequences.items():
        if len(seq) < 4:
            train[user_id], valid[user_id], test[user_id] = seq, [], []
        else:
            train[user_id], valid[user_id], test[user_id] = seq[:-2], [seq[-2]], [seq[-1]]

    return train, valid, test, user_num, item_num


def summarize_dataset_splits(dataset):
    train, valid, test, user_num, item_num = dataset
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

    return {
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

    def __init__(self, user_train, user_num, item_num, batch_size, maxlen, seed=42):
        self.user_train = user_train
        self.user_num = user_num
        self.item_num = item_num
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)
        self.users = [u for u in range(1, user_num + 1) if len(user_train.get(u, [])) > 1]
        if not self.users:
            raise ValueError("No users have enough training interactions for SASRec sampling.")

    def sample(self):
        users, seqs, poss, negs = [], [], [], []
        for _ in range(self.batch_size):
            user = int(self.rng.choice(self.users))
            seq, pos, neg = self._sample_user(user)
            users.append(user)
            seqs.append(seq)
            poss.append(pos)
            negs.append(neg)
        return np.array(users), np.array(seqs), np.array(poss), np.array(negs)

    def _sample_user(self, user: int):
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros(self.maxlen, dtype=np.int32)
        items = self.user_train[user]
        nxt = items[-1]
        idx = self.maxlen - 1
        rated = set(items)

        for item in reversed(items[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num + 1, rated)
            nxt = item
            idx -= 1
            if idx == -1:
                break
        return seq, pos, neg


def evaluate(model, dataset, args, split: str = "test"):
    train, valid, test, user_num, item_num = dataset
    target_dict = test if split == "test" else valid
    ndcg = 0.0
    hit = 0.0
    valid_user = 0

    users = list(range(1, user_num + 1))
    if args.eval_users > 0 and len(users) > args.eval_users:
        users = random.sample(users, args.eval_users)

    model.eval()
    with torch.no_grad():
        for user in users:
            if len(train.get(user, [])) < 1 or len(target_dict.get(user, [])) < 1:
                continue

            seq = np.zeros(args.maxlen, dtype=np.int32)
            idx = args.maxlen - 1
            eval_source = train[user] + (valid[user] if split == "test" else [])
            for item in reversed(eval_source):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break

            rated = set(eval_source)
            rated.add(0)
            item_idx = [target_dict[user][0]]
            for _ in range(args.num_negative_samples):
                item_idx.append(random_neq(1, item_num + 1, rated))

            predictions = -model.predict(np.array([user]), np.array([seq]), item_idx)[0]
            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1
            if rank < args.topk:
                ndcg += 1 / np.log2(rank + 2)
                hit += 1

    return (ndcg / valid_user, hit / valid_user) if valid_user else (0.0, 0.0)


def recommend_topk(model, user_id: int, dataset, args, topk: int | None = None):
    train, valid, test, _, item_num = dataset
    topk = topk or args.topk
    history = train.get(user_id, []) + valid.get(user_id, [])
    if not history:
        raise ValueError(f"User {user_id} has no history for inference.")

    seq = np.zeros(args.maxlen, dtype=np.int32)
    idx = args.maxlen - 1
    for item in reversed(history):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break

    seen = set(history)
    candidates = [item for item in range(1, item_num + 1) if item not in seen]
    model.eval()
    with torch.no_grad():
        scores = model.predict(np.array([user_id]), np.array([seq]), candidates)[0]
    top_indices = torch.topk(scores, k=min(topk, len(candidates))).indices.cpu().numpy()
    return [(candidates[i], float(scores[i].detach().cpu())) for i in top_indices]
