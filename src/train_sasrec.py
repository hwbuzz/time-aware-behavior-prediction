from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

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

    dataset = load_sasrec_dataset(args.interactions_path)
    user_train, _, _, user_num, item_num = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    avg_len = sum(len(v) for v in user_train.values()) / max(len(user_train), 1)
    print(f"users={user_num}, items={item_num}, avg_train_len={avg_len:.2f}, batches={num_batch}")

    model = init_model(user_num, item_num, args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if args.inference_only:
        val = evaluate(model, dataset, args, split="valid")
        test = evaluate(model, dataset, args, split="test")
        print(f"valid NDCG@{args.topk}: {val[0]:.4f}, HR@{args.topk}: {val[1]:.4f}")
        print(f"test  NDCG@{args.topk}: {test[0]:.4f}, HR@{args.topk}: {test[1]:.4f}")
        if args.recommend_user is not None:
            print("recommendations:", recommend_topk(model, args.recommend_user, dataset, args))
        return

    sampler = BatchSampler(user_train, user_num, item_num, args.batch_size, args.maxlen, args.seed)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val = -1.0
    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(model, sampler, optimizer, criterion, args, num_batch)
        print(f"epoch={epoch}, loss={loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            val = evaluate(model, dataset, args, split="valid")
            test = evaluate(model, dataset, args, split="test")
            print(
                f"epoch={epoch}, valid NDCG@{args.topk}: {val[0]:.4f}, HR@{args.topk}: {val[1]:.4f}, "
                f"test NDCG@{args.topk}: {test[0]:.4f}, HR@{args.topk}: {test[1]:.4f}"
            )
            if val[0] > best_val:
                best_val = val[0]
                ckpt = output_dir / "sasrec_best.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"saved checkpoint: {ckpt}")

    final_ckpt = output_dir / "sasrec_last.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"saved checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()

