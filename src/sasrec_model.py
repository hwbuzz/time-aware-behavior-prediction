from __future__ import annotations

import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    """Point-wise feed-forward block from SASRec.pytorch."""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        return outputs.transpose(-1, -2)


class SASRec(torch.nn.Module):
    """SASRec model adapted from pmixer/SASRec.pytorch."""

    def __init__(self, user_num: int, item_num: int, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.use_time_embedding = getattr(args, "use_time_embedding", False)
        self.use_time_attention_bias = getattr(args, "use_time_attention_bias", False)
        self.time_encoding = getattr(args, "time_encoding", "bucket")
        self.num_heads = args.num_heads
        self.time_bucket_zero_gap_separate = getattr(args, "time_bucket_zero_gap_separate", True)

        self.item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        if self.use_time_embedding:
            if self.time_encoding == "bucket":
                self.time_emb = torch.nn.Embedding(args.time_bucket_count, args.hidden_units, padding_idx=0)
            elif self.time_encoding == "continuous":
                self.time_proj = torch.nn.Linear(getattr(args, "time_feature_dim", 2), args.hidden_units)
            else:
                raise ValueError(f"Unknown time_encoding: {self.time_encoding}")
        if self.use_time_attention_bias:
            self.time_attn_boundaries = torch.tensor(
                getattr(args, "time_bucket_boundaries_parsed", []),
                dtype=torch.float32,
            )
            self.time_attn_bias = torch.nn.Embedding(
                getattr(args, "time_attention_bias_bucket_count", 0),
                1,
            )
        self.emb_dropout = torch.nn.Dropout(args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def _build_attention_mask(self, time_seqs: np.ndarray | None, seq_len: int) -> torch.Tensor:
        future_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=self.dev)
        future_mask.masked_fill_(
            torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev), diagonal=1),
            float("-inf"),
        )
        if not self.use_time_attention_bias:
            return future_mask
        if time_seqs is None:
            raise ValueError("time_seqs must be provided when use_time_attention_bias=True")

        time_tensor = torch.as_tensor(time_seqs, dtype=torch.float32, device=self.dev)
        if time_tensor.ndim == 3:
            time_tensor = time_tensor[..., 0]
        gap = time_tensor.unsqueeze(2) - time_tensor.unsqueeze(1)
        gap = torch.clamp(gap, min=0.0)

        boundaries = self.time_attn_boundaries.to(self.dev)
        bucket_idx = torch.bucketize(gap, boundaries, right=False)
        if self.time_bucket_zero_gap_separate:
            bucket_idx = bucket_idx + (gap > 0).long()

        bias = self.time_attn_bias(bucket_idx).squeeze(-1)
        bias = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        bias = bias.reshape(-1, seq_len, seq_len)
        return bias + future_mask.unsqueeze(0)

    def log2feats(self, log_seqs: np.ndarray, time_seqs: np.ndarray | None = None) -> torch.Tensor:
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        positions *= log_seqs != 0
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        if self.use_time_embedding:
            if time_seqs is None:
                raise ValueError("time_seqs must be provided when use_time_embedding=True")
            if self.time_encoding == "bucket":
                seqs += self.time_emb(torch.LongTensor(time_seqs).to(self.dev))
            else:
                time_tensor = torch.as_tensor(time_seqs, dtype=torch.float32, device=self.dev)
                if time_tensor.ndim == 2:
                    time_tensor = time_tensor.unsqueeze(-1)
                seqs += self.time_proj(time_tensor)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = self._build_attention_mask(time_seqs, tl)

        for i, attention_layer in enumerate(self.attention_layers):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = attention_layer(x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = attention_layer(seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, time_seqs=None):
        log_feats = self.log2feats(log_seqs, time_seqs=time_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, time_seqs=None):
        log_feats = self.log2feats(log_seqs, time_seqs=time_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
