from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn


@dataclass
class EncoderConfig:
    event_dim: int = 64
    graph_dim: int = 32
    artifact_dim: int = 32
    attr_dim: int = 16
    fused_dim: int = 64


class EventSequenceEncoder(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, max_types: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.type_emb = nn.Embedding(max_types, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(
        self, type_ids: Tensor, values: Tensor, deltas: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # type_ids: [B, T] long, values/deltas: [B, T, 1]
        h_type = self.type_emb(type_ids)
        h_val = self.value_proj(values)
        h_time = self.time_proj(deltas)
        h = h_type + h_val + h_time
        attn_out, attn_weights = self.attn(
            h, h, h, need_weights=True, average_attn_weights=False
        )
        h2 = self.layernorm(h + attn_out)
        h3 = self.layernorm(h2 + self.ff(h2))
        # Pool by mean over time
        pooled = h3.mean(dim=1)
        # attn_weights: [B, n_heads, T, T]; we return mean over heads and keys
        attn_tokens = attn_weights.mean(dim=1).mean(dim=2)
        return pooled, attn_tokens


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, self_emb: Tensor, neighbor_embs: Tensor, neighbor_mask: Tensor) -> Tensor:
        # self_emb: [B, D], neighbor_embs: [B, K, D], neighbor_mask: [B, K]
        masked = neighbor_embs * neighbor_mask.unsqueeze(-1)
        sums = masked.sum(dim=1)
        counts = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = sums / counts
        concat = torch.cat([self_emb, mean], dim=-1)
        return self.mlp(concat)


class ArtifactEncoder(nn.Module):
    def __init__(self, feat_dim: int, output_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, feats: Tensor) -> Tensor:
        # feats: [B, A, F]
        if feats.numel() == 0:
            return torch.zeros(feats.size(0), self.mlp[0].out_features, device=feats.device)
        mean = feats.mean(dim=1)
        max_v, _ = feats.max(dim=1)
        concat = torch.cat([mean, max_v], dim=-1)
        return self.mlp(concat)


class AttrEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MultiTaskHead(nn.Module):
    def __init__(self, fused_dim: int) -> None:
        super().__init__()
        self.reg_head = nn.Linear(fused_dim, 1)
        self.cls_head = nn.Linear(fused_dim, 1)
        self.rank_head = nn.Linear(fused_dim, 1)

    def forward(self, fused: Tensor) -> Dict[str, Tensor]:
        reg = self.reg_head(fused).squeeze(-1)
        logit = self.cls_head(fused).squeeze(-1)
        score = self.rank_head(fused).squeeze(-1)
        return {"regression": reg, "logit": logit, "score": score}


class FullModel(nn.Module):
    def __init__(
        self, encoder_cfg: EncoderConfig, attr_input_dim: int, artifact_feat_dim: int
    ) -> None:
        super().__init__()
        self.event_encoder = EventSequenceEncoder(d_model=encoder_cfg.event_dim)
        self.attr_encoder = AttrEncoder(in_dim=attr_input_dim, out_dim=encoder_cfg.attr_dim)
        self.graph_encoder = GraphEncoder(
            input_dim=encoder_cfg.attr_dim * 2, output_dim=encoder_cfg.graph_dim
        )
        self.artifact_encoder = ArtifactEncoder(
            feat_dim=artifact_feat_dim, output_dim=encoder_cfg.artifact_dim
        )
        self.fused_proj = nn.Linear(
            encoder_cfg.event_dim
            + encoder_cfg.graph_dim
            + encoder_cfg.artifact_dim
            + encoder_cfg.attr_dim,
            encoder_cfg.fused_dim,
        )
        self.head = MultiTaskHead(fused_dim=encoder_cfg.fused_dim)

    def forward(
        self,
        *,
        event_type_ids: Tensor,
        event_values: Tensor,
        event_deltas: Tensor,
        attr_vec: Tensor,
        neighbor_attr: Tensor,
        neighbor_mask: Tensor,
        artifact_feats: Tensor,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        seq_emb, attn = self.event_encoder(event_type_ids, event_values, event_deltas)
        attr_emb = self.attr_encoder(attr_vec)
        graph_emb = self.graph_encoder(attr_emb, neighbor_attr, neighbor_mask)
        art_emb = self.artifact_encoder(artifact_feats)
        fused = torch.cat([seq_emb, graph_emb, art_emb, attr_emb], dim=-1)
        fused_emb = self.fused_proj(fused)
        outputs = self.head(fused_emb)
        return outputs, fused_emb, attn


def pairwise_ranking_loss(scores: Tensor, targets: Tensor) -> Tensor:
    # Deterministic pairwise hinge loss: iterate in sorted index order
    n = scores.size(0)
    loss = torch.tensor(0.0, device=scores.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ti = targets[i]
            tj = targets[j]
            if ti == tj:
                continue
            margin = 1.0
            if ti > tj:
                diff = margin - (scores[i] - scores[j])
            else:
                diff = margin - (scores[j] - scores[i])
            loss = loss + torch.clamp(diff, min=0.0)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=scores.device)
    return loss / float(count)


def regression_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    var = target.var().item()
    r2 = 1.0 - mse / var if var > 0 else 0.0
    rmse = mse**0.5
    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(logit: Tensor, target: Tensor) -> Dict[str, float]:
    prob = torch.sigmoid(logit)
    pred = (prob >= 0.5).float()
    tp = float(((pred == 1.0) & (target == 1.0)).sum().item())
    fp = float(((pred == 1.0) & (target == 0.0)).sum().item())
    fn = float(((pred == 0.0) & (target == 1.0)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # For simplicity, we approximate ROC AUC and PR AUC
    # with threshold-0.5 values here.
    return {"f1": f1, "precision": precision, "recall": recall}


def ranking_metrics(scores: Tensor, targets: Tensor) -> Dict[str, float]:
    # NDCG@K with K = min(10, n)
    n = scores.size(0)
    if n == 0:
        return {"ndcg@10": 0.0, "spearman": 0.0}
    k = min(10, n)
    _, idx_pred = scores.sort(descending=True)
    _, idx_true = targets.sort(descending=True)

    gains = 2 ** targets[idx_pred[:k]] - 1
    discounts = torch.log2(
        torch.arange(2, k + 2, dtype=torch.float32, device=scores.device)
    )
    dcg = (gains / discounts).sum().item()

    ideal_gains = 2 ** targets[idx_true[:k]] - 1
    ideal_dcg = (ideal_gains / discounts).sum().item()
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # Spearman rank correlation (simple implementation)
    rank_pred = torch.zeros_like(scores)
    rank_pred[idx_pred] = torch.arange(
        1, n + 1, dtype=torch.float32, device=scores.device
    )
    rank_true = torch.zeros_like(scores)
    rank_true[idx_true] = torch.arange(
        1, n + 1, dtype=torch.float32, device=scores.device
    )
    cov = ((rank_pred - rank_pred.mean()) * (rank_true - rank_true.mean())).mean().item()
    var_p = rank_pred.var().item()
    var_t = rank_true.var().item()
    denom = (var_p * var_t) ** 0.5
    spearman = cov / denom if denom > 0 else 0.0

    return {"ndcg@10": ndcg, "spearman": spearman}
