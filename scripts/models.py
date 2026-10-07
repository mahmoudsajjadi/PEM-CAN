"""
PEM-CAN: Parameter-Efficient Multimodal Cross-Attention Network
PyTorch implementation of architectures, Stiefel-manifold low-rank adapters,
and clinical/PEFT baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubspaceLoRALinear(nn.Module):
    """
    Subspace-Constrained Low-Rank Adaptation (LoRA) Layer
    W = W_0 + (alpha / r) * (B @ A)
    constrained by Stiefel-manifold Orthogonal Regularization: || A @ A^T - I_r ||_F^2.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen foundation base weight W_0
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Trainable low-rank decomposition factors A and B
        # A is initialized with orthonormal rows (Stiefel manifold basis)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # QR orthonormal initialization for Stiefel projection
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(in_features, rank))
            self.lora_A.copy_(q.t())

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def orthogonal_loss(self) -> torch.Tensor:
        """Computes Stiefel-manifold orthogonality loss: || A @ A^T - I_r ||_F^2"""
        aat = torch.matmul(self.lora_A, self.lora_A.t())
        identity = torch.eye(self.rank, device=self.lora_A.device)
        return torch.norm(aat - identity, p='fro') ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class StandardLoRALinear(nn.Module):
    """Unregularized Standard LoRA baseline (Hu et al., 2022)."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class BottleneckAdapter(nn.Module):
    """Houlsby / Pfeiffer Bottleneck Adapter baseline."""
    def __init__(self, embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class TemporalConvEncoder(nn.Module):
    """Multiscale 1D Dilated Convolutional Encoder for CGM & Actigraphy."""
    def __init__(self, in_channels: int, embed_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 384, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(384)
        self.proj = nn.Linear(384, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        return self.proj(h.transpose(1, 2))


class ParameterEfficientCrossAttention(nn.Module):
    """Bidirectional Cross-Attention with Low-Rank Subspace Projections (98,304 adapter params)."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, rank: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = SubspaceLoRALinear(embed_dim, embed_dim, rank=rank)
        self.k_proj = SubspaceLoRALinear(embed_dim, embed_dim, rank=rank)
        self.v_proj = SubspaceLoRALinear(embed_dim, embed_dim, rank=rank)
        self.out_proj = SubspaceLoRALinear(embed_dim, embed_dim, rank=rank)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        B, N_q, D = query.shape
        _, N_kv, _ = key_value.shape

        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        return self.out_proj(out)

    def get_orthogonal_penalty(self) -> torch.Tensor:
        return (self.q_proj.orthogonal_loss() + self.k_proj.orthogonal_loss() +
                self.v_proj.orthogonal_loss() + self.out_proj.orthogonal_loss())


class PEMCAN(nn.Module):
    """
    Complete Parameter-Efficient Multimodal Cross-Attention Network.
    Tasks: [DAN, DR Grade >= 2, CKD Stage >= 3, DPN]
    """
    def __init__(self, embed_dim: int = 768, rank: int = 8, num_tasks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.cgm_encoder = TemporalConvEncoder(in_channels=1, embed_dim=embed_dim)
        self.act_encoder = TemporalConvEncoder(in_channels=3, embed_dim=embed_dim)
        
        # Temporal Bottleneck Fusion (0.45M parameter budget)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 192),
            nn.GELU(),
            nn.Linear(192, embed_dim)
        )
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_s = nn.LayerNorm(embed_dim)

        # Bidirectional cross-modal low-rank attention
        self.cross_attn_v2s = ParameterEfficientCrossAttention(embed_dim, rank=rank)
        self.cross_attn_s2v = ParameterEfficientCrossAttention(embed_dim, rank=rank)

        # Multi-task classification head (0.20M parameter budget)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_tasks)
        )

    def forward(self, x_v: torch.Tensor, x_g: torch.Tensor, x_a: torch.Tensor):
        z_v = self.norm_v(x_v)

        h_g = self.cgm_encoder(x_g)
        h_a = self.act_encoder(x_a)

        if h_g.size(1) != h_a.size(1):
            h_a = F.interpolate(h_a.transpose(1, 2), size=h_g.size(1), mode='linear', align_corners=False).transpose(1, 2)

        z_s = self.norm_s(self.temporal_fusion(torch.cat([h_g, h_a], dim=-1)))

        s_v2s = self.cross_attn_v2s(z_v, z_s)
        s_s2v = self.cross_attn_s2v(z_s, z_v)

        pooled_v = s_v2s.mean(dim=1)
        pooled_s = s_s2v.mean(dim=1)

        fused = torch.cat([pooled_v, pooled_s], dim=-1)
        logits = self.classifier(fused)
        return logits, pooled_v, pooled_s

    def compute_auxiliary_loss(self, pooled_v: torch.Tensor, pooled_s: torch.Tensor,
                               lambda_orth: float = 1e-3, lambda_align: float = 1e-2) -> torch.Tensor:
        """
        Computes Stiefel orthogonality loss and cross-modal cosine alignment loss.
        """
        loss_orth = self.cross_attn_v2s.get_orthogonal_penalty() + self.cross_attn_s2v.get_orthogonal_penalty()
        cos_sim = F.cosine_similarity(pooled_v, pooled_s, dim=-1).mean()
        loss_align = 1.0 - cos_sim
        return lambda_orth * loss_orth + lambda_align * loss_align


class ClinicalTabularBaseline(nn.Module):
    """Clinical Tabular Baseline (MLP on 10 risk factors)."""
    def __init__(self, in_features: int = 10, num_tasks: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_tasks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EarlyConcatenationBaseline(nn.Module):
    """Early Concatenation MLP baseline."""
    def __init__(self, embed_dim: int = 768, num_tasks: int = 4):
        super().__init__()
        self.cgm_pool = nn.AdaptiveAvgPool1d(embed_dim)
        self.act_pool = nn.AdaptiveAvgPool1d(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_tasks)
        )

    def forward(self, x_v: torch.Tensor, x_g: torch.Tensor, x_a: torch.Tensor):
        v = x_v.mean(dim=1)
        g = self.cgm_pool(x_g).squeeze(1)
        a = self.act_pool(x_a.mean(dim=1, keepdim=True)).squeeze(1)
        x = torch.cat([v, g, a], dim=-1)
        return self.mlp(x)


def parameter_breakdown(model: nn.Module, foundation_backbone_params: int = 86_600_000):
    """Prints fine-grained parameter breakdown across components."""
    total_model = foundation_backbone_params + sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    tokenizers = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad)
    fusion_clf = trainable - adapters - tokenizers
    return {
        'foundation_backbone': foundation_backbone_params,
        'total_parameters': total_model,
        'trainable_parameters': trainable,
        'adapter_parameters': adapters,
        'tokenizer_parameters': tokenizers,
        'fusion_and_heads': fusion_clf,
        'trainable_ratio_percent': (trainable / total_model) * 100.0 if total_model > 0 else 0.0
    }
