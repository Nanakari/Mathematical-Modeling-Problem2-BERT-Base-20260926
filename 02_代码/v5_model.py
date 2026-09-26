"""Composable, mask-aware multimodal model for the v5 teacher/student study.

Text tokens and the 50-position acoustic/visual streams are encoded separately.
Cross-attention works between sequences of different lengths; there is no
position-wise text/audio/video addition or fabricated token alignment.
"""
from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


MODEL_DEFAULTS = {
    "fusion_dim": 128,
    "temporal_width": 64,
    "audio_encoder": "baseline",
    "vision_encoder": "baseline",
    "pooling": "masked_mean",
    "text_pooling": "masked_mean",
    "fusion": "pooled",
    "num_heads": 4,
    "dropout": 0.1,
    "av_direct": True,
    "av_direct_scale": 0.1,
    "neutral_aux": True,
    "teacher_feature_dim": None,
    "joint_prediction": False,
}


def _mask_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Multiplication by zero does not sanitize NaN/Inf. Treat unavailable
    # positions as absent evidence even when upstream padding contains sentinels.
    return torch.where(mask.to(dtype=torch.bool).unsqueeze(-1), values,
                       torch.zeros((), dtype=values.dtype, device=values.device))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    clean = torch.where(mask.unsqueeze(-1), values,
                        torch.zeros((), dtype=values.dtype, device=values.device))
    return clean.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MaskedPool(nn.Module):
    def __init__(self, dimension: int, mode: str):
        super().__init__()
        if mode not in {"masked_mean", "masked_attention"}:
            raise ValueError(f"Unknown pooling mode: {mode}")
        self.mode = mode
        self.score = nn.Linear(dimension, 1, bias=False) if mode == "masked_attention" else None

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        if self.mode == "masked_mean":
            return masked_mean(values, mask)
        safe_values = torch.where(mask.unsqueeze(-1), values,
                                  torch.zeros((), dtype=values.dtype, device=values.device))
        scores = self.score(safe_values).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        empty = ~mask.any(dim=1)
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0.0
        weights = torch.softmax(scores, dim=1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return torch.bmm(weights.unsqueeze(1), safe_values).squeeze(1)


class _ConformerBlock(nn.Module):
    """Compact conformer block with masked attention and depthwise convolution."""

    def __init__(self, width: int, heads: int, dropout: float, kernel_size: int = 7):
        super().__init__()
        if width % heads:
            raise ValueError("temporal_width must be divisible by num_heads for Conformer")
        self.ffn1_norm = nn.LayerNorm(width)
        self.ffn1 = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.pointwise_in = nn.Conv1d(width, width * 2, 1)
        self.depthwise = nn.Conv1d(width, width, kernel_size,
                                   padding=kernel_size // 2, groups=width)
        self.pointwise_out = nn.Conv1d(width, width, 1)
        self.ffn2_norm = nn.LayerNorm(width)
        self.ffn2 = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        x = _mask_values(x + 0.5 * self.dropout(self.ffn1(self.ffn1_norm(x))), mask)
        q = self.attn_norm(x)
        attended = self.attn(q, q, q, key_padding_mask=~safe_mask, need_weights=False)[0]
        x = _mask_values(x + self.dropout(attended), mask)
        z = self.conv_norm(x).transpose(1, 2)
        z = F.glu(self.pointwise_in(z), dim=1)
        z = F.silu(self.depthwise(z))
        z = self.pointwise_out(z).transpose(1, 2)
        x = _mask_values(x + self.dropout(z), mask)
        x = _mask_values(x + 0.5 * self.dropout(self.ffn2(self.ffn2_norm(x))), mask)
        return x


class MaskedTemporalEncoder(nn.Module):
    """AV sequence encoder with independently selectable temporal structure."""

    MODES = {"baseline", "dilated_tcn", "transformer", "conformer"}

    def __init__(self, input_dim: int, width: int, mode: str, heads: int,
                 dropout: float, layers: int = 2):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown temporal encoder {mode!r}; expected {sorted(self.MODES)}")
        if width % heads and mode in {"transformer", "conformer"}:
            raise ValueError("temporal_width must be divisible by num_heads")
        self.mode = mode
        self.project = nn.Linear(input_dim + 2, width)
        self.dropout = nn.Dropout(dropout)
        if mode in {"baseline", "dilated_tcn"}:
            dilations = [1, 1] if mode == "baseline" else [1, 2, 4][:max(1, layers)]
            self.convs = nn.ModuleList([
                nn.Conv1d(width, width, kernel_size=3, padding=d, dilation=d)
                for d in dilations
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in dilations])
            self.blocks = None
        elif mode == "transformer":
            block = nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=width * 4,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
            self.blocks = nn.TransformerEncoder(block, num_layers=max(1, layers), enable_nested_tensor=False)
            self.convs = self.norms = None
        else:
            self.blocks = nn.ModuleList([
                _ConformerBlock(width, heads, dropout) for _ in range(max(1, layers))
            ])
            self.convs = self.norms = None

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        clean = _mask_values(values, mask)
        positions = torch.linspace(0.0, 1.0, values.shape[1], device=values.device,
                                  dtype=values.dtype).expand(values.shape[0], -1)
        x = self.project(torch.cat([clean, mask.to(values.dtype).unsqueeze(-1),
                                    positions.unsqueeze(-1)], dim=-1))
        x = _mask_values(x, mask)
        if self.mode in {"baseline", "dilated_tcn"}:
            for conv, norm in zip(self.convs, self.norms):
                update = self.dropout(F.gelu(conv(x.transpose(1, 2)).transpose(1, 2)))
                x = _mask_values(norm(x + update), mask)
            return x
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        if self.mode == "transformer":
            x = self.blocks(x, src_key_padding_mask=~safe_mask)
            return _mask_values(x, mask)
        for block in self.blocks:
            x = block(x, mask)
        return _mask_values(x, mask)


class V5Model(nn.Module):
    """Configurable text/AV classifier and intensity regressor.

    The standard output uses the three-class argmax; the neutral head is an
    auxiliary training signal and does not route or override predictions.
    """

    def __init__(self, backbone_dir, config: dict[str, Any] | None = None,
                 *, with_distill_heads: bool = True):
        super().__init__()
        cfg = dict(MODEL_DEFAULTS)
        cfg.update(config or {})
        if backbone_dir is None:
            backbone_dir = cfg.get("backbone_dir")
        if not backbone_dir:
            raise ValueError("backbone_dir is required for local AutoConfig initialization")
        self.backbone_dir = str(backbone_dir)
        self.model_config = cfg
        if cfg.get("load_pretrained", True):
            self.text_encoder = AutoModel.from_pretrained(
                self.backbone_dir, local_files_only=True)
        else:
            base_config = AutoConfig.from_pretrained(self.backbone_dir, local_files_only=True)
            self.text_encoder = AutoModel.from_config(base_config)
        if cfg.get("gradient_checkpointing", False):
            if not hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                raise ValueError("This local text backbone does not support gradient checkpointing")
            self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.text_encoder.config, "use_cache"):
                self.text_encoder.config.use_cache = False
        text_width = int(self.text_encoder.config.hidden_size)
        fusion_dim = int(cfg["fusion_dim"])
        width = int(cfg["temporal_width"])
        heads = int(cfg["num_heads"])
        dropout = float(cfg["dropout"])
        if fusion_dim % heads:
            raise ValueError("fusion_dim must be divisible by num_heads")
        self.fusion_dim = fusion_dim
        self.text_accepts_segments = "token_type_ids" in inspect.signature(self.text_encoder.forward).parameters
        self.text_projection = nn.Linear(text_width, fusion_dim)
        self.audio_encoder = MaskedTemporalEncoder(
            74, width, cfg["audio_encoder"], heads, dropout,
            layers=int(cfg.get("audio_layers", 2)))
        self.vision_encoder = MaskedTemporalEncoder(
            35, width, cfg["vision_encoder"], heads, dropout,
            layers=int(cfg.get("vision_layers", 2)))
        self.audio_projection = nn.Linear(width, fusion_dim)
        self.vision_projection = nn.Linear(width, fusion_dim)
        self.pooling_mode = cfg["pooling"]
        self.text_pool = MaskedPool(fusion_dim, self.pooling_mode)
        self.audio_pool = MaskedPool(fusion_dim, self.pooling_mode)
        self.vision_pool = MaskedPool(fusion_dim, self.pooling_mode)
        self.text_pooling_mode = str(cfg.get("text_pooling", "masked_mean"))
        if self.text_pooling_mode not in {"masked_mean", "cls", "cls_mean"}:
            raise ValueError("text_pooling must be one of 'masked_mean', 'cls', or 'cls_mean'")
        self.fusion_mode = cfg["fusion"]
        if self.fusion_mode not in {
                "pooled", "cross_attention", "mag_style", "content_gate", "availability_gate"}:
            raise ValueError("Unknown fusion mode")
        if self.fusion_mode == "cross_attention":
            self.text_to_audio = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)
            self.audio_to_text = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)
            self.text_to_vision = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)
            self.vision_to_text = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)
            self.text_cross_norm = nn.LayerNorm(fusion_dim)
            self.audio_cross_norm = nn.LayerNorm(fusion_dim)
            self.vision_cross_norm = nn.LayerNorm(fusion_dim)
            self.cross_dropout = nn.Dropout(dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU())
        self.task_head = nn.Linear(fusion_dim, 4)
        self.neutral_head = nn.Linear(fusion_dim, 1) if cfg.get("neutral_aux", True) else None
        self.av_direct = bool(cfg.get("av_direct", True))
        self.av_direct_scale = float(cfg.get("av_direct_scale", 0.1))
        if self.av_direct:
            self.av_direct_head = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(fusion_dim, 4))
        self.with_distill_heads = bool(with_distill_heads)
        teacher_dim = cfg.get("teacher_feature_dim")
        self.distill_adapters = nn.ModuleDict()
        if self.with_distill_heads and teacher_dim is not None and int(teacher_dim) != fusion_dim:
            self.distill_adapters = nn.ModuleDict({
                name: nn.Linear(fusion_dim, int(teacher_dim), bias=False)
                for name in ("text", "audio", "vision", "fused")
            })
            for adapter in self.distill_adapters.values():
                nn.init.xavier_uniform_(adapter.weight)
        self.distill_feature_dim = int(teacher_dim) if teacher_dim is not None else fusion_dim
        if self.fusion_mode == "mag_style":
            # Create the new fusion-only parameters after all shared modules,
            # while restoring every RNG stream so pooled and MAG-style models
            # start from identical shared weights and training randomness.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                self.audio_gate = nn.Linear(fusion_dim * 2, fusion_dim)
                self.vision_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        elif self.fusion_mode in {"content_gate", "availability_gate"}:
            # The scalar reliability head is appended after shared modules and
            # isolated from global CPU/CUDA RNG streams. Zero initialization
            # gives each available modality an initial weight of exactly 1.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                self.reliability_score = nn.Linear(fusion_dim * 3 + 6, 3)
                nn.init.zeros_(self.reliability_score.weight)
                nn.init.zeros_(self.reliability_score.bias)
        self.joint_prediction = bool(cfg.get("joint_prediction", False))
        if self.joint_prediction:
            # The optional class-conditional magnitude head is appended after
            # all shared modules without advancing global CPU/CUDA RNG state.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                self.mixture_amplitude_head = nn.Linear(fusion_dim, 2)

    @staticmethod
    def _safe_attention(mask: torch.Tensor) -> torch.Tensor:
        safe = mask.to(dtype=torch.bool).clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        return safe

    @staticmethod
    def _cross(query, source, query_mask, source_mask, attention):
        safe_source = V5Model._safe_attention(source_mask)
        context = attention(query, source, source,
                            key_padding_mask=~safe_source,
                            need_weights=False)[0]
        # A fully empty source still attends to a synthetic safe key to avoid
        # NaNs. Its output (including MHA output bias) must nevertheless be 0.
        context = _mask_values(context, source_mask.any(dim=1).view(-1, 1).expand(-1, context.shape[1]))
        return _mask_values(context, query_mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        text_ids = batch["text_ids"].long()
        text_mask = batch["text_mask"].to(dtype=torch.bool)
        text_pool_mask = batch.get("text_pool_mask", text_mask).to(dtype=torch.bool) & text_mask
        text_segments = batch.get("text_segments")
        safe_text_mask = self._safe_attention(text_mask)
        pad_id = int(getattr(self.text_encoder.config, "pad_token_id", 0) or 0)
        # Sanitize every masked ID, not only all-empty rows. This prevents
        # out-of-range padding sentinels from reaching the embedding table.
        safe_ids = torch.where(text_mask, text_ids, torch.full_like(text_ids, pad_id))
        empty = ~text_mask.any(dim=1)
        if empty.any():
            safe_ids[empty, 0] = pad_id
        encoder_kwargs = {"input_ids": safe_ids, "attention_mask": safe_text_mask.long()}
        if self.text_accepts_segments and text_segments is not None:
            safe_segments = torch.where(text_mask, text_segments.long(), torch.zeros_like(text_ids))
            encoder_kwargs["token_type_ids"] = safe_segments
        text_raw = self.text_encoder(**encoder_kwargs).last_hidden_state
        text_seq = _mask_values(self.text_projection(text_raw), text_mask)

        audio_mask = batch["audio_mask"].to(dtype=torch.bool)
        vision_mask = batch["vision_mask"].to(dtype=torch.bool)
        audio_raw = self.audio_encoder(batch["audio"], audio_mask)
        vision_raw = self.vision_encoder(batch["vision"], vision_mask)
        audio_seq = _mask_values(self.audio_projection(audio_raw), audio_mask)
        vision_seq = _mask_values(self.vision_projection(vision_raw), vision_mask)

        if self.fusion_mode == "cross_attention":
            text_update = self._cross(text_seq, audio_seq, text_pool_mask, audio_mask, self.text_to_audio)
            text_update = text_update + self._cross(text_seq, vision_seq, text_pool_mask, vision_mask, self.text_to_vision)
            audio_update = self._cross(audio_seq, text_seq, audio_mask, text_pool_mask, self.audio_to_text)
            vision_update = self._cross(vision_seq, text_seq, vision_mask, text_pool_mask, self.vision_to_text)
            text_seq = _mask_values(self.text_cross_norm(text_seq + self.cross_dropout(text_update)), text_pool_mask)
            audio_seq = _mask_values(self.audio_cross_norm(audio_seq + self.cross_dropout(audio_update)), audio_mask)
            vision_seq = _mask_values(self.vision_cross_norm(vision_seq + self.cross_dropout(vision_update)), vision_mask)

        text_available = text_pool_mask.any(dim=1)
        if self.text_pooling_mode == "masked_mean":
            pooled_text = self.text_pool(text_seq, text_pool_mask)
        else:
            # BERT's CLS can remain nonzero when all content tokens are
            # masked (special tokens still attend to each other), so gate it
            # by actual content availability rather than text attention_mask.
            cls = text_seq[:, 0, :]
            cls = torch.where(
                text_available.unsqueeze(-1), cls,
                torch.zeros((), dtype=cls.dtype, device=cls.device))
            if self.text_pooling_mode == "cls":
                pooled_text = cls
            else:
                mean = masked_mean(text_seq, text_pool_mask)
                pooled_text = 0.5 * cls + 0.5 * mean
        pooled = {
            "text": pooled_text,
            "audio": self.audio_pool(audio_seq, audio_mask),
            "vision": self.vision_pool(vision_seq, vision_mask),
        }
        text_for_fusion = pooled["text"]
        if self.fusion_mode == "mag_style":
            audio_available = audio_mask.any(dim=1)
            vision_available = vision_mask.any(dim=1)
            text_available = text_pool_mask.any(dim=1)
            audio_gate = torch.sigmoid(self.audio_gate(
                torch.cat([pooled["text"], pooled["audio"]], dim=-1)))
            vision_gate = torch.sigmoid(self.vision_gate(
                torch.cat([pooled["text"], pooled["vision"]], dim=-1)))
            audio_residual = torch.where(
                audio_available.unsqueeze(-1), audio_gate * pooled["audio"],
                torch.zeros((), dtype=pooled["audio"].dtype, device=pooled["audio"].device))
            vision_residual = torch.where(
                vision_available.unsqueeze(-1), vision_gate * pooled["vision"],
                torch.zeros((), dtype=pooled["vision"].dtype, device=pooled["vision"].device))
            residual = audio_residual + vision_residual
            # Accumulate norms in FP32 so mixed-precision inference cannot
            # overflow while enforcing MAG's bounded residual ratio.
            text_norm = torch.linalg.vector_norm(pooled["text"].float(), dim=-1, keepdim=True)
            residual_norm = torch.linalg.vector_norm(residual.float(), dim=-1, keepdim=True)
            scale = (0.1 * text_norm / (residual_norm + 1e-6)).clamp(max=1.0)
            scale = scale * text_available.unsqueeze(-1).to(scale.dtype)
            text_for_fusion = pooled["text"] + scale.to(residual.dtype) * residual
        fusion_features = torch.cat([text_for_fusion, pooled["audio"], pooled["vision"]], dim=-1)
        if self.fusion_mode in {"content_gate", "availability_gate"}:
            available = torch.stack([
                text_pool_mask.any(dim=1), audio_mask.any(dim=1), vision_mask.any(dim=1)
            ], dim=-1)
            # For the fixed 50-slot aligned input, dividing by 50 measures
            # visible content density without consulting the original extent.
            densities = torch.stack([
                text_pool_mask.sum(dim=1), audio_mask.sum(dim=1), vision_mask.sum(dim=1)
            ], dim=-1).to(dtype=pooled["text"].dtype) / 50.0
            availability_values = available.to(dtype=pooled["text"].dtype)
            mask_features = torch.cat([availability_values, densities], dim=-1)
            if self.fusion_mode == "content_gate":
                mask_features = torch.zeros_like(mask_features)
            score_inputs = torch.cat([
                pooled["text"], pooled["audio"], pooled["vision"], mask_features
            ], dim=-1)
            scores = self.reliability_score(score_inputs)
            available_count = available.sum(dim=1, keepdim=True)
            safe_scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
            # Softmax over an all-missing row is made finite; the subsequent
            # hard mask and zero count force every reliability weight to 0.
            safe_scores = torch.where(available_count > 0, safe_scores, torch.zeros_like(safe_scores))
            weights = torch.softmax(safe_scores.float(), dim=-1).to(dtype=scores.dtype)
            weights = weights * available.to(dtype=weights.dtype)
            weights = weights * available_count.to(dtype=weights.dtype)
            weighted = []
            for index, name in enumerate(("text", "audio", "vision")):
                candidate = pooled[name] * weights[:, index:index + 1]
                weighted.append(torch.where(
                    available[:, index:index + 1], candidate,
                    torch.zeros((), dtype=candidate.dtype, device=candidate.device)))
            fusion_features = torch.cat(weighted, dim=-1)
        fused = self.fusion_head(fusion_features)
        pooled["fused"] = fused
        task_values = self.task_head(fused)
        av_observed = (audio_mask.any(dim=1) | vision_mask.any(dim=1)).to(task_values.dtype)
        if self.av_direct:
            direct_values = self.av_direct_head(torch.cat([pooled["audio"], pooled["vision"]], dim=-1))
            task_values = task_values + self.av_direct_scale * av_observed.unsqueeze(-1) * direct_values
        sequence_features = {
            "text": text_seq,
            "audio": audio_seq,
            "vision": vision_seq,
            # A set-like concatenation of separately-positioned sequences.
            # The associated sequence_masks keep padding/availability explicit.
            "fused": torch.cat([text_seq, audio_seq, vision_seq], dim=1),
        }
        sequence_masks = {
            # Cross-modal participating text positions exclude CLS/SEP by the
            # dataset contract; BERT's own contextual encoder still sees them.
            "text": text_pool_mask,
            "audio": audio_mask,
            "vision": vision_mask,
            "fused": torch.cat([text_pool_mask, audio_mask, vision_mask], dim=1),
        }
        distill_features = {}
        for name, value in pooled.items():
            distill_features[name] = self.distill_adapters[name](value) if name in self.distill_adapters else value
        intensity_direct = task_values[:, 3]
        if self.joint_prediction:
            magnitude_raw = self.mixture_amplitude_head(fused)
            negative_magnitude = -3.0 * torch.sigmoid(magnitude_raw[:, 0])
            positive_magnitude = 3.0 * torch.sigmoid(magnitude_raw[:, 1])
            class_probabilities = torch.softmax(task_values[:, :3].float(), dim=-1)
            intensity_raw = (
                class_probabilities[:, 0] * negative_magnitude.float()
                + class_probabilities[:, 2] * positive_magnitude.float()
            )
        else:
            intensity_raw = intensity_direct
        result = {
            "logits": task_values[:, :3],
            "intensity_raw": intensity_raw,
            "neutral_logit": self.neutral_head(fused).squeeze(-1) if self.neutral_head is not None else None,
            "pooled_features": pooled,
            "sequence_features": sequence_features,
            "sequence_masks": sequence_masks,
            "distill_features": distill_features,
        }
        if self.joint_prediction:
            result["intensity_direct"] = intensity_direct
        return result


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def inference_state_dict(model: V5Model) -> dict[str, torch.Tensor]:
    """Return only inference parameters, excluding optional KD adapters."""
    return {name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
            if not name.startswith("distill_adapters.")}
