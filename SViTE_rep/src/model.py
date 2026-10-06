"""DeiT-Tiny with SViTE+ patch selection and DeiT distillation heads."""

import math
from functools import partial

import torch
from torch import nn
from timm.models.deit import VisionTransformerDistilled


class TokenSelector(nn.Module):
    """
    Hard top-k in the forward pass, softmax surrogate in the backward pass.
    The softmax surrogate simulates the gradient of the hard top-k selection, allowing gradients to flow through the selection process during backpropagation.
    Note that the softmax surrogate is only used during training; during evaluation, the hard top-k selection is used directly.

    The MLP width is an explicit implementation choice: the paper specifies an
    MLP scorer but not its hidden width. Prefix tokens never enter this module.
    """

    def __init__(self, dim: int, keep_rate: float = 0.9):
        super().__init__()
        if not 0 < keep_rate <= 1:
            raise ValueError("keep_rate must be in (0, 1].")
        self.keep_rate = keep_rate
        self.scorer = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)) # score gives one value per token, so scores has shape [B, N] (of [B, N, D] tokens). B is batch size, N is number of tokens, D is embedding dimension (position).
        if keep_rate == 1:
            self.scorer.requires_grad_(False)


    """
    forward value:       gate = hard
    backward derivative: ∂gate/∂scores = ∂soft/∂scores
    
    Gather usage to return selected tokens (ex):
    tokens:       [t0, t1, t2, t3, t4]
    hard gate:    [ 0,  1,  0,  1,  0] (mask for top k tokens)
    gated tokens: [ 0, t1,  0, t3,  0]
    gather:       [t1, t3]
    """
    def forward(self, tokens, temperature: float = 1.0):
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive.")
        batch, count, dim = tokens.shape
        k = max(1, math.floor(count * self.keep_rate))
        if k == count:
            indices = torch.arange(count, device=tokens.device).expand(batch, -1)
            return tokens, indices
        # Float32 softmax is useful even when the backbone runs under autocast.
        # Here wwe use a Gumbel trick to sample from the categorical distribution defined by the scores (shown to increase exploration).
        scores = self.scorer(tokens).squeeze(-1).float()
        if self.training:
            noise = -torch.empty_like(scores).exponential_().clamp_min_(1e-10).log()
            scores = scores + noise
        soft = (scores / temperature).softmax(dim=-1) # Produces probbaility distribution over tokens
        indices = scores.topk(k, dim=-1).indices.sort(dim=-1).values # Positions of top k tokens in the original sequence. Sort to preserve order of tokens in the original sequence.
        hard = torch.zeros_like(soft).scatter_(1, indices, 1.0) # Hard selection of top k tokens, represented as a one-hot vector (1 for retained tokens). This is used in the forward pass.
        gate = hard - soft.detach() + soft if self.training else hard
        # Gather actually shortens the sequence; a zero mask alone would not.
        selected = (tokens * gate.to(tokens.dtype).unsqueeze(-1)).gather(
            1, indices.unsqueeze(-1).expand(-1, -1, dim)
        )
        return selected, indices


class SViTETiny(VisionTransformerDistilled):
    """12 blocks, width 192, 3 attention heads, 16x16 patches, MLP ratio 4.

    Always return the two raw logit tensors, including during evaluation.
    prediction_probabilities implements the paper's probability-level fusion.
    Sparse connectivity is installed separately by SparseTopology.
    """

    def __init__(self, image_size=224, num_classes=1000, keep_rate=0.9, drop_path_rate=0.1):
        if image_size <= 0 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16.")
        super().__init__(
            img_size=image_size, patch_size=16, num_classes=num_classes,
            embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, # D -> 4D -> D
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0, pos_drop_rate=0.0, proj_drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=drop_path_rate,
        )
        self.selector = TokenSelector(192, keep_rate)
        for module in self.selector.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    @property
    def kept_patch_tokens(self):
        return max(1, math.floor(self.patch_embed.num_patches * self.selector.keep_rate))

    def forward(self, images, temperature=1.0):
        # Attach positions before selection so a kept patch retains its location.
        x = self._pos_embed(self.patch_embed(images))
        patches, _ = self.selector(x[:, 2:], temperature)
        x = torch.cat((x[:, :2], patches), dim=1)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0]), self.head_dist(x[:, 1])


def prediction_probabilities(outputs):
    """Late fusion specified in the supplied DeiT excerpt; teacher not needed."""
    class_logits, distillation_logits = outputs
    return (class_logits.float().softmax(-1) + distillation_logits.float().softmax(-1)) / 2
