"""Unstructured SViTE connectivity exploration using dense PyTorch kernels.

Masks are part of the model state_dict. Dense kernels keep this implementation
portable; masked zeros do not themselves accelerate matrix multiplication.
"""

import math

import torch
from torch import nn


class SparseLinear(nn.Linear):
    def __init__(self, original: nn.Linear):
        super().__init__(original.in_features, original.out_features, original.bias is not None,
                         device=original.weight.device, dtype=original.weight.dtype)
        self.weight = original.weight
        self.bias = original.bias
        self.register_buffer("mask", torch.ones_like(self.weight, dtype=torch.bool))
        self.collect_dense_gradient = False
        self.weight.register_hook(self._mask_gradient)

    def _mask_gradient(self, gradient):
        # On exploration steps preserve inactive derivatives until accumulation
        # is complete. Multiplying weight*mask in forward would destroy them.
        return gradient if self.collect_dense_gradient else gradient * self.mask


def erdos_renyi_counts(shapes, density):
    """Exact integer budget with ER densities proportional to (in+out)/(in*out), equality scaled by factor of density (epsilon)."""
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1].")
    sizes = [math.prod(shape) for shape in shapes]
    budget = round(sum(sizes) * density)
    if not sizes or budget < 1:
        raise ValueError("The sparse budget must contain at least one connection.")
    rates = [sum(shape) / size for shape, size in zip(shapes, sizes)] # calculates the relative density of each layer based on its in/out dimensions
    dense = set()
    while True:
        remaining = [i for i in range(len(sizes)) if i not in dense]
        if not remaining:
            return sizes
        scale = (budget - sum(sizes[i] for i in dense)) / sum(rates[i] * sizes[i] for i in remaining)
        saturated = {i for i in remaining if scale * rates[i] >= 1}
        if not saturated:
            break
        dense.update(saturated)
    expected = [float(size) if i in dense else scale * rates[i] * size for i, size in enumerate(sizes)]
    counts = [math.floor(value) for value in expected]
    # Largest remainders preserve the requested overall budget despite rounding.
    order = sorted(range(len(sizes)), key=lambda i: expected[i] - counts[i], reverse=True)
    for i in order[:budget - sum(counts)]:
        counts[i] += 1
    return counts


class SparseTopology:
    """Install block-only masks and manage magnitude pruning/gradient growth.

    Build before constructing the optimizer. Call begin_step before backward,
    prepare_update after unscaling gradients, and finish_step after AdamW.
    """

    # Consider using er_epsilon = 20 (parameter setting in SET Erdos-Renyo MLP paper).
    def __init__(self, model, density=0.5, update_interval=20000,
                 end_step=1200000, initial_drop_fraction=0.5, initialize=True,
                 er_epsilon=20): # was set to None
        if update_interval < 1 or end_step < 1 or not 0 <= initial_drop_fraction <= 1:
            raise ValueError("Invalid sparse update schedule.")
        self.update_interval = update_interval
        self.end_step = end_step
        self.initial_drop_fraction = initial_drop_fraction
        if er_epsilon is not None and (er_epsilon <= 0 or not math.isfinite(er_epsilon)):
            raise ValueError("ER epsilon must be finite and positive.")
        self.layers = {}
        # Patch projection, classifier heads, scorer, biases, and norms stay dense.
        for name, module in list(model.blocks.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, _, child = name.rpartition(".")
                parent = model.blocks.get_submodule(parent_name) if parent_name else model.blocks
                sparse = module if isinstance(module, SparseLinear) else SparseLinear(module)
                setattr(parent, child, sparse)
                self.layers["blocks." + name] = sparse
        counts = erdos_renyi_counts([m.weight.shape for m in self.layers.values()], density)
        if initialize:
            with torch.no_grad():
                for module, count in zip(self.layers.values(), counts):
                    if er_epsilon is None:
                        module.mask.zero_()
                        indices = torch.randperm(module.weight.numel(), device=module.weight.device)[:count]
                        module.mask.view(-1)[indices] = True
                    else:
                        # Literal SET Eq. (1): independent Bernoulli connections.
                        probability = min(1.0, er_epsilon * sum(module.weight.shape) / module.weight.numel())
                        module.mask.copy_(torch.rand_like(module.weight) < probability)
                    module.weight.mul_(module.mask)
        self.pending = {}
        self.current_step = 0

    def is_update_step(self, step):
        return 0 < step < self.end_step and step % self.update_interval == 0

    def drop_fraction(self, step):
        if step >= self.end_step:
            return 0.0
        return self.initial_drop_fraction * (1 + math.cos(math.pi * step / self.end_step)) / 2

    def begin_step(self, step):
        self.current_step = step
        self.pending.clear()
        for module in self.layers.values():
            module.collect_dense_gradient = self.is_update_step(step)

    @torch.no_grad()
    def prepare_update(self):
        """
        Chooses where connections could grow and prepares gradients before calling the optimizer.
        Keep only top candidate indices, then mask gradients before AdamW.

        With microbatches, selection uses the gradient of their accumulated loss.
        Candidates must be inactive in the OLD topology, as the paper specifies.
        """
        for name, module in self.layers.items():
            grad = module.weight.grad
            if grad is None:
                raise RuntimeError(f"Missing gradient for {name}.")
            if module.collect_dense_gradient:
                active = int(module.mask.sum())
                inactive = module.mask.numel() - active
                count = min(math.floor(active * self.drop_fraction(self.current_step)), inactive)
                if count:
                    scores = grad.detach().abs().flatten().masked_fill(module.mask.flatten(), -torch.inf)
                    self.pending[name] = scores.topk(count).indices
            grad.mul_(module.mask)
            module.collect_dense_gradient = False

    @torch.no_grad()
    def finish_step(self, optimizer, skipped=False):
        """
        Applies topology changes to each multi-self-attention head layer after the optimizer has updated weights. If skipped, the step is not counted for growth.
        """
        for name, module in self.layers.items():
            if not skipped and name in self.pending:
                grow = self.pending[name]
                old_mask = module.mask.flatten()
                scores = module.weight.abs().flatten().masked_fill(~old_mask, torch.inf)
                prune = scores.topk(grow.numel(), largest=False).indices
                old_mask[prune] = False
                old_mask[grow] = True
                module.weight.view(-1)[prune] = 0
                module.weight.view(-1)[grow] = 0
                # New connections must not inherit moments from an earlier life.
                for value in optimizer.state.get(module.weight, {}).values():
                    if isinstance(value, torch.Tensor) and value.shape == module.weight.shape:
                        value.view(-1)[prune] = 0
                        value.view(-1)[grow] = 0
            module.weight.mul_(module.mask)
            for value in optimizer.state.get(module.weight, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == module.weight.shape:
                    value.mul_(module.mask)
        self.pending.clear()

    def statistics(self):
        total = sum(m.mask.numel() for m in self.layers.values())
        active = sum(int(m.mask.sum()) for m in self.layers.values())
        return {"prunable_weights": total, "active_connections": active,
                "weight_sparsity": 1 - active / total}
