import copy
import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

from data import FlatImageNetValidation, build_transform
from distillation import HardDistillationLoss
from model import SViTETiny, TokenSelector, prediction_probabilities
from sparsity import SparseTopology, erdos_renyi_counts
from training import ModelEMA, learning_rate


class TokenTests(unittest.TestCase):
    def test_selection_reduces_tokens_and_scorer_learns(self):
        torch.manual_seed(4)
        selector = TokenSelector(8, 0.5).train()
        tokens = torch.randn(2, 6, 8, requires_grad=True)
        selected, indices = selector(tokens, temperature=1.0)
        self.assertEqual(selected.shape, (2, 3, 8))
        torch.testing.assert_close(selected, tokens.gather(1, indices[..., None].expand(-1, -1, 8)))
        selected.square().sum().backward()
        self.assertGreater(selector.scorer[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(selector.scorer[0].weight.grad).all())
        for row in indices:
            self.assertEqual(row.unique().numel(), 3)

    def test_eval_is_deterministic_and_keep_all_is_identity(self):
        tokens = torch.randn(2, 10, 8)
        selector = TokenSelector(8, 0.7).eval()
        first, first_idx = selector(tokens)
        torch.manual_seed(123)
        second, second_idx = selector(tokens)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first_idx, second_idx)
        all_tokens, _ = TokenSelector(8, 1.0)(tokens)
        torch.testing.assert_close(all_tokens, tokens, atol=0, rtol=0)

    def test_tiny_dimensions_and_prefix_tokens(self):
        model = SViTETiny(drop_path_rate=0).eval()
        self.assertEqual(len(model.blocks), 12)
        self.assertEqual(model.embed_dim, 192)
        self.assertEqual(model.blocks[0].attn.num_heads, 3)
        self.assertEqual(model.kept_patch_tokens, 176)
        lengths = []
        hook = model.blocks[0].register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 224, 224))
        hook.remove()
        self.assertEqual(lengths, [178])
        self.assertEqual(outputs[0].shape, (1, 1000))
        self.assertEqual(outputs[1].shape, (1, 1000))
        probabilities = prediction_probabilities(outputs)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))

    def test_dense_backbone_matches_timm_without_selection(self):
        from timm.models.deit import VisionTransformerDistilled
        model = SViTETiny(image_size=32, keep_rate=1, drop_path_rate=0).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = VisionTransformerDistilled.forward_features(model, images)
            class_logits, dist_logits = model(images)
        torch.testing.assert_close(class_logits, model.head(expected[:, 0]))
        torch.testing.assert_close(dist_logits, model.head_dist(expected[:, 1]))


class SparseTests(unittest.TestCase):
    @staticmethod
    def setup_layer():
        model = nn.Module()
        model.blocks = nn.Sequential(nn.Linear(4, 2, bias=False))
        model.head = nn.Linear(2, 2)
        topology = SparseTopology(model, density=0.5, update_interval=1, end_step=100)
        module = model.blocks[0]
        with torch.no_grad():
            module.mask.copy_(torch.tensor([[True, True, False, False], [True, True, False, False]]))
            module.weight.copy_(torch.tensor([[0.01, 0.1, 0., 0.], [0.2, 0.3, 0., 0.]]))
        return model, topology, module

    def test_er_exact_budget_and_small_layer_density(self):
        shapes = [(4, 4), (40, 40)]
        counts = erdos_renyi_counts(shapes, 0.2)
        self.assertEqual(sum(counts), round((16 + 1600) * 0.2))
        self.assertGreater(counts[0] / 16, counts[1] / 1600)
        self.assertEqual(erdos_renyi_counts(shapes, 1), [16, 1600])

    def test_inactive_gradients_drive_growth_and_budget_is_fixed(self):
        model, topology, layer = self.setup_layer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        old_mask = layer.mask.clone()
        topology.begin_step(1)
        layer(torch.tensor([[1., 2., 9., 3.]])).sum().backward()
        self.assertGreater(layer.weight.grad[0, 2].item(), 0)
        topology.prepare_update()
        grow = topology.pending["blocks.0"].clone()
        self.assertFalse(old_mask.flatten()[grow].any())
        self.assertTrue((layer.weight.grad[~old_mask] == 0).all())
        optimizer.step()
        topology.finish_step(optimizer)
        self.assertEqual(int(layer.mask.sum()), int(old_mask.sum()))
        self.assertEqual(int((layer.mask != old_mask).sum()), 2 * grow.numel())
        self.assertTrue((layer.weight.flatten()[grow] == 0).all())
        self.assertTrue((layer.weight[~layer.mask] == 0).all())
        self.assertTrue((optimizer.state[layer.weight]["exp_avg"].flatten()[grow] == 0).all())

    def test_gradient_accumulation_selects_sum_not_last_microbatch(self):
        _, topology, layer = self.setup_layer()
        topology.begin_step(1)
        layer(torch.tensor([[1., 1., 100., 0.]])).sum().backward()
        layer(torch.tensor([[1., 1., 0., 1.]])).sum().backward()
        topology.prepare_update()
        # The largest accumulated inactive derivatives are in column 2.
        self.assertTrue(((topology.pending["blocks.0"] % 4) == 2).all())

    def test_no_exploration_at_or_after_end(self):
        model, topology, layer = self.setup_layer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        old_mask = layer.mask.clone()
        topology.begin_step(100)
        layer(torch.ones(1, 4)).sum().backward()
        self.assertTrue((layer.weight.grad[~old_mask] == 0).all())
        topology.prepare_update()
        optimizer.step()
        topology.finish_step(optimizer)
        torch.testing.assert_close(layer.mask, old_mask)
        self.assertEqual(topology.drop_fraction(100), 0)

    def test_masks_checkpoint_and_ema_follow_connectivity(self):
        model, topology, layer = self.setup_layer()
        ema = ModelEMA(model, 0.9)
        with torch.no_grad():
            layer.mask[0, 0] = False
            layer.weight[0, 0] = 0
        ema.update(model)
        self.assertFalse(ema.module.blocks[0].mask[0, 0])
        self.assertEqual(ema.module.blocks[0].weight[0, 0].item(), 0)
        state = copy.deepcopy(model.state_dict())
        restored, _, _ = self.setup_layer()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.blocks[0].mask, layer.mask)

    def test_only_transformer_weight_matrices_are_sparse(self):
        model = SViTETiny(image_size=32)
        topology = SparseTopology(model, density=0.5)
        self.assertEqual(len(topology.layers), 48)
        self.assertFalse(hasattr(model.patch_embed.proj, "mask"))
        self.assertFalse(hasattr(model.head, "mask"))
        self.assertFalse(hasattr(model.head_dist, "mask"))
        self.assertFalse(hasattr(model.selector.scorer[0], "mask"))


class TrainingTests(unittest.TestCase):
    def test_mixup_targets_are_not_smoothed_twice(self):
        criterion = HardDistillationLoss(nn.Identity())
        logits = torch.zeros(1, 2, requires_grad=True)
        teacher_logits = torch.tensor([[0., 3.]])
        criterion(teacher_logits, logits, torch.tensor([[0.8, 0.2]])).backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0.1, -0.1]]))

    def test_lr_warmup_and_cosine(self):
        config = dict(learning_rate=0.0005, effective_batch_size=512, warmup_epochs=5,
                      warmup_learning_rate=1e-6, min_learning_rate=1e-5, epochs=600)
        self.assertEqual(learning_rate(0, config), 1e-6)
        self.assertGreater(learning_rate(4, config), learning_rate(1, config))
        self.assertGreater(learning_rate(5, config), learning_rate(300, config))
        self.assertAlmostEqual(learning_rate(600, config), 1e-5)

    def test_flat_validation_uses_training_synset_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (32, 32)).save(root / "example.JPEG")
            labels = root / "labels.csv"
            with labels.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ImageId", "PredictionString"])
                writer.writerow(["example", "n00000002 0 0 10 10"])
            dataset = FlatImageNetValidation(root, labels, {"n00000001": 0, "n00000002": 1}, build_transform(False, 32))
            image, label = dataset[0]
            self.assertEqual(label, 1)
            self.assertEqual(image.shape, (3, 32, 32))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
