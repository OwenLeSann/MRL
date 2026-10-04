"""Small synthetic checks; no ImageNet data or pretrained download required."""

import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from data import build_transform
from distillation import HardDistillationLoss
from teacher import TEACHER_MODEL, build_teacher


class DistillationTests(unittest.TestCase):
    def test_equal_weight_when_teacher_disagrees(self):
        # True class is 0, teacher class is 1. With no smoothing, the target
        # distribution must be exactly [0.5, 0.5], regardless of confidence.
        loss_fn = HardDistillationLoss(nn.Identity(), label_smoothing=0)
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        loss = loss_fn(torch.tensor([[-3.0, 9.0]]), logits, torch.tensor([0]))
        expected = -logits.log_softmax(-1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - 0.5)

    def test_two_heads_receive_separate_targets(self):
        loss_fn = HardDistillationLoss(nn.Identity())
        class_logits = torch.zeros(1, 2, requires_grad=True)
        dist_logits = torch.zeros(1, 2, requires_grad=True)
        loss_fn(
            torch.tensor([[0.0, 10.0]]),
            (class_logits, dist_logits),
            torch.tensor([0]),
        ).backward()
        torch.testing.assert_close(class_logits.grad, torch.tensor([[-0.225, 0.225]]))
        torch.testing.assert_close(dist_logits.grad, torch.tensor([[0.25, -0.25]]))

    def test_teacher_is_frozen_and_batchnorm_unchanged(self):
        teacher = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        loss_fn = HardDistillationLoss(teacher).train()
        before_mean = teacher[0].running_mean.clone()
        before_var = teacher[0].running_var.clone()
        images = torch.randn(4, 3, requires_grad=True)
        logits = torch.randn(4, 2, requires_grad=True)
        loss_fn(images, logits, torch.tensor([0, 1, 0, 1])).backward()
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assertIsNone(images.grad)
        self.assertIsNotNone(logits.grad)
        torch.testing.assert_close(teacher[0].running_mean, before_mean)
        torch.testing.assert_close(teacher[0].running_var, before_var)

    def test_teacher_targets_follow_input_each_call(self):
        loss_fn = HardDistillationLoss(nn.Identity(), label_smoothing=0)
        logits = torch.tensor([[4.0, -4.0]])
        labels = torch.tensor([0])
        first = loss_fn(torch.tensor([[5.0, 0.0]]), logits, labels)
        second = loss_fn(torch.tensor([[0.0, 5.0]]), logits, labels)
        self.assertGreater(second.item(), first.item())

    def test_incompatible_class_counts_fail(self):
        loss_fn = HardDistillationLoss(nn.Identity())
        with self.assertRaisesRegex(ValueError, "Teacher and student"):
            loss_fn(torch.ones(2, 3), torch.ones(2, 2), torch.tensor([0, 1]))


class InputTests(unittest.TestCase):
    def test_train_and_validation_transforms(self):
        image = Image.new("RGB", (320, 256), (30, 120, 210))
        for training in (True, False):
            transform = build_transform(training)
            result = transform(image)
            self.assertEqual(result.shape, (3, 224, 224))
            self.assertTrue(torch.isfinite(result).all())
            if not training:
                torch.testing.assert_close(result, transform(image), rtol=0, atol=0)

    def test_teacher_factory_selects_authors_checkpoint(self):
        with patch("teacher.timm.create_model", return_value=nn.Linear(3, 1000)) as create:
            teacher = build_teacher()
        create.assert_called_once_with(TEACHER_MODEL, pretrained=True, num_classes=1000)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
