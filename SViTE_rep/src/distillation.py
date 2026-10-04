"""DeiT hard-label supervision, independent of the student's architecture."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HardDistillationLoss(nn.Module):
    """Equal weighting of ground-truth and teacher-label cross entropy.

    A single [batch, classes] tensor implements equation (3). A pair
    (class_logits, distillation_logits) supervises two separate output heads.
    Supply raw logits, since cross_entropy computes log-softmax internally.
    """

    def __init__(self, teacher: nn.Module, label_smoothing: float = 0.1): # Follows epsilon=0.1 redistribution of target probability applied to every ground-truth label (label smoothing) used in DeiT paper.
        super().__init__()
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.teacher = teacher.requires_grad_(False).eval()
        self.label_smoothing = label_smoothing

    def train(self, mode: bool = True):
        # Calling criterion.train() must not update the teacher's BatchNorm stats.
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        images: Tensor,
        student_outputs: Tensor | tuple[Tensor, Tensor],
        labels: Tensor,
    ) -> Tensor:
        """images must be the exact augmented batch passed to the student.

        labels are integer class indices [batch], or already smoothed Mixup/
        CutMix probabilities [batch, classes]. Soft targets are not smoothed twice.
        Teacher targets are recomputed each call, because augmentation changes
        the input and can therefore change the teacher's predicted class.
        """
        if isinstance(student_outputs, Tensor):
            class_logits = distillation_logits = student_outputs
        else:
            if len(student_outputs) != 2:
                raise ValueError("Expected (class_logits, distillation_logits).")
            class_logits, distillation_logits = student_outputs
        if class_logits.ndim != 2 or distillation_logits.shape != class_logits.shape:
            raise ValueError("Both student outputs must have shape [batch, classes].")
        soft_targets = labels.is_floating_point() and labels.shape == class_logits.shape
        hard_targets = labels.dtype == torch.long and labels.shape == class_logits.shape[:1]
        if not (soft_targets or hard_targets):
            raise ValueError("labels must be integer [batch] or floating [batch, classes].")
        if images.shape[0] != class_logits.shape[0]:
            raise ValueError("Images and logits must have the same batch size.")

        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(images)
            if teacher_logits.shape != class_logits.shape:
                raise ValueError("Teacher and student must share batch and class dimensions.")
            teacher_labels = teacher_logits.argmax(dim=-1)

        # Match the released implementation: smooth true labels only.
        supervised_loss = F.cross_entropy(
            class_logits, labels, label_smoothing=0.0 if soft_targets else self.label_smoothing
        )
        teacher_loss = F.cross_entropy(distillation_logits, teacher_labels)
        return 0.5 * supervised_loss + 0.5 * teacher_loss
