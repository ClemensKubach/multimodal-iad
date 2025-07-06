"""Multimodal pre-processor."""

import torch
from anomalib.data import Batch, DepthBatch
from anomalib.pre_processing import PreProcessor
from lightning.pytorch import LightningModule, Trainer
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image, Mask
from typing_extensions import override


def depth_to_rgb(depth_map: torch.Tensor | None) -> torch.Tensor | None:
    """Convert depth map to RGB by repeating the channel dimension."""
    if depth_map is None:
        return None
    if depth_map.shape[-3] == 1:
        depth_map = depth_map.repeat(1, 3, 1, 1)
    if not isinstance(depth_map, Mask):
        depth_map = Image(depth_map)
    return depth_map


class MultimodalPreProcessor(PreProcessor):
    """Pre-processor for multimodal data."""

    @override
    def __init__(
        self,
        transform: Transform | None = None,
    ) -> None:
        super().__init__(transform)

    @override
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            if isinstance(batch, DepthBatch) and batch.depth_map is not None:
                target = (batch.gt_mask, depth_to_rgb(batch.depth_map))
                batch.image, target = self.transform(batch.image, target)
                batch.gt_mask, batch.depth_map = target
            else:
                batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    @override
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            if isinstance(batch, DepthBatch) and batch.depth_map is not None:
                target = (batch.gt_mask, depth_to_rgb(batch.depth_map))
                batch.image, target = self.transform(batch.image, target)
                batch.gt_mask, batch.depth_map = target
            else:
                batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    @override
    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            if isinstance(batch, DepthBatch) and batch.depth_map is not None:
                target = (batch.gt_mask, depth_to_rgb(batch.depth_map))
                batch.image, target = self.transform(batch.image, target)
                batch.gt_mask, batch.depth_map = target
            else:
                batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    @override
    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            if isinstance(batch, DepthBatch) and batch.depth_map is not None:
                target = (batch.gt_mask, depth_to_rgb(batch.depth_map))
                batch.image, target = self.transform(batch.image, target)
                batch.gt_mask, batch.depth_map = target
            else:
                batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
