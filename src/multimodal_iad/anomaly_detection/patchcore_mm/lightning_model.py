"""Patchcore multimodal lightning model."""

import logging
from collections.abc import Sequence
from typing import Any

import torch
from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize
from typing_extensions import override

from multimodal_iad.anomaly_detection.components.feature_extractors.multimodal_utils import (
    FeatureExtractorConfig,
    FeatureExtractorModality,
    get_multimodal_batch_tensor,
)
from multimodal_iad.anomaly_detection.components.multimodal_pre_processor import MultimodalPreProcessor
from multimodal_iad.anomaly_detection.patchcore_mm.torch_model import PatchcoreMultimodalModel

logger = logging.getLogger(__name__)


class PatchcoreMultimodal(MemoryBankMixin, AnomalibModule):
    """Patchcore multimodal model."""

    def __init__(
        self,
        feature_extractor_configs: Sequence[FeatureExtractorConfig] = (FeatureExtractorModality.RGB.value,),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        """Initialize the PatchcoreMultimodal model.

        Args:
            feature_extractor_configs: Sequence of feature extractor configurations.
            coreset_sampling_ratio: Ratio of the coreset sampling.
            num_neighbors: Number of neighbors to use for the coreset sampling.
            pre_processor: Pre-processor to use.
            post_processor: Post-processor to use.
            evaluator: Evaluator to use.
            visualizer: Visualizer to use.

        """
        self.feature_extractor_configs = feature_extractor_configs

        if pre_processor:
            # override the default pre-processor with a multimodal pre-processor and transform
            pre_processor = self._dynamic_default_pre_processor(feature_extractor_configs=feature_extractor_configs)

        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: PatchcoreMultimodalModel = PatchcoreMultimodalModel(
            feature_extractor_configs=feature_extractor_configs,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []

    @classmethod
    def _dynamic_default_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
        feature_extractor_configs: Sequence[FeatureExtractorConfig] = (FeatureExtractorModality.RGB.value,),
    ) -> MultimodalPreProcessor:
        """Dynamic default pre-processor."""
        image_size = image_size or (256, 256)

        norm_mean = []
        norm_std = []
        rgb_norm_mean = [0.485, 0.456, 0.406]
        rgb_norm_std = [0.229, 0.224, 0.225]
        # for config in feature_extractor_configs:
        #     if config.channel_mode == "adjust_arch":
        #         msg = "Adjusting the architecture is not yet implemented for the default pre-processor."
        #         raise NotImplementedError(msg)
        #     if config.modality_marker in (
        #         FeatureExtractorModality.RGB.value.modality_marker,
        #         FeatureExtractorModality.DEPTH.value.modality_marker,
        #     ):
        #         norm_mean.extend(rgb_norm_mean)
        #         norm_std.extend(rgb_norm_std)
        #     else:
        #         msg = f"Mobility marker {config.modality_marker} is not yet implemented for the default pre-processor."
        #         raise NotImplementedError(msg)
        norm_mean.extend(rgb_norm_mean)
        norm_std.extend(rgb_norm_std)
        logger.info(f"CLEMENS Preprocessor uses norm_mean: {norm_mean}")
        logger.info(f"CLEMENS Preprocessor uses norm_std: {norm_std}")

        if center_crop_size is not None:
            if center_crop_size[0] > image_size[0] or center_crop_size[1] > image_size[1]:
                msg = f"Center crop size {center_crop_size} cannot be larger than image size {image_size}."
                raise ValueError(msg)
            transform = Compose(
                [
                    Resize(image_size, antialias=True),
                    CenterCrop(center_crop_size),
                    Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        else:
            transform = Compose(
                [
                    Resize(image_size, antialias=True),
                    Normalize(mean=norm_mean, std=norm_std),
                ]
            )

        return MultimodalPreProcessor(transform=transform)

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for PatchCore.

        If valid center_crop_size is provided, the pre-processor will
        also perform center cropping, according to the paper.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.
            center_crop_size (tuple[int, int] | None, optional): Size for center
                cropping. Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Raises:
            ValueError: If at least one dimension of ``center_crop_size`` is larger
                than correspondent ``image_size`` dimension.

        Example:
            >>> pre_processor = Patchcore.configure_pre_processor(image_size=(256, 256))
            >>> transformed_image = pre_processor(image)

        """
        return cls._dynamic_default_pre_processor(image_size=image_size, center_crop_size=center_crop_size)

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        Returns:
            None: PatchCore requires no optimization.

        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Generate feature embedding of the batch.

        Args:
            batch (Batch): Input batch containing image and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility

        Note:
            The method stores embeddings in ``self.embeddings`` for later use in
            ``fit()``.

        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(get_multimodal_batch_tensor(batch))
        self.embeddings.append(embedding)
        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Apply subsampling to the embedding collected from the training set.

        This method:
        1. Aggregates embeddings from all training batches
        2. Applies coreset subsampling to reduce memory requirements
        """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for a batch of images.

        Args:
            batch (Batch): Input batch containing images and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Batch with added predictions

        Note:
            Predictions include anomaly maps and scores computed using nearest
            neighbor search.

        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(get_multimodal_batch_tensor(batch))

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get default trainer arguments for PatchCore.

        Returns:
            dict[str, Any]: Trainer arguments
                - ``gradient_clip_val``: ``0`` (no gradient clipping needed)
                - ``max_epochs``: ``1`` (single pass through training data)
                - ``num_sanity_val_steps``: ``0`` (skip validation sanity checks)

        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS`` as PatchCore only
                trains on normal samples

        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Post-processor for one-class models that
                converts raw scores to anomaly predictions

        """
        return PostProcessor()

    @override
    def forward(self, batch: torch.Tensor, *args, **kwargs) -> InferenceBatch:
        del args, kwargs  # These variables are not used.
        batch = self.pre_processor(batch) if self.pre_processor else batch
        batch = self.model(get_multimodal_batch_tensor(batch))  # type: ignore[arg-type]
        return self.post_processor(batch) if self.post_processor else batch  # type: ignore[return-value]
