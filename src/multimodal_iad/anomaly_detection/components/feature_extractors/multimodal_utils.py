"""Multimodal feature extractor utils and configurations."""

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import torch
from anomalib.data import Batch, DepthBatch
from pydantic import BaseModel, Field, StringConstraints

ChannelMarker: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[A-Z]$")]
ModalityMarker: TypeAlias = Annotated[Sequence[ChannelMarker], Field(min_length=1)]


class FeatureExtractorConfig(BaseModel):
    """Feature extractor configuration."""

    modality_marker: ModalityMarker
    backbone: str = "wide_resnet50_2"
    layers: Sequence[str] = ("layer2", "layer3")
    pre_trained: bool = True
    channel_mode: Literal["adjust_arch", "adjust_input"] = "adjust_input"


class FeatureExtractorModality(Enum):
    """Feature extractor configurations."""

    RGB = FeatureExtractorConfig(modality_marker=("R", "G", "B"))
    DEPTH = FeatureExtractorConfig(modality_marker=("D",))


def get_multimodal_batch_tensor(
    batch: Batch | DepthBatch,
) -> torch.Tensor:
    """Aggregate data tensors from a batch.

    Args:
        batch: Batch of data.
        feature_extractor_configs: Sequence of feature extractor configurations.

    Returns:
        Aggregated tensor.

    """
    rgb = batch.image
    depth = batch.depth_map if isinstance(batch, DepthBatch) else None
    if depth is not None:
        if depth.shape[-3] != 3:  # noqa: PLR2004
            msg = f"Depth map must have been converted to RGB already but has channel size {depth.shape[-3]}."
            # because currently only adjust_input is just implemented
            raise ValueError(msg)
        return torch.cat([rgb, depth], dim=1, device=rgb.device)  # type: ignore[arg-type]
    return rgb  # type: ignore[return-value]
