"""Multimodal Timm feature extractor."""

from collections.abc import Sequence

import timm
from anomalib.models.components.feature_extractors.timm import TimmFeatureExtractor
from torch import nn
from typing_extensions import override


class MultimodalTimmFeatureExtractor(TimmFeatureExtractor):
    """Multimodal Timm feature extractor.

    Same as the original TimmFeatureExtractor, but with parameterizable number of input channels.
    """

    @override
    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
        in_channels: int | None = None,
    ) -> None:
        super().__init__(backbone, layers, pre_trained, requires_grad)
        if isinstance(backbone, str):
            self.idx = self._map_layer_to_idx()
            if in_channels is None:
                self.feature_extractor = timm.create_model(
                    backbone,
                    pretrained=pre_trained,
                    pretrained_cfg=None,
                    features_only=True,
                    exportable=True,
                    out_indices=self.idx,
                )
            else:
                self.feature_extractor = timm.create_model(
                    backbone,
                    pretrained=pre_trained,
                    pretrained_cfg=None,
                    features_only=True,
                    exportable=True,
                    out_indices=self.idx,
                    in_chans=in_channels,
                )
            self.out_dims = self.feature_extractor.feature_info.channels()
