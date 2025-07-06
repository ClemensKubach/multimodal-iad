"""Multimodal Patchcore torch model."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torch import nn
from torch.nn import functional as F  # noqa: N812

from multimodal_iad.anomaly_detection.components.feature_extractors.multimodal_utils import FeatureExtractorConfig

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler

logger = logging.getLogger(__name__)


class PatchcoreMultimodalModel(DynamicBufferMixin, nn.Module):
    """Anomalib torch implementation of Patchcore but for multimodal inputs."""

    def __init__(
        self,
        feature_extractor_configs: Sequence[FeatureExtractorConfig],
        num_neighbors: int = 9,
    ) -> None:
        """Initialize the PatchcoreMultimodalModel.

        Args:
            feature_extractor_configs: Sequence of feature extractor configurations.
            num_neighbors: Number of neighbors to use for the coreset sampling.

        """
        super().__init__()
        self.tiler: Tiler | None = None

        self.feature_extractor_configs = feature_extractor_configs
        self.backbone = self.feature_extractor_configs[0].backbone
        self.layers = self.feature_extractor_configs[0].layers
        self.num_neighbors = num_neighbors

        self.feature_extractors = nn.ModuleList()
        self.input_channel_slices = []
        current_slice_start = 0

        for config in self.feature_extractor_configs:
            self.feature_extractors.append(
                TimmFeatureExtractor(  # MultimodalTimmFeatureExtractor(
                    backbone=config.backbone,
                    pre_trained=config.pre_trained,
                    layers=config.layers,
                    # in_channels=len(config.modality_marker) if config.channel_mode == "adjust_arch" else None,
                ).eval()
            )
            if config.backbone != self.backbone:
                msg = "Backbones must be the same for all feature extractors"
                raise ValueError(msg)
            if config.layers != self.layers:
                msg = "Layers must be the same for all feature extractors"
                raise ValueError(msg)

            channel_count = len(config.modality_marker) if config.channel_mode == "adjust_arch" else 3  # default to rgb
            current_slice_end = current_slice_start + channel_count
            self.input_channel_slices.append(slice(current_slice_start, current_slice_end))
            current_slice_start = current_slice_end
        logger.info("Input channel slices: %s", self.input_channel_slices)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input tensor through the model.

        During training, returns embeddings extracted from the input. During
        inference, returns anomaly maps and scores computed by comparing input
        embeddings against the memory bank.

        Args:
            input_tensor (torch.Tensor): Input images of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns embeddings.
                During inference, returns ``InferenceBatch`` containing anomaly
                maps and scores.

        Example:
            >>> model = PatchcoreModel(layers=["layer1"])
            >>> input_tensor = torch.randn(32, 3, 224, 224)
            >>> output = model(input_tensor)
            >>> if model.training:
            ...     assert isinstance(output, torch.Tensor)
            ... else:
            ...     assert isinstance(output, InferenceBatch)

        """
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        multimodal_features = []
        if self.input_channel_slices[-1].stop > input_tensor.shape[1]:
            logger.warning("Input channel slices: %s", self.input_channel_slices)
            logger.warning("Input tensor: %s", input_tensor.shape)
        with torch.no_grad():
            for feature_extractor, input_channel_slice in zip(
                self.feature_extractors,
                self.input_channel_slices,
                strict=False,
            ):
                sliced_tensor = input_tensor[:, input_channel_slice, ...]
                multimodal_features.append(feature_extractor(sliced_tensor))

        features = {}
        layer_features = {}
        for features in multimodal_features:
            for layer in features:
                if layer not in layer_features:
                    layer_features[layer] = []
                layer_features[layer].append(features[layer])
        for layer, features_list in layer_features.items():
            features[layer] = torch.cat(features_list, dim=1)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = PatchcoreModel.reshape_embedding(embedding)

        if self.training:
            return embedding
        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding by concatenating multi-scale feature maps.

        Combines feature maps from different CNN layers by upsampling them to a
        common size and concatenating along the channel dimension.

        Args:
            features (dict[str, torch.Tensor]): Dictionary mapping layer names to
                feature tensors extracted from the backbone CNN.

        Returns:
            torch.Tensor: Concatenated feature embedding of shape
                ``(batch_size, num_features, height, width)``.

        Example:
            >>> features = {"layer1": torch.randn(32, 64, 56, 56), "layer2": torch.randn(32, 128, 28, 28)}
            >>> embedding = model.generate_embedding(features)
            >>> embedding.shape
            torch.Size([32, 192, 56, 56])

        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embeddings using coreset selection.

        Copied from anomalib Patchcore without (logical) modifications.

        Uses k-center-greedy coreset subsampling to select a representative
        subset of patch embeddings to store in the memory bank.

        Args:
            embedding (torch.Tensor): Embedding tensor to subsample from.
            sampling_ratio (float): Fraction of embeddings to keep, in range (0,1].

        Example:
            >>> embedding = torch.randn(1000, 512)
            >>> model.subsample_embedding(embedding, sampling_ratio=0.1)
            >>> model.memory_bank.shape
            torch.Size([100, 512])

        """
        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest neighbors in memory bank for input embeddings.

        Copied from anomalib Patchcore without (logical) modifications.

        Uses brute force search with Euclidean distance to find the closest
        matches in the memory bank for each input embedding.

        Args:
            embedding (torch.Tensor): Query embeddings to find neighbors for.
            n_neighbors (int): Number of nearest neighbors to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Distances to nearest neighbors (shape: ``(n, k)``)
                - Indices of nearest neighbors (shape: ``(n, k)``)
                where ``n`` is number of query embeddings and ``k`` is
                ``n_neighbors``.

        Example:
            >>> embedding = torch.randn(100, 512)
            >>> # Assuming memory_bank is already populated
            >>> scores, locations = model.nearest_neighbors(embedding, n_neighbors=5)
            >>> scores.shape, locations.shape
            (torch.Size([100, 5]), torch.Size([100, 5]))

        """
        distances = PatchcoreModel.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image-level anomaly scores.

        Copied from anomalib Patchcore without (logical) modifications.

        Implements the paper's weighted scoring mechanism that considers both
        the distance to nearest neighbors and the local neighborhood structure
        in the memory bank.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores.
            locations (torch.Tensor): Memory bank indices of nearest neighbors.
            embedding (torch.Tensor): Input embeddings that generated the scores.

        Returns:
            torch.Tensor: Image-level anomaly scores.

        Example:
            >>> patch_scores = torch.randn(32, 49)  # 7x7 patches
            >>> locations = torch.randint(0, 1000, (32, 49))
            >>> embedding = torch.randn(32 * 49, 512)
            >>> scores = model.compute_anomaly_score(patch_scores, locations, embedding)
            >>> scores.shape
            torch.Size([32])

        Note:
            When ``num_neighbors=1``, returns the maximum patch score directly.
            Otherwise, computes weighted scores using neighborhood information.

        """
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = PatchcoreModel.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper
