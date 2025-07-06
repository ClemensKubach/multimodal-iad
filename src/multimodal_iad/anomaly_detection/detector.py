"""Anomaly detection model abstraction for multimodal-IAD using anomalib 2.0 models."""

import logging
from enum import auto
from pathlib import Path

from anomalib.data import (
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    MVTec3D,
    MVTecAD,
    MVTecLOCO,
    NumpyImageItem,
)
from anomalib.data.dataclasses.numpy.depth import NumpyDepthItem
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore
from strenum import StrEnum

from multimodal_iad.anomaly_detection.components.feature_extractors.multimodal_utils import FeatureExtractorModality
from multimodal_iad.anomaly_detection.explainer import TextualAnomalyExplainer
from multimodal_iad.anomaly_detection.patchcore_mm.lightning_model import PatchcoreMultimodal
from multimodal_iad.anomaly_detection.predict_dataset import MultimodalPredictDataset
from multimodal_iad.utils.constants import DATASETS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class SupportedAdModels(StrEnum):
    """Supported anomaly detection models."""

    Patchcore = auto()
    PatchcoreMultimodal = auto()


class SupportedDatamodules(StrEnum):
    """Supported datamodules."""

    MVTecAD = auto()
    MVTec3D = auto()
    MVTecAD_LOCO = auto()


class AnomalyDetector:
    """Anomaly detection using anomalib 2.0 models with GUI support."""

    def __init__(
        self,
        dataset_category: str = "bottle",
        datasets_dir: Path = DATASETS_DIR,
        datamodule: SupportedDatamodules = SupportedDatamodules.MVTecAD,
        model: SupportedAdModels = SupportedAdModels.Patchcore,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            dataset_category: Category if any in the dataset to use.
            datasets_dir: Directory containing the datasets.
            datamodule: Datamodule to use.
            model: Anomaly detection model to use.

        """
        self.datasets_dir = datasets_dir
        self.dataset_category = dataset_category
        self.selected_model = model
        self.selected_datamodule = datamodule
        self.results_dir = RESULTS_DIR
        self.checkpoint_path: Path | None = None

        if self.selected_model == SupportedAdModels.Patchcore:
            self.model = Patchcore()  # post_processor=PostProcessor(enable_thresholding=False))
        elif self.selected_model == SupportedAdModels.PatchcoreMultimodal:
            fe_configs = [FeatureExtractorModality.RGB.value]
            if self.selected_datamodule == SupportedDatamodules.MVTec3D:
                fe_configs.append(FeatureExtractorModality.DEPTH.value)
            self.model = PatchcoreMultimodal(feature_extractor_configs=fe_configs)
        else:
            msg = f"Model {self.selected_model} not supported"
            raise ValueError(msg)

        if self.selected_datamodule == SupportedDatamodules.MVTecAD:
            self.datamodule = MVTecAD(
                root=self.datasets_dir / "mvtec-ad",
                category=self.dataset_category,
                train_batch_size=32,
                eval_batch_size=32,
            )
        elif self.selected_datamodule == SupportedDatamodules.MVTec3D:
            self.datamodule = MVTec3D(
                root=self.datasets_dir / "mvtec_3d_anomaly_detection",
                category=self.dataset_category,
                train_batch_size=32,
                eval_batch_size=32,
            )
        elif self.selected_datamodule == SupportedDatamodules.MVTecAD_LOCO:
            self.datamodule = MVTecLOCO(
                root=self.datasets_dir / "mvtec-loco-ad",
                category=self.dataset_category,
                train_batch_size=32,
                eval_batch_size=32,
            )
        else:
            msg = f"Datamodule {self.selected_datamodule} not supported"
            raise ValueError(msg)

        self.explainer = TextualAnomalyExplainer(dataset_category=self.dataset_category, datamodule=self.datamodule)

        self.engine = Engine()
        self.trained = False

    def find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint for the current model and dataset."""
        model_name = self.model.__class__.__name__
        datamodule_class_name = self.datamodule.__class__.__name__

        experiment_path = self.results_dir / model_name / datamodule_class_name / self.dataset_category

        latest_version_dir = experiment_path / "latest"

        if not latest_version_dir.exists():
            logger.warning("No 'latest' checkpoint directory found in %s", experiment_path)
            return None

        checkpoints_dir = latest_version_dir / "weights" / "lightning"
        if not checkpoints_dir.exists():
            logger.warning("Weights directory not found in %s", checkpoints_dir)
            return None

        checkpoints = list(checkpoints_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.warning("No checkpoint file found in %s", checkpoints_dir)
            return None

        return checkpoints[0]

    def load_checkpoint(self) -> bool:
        """Load model from the latest available checkpoint.

        Returns:
            True if a checkpoint was found and loaded, False otherwise.

        """
        self.checkpoint_path = self.find_latest_checkpoint()
        if self.checkpoint_path:
            logger.info("Found checkpoint: %s", self.checkpoint_path)
            self.trained = True
            return True

        model_name = self.model.__class__.__name__
        datamodule_class_name = self.datamodule.__class__.__name__
        experiment_path = self.results_dir / model_name / datamodule_class_name / self.dataset_category
        logger.warning(
            "No checkpoint found for %s in %s.",
            model_name,
            experiment_path,
        )
        self.trained = False
        return False

    def train(self) -> None:
        """Train the anomaly detection model.

        Args:
            progress_callback: Optional callback for progress updates

        """
        logger.info("Training %s on %s/%s...", self.selected_model, self.selected_datamodule, self.dataset_category)
        self.checkpoint_path = None  # Reset checkpoint path before training
        self.datamodule.setup()
        self.engine.fit(datamodule=self.datamodule, model=self.model)
        self.trained = True

    def test(self) -> dict[str, float]:
        """Test the model and return metrics."""
        if not self.trained:
            msg = "Model must be trained first!"
            raise RuntimeError(msg)

        logger.info("Testing model...")
        test_results = self.engine.test(
            datamodule=self.datamodule,
            model=self.model,
            ckpt_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )
        first_dataloader_results = test_results[0]
        return dict(first_dataloader_results)

    def predict_image(
        self, sample: ImageItem | DepthItem | None = None, image_path: str | None = None
    ) -> NumpyImageItem | NumpyDepthItem | None:
        """Apply anomaly detection for a single image.

        Args:
            sample: Sample from the dataset.
            image_path: Path to the image file.

        Returns:
            Dictionary containing prediction results

        """
        if not self.trained:
            msg = "Model must be trained first!"
            raise RuntimeError(msg)

        if sample is None and image_path is None:
            msg = "Either sample or image_path must be provided"
            raise ValueError(msg)

        image_path = sample.image_path if sample is not None and sample.image_path is not None else image_path
        if image_path is None:
            msg = "Image path is required"
            raise ValueError(msg)

        if self.selected_datamodule == SupportedDatamodules.MVTec3D:
            depth_path = image_path.replace("rgb", "xyz").replace(".png", ".tiff")
        else:
            depth_path = None

        logger.info("Predicting image %s...", image_path)
        dataset = MultimodalPredictDataset(
            path=image_path,
            depth_path=depth_path,
            image_size=(256, 256),
        )

        # Get predictions
        predictions = self.engine.predict(
            model=self.model,
            dataset=dataset,
            ckpt_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )

        if predictions and len(predictions) > 0:
            pred = predictions[0]
            if isinstance(pred, list):
                pred = pred[0]

            if isinstance(pred, ImageBatch | DepthBatch):
                item = pred.items[0]
            elif isinstance(pred, ImageItem | DepthItem):
                item = pred
            else:
                msg = f"Unsupported prediction type: {type(pred)}"
                raise ValueError(msg)

            # augmentate the predictions if sample is provided
            if sample is not None:
                item.gt_label = sample.gt_label
                item.gt_mask = sample.gt_mask

            if isinstance(item, DepthItem):
                # quick fix because DepthItem extends NumpyImageItem
                item.numpy_class = NumpyDepthItem  # type: ignore[reportAttributeAccessIssue]
                if item.depth_map is not None:
                    # quick fix order of channels
                    item.depth_map = None
            return item.to_numpy()  # type: ignore[reportUnknownReturnType]

        return None

    def get_sample_from_dataset(self, split: str = "test", index: int = 0) -> NumpyImageItem | NumpyDepthItem | None:
        """Get a sample from the dataset with ground truth.

        Args:
            split: Dataset split ('train', 'val', 'test')
            index: Index of the sample

        Returns:
            Dictionary containing sample data

        """
        self.datamodule.setup()

        if split == "test":
            dataset = self.datamodule.test_data
        elif split == "val":
            dataset = self.datamodule.val_data
        else:
            dataset = self.datamodule.train_data

        if index >= len(dataset):
            logger.warning(
                "Index %s is out of range for %s dataset with length %s. Resetting to 0.",
                index,
                split,
                len(dataset),
            )
            index = 0

        sample: ImageItem | DepthItem = dataset[index]  # type: ignore[reportUnknownReturnType]
        return self.predict_image(sample=sample)

    def generate_explanation(self, _result: NumpyImageItem | NumpyDepthItem) -> str:
        """Generate textual explanation for the prediction."""
        explanation = self.explainer.explain(_result)
        _result.explanation = explanation  # type: ignore[reportAttributeAccessIssue]
        return explanation or "This image is predicted to be normal. No explanation is needed."


if __name__ == "__main__":
    detector = AnomalyDetector(dataset_category="bottle")
    detector.predict_image(image_path="/Users/clemens/Datasets/mvtec-ad/bottle/test/broken_small/001.png")
