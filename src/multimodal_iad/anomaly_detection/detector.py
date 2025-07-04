"""Anomaly detection model abstraction for multimodal-IAD using anomalib 2.0 models."""

import logging
from pathlib import Path

import numpy as np
from anomalib.data import MVTec3D, MVTecAD, PredictDataset
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore
from PIL import Image
from pydantic import BaseModel
from strenum import StrEnum

from multimodal_iad.utils.constants import DATASETS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class SupportedAdModels(StrEnum):
    """Supported anomaly detection models."""

    Patchcore = "patchcore"


class SupportedDatamodules(StrEnum):
    """Supported datamodules."""

    MVTecAD = "mvtec-ad"
    MVTec3D = "mvtec_3d_anomaly_detection"


class AnomalyDetectorResult(BaseModel):
    """Result of anomaly detection."""

    model_config = {"arbitrary_types_allowed": True}

    image: np.ndarray | None = None
    pred_label: int | None = None
    pred_score: float | None = None
    anomaly_map: np.ndarray | None = None
    pred_mask: np.ndarray | None = None
    gt_label: int | None = None
    gt_mask: np.ndarray | None = None
    image_path: str | None = None


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
            self.model = Patchcore()
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
                root=self.datasets_dir / "mvtec-3d",
                category=self.dataset_category,
                train_batch_size=32,
                eval_batch_size=32,
            )
        else:
            msg = f"Datamodule {self.selected_datamodule} not supported"
            raise ValueError(msg)

        self.engine = Engine()
        self.trained = False

    def find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint for the current model and dataset."""
        model_name = self.model.__class__.__name__
        datamodule_class_name = self.datamodule.__class__.__name__

        # Correct path structure based on user feedback.
        # Example: results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt
        experiment_path = self.results_dir / model_name / datamodule_class_name / self.dataset_category

        # The 'latest' directory is a symlink to the latest version.
        latest_version_dir = experiment_path / "latest"

        if not latest_version_dir.exists():
            logger.warning("No 'latest' checkpoint directory found in %s", experiment_path)
            return None

        # Checkpoint is in weights/lightning/
        checkpoints_dir = latest_version_dir / "weights" / "lightning"
        if not checkpoints_dir.exists():
            logger.warning("Weights directory not found in %s", checkpoints_dir)
            return None

        # Find the checkpoint file (usually named 'model.ckpt' or similar)
        checkpoints = list(checkpoints_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.warning("No checkpoint file found in %s", checkpoints_dir)
            return None

        # Assuming there is one .ckpt file, or we take the first one
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
        logger.info("Training completed!")

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

    def predict_image(self, image_path: str) -> AnomalyDetectorResult | None:
        """Apply anomaly detection for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing prediction results

        """
        if not self.trained:
            msg = "Model must be trained first!"
            raise RuntimeError(msg)

        logger.info("Predicting image %s...", image_path)
        image = Image.open(image_path).convert("RGB")
        dataset = PredictDataset(
            path=image_path,
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

            # Extract results
            return AnomalyDetectorResult(
                image=np.array(image),
                pred_label=pred.pred_label.item() if hasattr(pred, "pred_label") else 0,
                pred_score=pred.pred_score.item() if hasattr(pred, "pred_score") else 0.0,
                anomaly_map=pred.anomaly_map.cpu().numpy() if hasattr(pred, "anomaly_map") else None,
                pred_mask=pred.pred_mask.cpu().numpy() if hasattr(pred, "pred_mask") else None,
                image_path=image_path,
            )

        return None

    def get_sample_from_dataset(self, split: str = "test", index: int = 0) -> AnomalyDetectorResult | None:
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

        sample = dataset[index]

        # Get image path and ground truth
        image_path = getattr(sample, "image_path", None)
        if image_path is None:
            logger.warning("Sample at index %s has no image_path.", index)
            return None

        label = getattr(sample, "label", None)
        gt_label = label.item() if label is not None else 0

        mask = getattr(sample, "mask", None)
        gt_mask = mask.cpu().numpy() if mask is not None else None

        # Get prediction
        result = self.predict_image(image_path)
        if result is None:
            return None

        # Add ground truth info
        result.gt_label = int(gt_label)
        result.gt_mask = gt_mask
        result.image_path = image_path

        return result

    def generate_explanation(self, _result: AnomalyDetectorResult) -> str:
        """Generate textual explanation for the prediction."""
        return "This is a placeholder explanation"
