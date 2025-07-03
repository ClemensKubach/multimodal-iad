"""Anomaly detection model abstraction for multimodal-IAD using anomalib 2.0 models."""

import warnings
from typing import Any, ClassVar

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models.components import AnomalibModule
from anomalib.models.image.patchcore import Patchcore
from pydantic import BaseModel

from multimodal_iad.utils.constants import DATASETS_DIR

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")


class PatchcoreConfig(BaseModel):
    """Configuration for Patchcore model."""

    model_cls: ClassVar[type[AnomalibModule]] = Patchcore

    backbone: str = "wide_resnet50_2"
    layers: list[str] = ["layer2", "layer3"]
    pre_trained: bool = True


class AnomalyDetector:
    """General anomaly detection logic using anomalib 2.0 models.

    Supports training and inference on MVTec AD v2 or compatible datasets.
    Allows selection of model type and model-specific kwargs.
    """

    def __init__(
        self,
        model_config: PatchcoreConfig | None = None,
        datamodule_config: None = None,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            model_config: Model-specific configuration.
            datamodule_config: Datamodule-specific configuration.

        """
        self.model_config = model_config or PatchcoreConfig()
        self.datamodule_config = datamodule_config

        self.engine = Engine()
        self.trained = False
        self.model = Patchcore()
        self.datamodule = MVTecAD(
            root=DATASETS_DIR / "mvtec-ad",
            category="bottle",
        )

    def train(self) -> None:
        """Train the anomaly detection model using the provided datamodule.

        Args:
            datamodule: Anomalib datamodule or compatible.

        """
        self.engine.fit(datamodule=self.datamodule, model=self.model)
        self.trained = True

    def test(self) -> None:
        """Test the anomaly detection model using the provided datamodule."""
        self.engine.test(datamodule=self.datamodule, model=self.model)

    def predict(self) -> Any:  # noqa: ANN401
        """Run inference using the trained anomaly detection model.

        Args:
            datamodule: Anomalib datamodule or compatible.

        Returns:
            predictions: List of anomaly predictions (dicts with image, score, label, mask, etc.)

        """
        if not self.trained:
            msg = "Model must be trained first! Call train() before predict()."
            raise RuntimeError(msg)
        return self.engine.predict(datamodule=self.datamodule, model=self.model)


def main() -> None:
    detector = AnomalyDetector()
    detector.train()
    detector.test()
    detector.predict()


if __name__ == "__main__":
    main()
