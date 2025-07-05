"""Textual explainer for the anomaly detection results."""

from anomalib.data.dataclasses.numpy.depth import NumpyDepthItem
from anomalib.data.dataclasses.numpy.image import NumpyImageItem


class TextualAnomalyExplainer:
    """Textual explainer for the anomaly detection results.

    Leverages a LLM to generate a textual explanation for the anomaly detection results.
    """

    def __init__(self) -> None:
        """Initialize the explainer."""

    def explain(self, item: NumpyImageItem | NumpyDepthItem) -> str:
        """Explain the anomaly detection results."""
        return "This is a placeholder explanation"
