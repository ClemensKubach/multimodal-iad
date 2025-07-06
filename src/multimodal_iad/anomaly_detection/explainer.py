"""Textual explainer for the anomaly detection results."""

import logging
import os
import random
from typing import Any

from anomalib.data import DepthItem, ImageItem, MVTec3D, MVTecAD, MVTecLOCO
from anomalib.data.dataclasses.numpy.depth import NumpyDepthItem
from anomalib.data.dataclasses.numpy.image import NumpyImageItem
from anomalib.visualization import visualize_image_item
from dotenv import load_dotenv
from google import genai
from google.genai import types
from strenum import StrEnum
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


class SupportedLLMs(StrEnum):
    """Supported LLM models."""

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
    GEMINI_2_5_FLASH_TTS = "gemini-2.5-flash-preview-tts"
    GEMINI_2_5_PRO_TTS = "gemini-2.5-pro-preview-tts"


class TextualAnomalyExplainer:
    """Textual explainer for the anomaly detection results.

    Leverages a LLM to generate a textual explanation for the anomaly detection results.
    """

    def __init__(
        self,
        dataset_category: str,
        datamodule: MVTecAD | MVTec3D | MVTecLOCO,
        num_normal_examples: int = 3,
        model: SupportedLLMs = SupportedLLMs.GEMINI_2_5_FLASH_LITE,
    ) -> None:
        """Initialize the explainer.

        Args:
            dataset_category: The category of the dataset.
            datamodule: The datamodule of the dataset.
            num_normal_examples: The number of normal examples to include in the prompt.
            model: The model to use for the explanation. Currently only supports models part of the
                Google "google-genai" api.

        """
        self.dataset_category = dataset_category
        self.datamodule = datamodule
        self.num_normal_examples = num_normal_examples
        if "tts" in model.value:
            msg = "TTS models are not supported for explanation. They only support audio output."
            raise ValueError(msg)
        self.model = model

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY not found in environment variables."
            raise ValueError(msg)

        self.client = genai.Client(api_key=api_key)
        random.seed(42)
        self.normal_example_files = self._upload_normal_samples()

    def _get_normal_sample_paths(self) -> list[str]:
        """Get random normal sample paths from the training dataset."""
        self.datamodule.setup(stage="fit")
        normal_sample_paths = []
        train_dataset = self.datamodule.train_data
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for i in indices:
            sample: ImageItem | DepthItem = train_dataset[i]  # type: ignore[reportUnknownMemberType]
            if sample.pred_label == 0 and hasattr(sample, "image_path"):
                normal_sample_paths.append(str(sample.image_path))
            if len(normal_sample_paths) == self.num_normal_examples:
                break
        return normal_sample_paths

    def _upload_normal_samples(self) -> list[types.File]:
        """Upload normal samples and return their references."""
        normal_sample_paths = self._get_normal_sample_paths()
        logger.info("Found %d normal samples to upload.", len(normal_sample_paths))
        uploaded_files = []
        for path in normal_sample_paths:
            logger.info("Uploading normal sample: %s", path)
            uploaded_files.append(self.client.upload_file(path=path))  # type: ignore[reportUnknownMemberType]
        return uploaded_files

    def explain(self, item: NumpyImageItem | NumpyDepthItem) -> str | None:
        """Explain the anomaly detection results."""
        if item.pred_label != 1:  # Only explain anomalies
            return None

        try:
            # 1. Prepare data for the prompt
            pred_label_str = "abnormal" if item.pred_label == 1 else "normal"
            pred_score_str = f"{item.pred_score:.4f}" if item.pred_score is not None else "N/A"

            if item.image is None:
                msg = "Input image is None"
                raise ValueError(msg)  # noqa: TRY301
            input_image_pil = to_pil_image(item.image)
            anomaly_map_pil = visualize_image_item(
                item,  # type: ignore[reportUnknownReturnType]
                overlay_fields=[("image", ["anomaly_map", "gt_mask"])],
                fields_config={
                    "anomaly_map": {"normalize": True, "colormap": True},
                    "gt_mask": {"mode": "contour", "color": (255, 255, 255), "alpha": 0.9},
                },
                text_config={"enable": False},
            )

            # 2. Construct the multimodal prompt
            system_instruction = (
                f"You are an expert in industrial anomaly detection in {self.dataset_category} images."
                "Your task is to explain in a concise way (max 1 sentences) why the provided image is "
                f"classified as {pred_label_str} with a score of {pred_score_str} (of range 0-1). "
                "Only explain the anomaly by describing what is wrong with the product not the image."
                f"Always understand first the image and perspective of the {self.dataset_category} product image "
                "before explaining the anomaly."
            )
            user_prompt_parts: list[Any] = []
            if anomaly_map_pil is not None:
                user_prompt_parts.extend(
                    [
                        "The input image and the predicted anomaly map are provided, where brighter areas indicate a "
                        "higher likelihood of anomaly.",
                        "Be factual and base your explanation on the visual evidence but do not use the anomaly map "
                        "in your explanation. Only use the anomaly map to understand the anomaly.",
                        "---",
                        "Input Image:",
                        input_image_pil,
                        "Anomaly Map:",
                        anomaly_map_pil,
                    ],
                )
            else:
                user_prompt_parts.extend(
                    [
                        "The input image is provided.",
                        "Be factual and base your explanation on the visual evidence.",
                        "---",
                        "Input Image:",
                        input_image_pil,
                    ],
                )

            user_prompt_parts.extend(
                [
                    "---",
                    "For reference, here are some examples of ground truth normal images of the same product category:",
                    *self.normal_example_files,
                    "---",
                    f"Start your explanation with 'The {self.dataset_category} is {pred_label_str} because...'",
                ],
            )

            # 3. Generate explanation
            logger.info("Generating textual explanation for anomaly...")
            response = self.client.models.generate_content(
                model=self.model.value,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_modalities=["Text"],
                ),
                contents=user_prompt_parts,  # type: ignore[reportUnknownMemberType]
            )
            logger.info("Explanation generated successfully.")
        except Exception as e:
            logger.exception("Failed to generate explanation from LLM.")
            return f"Error generating explanation: {e}"
        else:
            return response.text
