"""Dataset for performing inference on (depth) images.

This module provides a dataset class for loading and preprocessing (depth) images for
inference in anomaly detection tasks.

Example:
    >>> from pathlib import Path
    >>> from anomalib.data import PredictDataset
    >>> dataset = PredictDataset(path="path/to/images")
    >>> item = dataset[0]
    >>> item.image.shape  # doctest: +SKIP
    torch.Size([3, 256, 256])

"""

from collections.abc import Callable
from pathlib import Path

from anomalib.data import DepthBatch, DepthItem, PredictDataset
from anomalib.data.utils import get_image_filenames, read_depth_image, read_image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image
from typing_extensions import override


class MultimodalPredictDataset(PredictDataset):
    """Dataset for performing inference on (depth) images.

    Args:
        path (str | Path): Path to an image or directory containing images.
        transform (Transform | None, optional): Transform object describing the
            transforms to be applied to the inputs. Defaults to ``None``.
        image_size (int | tuple[int, int], optional): Target size to which input
            images will be resized. If int, a square image of that size will be
            created. Defaults to ``(256, 256)``.

    Examples:
        >>> from pathlib import Path
        >>> dataset = PredictDataset(
        ...     path=Path("path/to/images"),
        ...     image_size=(224, 224),
        ... )
        >>> len(dataset)  # doctest: +SKIP
        10
        >>> item = dataset[0]  # doctest: +SKIP
        >>> item.image.shape  # doctest: +SKIP
        torch.Size([3, 224, 224])

    """

    @override
    def __init__(
        self,
        path: str | Path,
        depth_path: str | Path | None = None,
        transform: Transform | None = None,
        image_size: int | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__(path=path, transform=transform, image_size=image_size)

        self.image_filenames = get_image_filenames(path)
        if depth_path:
            self.depth_filenames = get_image_filenames(depth_path)
        else:
            self.depth_filenames = None

    def __len__(self) -> int:
        """Get number of images in dataset.

        Returns:
            int: Number of images in the dataset.

        """
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> DepthItem:
        """Get image item at specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            ImageItem: Object containing the loaded image and its metadata.

        """
        image_filename = self.image_filenames[index]
        if self.depth_filenames:
            depth_filename = self.depth_filenames[index]
            depth_map = Image(to_tensor(read_depth_image(depth_filename)))
        else:
            depth_map = None
        image = Image(read_image(image_filename, as_tensor=True))
        if self.transform:
            if depth_map is not None:
                image, depth_map = self.transform(image, depth_map)
            else:
                image = self.transform(image)

        return DepthItem(
            image=image,
            depth_map=depth_map,
            image_path=str(image_filename),
            depth_path=str(depth_filename) if depth_map else None,
        )

    @property
    def collate_fn(self) -> Callable:
        """Get collate function for creating batches.

        Returns:
            Callable: Function that collates multiple ``ImageItem`` instances into
                a batch.

        """
        return DepthBatch.collate
