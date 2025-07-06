"""Main window for the Multimodal-IAD PyQt6 GUI application."""

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from anomalib.data import NumpyImageItem
from anomalib.data.dataclasses.numpy.depth import NumpyDepthItem
from anomalib.visualization import visualize_image_item
from PIL import Image
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import override

mpl.use("QtAgg")
from anomalib.data.datasets.depth.mvtec_3d import CATEGORIES as MVTEC3D_CATEGORIES
from anomalib.data.datasets.image.mvtec_loco import CATEGORIES as MVTEC_LOCO_CATEGORIES
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTECAD_CATEGORIES
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from multimodal_iad.anomaly_detection.detector import (
    AnomalyDetector,
    SupportedAdModels,
    SupportedDatamodules,
)
from multimodal_iad.utils.constants import DATASETS_DIR, GRAYSCALE_IMAGE_DIMS

logger = logging.getLogger(__name__)

TOP_BOTTOM_SECTION_HEIGHT = 150


class QtStream(QObject):
    """Stream to redirect stdout/stderr to a QTextEdit."""

    new_text = pyqtSignal(str)

    def write(self, text: str) -> None:
        """Write text to the stream."""
        self.new_text.emit(str(text))

    def flush(self) -> None:
        """Flush the stream."""
        # No-op, as signals are dispatched immediately.


class TrainingThread(QThread):
    """Thread for training the model without blocking the GUI."""

    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, detector: AnomalyDetector) -> None:
        """Initialize the training thread."""
        super().__init__()
        self.detector = detector

    def run(self) -> None:
        """Run the training process."""
        try:
            self.progress.emit("Starting training...")
            self.detector.train()
            self.progress.emit("Training completed!")
            self.finished.emit()
        except Exception as e:
            logger.exception("Training failed.")
            self.error.emit(str(e))


class ExplanationThread(QThread):
    """Thread for generating explanations without blocking the GUI."""

    explanation_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, detector: AnomalyDetector, result: NumpyImageItem | NumpyDepthItem) -> None:
        """Initialize the explanation thread."""
        super().__init__()
        self.detector = detector
        self.result = result
        self.is_cancelled = False

    def run(self) -> None:
        """Run the explanation generation process."""
        try:
            explanation = self.detector.generate_explanation(self.result)
            if not self.is_cancelled:
                self.explanation_ready.emit(explanation)
        except Exception as e:
            if not self.is_cancelled:
                logger.exception("Explanation generation failed.")
                self.error.emit(str(e))

    def cancel(self) -> None:
        """Mark the thread for cancellation."""
        self.is_cancelled = True


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling."""

    def __init__(self, title: str = "") -> None:
        """Initialize the image label."""
        super().__init__()
        self.title = title
        self.setMinimumSize(300, 300)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: #f5f5f5;
                padding: 10px;
            }
        """)

    def set_image(self, image: np.ndarray | None) -> None:
        """Set image from numpy array."""
        if image is None:
            self.setText(f"No {self.title}")
            self.setPixmap(QPixmap())
            return

        if len(image.shape) == GRAYSCALE_IMAGE_DIMS:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(
                image.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            )
        else:  # RGB
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                image.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled_pixmap)


class HeatmapWidget(FigureCanvas):
    """Widget for displaying anomaly heatmap using matplotlib."""

    def __init__(self) -> None:
        """Initialize the heatmap widget."""
        self.figure = Figure(figsize=(5, 5), dpi=100)
        super().__init__(self.figure)
        self.setMinimumSize(300, 300)

    def update_heatmap(
        self,
        item: NumpyImageItem | NumpyDepthItem | None,
    ) -> None:
        """Update the heatmap display."""
        visualization = None
        if item is not None:
            visualization = visualize_image_item(
                item,  # type: ignore[reportUnknownReturnType]
                overlay_fields=[("image", ["anomaly_map", "gt_mask"])],
                fields_config={
                    "anomaly_map": {"normalize": True, "colormap": True},
                    "gt_mask": {"mode": "contour", "color": (255, 255, 255), "alpha": 0.9},
                },
                text_config={"enable": False},
            )

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if visualization is None:
            ax.text(
                0.5,
                0.5,
                "No anomaly map available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.imshow(visualization)
            ax.axis("off")

        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    """Main window for the Multimodal-IAD application."""

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.detector: AnomalyDetector | None = None
        self.current_result: NumpyImageItem | NumpyDepthItem | None = None
        self.current_sample_index = 0
        self.explanation_thread: ExplanationThread | None = None

        self.status_bar: QStatusBar | None = self.statusBar()
        self.init_ui()
        self.apply_light_theme()
        self.setup_logging()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Multimodal Industrial Anomaly Detection Interface")

        # Set geometry based on screen size to ensure it fits
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            width = int(screen_geometry.width() * 0.85)
            height = int(screen_geometry.height() * 0.85)
            self.setGeometry(
                screen_geometry.x() + (screen_geometry.width() - width) // 2,
                screen_geometry.y() + (screen_geometry.height() - height) // 2,
                width,
                height,
            )
        else:
            self.setGeometry(100, 100, 1280, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header with title
        header_label = QLabel("Multimodal Industrial Anomaly Detection")
        header_font = QFont()
        header_font.setPointSize(24)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #1976d2; margin: 10px;")
        main_layout.addWidget(header_label)

        # Top section: Config and Logs
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        config_group = self.create_config_section()
        log_group = self.create_log_section()
        top_splitter.addWidget(config_group)
        top_splitter.addWidget(log_group)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)  # Give more space to log
        main_layout.addWidget(top_splitter)

        # Middle section: Images
        images_splitter = self.create_images_section()
        main_layout.addWidget(images_splitter, 1)  # Give it stretch factor

        # Bottom section: Results
        results_splitter = self.create_results_section()
        results_splitter.setFixedHeight(TOP_BOTTOM_SECTION_HEIGHT)
        main_layout.addWidget(results_splitter)  # No stretch

        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        if self.status_bar:
            self.status_bar.addPermanentWidget(self.progress_bar, 1)  # Add with stretch
        self.update_status("Ready")

    def create_config_section(self) -> QGroupBox:
        """Create the configuration section."""
        group = QGroupBox("Model Configuration")
        group.setStyleSheet(
            """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """,
        )

        layout = QHBoxLayout()
        layout.setSpacing(10)

        self._add_config_selectors(layout)
        self._add_action_buttons(layout)

        group.setLayout(layout)
        return group

    def _add_config_selectors(self, layout: QHBoxLayout) -> None:
        """Add model, category, and mode selectors to the layout."""
        min_combo_width = 150

        # Label column
        label_layout = QVBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-weight: normal;")
        label_layout.addWidget(model_label)
        data_label = QLabel("Dataset:")
        data_label.setStyleSheet("font-weight: normal;")
        label_layout.addWidget(data_label)
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: normal;")
        label_layout.addWidget(category_label)
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: normal;")
        label_layout.addWidget(mode_label)
        layout.addLayout(label_layout)

        # Selector column
        selector_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([model.name for model in SupportedAdModels])
        self.model_combo.setMinimumWidth(min_combo_width)
        selector_layout.addWidget(self.model_combo)
        self.datamodule_combo = QComboBox()
        self.datamodule_combo.addItems([datamodule.name for datamodule in SupportedDatamodules])
        self.datamodule_combo.setMinimumWidth(min_combo_width)
        selector_layout.addWidget(self.datamodule_combo)
        self.category_combo = QComboBox()
        self.category_combo.setMinimumWidth(min_combo_width)
        selector_layout.addWidget(self.category_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Predict", "Train"])
        self.mode_combo.setMinimumWidth(min_combo_width)
        selector_layout.addWidget(self.mode_combo)
        layout.addLayout(selector_layout)

        self.datamodule_combo.currentIndexChanged.connect(self._update_categories)
        self._update_categories()

        # Connect all config combos to reset state on change
        self.model_combo.currentIndexChanged.connect(self._on_config_changed)
        self.datamodule_combo.currentIndexChanged.connect(self._on_config_changed)
        self.category_combo.currentIndexChanged.connect(self._on_config_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_config_changed)

        # Add a spacer item to the layout
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

    def _update_categories(self) -> None:
        """Update the category dropdown based on the selected datamodule."""
        selected_datamodule = SupportedDatamodules(self.datamodule_combo.currentText())
        self.category_combo.clear()
        if selected_datamodule == SupportedDatamodules.MVTecAD:
            self.category_combo.addItems(MVTECAD_CATEGORIES)
        elif selected_datamodule == SupportedDatamodules.MVTec3D:
            self.category_combo.addItems(MVTEC3D_CATEGORIES)
        elif selected_datamodule == SupportedDatamodules.MVTecAD_LOCO:
            self.category_combo.addItems(MVTEC_LOCO_CATEGORIES)

    def _cancel_explanation_thread(self) -> None:
        """Cancel any running explanation thread to prevent outdated results."""
        if self.explanation_thread and self.explanation_thread.isRunning():
            self.explanation_thread.cancel()

    def _on_config_changed(self) -> None:
        """Reset the UI and state when the configuration changes."""
        # Cancel any running explanation thread to prevent outdated results
        self._cancel_explanation_thread()

        self.load_image_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        # Clear all displays to avoid showing stale data
        self.input_image_label.set_image(None)
        self.image_path_label.setText("Image Path: N/A")
        self.heatmap_widget.update_heatmap(None)
        self.gt_label_value.setText("N/A")
        self.pred_label_value.setText("N/A")
        self.pred_label_value.setStyleSheet("font-size: 18px;")  # Reset color
        self.score_value.setText("N/A")
        self.explanation_text.clear()

        # Reset internal state
        self.detector = None
        self.current_result = None
        self.current_sample_index = 0

        self.update_status("Configuration changed. Click 'Execute' to apply.")

    def _add_action_buttons(self, layout: QHBoxLayout) -> None:
        """Add action and navigation buttons to the layout."""
        min_button_height = 40
        # Buttons
        button_layout = QVBoxLayout()
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setMinimumHeight(min_button_height)
        self.execute_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """,
        )
        self.execute_btn.clicked.connect(self.execute_detection)

        self.load_image_btn = QPushButton("Load Custom Image")
        self.load_image_btn.setMinimumHeight(min_button_height)
        self.load_image_btn.setEnabled(False)
        self.load_image_btn.clicked.connect(self.load_custom_image)

        button_layout.addWidget(self.execute_btn)
        button_layout.addWidget(self.load_image_btn)
        layout.addLayout(button_layout)

        # Navigation buttons for dataset samples
        nav_layout = QVBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setMinimumHeight(min_button_height)
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("Next →")
        self.next_btn.setMinimumHeight(min_button_height)
        self.next_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.show_previous_sample)
        self.next_btn.clicked.connect(self.show_next_sample)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

    def create_images_section(self) -> QSplitter:
        """Create the images display section."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Input image
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout()
        self.input_image_label = ImageLabel("Input Image")
        self.image_path_label = QLabel("Image Path: N/A")
        self.image_path_label.setStyleSheet("font-size: 10px; color: gray; qproperty-alignment: 'AlignCenter';")
        input_layout.addWidget(self.input_image_label)
        input_layout.addWidget(self.image_path_label)
        input_group.setLayout(input_layout)
        splitter.addWidget(input_group)

        # Heatmap
        heatmap_group = QGroupBox("Anomaly Heatmap")
        heatmap_layout = QVBoxLayout()
        self.heatmap_widget = HeatmapWidget()
        heatmap_layout.addWidget(self.heatmap_widget)
        heatmap_group.setLayout(heatmap_layout)
        splitter.addWidget(heatmap_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        return splitter

    def create_results_section(self) -> QSplitter:
        """Create the results display section."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Labels group
        labels_group = QGroupBox("Classification Results")
        labels_layout = QGridLayout()

        # Ground truth label
        self.gt_label_title = QLabel("Ground Truth:")
        self.gt_label_title.setStyleSheet("font-weight: bold;")
        self.gt_label_value = QLabel("N/A")
        self.gt_label_value.setStyleSheet("font-size: 18px;")
        labels_layout.addWidget(self.gt_label_title, 0, 0)
        labels_layout.addWidget(self.gt_label_value, 0, 1)

        # Predicted label
        self.pred_label_title = QLabel("Prediction:")
        self.pred_label_title.setStyleSheet("font-weight: bold;")
        self.pred_label_value = QLabel("N/A")
        self.pred_label_value.setStyleSheet("font-size: 18px;")
        labels_layout.addWidget(self.pred_label_title, 1, 0)
        labels_layout.addWidget(self.pred_label_value, 1, 1)

        # Anomaly score
        self.score_title = QLabel("Anomaly Score:")
        self.score_title.setStyleSheet("font-weight: bold;")
        self.score_value = QLabel("N/A")
        self.score_value.setStyleSheet("font-size: 18px;")
        labels_layout.addWidget(self.score_title, 2, 0)
        labels_layout.addWidget(self.score_value, 2, 1)

        labels_group.setLayout(labels_layout)
        splitter.addWidget(labels_group)

        # Explanation text
        explanation_group = QGroupBox("Textual Explanation")
        explanation_layout = QVBoxLayout()
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(TOP_BOTTOM_SECTION_HEIGHT)
        explanation_layout.addWidget(self.explanation_text)
        explanation_group.setLayout(explanation_layout)
        splitter.addWidget(explanation_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)  # Give more space to explanation

        return splitter

    def create_log_section(self) -> QGroupBox:
        """Create the log output section."""
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()
        self.log_output_text = QTextEdit()
        self.log_output_text.setReadOnly(True)
        self.log_output_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.log_output_text)
        group.setLayout(layout)
        return group

    def apply_light_theme(self) -> None:
        """Apply a light theme to the application."""
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
            QMainWindow {
                background-color: #ffffff;
            }
            QDialog {
                background-color: #ffffff;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #424242;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #424242;
            }
            QComboBox {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 5px;
                background-color: #ffffff;
                color: #424242;
            }
            QComboBox:hover {
                border: 1px solid #1976d2;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #424242;
                selection-background-color: #1976d2;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                color: #424242;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #bdbdbd;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #9e9e9e;
            }
            QTextEdit {
                font-size: 14px;
                padding: 10px;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                color: #424242;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #f5f5f5;
                text-align: center;
                color: #424242;
            }
            QProgressBar::chunk {
                background-color: #1976d2;
                width: 10px;
                margin: 0.5px;
            }
        """)

    def execute_detection(self) -> None:
        """Execute the anomaly detection based on selected configuration."""
        self._cancel_explanation_thread()
        try:
            # Get configuration
            model_name = self.model_combo.currentText()
            category = self.category_combo.currentText()
            mode = self.mode_combo.currentText()

            # Initialize detector
            self.update_status(f"Initializing {model_name} for {category}...")
            self.detector = AnomalyDetector(
                dataset_category=category,
                datamodule=SupportedDatamodules(self.datamodule_combo.currentText()),
                model=SupportedAdModels(model_name),
            )

            if mode == "Train":
                # Start training in separate thread
                self.execute_btn.setEnabled(False)
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress

                self.training_thread = TrainingThread(self.detector)
                self.training_thread.progress.connect(self.update_status)
                self.training_thread.finished.connect(self.on_training_only_finished)
                self.training_thread.error.connect(self.on_training_error)
                self.training_thread.start()
            elif mode == "Predict":
                self.update_status("Predict mode - loading latest pre-trained model...")
                if self.detector.load_checkpoint():
                    self.on_prediction_ready()
                    self.update_status(f"Loaded checkpoint: {self.detector.checkpoint_path}")
                else:
                    self.update_status("No pre-trained model found for this configuration.")
                    QMessageBox.warning(
                        self,
                        "No Checkpoint Found",
                        "Could not find a pre-trained model checkpoint for the selected configuration. "
                        "Please execute Train mode first.",
                    )

        except Exception as e:
            logger.exception("Failed to initialize detector:")
            QMessageBox.critical(self, "Error", f"Failed to initialize detector: {e}")

    def on_prediction_ready(self) -> None:
        """Handle completion of setup for prediction."""
        self.progress_bar.setVisible(False)
        self.execute_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

        self.update_status("Model ready for prediction. Showing test samples...")
        self.current_sample_index = 0
        self.show_sample()

    def on_training_only_finished(self) -> None:
        """Handle training completion for 'Train' mode."""
        self.progress_bar.setVisible(False)
        self.execute_btn.setEnabled(True)
        self.update_status("Training completed! You can now switch to 'Predict' mode.")

    def on_training_error(self, error: str) -> None:
        """Handle training error."""
        self.progress_bar.setVisible(False)
        self.execute_btn.setEnabled(True)
        self.update_status("Training failed!")
        QMessageBox.critical(self, "Training Error", f"Training failed: {error}")

    def show_sample(self) -> None:
        """Show a sample from the test dataset."""
        if not self.detector or not self.detector.trained:
            return

        try:
            # Get sample with prediction
            self.current_result = self.detector.get_sample_from_dataset(
                split="test",
                index=self.current_sample_index,
            )

            # Update displays
            self.update_displays()

        except Exception as e:
            logger.exception("Failed to load sample:")
            QMessageBox.critical(self, "Error", f"Failed to load sample: {e}")

    def show_previous_sample(self) -> None:
        """Show the previous sample in the dataset."""
        self._cancel_explanation_thread()
        if self.current_sample_index > 0:
            self.current_sample_index -= 1
            self.show_sample()

    def show_next_sample(self) -> None:
        """Show the next sample in the dataset."""
        self._cancel_explanation_thread()
        self.current_sample_index += 1
        self.show_sample()

    def load_custom_image(self) -> None:
        """Load a custom image for prediction."""
        self._cancel_explanation_thread()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path and self.detector and self.detector.trained:
            try:
                self.current_result = self.detector.predict_image(image_path=file_path)
                self.update_displays()
                self.update_status(f"Loaded custom image: {Path(file_path).name}")
            except Exception as e:
                logger.exception("Failed to process image:")
                QMessageBox.critical(self, "Error", f"Failed to process image: {e}")

    def update_displays(self) -> None:
        """Update all display widgets with current result."""
        if not self.current_result:
            return

        # Update input image
        if self.current_result.image is None or self.current_result.image_path is None:
            msg = "Image or path is None"
            raise ValueError(msg)

        image = Image.open(self.current_result.image_path).convert("RGB")
        self.input_image_label.set_image(np.array(image))
        if self.current_result.image_path:
            self.image_path_label.setText(f"Path: {Path(self.current_result.image_path).relative_to(DATASETS_DIR)}")
        else:
            self.image_path_label.setText("Path: N/A")

        # Update labels
        if self.current_result.gt_label is None:
            gt_label_str = "Unknown (Custom Image)"
        else:
            gt_label_str = "Abnormal" if self.current_result.gt_label == 1 else "Normal"

        pred_label_str = "Abnormal" if self.current_result.pred_label == 1 else "Normal"
        score = self.current_result.pred_score or 0.0

        self.gt_label_value.setText(gt_label_str)
        self.pred_label_value.setText(pred_label_str)
        self.score_value.setText(f"{score:.4f}")

        # Color code based on correctness and label
        if gt_label_str != "Unknown (Custom Image)" and gt_label_str == pred_label_str:
            color = "#4caf50"  # Green for correct
        else:
            color = "#f44336" if pred_label_str == "Abnormal" else "#2196f3"  # Red for abnormal, blue for normal

        self.pred_label_value.setStyleSheet(f"font-size: 18px; color: {color}; font-weight: bold;")

        self.heatmap_widget.update_heatmap(self.current_result)

        # Update explanation
        if self.detector and self.current_result:
            self.explanation_text.setText("Generating explanation, please wait...")
            self.update_status("Generating explanation...")
            self.explanation_thread = ExplanationThread(self.detector, self.current_result)
            self.explanation_thread.explanation_ready.connect(self.on_explanation_ready)
            self.explanation_thread.error.connect(self.on_explanation_error)
            self.explanation_thread.start()

    def on_explanation_ready(self, explanation: str) -> None:
        """Handle completion of explanation generation."""
        self.explanation_text.setText(explanation)
        self.update_status("Explanation ready.")

    def on_explanation_error(self, error_message: str) -> None:
        """Show an error if explanation generation fails."""
        self.explanation_text.setText(f"Could not generate explanation: {error_message}")
        self.update_status("Failed to generate explanation.")

    def update_status(self, message: str) -> None:
        """Update the status bar message."""
        if self.status_bar:
            self.status_bar.showMessage(message)
        logger.info(message)

    def setup_logging(self) -> None:
        """Redirect stdout and stderr to the log widget."""
        self.log_stream = QtStream()
        self.log_stream.new_text.connect(self.append_log_text)
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logger.info("--- Log started ---")

    def append_log_text(self, text: str) -> None:
        """Append text to the log output widget."""
        cursor = self.log_output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_output_text.setTextCursor(cursor)
        self.log_output_text.ensureCursorVisible()

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        """Restore stdout/stderr on close."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        super().closeEvent(event)
