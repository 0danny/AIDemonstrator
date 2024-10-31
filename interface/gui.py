# System Imports
import yaml
import subprocess
from pathlib import Path

import glob
import os

# Library Imports
import cv2
import torch
from collections import defaultdict

from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QHBoxLayout, QTabWidget, QGridLayout, QScrollArea, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


# Project Imports
import utilities
import model_statistics
from detection_box import DetectionBox

# Constants
WINDOW_X = 100
WINDOW_Y = 100
WINDOW_WIDTH = 1310
WINDOW_HEIGHT = 890

WINDOW_TITLE = "AI For Engineering [Theme 1] - Sign Detection Viewer (Group Project)"
MODEL_PATH = "models/best_yolom.pt"
DATASET_YAML_PATH = "data/dataset.yaml"
TEST_DATA_PATH = "data/test"

class SignViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

        with open(DATASET_YAML_PATH, 'r') as f:
            data_yaml = yaml.safe_load(f)

        self.class_names = data_yaml['names']
        print(f"Loaded class names: {self.class_names}")

        self.class_colors = utilities.generate_class_colors(self.class_names)
        self.model_stats = model_statistics.extract_model_statistics(MODEL_PATH, self.class_names, DATASET_YAML_PATH)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.current_image = None
        self.current_results = None
        self.current_filename = None

        self.setup_viewer_tab()
        self.setup_video_tab()
        self.setup_statistics_tab()

    def setup_viewer_tab(self):
        viewer_tab = QWidget()
        layout = QVBoxLayout(viewer_tab)

        # Top section - Controls and file info
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)

        # Import button
        import_button = QPushButton("Import Image")
        import_button.clicked.connect(self.import_image)
        import_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # File info section
        file_info = QWidget()
        file_info_layout = QVBoxLayout(file_info)
        self.filename_label = QLabel("No file selected")
        self.status_label = QLabel("Status: Ready")
        file_info_layout.addWidget(self.filename_label)
        file_info_layout.addWidget(self.status_label)

        top_layout.addWidget(import_button)
        top_layout.addWidget(file_info)
        top_layout.addStretch()

        # Images section
        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)  # Changed to QHBoxLayout for horizontal placement

        # Original image preview
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setAlignment(Qt.AlignTop)
        preview_label = QLabel("Original Image")
        preview_label.setAlignment(Qt.AlignCenter)
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        preview_layout.addWidget(preview_label)
        preview_layout.addWidget(self.original_image_label)

        # Prediction results
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        results_layout = QVBoxLayout(results_frame)
        results_label = QLabel("Detection Results")
        results_label.setAlignment(Qt.AlignCenter)
        self.predicted_image_label = QLabel()
        self.predicted_image_label.setAlignment(Qt.AlignCenter)
        self.predicted_image_label.setMinimumSize(800, 600)

        # Detection boxes container
        detection_boxes_widget = QWidget()
        detection_boxes_layout = QHBoxLayout(detection_boxes_widget)
        detection_boxes_layout.setSpacing(10)
        detection_boxes_layout.setAlignment(Qt.AlignCenter)

        # Create detection boxes for each class
        self.detection_boxes = {}

        for class_name in self.class_names:
            box = DetectionBox(class_name, self.class_colors[class_name])
            self.detection_boxes[class_name] = box
            detection_boxes_layout.addWidget(box)

        results_layout.addWidget(results_label)
        results_layout.addWidget(self.predicted_image_label)
        results_layout.addWidget(detection_boxes_widget)

        # Place the preview and results side by side in the horizontal layout
        images_layout.addWidget(preview_frame)
        images_layout.addWidget(results_frame)

        # Add all sections to main layout
        layout.addWidget(top_section)
        layout.addWidget(images_widget)

        self.tab_widget.addTab(viewer_tab, "Viewer")

    def import_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if file_name:
            self.current_video_path = file_name
            self.video_filename_label.setText(f"File: {Path(file_name).name}")
            self.video_status_label.setText("Status: Video loaded")
            self.process_video_button.setEnabled(True)
            
            # Load and display first frame of original video
            self.display_frame(file_name, self.original_frame_label)

    def display_frame(self, video_path, label):
        """Display the first frame of a video in a QLabel."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to QImage
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)

    def process_video(self):
        if hasattr(self, 'current_video_path'):
            self.video_status_label.setText("Status: Processing video...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress bar
            self.process_video_button.setEnabled(False)

            # Run video processing script
            subprocess.run(['python', 'yolov5/detect.py', 
                        '--weights', MODEL_PATH, 
                        '--source', self.current_video_path, 
                        '--conf', '0.4', 
                        '--save-txt', 
                        '--save-conf'])

            # Find the latest exp folder
            runs_dir = 'yolov5/runs/detect'
            exp_folders = glob.glob(os.path.join(runs_dir, 'exp*'))
            if exp_folders:
                latest_exp = max(exp_folders, key=os.path.getctime)
                processed_video = os.path.join(latest_exp, os.path.basename(self.current_video_path))
                
                if os.path.exists(processed_video):
                    # Display first frame of processed video
                    self.display_frame(processed_video, self.processed_frame_label)
                    self.video_status_label.setText("Status: Processing complete")
                else:
                    self.video_status_label.setText("Status: Error - Processed video not found")
            
            self.progress_bar.setVisible(False)
            self.process_video_button.setEnabled(True)

    def setup_video_tab(self):
        video_tab = QWidget()
        layout = QVBoxLayout(video_tab)

        # Controls section
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        # Import video button
        import_video_button = QPushButton("Import Video")
        import_video_button.clicked.connect(self.import_video)
        import_video_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Process video button
        self.process_video_button = QPushButton("Process Video")
        self.process_video_button.clicked.connect(self.process_video)
        self.process_video_button.setEnabled(False)
        self.process_video_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)

        # Video info section
        video_info = QWidget()
        video_info_layout = QVBoxLayout(video_info)
        self.video_filename_label = QLabel("No video selected")
        self.video_status_label = QLabel("Status: Ready")
        video_info_layout.addWidget(self.video_filename_label)
        video_info_layout.addWidget(self.video_status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        controls_layout.addWidget(import_video_button)
        controls_layout.addWidget(self.process_video_button)
        controls_layout.addWidget(video_info)
        controls_layout.addStretch()

        # Frame display section
        frame_display = QWidget()
        frame_layout = QHBoxLayout(frame_display)

        # Original frame widget
        original_frame = QFrame()
        original_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        original_layout = QVBoxLayout(original_frame)
        original_label = QLabel("Original Frame")
        original_label.setAlignment(Qt.AlignCenter)
        
        self.original_frame_label = QLabel()
        self.original_frame_label.setMinimumSize(400, 300)
        self.original_frame_label.setStyleSheet("background-color: #f0f0f0;")
        
        original_layout.addWidget(original_label)
        original_layout.addWidget(self.original_frame_label)

        # Processed frame widget
        processed_frame = QFrame()
        processed_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        processed_layout = QVBoxLayout(processed_frame)
        processed_label = QLabel("Processed Frame")
        processed_label.setAlignment(Qt.AlignCenter)
        
        self.processed_frame_label = QLabel()
        self.processed_frame_label.setMinimumSize(800, 600)
        self.processed_frame_label.setStyleSheet("background-color: #f0f0f0;")
        
        processed_layout.addWidget(processed_label)
        processed_layout.addWidget(self.processed_frame_label)

        frame_layout.addWidget(original_frame)
        frame_layout.addWidget(processed_frame)

        # Add all sections to main layout
        layout.addWidget(controls_widget)
        layout.addWidget(self.progress_bar)
        layout.addWidget(frame_display)

        self.tab_widget.addTab(video_tab, "Video Processing")

    def setup_statistics_tab(self):
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Display model statistics
        sections = {
            "Model Overview": [
                ("Number of Classes", str(self.model_stats.get('num_classes', 'N/A'))),
                ("Class Names", ", ".join(self.model_stats.get('class_names', ['N/A']))),
                ("Total Training Epochs", str(self.model_stats.get('training_options', {}).get('epochs', 'N/A'))),
                ("Best Epoch", str(self.model_stats.get('best_epoch', 'N/A')))
            ],
            "Training Performance": [
                ("Best mAP@0.5", f"{self.model_stats.get('best_map50', 0):.4f}"),
                ("Best mAP@0.5:0.95", f"{self.model_stats.get('best_map', 0):.4f}"),
                ("Best Precision", f"{self.model_stats.get('best_precision', 0):.4f}"),
                ("Best Recall", f"{self.model_stats.get('best_recall', 0):.4f}")
            ],
            "Test Data - IOU Metrics": [
                ("Mean IOU", f"{self.model_stats.get('mean_iou', 0):.4f}",
                "Average Intersection over Union across all predictions - measures overall localization accuracy"),
                ("Maximum IOU", f"{self.model_stats.get('max_iou', 0):.4f}",
                "Best achieved IOU score - indicates peak localization performance"),
                ("Minimum IOU", f"{self.model_stats.get('min_iou', 0):.4f}",
                "Lowest IOU score - helps identify worst-case detection scenarios")
            ],
            "Test Data - Per-Class IOU Performance": [
                (f"Class: {class_name}",
                f"Mean: {metrics.get('mean', 0):.4f}, Max: {metrics.get('max', 0):.4f}, Min: {metrics.get('min', 0):.4f}",
                f"IOU metrics specific to {class_name} class showing average, best, and worst localization performance")
                for class_name, metrics in self.model_stats.get('class_iou', {}).items()
            ],
            "Training Metrics": [
                ("Training Box Regression Loss", f"{self.model_stats.get('final_train_box_loss', 0):.4f}"),
                ("Training Objectness Loss", f"{self.model_stats.get('final_train_obj_loss', 0):.4f}"),
                ("Training Classification Loss", f"{self.model_stats.get('final_train_cls_loss', 0):.4f}")  
            ],
            "Validation Metrics": [
                ("Validation Box Regression", f"{self.model_stats.get('final_val_box_loss', 0):.4f}"),
                ("Validation Objectness Loss", f"{self.model_stats.get('final_val_obj_loss', 0):.4f}"),
                ("Validation Classification Loss", f"{self.model_stats.get('final_val_cls_loss', 0):.4f}")
            ],
            "Training Options": [
                ("Weights", str(self.model_stats.get('training_options', {}).get('weights', 'N/A'))),
                ("Batch Size", str(self.model_stats.get('training_options', {}).get('batch_size', 'N/A'))),
                ("Image Size", str(self.model_stats.get('training_options', {}).get('imgsz', 'N/A'))),
                ("Optimizer", str(self.model_stats.get('training_options', {}).get('optimizer', 'N/A'))),
                ("Learning Rate Initial", f"{self.model_stats.get('hyperparameters', {}).get('lr0', 0):.4f}"),
                ("Learning Rate Final", f"{self.model_stats.get('hyperparameters', {}).get('lrf', 0):.4f}"),
                ("Momentum", f"{self.model_stats.get('hyperparameters', {}).get('momentum', 0):.4f}"),
                ("Weight Decay", f"{self.model_stats.get('hyperparameters', {}).get('weight_decay', 0):.4f}"),
                ("Warmup Epochs", f"{self.model_stats.get('hyperparameters', {}).get('warmup_epochs', 0):.2f}"),
                ("Warmup Momentum", f"{self.model_stats.get('hyperparameters', {}).get('warmup_momentum', 0):.2f}"),
                ("Warmup Bias LR", f"{self.model_stats.get('hyperparameters', {}).get('warmup_bias_lr', 0):.4f}")
            ],
            "Hyperparameters": [
                ("Box Loss", f"{self.model_stats.get('hyperparameters', {}).get('box', 0):.4f}"),
                ("Class Loss", f"{self.model_stats.get('hyperparameters', {}).get('cls', 0):.4f}"),
                ("Class Power", f"{self.model_stats.get('hyperparameters', {}).get('cls_pw', 0):.4f}"),
                ("Object Loss", f"{self.model_stats.get('hyperparameters', {}).get('obj', 0):.4f}"),
                ("Object Power", f"{self.model_stats.get('hyperparameters', {}).get('obj_pw', 0):.4f}"),
                ("IOU Threshold", f"{self.model_stats.get('hyperparameters', {}).get('iou_t', 0):.2f}"),
                ("Anchor Threshold", f"{self.model_stats.get('hyperparameters', {}).get('anchor_t', 0):.2f}"),
                ("Focal Loss Gamma", f"{self.model_stats.get('hyperparameters', {}).get('fl_gamma', 0):.2f}")
            ],
            "Augmentation Parameters": [
                ("HSV Saturation", f"{self.model_stats.get('hyperparameters', {}).get('hsv_s', 0):.2f}"),
                ("HSV Hue", f"{self.model_stats.get('hyperparameters', {}).get('hsv_h', 0):.2f}"),
                ("HSV Value", f"{self.model_stats.get('hyperparameters', {}).get('hsv_v', 0):.2f}"),
                ("Flip Horizontal", f"{self.model_stats.get('hyperparameters', {}).get('fliplr', 0):.2f}"),
                ("Flip Vertical", f"{self.model_stats.get('hyperparameters', {}).get('flipud', 0):.2f}"),
                ("Mosaic", f"{self.model_stats.get('hyperparameters', {}).get('mosaic', 0):.2f}"),
                ("Mixup", f"{self.model_stats.get('hyperparameters', {}).get('mixup', 0):.2f}"),
                ("Copy Paste", f"{self.model_stats.get('hyperparameters', {}).get('copy_paste', 0):.2f}")
            ],
            "Miscellaneous": [
                ("Trained on", str(self.model_stats.get('date', 'N/A')))
            ]
        }

      
        for section_name, items in sections.items():
            header = QLabel(section_name)
            header.setStyleSheet("""
                QLabel {
                    font-size: 22px;
                    font-weight: bold;
                    color: #2c3e50;
                    padding: 10px 0;
                }
            """)
            scroll_layout.addWidget(header)

            # Add section content
            for item in items:
                stat_widget = QWidget()
                stat_layout = QHBoxLayout(stat_widget)

                # Handle both 2-value and 3-value items
                label = item[0]
                value = item[1]
                tooltip = item[2] if len(item) > 2 else None

                label_widget = QLabel(label + ":")
                label_widget.setStyleSheet("font-weight: bold; font-size: 16px;")

                value_widget = QLabel(value)
                value_widget.setStyleSheet("font-size: 16px;")

                if tooltip:
                    # Add tooltip to both label and value
                    label_widget.setToolTip(tooltip)
                    value_widget.setToolTip(tooltip)

                stat_layout.addWidget(label_widget)
                stat_layout.addWidget(value_widget)
                stat_layout.addStretch()

                scroll_layout.addWidget(stat_widget)

            # Add separator
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setStyleSheet("background-color: #bdc3c7;")
            scroll_layout.addWidget(separator)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.tab_widget.addTab(stats_tab, "Statistics")

    def update_detection_stats(self):
        if self.current_results is not None:
            # Reset all counts
            for box in self.detection_boxes.values():
                box.update_count(0)

            # Update counts based on detections
            class_counts = defaultdict(int)

            for detection in self.current_results.pred[0]:
                class_idx = int(detection[-1])
                class_name = self.class_names[class_idx]  # Updated to use self.class_names
                class_counts[class_name] += 1

            # Update detection boxes
            for class_name, count in class_counts.items():
                if class_name in self.detection_boxes:
                    self.detection_boxes[class_name].update_count(count)

    def import_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.webp)")

        if file_name:
            self.current_filename = Path(file_name).name
            self.filename_label.setText(f"File: {self.current_filename}")
            self.status_label.setText("Status: Processing...")

            # Read and display original image
            self.current_image = cv2.imread(file_name)
            self.display_original_image()

            # Make prediction and display result
            self.make_prediction(file_name)

            self.status_label.setText("Status: Ready")

    def display_original_image(self):
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qt_image = QImage(rgb_image.data, width, height,
                              bytes_per_line, QImage.Format_RGB888)

            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.original_image_label.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.original_image_label.setPixmap(scaled_pixmap)

    def make_prediction(self, image_path):
        # Run inference
        self.current_results = self.model(image_path)

        self.current_results.names = self.class_names  # Ensure class names match

        # Get the rendered image with predictions
        rendered_img = self.current_results.render()[0]

        # Convert to QPixmap and display
        height, width = rendered_img.shape[:2]
        bytes_per_line = 3 * width
        qt_image = QImage(rendered_img.data, width, height,
                          bytes_per_line, QImage.Format_RGB888)

        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.predicted_image_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.predicted_image_label.setPixmap(scaled_pixmap)

        # Update detection statistics
        self.update_detection_stats()
