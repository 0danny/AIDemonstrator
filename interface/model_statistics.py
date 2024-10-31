import torch
import pandas as pd
import numpy as np
from pathlib import Path
from subprocess import run
import yaml
import sys

def calculate_iou(box1, box2):
    def yolo_to_xyxy(box, width=640, height=640):
        # For prediction boxes, remove confidence score if present
        if len(box) == 5:
            box = box[:4]
            
        x_center, y_center, w, h = box
        x1 = (x_center - w/2) * width
        y1 = (y_center - h/2) * height
        x2 = (x_center + w/2) * width
        y2 = (y_center + h/2) * height
        return [x1, y1, x2, y2]
    
    # If boxes are in YOLO format, convert them
    if len(box1) >= 4 and max(box1[:4]) <= 1.0:  # Handle 4 or 5 element boxes
        box1 = yolo_to_xyxy(box1)
    if len(box2) >= 4 and max(box2[:4]) <= 1.0:
        box2 = yolo_to_xyxy(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    # Add debug printing
    print(f"Converted boxes: {box1} and {box2}")
    print(f"Intersection: {intersection}, Union: {union}, IoU: {iou}")
    
    return iou

def calculate_test_ious(pred_files, label_files, class_names):
    ious_by_class = {class_name: [] for class_name in class_names}
    all_ious = []
    
    # Create a mapping of base names to label files for faster lookup
    label_map = {Path(label_file).name: label_file for label_file in label_files}
    
    for pred_file in pred_files:
        base_name = Path(pred_file).name
        
        if base_name not in label_map:
            continue
            
        label_file = label_map[base_name]
        
        pred_boxes = parse_labels_file(pred_file)
        true_boxes = parse_labels_file(label_file)
        
        # Print debug information
        print(f"\nProcessing {base_name}")
        print(f"Found {len(pred_boxes)} predicted boxes and {len(true_boxes)} true boxes")
        
        for pred_class, pred_box in pred_boxes:
            best_iou = 0
            
            print(f"\nPredicted box for class {class_names[pred_class]}: {pred_box}")
            
            for true_class, true_box in true_boxes:
                if pred_class == true_class:
                    print(f"Comparing with true box: {true_box}")
                    iou = calculate_iou(pred_box, true_box)
                    best_iou = max(best_iou, iou)
            
            if best_iou > 0:
                print(f"Best IOU for class {class_names[pred_class]}: {best_iou}")
                class_name = class_names[pred_class]
                ious_by_class[class_name].append(best_iou)
                all_ious.append(best_iou)
    
    # Calculate statistics
    iou_metrics = {
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'max_iou': np.max(all_ious) if all_ious else 0,
        'min_iou': np.min(all_ious) if all_ious else 0,
        'class_iou': {}
    }
    
    for class_name, ious in ious_by_class.items():
        if ious:
            iou_metrics['class_iou'][class_name] = {
                'mean': np.mean(ious),
                'max': np.max(ious),
                'min': np.min(ious)
            }
        else:
            iou_metrics['class_iou'][class_name] = {
                'mean': 0,
                'max': 0,
                'min': 0
            }
    
    return iou_metrics

def parse_labels_file(file_path):
    """Parse YOLO format label file."""
    boxes = []
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            for line in f:
                class_id, *box = map(float, line.strip().split())
                boxes.append((int(class_id), box))
    return boxes

def get_iou(model_path, dataset_yaml_path):
    print("Running validation script...")

    cmd = [
        sys.executable,
        f"yolov5/val.py",
        "--data", dataset_yaml_path,
        "--weights", model_path,
        "--batch-size", "16",
        "--imgsz", "1024",
        "--task", "test",
        "--save-txt",
        "--save-conf",
        "--verbose",
        "--save-json"
    ]
                
    result = run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Get a list of predicted label file names from latest exp folder
        predicted_labels = Path('yolov5/runs/val').iterdir()
        predicted_labels = sorted([str(label) for label in predicted_labels if label.is_dir()])[-1]
        predicted_labels = (Path(predicted_labels) / 'labels').iterdir()
        predicted_labels = sorted([str(label) for label in predicted_labels if label.is_file()])

        # Get a list of original label file names
        original_labels = Path('data/labels/test').iterdir()
        original_labels = sorted([str(label) for label in original_labels if label.is_file()])

        print(f"Loaded {len(predicted_labels)} predicted labels and {len(original_labels)} original labels.")
    
        # Get class names from dataset YAML
        with open(dataset_yaml_path, 'r') as f:
            dataset = yaml.safe_load(f)
            class_names = dataset['names']

            iou_metrics = calculate_test_ious(predicted_labels, original_labels, class_names)
        
            print("IOU Metrics -> ", iou_metrics)

            return iou_metrics
    else:
        print("Validation script failed.")

        print("Output:", result.stdout)
        print("Error:", result.stderr)

        return {}

def extract_model_statistics(model_path, class_names, dataset_yaml_path):

    # Check if models/model_stats.yaml exists.
    # If it does, load the file and return the metrics.
    try:
        with open('models/metrics.yaml', 'r') as f:
            metrics = yaml.load(f, Loader=yaml.Loader)
            print("Loaded model statistics from file.")
            return metrics
    except Exception as e:
        print(f"Error loading model statistics from file: {str(e)}")

    try:
        model_data = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        metrics = {}

        # Basic model info
        metrics['best_epoch'] = int(model_data.get('epoch', 0))

        if 'model' in model_data and hasattr(model_data['model'], 'yaml'):
            model_config = model_data['model'].yaml
            if model_config and 'nc' in model_config:
                metrics['num_classes'] = model_config['nc']
            else:
                metrics['num_classes'] = len(class_names)
            metrics['class_names'] = class_names

        # Get options and hyperparameters
        if 'opt' in model_data:
            opt = model_data['opt']
            metrics['training_options'] = opt
            if 'hyp' in opt:
                metrics['hyperparameters'] = opt['hyp']

        if 'date' in model_data:
            metrics['date'] = model_data['date']

        # Load CSV metrics if available
        try:
            best_csv_path = 'models/best_yolom.csv'
            if Path(best_csv_path).exists():
                df = pd.read_csv(best_csv_path)
                df.columns = df.columns.str.strip().str.replace(' ', '_')

                metrics['best_precision'] = df['metrics/precision'].max()
                metrics['best_recall'] = df['metrics/recall'].max()
                metrics['best_map50'] = df['metrics/mAP_0.5'].max()
                metrics['best_map'] = df['metrics/mAP_0.5:0.95'].max()

                final_row = df.iloc[-1]
                metrics['final_train_box_loss'] = final_row['train/box_loss']
                metrics['final_train_obj_loss'] = final_row['train/obj_loss']
                metrics['final_train_cls_loss'] = final_row['train/cls_loss']
                metrics['final_val_box_loss'] = final_row['val/box_loss']
                metrics['final_val_obj_loss'] = final_row['val/obj_loss']
                metrics['final_val_cls_loss'] = final_row['val/cls_loss']

        except Exception as e:
            print(f"Error loading best.csv file: {str(e)}")

        # Add the IOU metrics to the rest of the metrics
        iou_metrics = get_iou(model_path, dataset_yaml_path)
        metrics.update(iou_metrics)

        # Save the metrics to a file, overwriting any existing file
        with open('models/metrics.yaml', 'w') as f:
            yaml.dump(metrics, f)

        print("Model Stats -> ", metrics)

        return metrics

    except Exception as e:
        print(f"Error extracting model statistics: {str(e)}")
        return {}