import sys
sys.dont_write_bytecode = True

import subprocess
import os

def evaluate_yolov5(weights_path, data_config_path, img_size=640, iou_threshold=0.65):
    # Construct the command
    command = [
        'python', 'yolov5/val.py',  # The evaluation script in YOLOv5 repository
        '--weights', weights_path,  # Path to the model weights
        '--data', data_config_path,  # Path to the data config file
        '--img', str(img_size),  # Image size
        '--iou', str(iou_threshold)  # IoU threshold
    ]

    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output from the command
        print(result.stdout)

        if result.stderr:
            print("Error encountered:", result.stderr)

    except Exception as e:
        print(f"Failed to run evaluation: {str(e)}")

if __name__ == "__main__":
    weights_file = 'models/best_yolom.pt'
    data_config_file = 'data/dataset.yaml'

    evaluate_yolov5(weights_path=weights_file, data_config_path=data_config_file, img_size=1024, iou_threshold=0.65)
