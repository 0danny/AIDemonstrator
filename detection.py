import sys
sys.dont_write_bytecode = True

import subprocess
import random
from pathlib import Path
import shutil
import yaml

def load_and_print_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"Class names from {yaml_path}:")
    for i, name in enumerate(data.get('names', [])):
        print(f"  {i}: {name}")
    return data.get('names', [])

def run_yolo_detection(model_path, data_path, yolo_path):
    # Ensure YOLO path exists
    yolo_path = Path(yolo_path)
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLOv5 directory not found at {yolo_path}")

    # Load and print class names from the dataset YAML
    dataset_yaml = next(Path(data_path).glob('*.yaml'))

    # Get a list of all images in the dataset
    data_path = Path(data_path)
    image_files = list(data_path.glob('images/**/*.jpg'))  # Adjust if your images are not .jpg
    
    if not image_files:
        raise FileNotFoundError("No images found in the dataset")

    # Select a random image
    random_image_path = random.choice(image_files)
    print(f"Selected image for detection: {random_image_path}")

    # Create a temporary directory for the output
    output_dir = data_path / "temp_output"
    output_dir.mkdir(exist_ok=True)

    # Construct the command to run detect.py
    command = [
        sys.executable,
        str(yolo_path / "detect.py"),
        "--weights", str(model_path),
        "--source", str(random_image_path),
        "--img", "1024",
        "--conf", "0.25",
        "--project", str(output_dir),
        "--name", "detection",
        "--exist-ok",
        "--data", str(dataset_yaml)  # Use the dataset's YAML file
    ]

    # Run the detection
    print("Running YOLOv5 detection...")
    print(f"Command: {' '.join(map(str, command))}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output and error (if any)
    print("Detection Output:")
    print(result.stdout)
    if result.stderr:
        print("Detection Errors:")
        print(result.stderr)

    # Move the output image to a permanent location
    output_images = list(output_dir.glob("detection/*.jpg"))
    if output_images:
        output_image = output_images[0]
        permanent_output = data_path / "output" / f"detection_{random_image_path.name}"
        permanent_output.parent.mkdir(exist_ok=True)
        shutil.move(str(output_image), str(permanent_output))
        print(f"Detection complete. Output image saved to: {permanent_output}")
        print("Please open the output image to view the detection results.")
    else:
        print("No output image was generated. Check the detection output for errors.")

    # Clean up temporary directory
    shutil.rmtree(output_dir)

if __name__ == "__main__":
    model_path = "models/best.pt"
    data_path = "data"
    yolo_path = "yolov5"
    run_yolo_detection(model_path, data_path, yolo_path)