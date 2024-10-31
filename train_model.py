import sys
sys.dont_write_bytecode = True

import yaml
import argparse
import shutil
import subprocess
import os

from pathlib import Path

def setup_environment():
    if not Path("yolov5").exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"])

def prepare_data(data_path):
    data_path = Path(data_path).resolve()
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(data_path),
        'train': str(data_path / 'images/train'),
        'val': str(data_path / 'images/val'),
        'test': str(data_path / 'images/test'),  # Added test path
        'nc': len(list(data_path.glob('classes.txt'))[0].read_text().splitlines()),
        'names': [name.strip() for name in (data_path / 'classes.txt').read_text().splitlines()]
    }
    
    with open(data_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Create train, val, and test splits
    for split in ['train', 'val', 'test']:
        (data_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (data_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Move files according to new split ratios: 85% train, 10% val, 5% test
    images = list(data_path.glob('images/*.jpg'))
    labels = list(data_path.glob('labels/*.txt'))
    
    total_files = len(images)
    train_threshold = int(total_files * 0.85)
    val_threshold = int(total_files * 0.95)  # 85% + 10% = 95%
    
    for i, (image, label) in enumerate(zip(images, labels)):
        if i < train_threshold:
            split = 'train'
        elif i < val_threshold:
            split = 'val'
        else:
            split = 'test'
        
        shutil.move(str(image), str(data_path / 'images' / split / image.name))
        shutil.move(str(label), str(data_path / 'labels' / split / label.name))

def train_model(data_path):
    data_yaml = str(Path(data_path) / 'dataset.yaml')
    
    subprocess.run([
        sys.executable, "yolov5/train.py",
        "--img", "1024",
        "--batch", "16",
        "--epochs", "100",
        "--data", data_yaml,
        "--weights", "yolov5s.pt",
        "--workers", "3",
        "--noplots"
    ])

def main():
    parser = argparse.ArgumentParser()

    # Argument for mode, (train, prepare)
    parser.add_argument("mode", type=str, help="Mode to run the script in (train, prepare)")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path}")
    
    if args.mode == "prepare":
        setup_environment()
        prepare_data(data_path)
    elif args.mode == "train":
        os.chdir(script_dir)
        train_model(data_path)
    else:
        print("Invalid mode. Please choose either 'prepare' or 'train'.")

if __name__ == "__main__":
    main()