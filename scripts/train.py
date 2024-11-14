import argparse
from pathlib import Path
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default='outputs', help='Output directory')
    return parser.parse_args()

def main():
    print("Starting Training... 👀")
    args = parse_args()
    
    # TODO: Load config
    # TODO: Initialize data loaders
    # TODO: Set up model
    # TODO: Initialize trainer
    # TODO: Run training

if __name__ == '__main__':
    main()