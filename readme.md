# Custom VQA Model

## Setup and Usage
1. Run `make setup_environment` to create the conda env with the correct python version and pip-tools
2. Activate the newly created conda env with `conda activate vqa`
3. Run `make install_deps` to install the project dependencies

To add new dependency:
Add it to the `requirements/requirements.in` file and run `make generate_new_requirements` and commit the new requirements.txt file it generates. 

## Components

### Main Scripts
- `scripts/train.py`: Main entry point for running the training process.
- `scripts/inference.py`: Run inference on provided image and question
- `scripts/prepare_data.py`: Data generation pipeline
- `scripts/evaluate.py`: Run evaluation on test date. 

Usage should look something like:

Data Prep: `python scripts/prepare_data.py --config configs/config.yaml`

Training: `python scripts/train.py --config config/config.yaml`

Evaluation: `python scripts/evaluate.py --model-path checkpoints/best_model.pt --test-data data/processed/test`

Inference: `python scripts/predict.py --model-path checkpoints/best_model.pt --image path/to/image.jpg --question "What abnormality is visible?"`


### Config File Example (Placeholder)

```yaml
data:
  data_dir: "data/processed"
  image_size: [224, 224]
  max_question_length: 128

model:
  vision_encoder:
    type: "resnet50"
    pretrained: true
  text_encoder:
    type: "bert-base-uncased"
    freeze_layers: 6
  fusion:
    type: "attention"
    hidden_size: 768

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  warmup_steps: 1000
  gradient_clip_val: 1.0
  early_stopping_patience: 5

  # NOTE: we may need configurations for PEFT/LORA/any additional training for the large encoder
```




