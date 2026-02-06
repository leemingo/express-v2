# exPress-v2: Pressing Evaluation Model

This repository is the official implementation of [Contextual Evaluation of Individual Contributions from Pressing Situations in Football](https://dtai.cs.kuleuven.be/events/MLSA25/papers/MLSA25_paper_248.pdf) (Lee et al., MLSA 2025).

<img src="figs/framework.png" alt="Framework" width="600">

## Project Overview

This project provides a pipeline for preprocessing raw tracking data from football matches, validating event data and converting it to SPADL format, calculating pressing intensity for each match and frame, and training a pressing evaluation model.

## Project Structure

```
express-v2/
├── preprocess
│   ├── bepro.py         # BePro data preprocessing
│   ├── dfl.py           # DFL data preprocessing
│   └── ...
├── assertion/               # Event data validation and SPADL conversion
│   ├── assert.py           # Main validation script
│   ├── bepro.py            # BePro data conversion
│   ├── validator.py        # Validation logic
│   └── ...
├── pressing_intensity.py    # Pressing intensity calculation
├── train.py                # Model training and testing
├── model.py                # Model architecture
├── datasets.py             # Dataset classes
├── config.py               # Configuration file
├── params.json             # Model hyperparameters
└── ...
```

## Data Structure

### Required Data Organization

The project expects data to be organized in the following structure:

```
/data/
├── bepro/
│   ├── raw/                    # Raw BePro data
│   │   ├── {match_id}/
│   │   │   ├── {match_id}_metadata.json
│   │   │   ├── {match_id}_1st Half.json      # First half events
│   │   │   ├── {match_id}_2nd Half.json      # Second half events
│   │   │   └── tracking_data.jsonl           # Tracking data
│   │   └── ...

### File Naming Conventions

#### BePro Data
- **Raw Event Files**: `{match_id}_1st Half.json`, `{match_id}_2nd Half.json`
- **Metadata**: `{match_id}_metadata.json`
- **Tracking Data**: `tracking_data.jsonl`
```

## Installation and Setup

### 1. Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install PyTorch Lightning

```bash
python -m pip install lightning
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Raw Tracking Data Preprocessing

#### BePro Data Preprocessing
```bash
# Set custom path
python preprocess/bepro.py --data_path /path/to/bepro/raw/data
```

#### DFL Data Preprocessing
```bash
# Set custom path
python preprocess/dfl.py --data_path /path/to/dfl/raw/data
```

Alternatively,
```bash
# Set custom path
python -m preprocess.dfl --data_path /path/to/dfl/raw/data
```

This step includes:
- Loading and cleaning raw tracking data
- Calculating player and ball positions, velocities, and accelerations
- Coordinate system normalization and smoothing
- Saving preprocessed data in pickle format

### 2. Event Data Validation and SPADL Conversion

```bash
python assertion/assert.py --data_path /path/to/process/data
```

This step includes:
- Loading preprocessed event data
- Validating logical consistency of event sequences
- Converting to SPADL (Soccer Player Action Description Language) format
- Saving validated event data in CSV format

### 3. Pressing Intensity Calculation

```bash
# BePro data
python pressing_intensity.py --source bepro --data_path /path/to/bepro/processed

# DFL-SPoHo data
python pressing_intensity.py --source dfl-spoho --data_path /path/to/dfl-spoho/

# DFL additional data
python pressing_intensity.py --source dfl-confidential --data_path /path/to/dfl-additional/
```

This step includes:
- Calculating pressing intensity for all frames of each match
- Generating pressing metrics considering player distances, velocities, and directions
- Saving calculated pressing intensity in pickle format

#### Available Data Sources
- `bepro`: BePro data (Not open-source)
- `dfl-spoho`: DFL-SPoHo data (Open-source, [check here](https://github.com/spoho-datascience/idsse-data))
- `dfl-additional`: DFL confidential data (Not open-source)

### 4. Dataset Generation

Generate training datasets for model training. This step must be performed before model training.

```bash
# Generate cross validation datasets (recommended)
python datasets.py --data_path /path/to/bepro/processed --save_path /path/to/bepro/processed --cross_validation
```

#### Available Options
- `--data_path`: Path to processed data directory
- `--save_path`: Path to save generated datasets
- `--cross_validation`: Generate cross validation datasets (default: 6-fold)
- `--n_folds`: Number of folds for cross validation (default: 6)
- `--train_ratio`, `--valid_ratio`, `--test_ratio`: Data split ratios
- `--press_threshold`: Pressing intensity threshold (default: 0.9)
- `--num_frames_to_sample`: Number of frames to sample per sequence (default: 10)
- `--high_only`: Only consider high-intensity pressing situations
- `--exclude_matches`: List of match IDs to exclude

Generated files:
- `train_dataset.pkl`: Training dataset
- `valid_dataset.pkl`: Validation dataset  
- `test_dataset.pkl`: Test dataset
- `fold_info.pkl`: Cross validation fold information (when using cross validation)

### 5. Model Training and Testing

#### Training Mode
```bash
python main.py --config-name=config_exPress data.root_path=/path/to/train-dataset
```

#### Testing Mode
```bash
python main.py --config-name=config_exPress data.root_path=/path/to/test-dataset mode=test ckpt_path=/path/to/checkpoint.ckpt
```

#### Available Model Types
- `exPress`: GNN + LSTM based pressing evaluation model (default)
- `soccermap`: SoccerMap based model
- `xgboost`: XGBoost based model

## License

This project is developed for research purposes.

## Contact

If you have any questions about the project, please contact to minho.lee@uni-saarland.de. 
