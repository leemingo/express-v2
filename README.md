# Express-v2: Pressing Evaluation Model

A machine learning project for evaluating pressing intensity in football matches.

## Project Overview

This project provides a pipeline for preprocessing raw tracking data from football matches, validating event data and converting it to SPADL format, calculating pressing intensity for each match and frame, and training a pressing evaluation model.

## Project Structure

```
express-v2/
├── preprocess_bepro.py      # BePro data preprocessing
├── preprocess_dfl.py        # DFL data preprocessing
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
/data/MHL/
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

<!-- ## Installation and Setup -->

<!-- ### 1. Install Dependencies -->

<!-- ```bash
pip install -r requirements.txt
``` -->

<!-- ### 2. Configure Data Path

Set the data path in `config.py`:

```python
# Data path configuration
data_path = "/path/to/your/data"
``` -->

## Usage

### 1. Raw Tracking Data Preprocessing

#### BePro Data Preprocessing
```bash
# Use default path
python preprocess_bepro.py

# Use custom path
python preprocess_bepro.py --data_path /path/to/bepro/raw/data
```

#### DFL Data Preprocessing
```bash
# Use default path
python preprocess_dfl.py

# Use custom path
python preprocess_dfl.py --data_path /path/to/dfl/raw/data
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
python pressing_intensity.py --source bepro

# DFL-SPoHo data
python pressing_intensity.py --source dfl-spoho

# DFL confidential data
python pressing_intensity.py --source dfl-confidential
```

This step includes:
- Calculating pressing intensity for all frames of each match
- Generating pressing metrics considering player distances, velocities, and directions
- Saving calculated pressing intensity in pickle format

#### Available Data Sources
- `bepro`: BePro data (default)
- `dfl-spoho`: DFL-SPoHo data
- `dfl-confidential`: DFL confidential data

### 4. Model Training and Testing

#### Training Mode
```bash
python train.py --config-name=config_exPress data.root_path=/path/to/dataset
```

#### Testing Mode
```bash
python train.py --config-name=config_exPress data.root_path=/path/to/dataset mode=test ckpt_path=/path/to/checkpoint.ckpt
```

#### Available Model Types
- `exPress`: GNN + LSTM based pressing evaluation model (default)
- `soccermap`: SoccerMap based model
- `xgboost`: XGBoost based model

## Model Architecture

### exPress Model
- **Input**: 19-dimensional feature vector (player positions, velocities, accelerations, pressing-related features)
- **Structure**: 
  - GNN layers (2 layers)
  - LSTM layers (2 layers, bidirectional)
  - Output layer
- **Output**: Pressing effectiveness score

### Hyperparameter Configuration

You can configure hyperparameters for each model in `params.json`:


## License

This project is developed for research purposes.

## Contact

If you have any questions about the project, please create an issue. 