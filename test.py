import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import os
import argparse # To accept checkpoint path as argument

# Import project modules
# import config  # Import static configurations
from model import PytorchSoccerMapModel, exPressModel # Import Lightning model
from datasets import PressingSequenceDataset, exPressInputDataset, SoccerMapInputDataset 

def test(ckpt_path: str):
    """Loads a checkpoint and runs testing on the test dataset."""

    pl.seed_everything(42, workers=True) # Ensure reproducibility

    # # --- 1. Load Configurations (Needed for Model/Dataloader setup) ---
    # try:
    #     with open("params.json", 'r') as f: params = json.load(f)
    #     soccermap_params = params.get('soccermap', {})
    #     data_cfg = soccermap_params.get('DataConfig', {})
    #     trainer_cfg = soccermap_params.get('TrainerConfig', {}) # Need accelerator/devices
    #     model_cfg = soccermap_params.get('ModelConfig', {})    # Need model params for init
    #     optimizer_cfg = soccermap_params.get('OptimizerConfig', {}) # Need lr for model init signature
    # except Exception as e:
    #     print(f"Warning: Failed to load or parse params.json ({e}). Using minimal defaults.")
    #     data_cfg = {"batch_size": 32, "num_workers": 4}
    #     trainer_cfg = {"devices": 1, "accelerator": "auto"}
    #     model_cfg = {"in_channels": config.NUM_FEATURE_CHANNELS, "criterion_name": "bce"}
    #     optimizer_cfg = {"lr": 1e-4} # Default lr for model init

    # # --- 2. Prepare Test Data ---
    # test_pickle_path = os.path.join(config.DATA_PATH, config.TEST_PICKLE_NAME)
    # if not os.path.exists(test_pickle_path):
    #     print(f"Error: Test data file not found at {test_pickle_path}")
    #     return
    DATA_PATH = "/data/MHL/pressing-intensity-feat" # Path where pickled datasets are saved
    test_dataset = exPressInputDataset(os.path.join(DATA_PATH, "test_dataset.pkl"))
    
    if len(test_dataset) == 0:
        print("Loaded test dataset is empty. Exiting.")
        return

    # Custom collate function to handle potential None values from dataset errors
    def collate_fn_skip_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        try: return torch.utils.data.dataloader.default_collate(batch)
        except RuntimeError: return None # Skip batch if collation error

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        # collate_fn=collate_fn_skip_none
    )
    print(f"Test loader created with {len(test_dataset)} samples.")

    # --- 3. Initialize Model ---
    # Instantiate the model structure. The weights will be loaded from the checkpoint.
    # Note: Provide necessary arguments expected by __init__, even if optimizer isn't used.
    optimizer_params = {
        "optimizer_params": {
            "lr": 1e-4,
            "weight_decay": 1e-5
        }
    }
    model_config = {
        "in_channels": 10,
            "num_gnn_layers": 2,
            "gnn_hidden_dim": 64,
            "num_lstm_layers": 2,
            "lstm_hidden_dim": 64,
            "lstm_dropout": 0.4,
            "lstm_bidirectional": True,
            "use_pressing_features": False,
            "gnn_head": 4
    }

    model = exPressModel(model_config=model_config, optimizer_params=optimizer_params)
    model = model.to("cuda")


    # --- 4. Initialize Trainer ---
    # For testing, only device/accelerator settings are strictly necessary.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1], # Use configured devices
        logger=False, # Disable logging for testing typically
        enable_checkpointing=False, # No need to save checkpoints during test
        enable_progress_bar=True # Show progress
    )

    # --- 5. Run Testing ---
    if not os.path.exists(ckpt_path):
         print(f"Error: Checkpoint file not found at {ckpt_path}")
         return

    print(f"Starting testing using checkpoint: {ckpt_path}")
    # The trainer.test method loads the checkpoint, runs test_step, and aggregates results
    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)

    print("--- Test Results ---")
    # test_results is a list containing one dictionary per test dataloader
    if test_results:
        print(test_results[0]) # Print results for the first (and only) test dataloader
    else:
        print("Testing completed but no results were returned.")
    print("--------------------")
    print("Testing finished.")

if __name__ == "__main__":
    # Setup argument parser to accept checkpoint path from command line
    parser = argparse.ArgumentParser(description="Test a trained SoccerMap model.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the model checkpoint (.ckpt) file saved during training."
    )
    args = parser.parse_args()

    # Run the testing function with the provided checkpoint path
    test(args.ckpt_path)