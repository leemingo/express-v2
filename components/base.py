from abc import ABC, abstractmethod
import json
import os
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

from datasets import PressingSequenceDataset, SoccerMapInputDataset # Assuming dataset.py contains this


class BaseComponent(ABC):
    """Handles the setup, training, and testing pipeline for a model."""
    def __init__(self, args):
        """
        Initializes the pipeline with command line arguments.

        Args:
            args (argparse.Namespace): Parsed command line arguments.
        """
        pl.seed_everything(args.seed, workers=True)
        self.args = args
        self.model_params = self._load_params()
        self.data_cfg = self.model_params.get('DataConfig', {})
        self.trainer_cfg = self.model_params.get('TrainerConfig', {})
        self.optimizer_cfg = self.model_params.get('OptimizerConfig', {})
        self.model_cfg = self.model_params.get('ModelConfig', {})
        self.early_stop_cfg = self.model_params.get('EarlyStopConfig', {})
        self.checkpoint_cfg = self.model_params.get('ModelCheckpointConfig', {})

    def _load_params(self):
        """Loads parameters from the JSON file specified in args."""
        try:
            with open(self.args.params_path, 'r') as f:
                params = json.load(f)
            print(f"Configurations loaded from {self.args.params_path}.")
            if self.args.model_type not in params:
                raise ValueError(f"Model type '{self.args.model_type}' not found in {self.args.params_path}.")
            model_params = params.get(self.args.model_type, {})
            return model_params
        except FileNotFoundError:
            print(f"Error: Parameters file not found at {self.args.params_path}")
            exit()
        except json.JSONDecodeError:
            print(f"Error: {self.args.params_path} is not valid JSON.")
            exit()
        except ValueError as ve:
            print(ve)
            exit()
        except Exception as e:
            print(f"An error occurred loading configurations: {e}")
            exit()

    def _setup_data(self, stage='fit'):
        """Sets up datasets and dataloaders for fit (train/val) or test stage."""
        print(f"Setting up data for stage: {stage}")
        val_split_ratio = self.data_cfg.get('val_split_ratio', 0.2)
        num_workers = self.data_cfg.get("num_workers", 0)
        batch_size = self.data_cfg.get("batch_size", 32)
        pin_memory = self.data_cfg.get("pin_memory", True)

        def collate_fn_skip_none(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if not batch: return None
            try: return torch.utils.data.dataloader.default_collate(batch)
            except RuntimeError: return None # Skip batch if collation error

        common_loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": True if num_workers > 0 else False,
            # "collate_fn": collate_fn_skip_none
        }
        
        if stage == 'fit':
            train_pkl_path = f"{self.args.root_path}/train_dataset.pkl"
            if not os.path.exists(train_pkl_path):
                full_train_dataset = PressingSequenceDataset(self.args.root_path)
            else:
                with open(train_pkl_path, "rb") as f:
                    full_train_dataset = pickle.load(f)
                
            if len(full_train_dataset) == 0: print("Loaded training dataset is empty."); exit()

            total_samples = len(full_train_dataset)
            val_size = int(total_samples * val_split_ratio)
            train_size = total_samples - val_size
            print(f"Total train samples: {total_samples}, Training: {train_size}, Validation: {val_size}")
            train_dataset, val_dataset = None, None
            if train_size > 0 and val_size > 0:
                train_dataset, val_dataset = random_split(
                    full_train_dataset, [train_size, val_size],
                    # generator=torch.Generator().manual_seed(self.args.seed)
                    )
            elif train_size > 0:
                print("Warning: Not enough data for validation split."); train_dataset = full_train_dataset
            else: print("Error: No training samples."); exit()

            self.train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
            if val_dataset:
                self.val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
            else:
                self.val_loader = None

        elif stage == 'test':
            test_pkl_path = f"{self.args.root_path}/test_dataset.pkl"
            if not os.path.exists(test_pkl_path):
                test_dataset = PressingSequenceDataset(self.args.root_path)
            else:
                with open(test_pkl_path, "rb") as f:
                    test_dataset = pickle.load(f)
            
            if len(test_dataset) > 0:
                self.test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)
                print(f"Test loader created with {len(test_dataset)} samples.")
            else:
                print("Test dataset loaded but is empty.")
                self.test_loader = None
        else:
             print(f"Unknown stage: {stage}")


    @abstractmethod
    def _setup_model(self):
        pass

    def _setup_trainer(self, stage="fit"):
        """Initializes the PyTorch Lightning Trainer."""
        print("Setting up Trainer...")
        # Callbacks
        ckpt_path = os.path.join(self.args.root_path, "checkpoints")
        os.makedirs(ckpt_path, exist_ok=True)
        monitor_metric = self.checkpoint_cfg.get('monitor','val_loss')
        ckpt_filename = f"{self.args.model_type}-" + "{epoch:02d}-{val_loss:.2f}"

        callbacks_list = []
        if stage == "fit":
            if self.val_loader: # Only add monitoring callbacks if validation occurs
                print(f"Configuring Checkpoint and Early Stopping based on '{monitor_metric}'")
                self.checkpoint_cb = pl.callbacks.ModelCheckpoint( # Store reference
                    monitor=monitor_metric, dirpath=ckpt_path, filename=ckpt_filename,
                    save_top_k=self.checkpoint_cfg.get('save_top_k', 1), mode=self.checkpoint_cfg.get('mode','min'),
                    verbose=self.checkpoint_cfg.get('verbose', True) )
                early_stop_cb = pl.callbacks.EarlyStopping(
                    monitor=self.early_stop_cfg.get('monitor', 'val_loss'), patience=self.early_stop_cfg.get('patience', 5),
                    min_delta=self.early_stop_cfg.get('min_delta', 1e-3), verbose=self.early_stop_cfg.get('verbose', True),
                    mode=self.early_stop_cfg.get('mode','min'),
                    # strict=self.early_stop_cfg.get('strict', False)
                    )
                callbacks_list.extend([self.checkpoint_cb, early_stop_cb])
            else: print("Validation loader not available. Skipping ModelCheckpoint and EarlyStopping.")

        # Add progress bar callback (adjust refresh rate if needed)
        callbacks_list.append(TQDMProgressBar(refresh_rate=50)) # Example refresh rate

        # Logger
        tensorboard_logger = TensorBoardLogger(self.args.root_path, name=f"{self.args.model_type}_pressing")

        # Strategy (DDP)
        # strategy = None
        # devices_cfg = self.trainer_cfg.get('devices', [1])
        # if isinstance(devices_cfg, int) and devices_cfg > 1 or isinstance(devices_cfg, list) and len(devices_cfg) > 1:
        #      try:
        #          # Set find_unused_parameters=False first, change if needed
        #          strategy = DDPStrategy(find_unused_parameters=False)
        #          print(f"Using DDP Strategy for {devices_cfg} devices.")
        #      except ImportError: print("Warning: DDPStrategy not found.")

        # Trainer Initialization
        
        self.trainer = pl.Trainer(
            max_epochs=self.trainer_cfg.get('max_epochs', 10),
            # min_epochs=self.trainer_cfg.get('min_epochs', 1),
            accelerator=self.trainer_cfg.get('accelerator', "auto"),
            devices=self.trainer_cfg.get('devices', [1]),
            # strategy=strategy,
            gradient_clip_val=self.trainer_cfg.get('gradient_clip_val', 0.0),
            callbacks=callbacks_list,
            logger=tensorboard_logger,
            log_every_n_steps=max(1, len(self.train_loader)//20) if stage=='fit' else 1,
            deterministic=True
        )

    def train(self):
        """Runs the training process."""
        self._setup_data(stage='fit') # Setup train/val data
        if not self.train_loader: print("Training loader not created. Aborting train."); return
        self._setup_model()
        self._setup_trainer(stage='fit')

        print(f"Starting training for model type: {self.args.model_type}")
        try:
            self.trainer.fit(self.model, self.train_loader, self.val_loader) # Pass loaders directly
            print("Training finished.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()

    def test(self, ckpt_path):
        """Runs the testing process using the best checkpoint."""
        self._setup_data(stage='test') # Setup test data
        self._setup_model()
        self._setup_trainer(stage='test')
        if not self.test_loader: print("Test loader not created. Aborting test."); return
        
        if not os.path.exists(ckpt_path):
             print(f"Error: Checkpoint file not found at {ckpt_path}")
             return

        print(f"Starting testing using checkpoint: {ckpt_path}")
        try:
            test_results = self.trainer.test(self.model, dataloaders=self.test_loader, ckpt_path=ckpt_path)
            print("--- Test Results ---")
            if test_results: print(test_results)
            else: print("Testing completed but no results returned.")
            print("--------------------")
        except Exception as e:
            print(f"An error occurred during testing: {e}")
            import traceback
            traceback.print_exc()
        print("Testing finished.")

