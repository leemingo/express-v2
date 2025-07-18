from abc import ABC, abstractmethod
import json
import os
import pickle
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

from datasets import PressingSequenceDataset, SoccerMapInputDataset


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
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: {self.args.params_path} is not valid JSON.")
            sys.exit(1)
        except ValueError as ve:
            print(ve)
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred loading configurations: {e}")
            sys.exit(1)

    def _get_common_loader_args(self):
        """Get common DataLoader arguments."""
        return {
            "batch_size": self.data_cfg.get("batch_size", 32),
            "num_workers": self.data_cfg.get("num_workers", 0),
            "pin_memory": self.data_cfg.get("pin_memory", True),
            "persistent_workers": self.data_cfg.get("num_workers", 0) > 0
        }

    def _load_dataset_from_pkl(self, pkl_path, dataset_class=None, dataset_args=None):
        """Load dataset from pickle file or create new dataset if file doesn't exist."""
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        elif dataset_class and dataset_args:
            return dataset_class(*dataset_args)
        else:
            raise FileNotFoundError(f"Dataset file not found at {pkl_path} and no fallback dataset class provided.")

    def _setup_data(self, stage='fit'):
        """Sets up datasets and dataloaders for fit (train/val) or test stage."""
        print(f"Setting up data for stage: {stage}")
        
        common_loader_args = self._get_common_loader_args()
        
        if stage == 'fit':
            train_pkl_path = f"{self.args.root_path}/train_dataset.pkl"
            
            try:
                full_train_dataset = self._load_dataset_from_pkl(
                    train_pkl_path, 
                    PressingSequenceDataset, 
                    [self.args.root_path]
                )
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return False
                
            if len(full_train_dataset) == 0:
                print("Loaded training dataset is empty.")
                return False

            # Split dataset
            val_split_ratio = self.data_cfg.get('val_split_ratio', 0.2)
            total_samples = len(full_train_dataset)
            val_size = int(total_samples * val_split_ratio)
            train_size = total_samples - val_size
            
            print(f"Total train samples: {total_samples}, Training: {train_size}, Validation: {val_size}")
            
            if train_size > 0 and val_size > 0:
                train_dataset, val_dataset = random_split(
                    full_train_dataset, [train_size, val_size]
                )
            elif train_size > 0:
                print("Warning: Not enough data for validation split.")
                train_dataset = full_train_dataset
                val_dataset = None
            else:
                print("Error: No training samples.")
                return False

            self.train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
            self.val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args) if val_dataset else None

        elif stage == 'test':
            test_pkl_path = f"{self.args.root_path}/test_dataset.pkl"
            
            try:
                test_dataset = self._load_dataset_from_pkl(
                    test_pkl_path,
                    PressingSequenceDataset,
                    [self.args.root_path]
                )
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return False
            
            if len(test_dataset) > 0:
                self.test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)
                print(f"Test loader created with {len(test_dataset)} samples.")
            else:
                print("Test dataset loaded but is empty.")
                self.test_loader = None
        else:
            print(f"Unknown stage: {stage}")
            return False
            
        return True

    @abstractmethod
    def _setup_model(self):
        """Abstract method to be implemented by subclasses."""
        pass

    def _setup_trainer(self, stage="fit"):
        """Initializes the PyTorch Lightning Trainer."""
        print("Setting up Trainer...")
        
        # Setup checkpoint path
        ckpt_path = os.path.join(self.args.root_path, f"checkpoints_{self.args.exp_name}")
        os.makedirs(ckpt_path, exist_ok=True)
        
        # Setup callbacks
        callbacks_list = []
        
        if stage == "fit" and self.val_loader:
            monitor_metric = self.checkpoint_cfg.get('monitor', 'val_loss')
            ckpt_filename = f"{self.args.model_type}-" + "{epoch:02d}-{val_loss:.2f}"
            
            print(f"Configuring Checkpoint and Early Stopping based on '{monitor_metric}'")
            
            self.checkpoint_cb = pl.callbacks.ModelCheckpoint(
                monitor=monitor_metric, 
                dirpath=ckpt_path, 
                filename=ckpt_filename,
                save_top_k=self.checkpoint_cfg.get('save_top_k', 1), 
                mode=self.checkpoint_cfg.get('mode', 'min'),
                verbose=self.checkpoint_cfg.get('verbose', True)
            )
            
            early_stop_cb = pl.callbacks.EarlyStopping(
                monitor=self.early_stop_cfg.get('monitor', 'val_loss'), 
                patience=self.early_stop_cfg.get('patience', 5),
                min_delta=self.early_stop_cfg.get('min_delta', 1e-3), 
                verbose=self.early_stop_cfg.get('verbose', True),
                mode=self.early_stop_cfg.get('mode', 'min')
            )
            
            callbacks_list.extend([self.checkpoint_cb, early_stop_cb])
        elif stage == "fit":
            print("Validation loader not available. Skipping ModelCheckpoint and EarlyStopping.")

        # Add progress bar callback
        callbacks_list.append(TQDMProgressBar(refresh_rate=50))

        # Setup logger
        tensorboard_logger = TensorBoardLogger(
            self.args.root_path, 
            name=f"{self.args.model_type}_pressing"
        )

        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=self.trainer_cfg.get('max_epochs', 10),
            accelerator=self.trainer_cfg.get('accelerator', "auto"),
            devices=self.trainer_cfg.get('devices', [1]),
            gradient_clip_val=self.trainer_cfg.get('gradient_clip_val', 0.0),
            callbacks=callbacks_list,
            logger=tensorboard_logger,
            log_every_n_steps=max(1, len(self.train_loader)//20) if stage=='fit' else 1,
            deterministic=True
        )

    def train(self):
        """Runs the training process."""
        if not self._setup_data(stage='fit'):
            print("Failed to setup training data. Aborting training.")
            return
            
        if not self.train_loader:
            print("Training loader not created. Aborting training.")
            return
            
        self._setup_model()
        self._setup_trainer(stage='fit')

        print(f"Starting training for model type: {self.args.model_type}")
        try:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
            print("Training finished.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()

    def test(self, ckpt_path):
        """Runs the testing process using the best checkpoint."""
        if not self._setup_data(stage='test'):
            print("Failed to setup test data. Aborting testing.")
            return
            
        self._setup_model()
        self._setup_trainer(stage='test')
        
        if not self.test_loader:
            print("Test loader not created. Aborting testing.")
            return
        
        if not os.path.exists(ckpt_path):
             print(f"Error: Checkpoint file not found at {ckpt_path}")
             return

        print(f"Starting testing using checkpoint: {ckpt_path}")
        try:
            test_results = self.trainer.test(
                self.model, 
                dataloaders=self.test_loader, 
                ckpt_path=ckpt_path
            )
            print("--- Test Results ---")
            if test_results:
                print(test_results)
            else:
                print("Testing completed but no results returned.")
            print("--------------------")
        except Exception as e:
            print(f"An error occurred during testing: {e}")
            import traceback
            traceback.print_exc()
        print("Testing finished.")

