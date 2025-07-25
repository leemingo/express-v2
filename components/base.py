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
    
    def __init__(self, cfg):
        """
        Initializes the pipeline with Hydra configuration.

        Args:
            cfg (DictConfig): Hydra configuration object.
        """
        pl.seed_everything(cfg.seed, workers=True)
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.trainer_cfg = cfg.trainer
        self.optimizer_cfg = cfg.optimizer
        self.model_cfg = cfg.model
        self.early_stop_cfg = cfg.early_stop
        self.checkpoint_cfg = cfg.checkpoint

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
            train_pkl_path = f"{self.data_cfg.root_path}/train_dataset.pkl"
            
            try:
                full_train_dataset = self._load_dataset_from_pkl(
                    train_pkl_path, 
                    PressingSequenceDataset, 
                    [self.data_cfg.root_path]
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
            test_pkl_path = f"{self.data_cfg.root_path}/test_dataset.pkl"
            
            try:
                test_dataset = self._load_dataset_from_pkl(
                    test_pkl_path,
                    PressingSequenceDataset,
                    [self.data_cfg.root_path]
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
        ckpt_path = os.path.join(self.cfg.data.root_path, f"checkpoints_{self.cfg.exp_name}")
        os.makedirs(ckpt_path, exist_ok=True)
        
        # Setup callbacks
        callbacks_list = []
        
        if stage == "fit" and self.val_loader:
            monitor_metric = self.checkpoint_cfg.monitor_metric
            ckpt_filename = f"{self.cfg.model_type}-" + "{epoch:02d}-{val_loss:.2f}"
            
            print(f"Configuring Checkpoint and Early Stopping based on '{monitor_metric}'")
            
            self.checkpoint_cb = pl.callbacks.ModelCheckpoint(
                monitor=monitor_metric, 
                dirpath=ckpt_path, 
                filename=ckpt_filename,
                save_top_k=self.checkpoint_cfg.save_top_k, 
                mode=self.checkpoint_cfg.mode,
                verbose=self.checkpoint_cfg.verbose
            )
            
            early_stop_cb = pl.callbacks.EarlyStopping(
                monitor=self.early_stop_cfg.monitor_metric, 
                patience=self.early_stop_cfg.patience,
                min_delta=self.early_stop_cfg.min_delta, 
                verbose=self.early_stop_cfg.verbose,
                mode=self.early_stop_cfg.mode
            )
            
            callbacks_list.extend([self.checkpoint_cb, early_stop_cb])
        elif stage == "fit":
            print("Validation loader not available. Skipping ModelCheckpoint and EarlyStopping.")

        # Add progress bar callback
        callbacks_list.append(TQDMProgressBar(refresh_rate=50))

        # Setup logger based on configuration
        if self.cfg.logging.use_wandb:
            # Use Weights & Biases logger
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=self.cfg.logging.wandb.project,
                entity=self.cfg.logging.wandb.entity,
                name=self.cfg.exp_name,
                log_model=self.cfg.logging.wandb.log_model,
                save_code=self.cfg.logging.wandb.save_code
            )
        else:
            # Use TensorBoard logger as fallback
            from pytorch_lightning.loggers import TensorBoardLogger
            logger = TensorBoardLogger(
                save_dir=self.cfg.data.root_path,
                name=f"{self.cfg.model_type}_pressing"
            )

        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=self.trainer_cfg.max_epochs,
            accelerator=self.trainer_cfg.accelerator,
            devices=self.trainer_cfg.devices,
            gradient_clip_val=self.trainer_cfg.gradient_clip_val,
            callbacks=callbacks_list,
            logger=logger,
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

        print(f"Starting training for model type: {self.cfg.model_type}")
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

