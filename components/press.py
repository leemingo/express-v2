from abc import ABC, abstractmethod
import json
import os
import pickle 
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from datasets import PressingSequenceDataset, SoccerMapInputDataset, exPressInputDataset # Assuming dataset.py contains this
from components.base import BaseComponent
from model import PytorchSoccerMapModel, TemporalSoccerMapModel # Assuming model.py contains this

class SoccerMapComponent(BaseComponent):
    """Handles SoccerMap model training and testing."""
    def __init__(self, args):
        super().__init__(args)

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
            full_train_dataset = SoccerMapInputDataset(train_pkl_path)   
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
            test_dataset = SoccerMapInputDataset(test_pkl_path)   

            if len(test_dataset) > 0:
                self.test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)
                print(f"Test loader created with {len(test_dataset)} samples.")
            else:
                print("Test dataset loaded but is empty.")
                self.test_loader = None
        else:
             print(f"Unknown stage: {stage}")


    def _setup_model(self):
        """Initializes the Lightning model."""
        print(f"Initializing model type: {self.args.model_type}")
        if self.args.model_type == 'soccermap':
            # self.model = PytorchSoccerMapModel(self.model_cfg, self.optimizer_cfg)
            self.model = TemporalSoccerMapModel(self.model_cfg, self.optimizer_cfg)
        else:
            print(f"Error: Model type '{self.args.model_type}' setup not implemented.")
            exit()

class XGBoostComponent(BaseComponent):
    """Handles XGBoost model training and testing."""

    def __init__(self, args):
        super().__init__(args)

    def _setup_model(self):
        pass

class exPressComponent(BaseComponent):
    """Handles exPress model training and testing."""

    def __init__(self, args):
        super().__init__(args)

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
            full_train_dataset = exPressInputDataset(train_pkl_path)   
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
            test_dataset = exPressInputDataset(test_pkl_path)   

            if len(test_dataset) > 0:
                self.test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)
                print(f"Test loader created with {len(test_dataset)} samples.")
            else:
                print("Test dataset loaded but is empty.")
                self.test_loader = None
        else:
             print(f"Unknown stage: {stage}")

    
    def _setup_model(self):
        """Initializes the Lightning model."""
        print(f"Initializing model type: {self.args.model_type}")
        if self.args.model_type == 'exPress':
            # self.model = PytorchSoccerMapModel(self.model_cfg, self.optimizer_cfg)
            self.model = exPressModel(self.model_cfg, self.optimizer_cfg)
        else:
            print(f"Error: Model type '{self.args.model_type}' setup not implemented.")
            exit()

    
