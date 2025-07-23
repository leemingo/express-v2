from abc import ABC, abstractmethod
import json
import os
import pickle 
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from datasets import SoccerMapInputDataset, exPressInputDataset # Assuming dataset.py contains this
from components.base import BaseComponent
from model import PytorchSoccerMapModel, TemporalSoccerMapModel, exPressModel # Assuming model.py contains this
from utils_data import custom_temporal_collate


class SoccerMapComponent(BaseComponent):
    """Handles SoccerMap model training and testing."""
    
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_common_loader_args(self):
        """Get common DataLoader arguments with custom collate function."""
        common_args = super()._get_common_loader_args()
        common_args["collate_fn"] = self._collate_fn_skip_none
        return common_args

    def _collate_fn_skip_none(self, batch):
        """Custom collate function that skips None values."""
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except RuntimeError:
            return None

    def _setup_data(self, stage='fit'):
        """Sets up datasets and dataloaders for fit (train/val) or test stage."""
        print(f"Setting up data for stage: {stage}")
        
        common_loader_args = self._get_common_loader_args()
        
        if stage == 'fit':
            train_pkl_path = f"{self.data_cfg.root_path}/train_dataset.pkl"
            
            try:
                full_train_dataset = SoccerMapInputDataset(self.data_cfg, train_pkl_path)
            except Exception as e:
                print(f"Error loading training dataset: {e}")
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
                test_dataset = SoccerMapInputDataset(self.data_cfg, test_pkl_path)
            except Exception as e:
                print(f"Error loading test dataset: {e}")
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

    def _setup_model(self):
        """Initializes the Lightning model."""
        print(f"Initializing model type: {self.cfg.model_type}")
        
        if self.cfg.model_type == 'soccermap':
            self.model = PytorchSoccerMapModel(self.model_cfg, self.optimizer_cfg)
        elif self.cfg.model_type == 'temporal_soccermap':
            self.model = TemporalSoccerMapModel(self.model_cfg, self.optimizer_cfg)
        else:
            raise ValueError(f"Model type '{self.cfg.model_type}' setup not implemented.")


class XGBoostComponent(BaseComponent):
    """Handles XGBoost model training and testing."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def _setup_model(self):
        """Initializes the XGBoost model."""
        print(f"Initializing model type: {self.cfg.model_type}")
        # TODO: Implement XGBoost model initialization
        raise NotImplementedError("XGBoost model setup not yet implemented.")


class exPressComponent(BaseComponent):
    """Handles exPress model training and testing."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_common_loader_args(self):
        """Get common DataLoader arguments with temporal collate function."""
        common_args = super()._get_common_loader_args()
        common_args["collate_fn"] = custom_temporal_collate
        return common_args

    def _setup_data(self, stage='fit'):
        """Sets up datasets and dataloaders for fit (train/val) or test stage."""
        print(f"Setting up data for stage: {stage}")
        
        common_loader_args = self._get_common_loader_args()

        # Load datasets
        train_pkl_path = f"{self.data_cfg.root_path}/train_dataset.pkl"
        valid_pkl_path = f"{self.data_cfg.root_path}/valid_dataset.pkl"
        test_pkl_path = f"{self.data_cfg.root_path}/test_dataset.pkl"

        try:
            train_dataset = exPressInputDataset(train_pkl_path, wo_vel=self.data_cfg.wo_vel)
            val_dataset = exPressInputDataset(valid_pkl_path, wo_vel=self.data_cfg.wo_vel)
            test_dataset = exPressInputDataset(test_pkl_path, wo_vel=self.data_cfg.wo_vel)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False

        if len(train_dataset) == 0:
            print("Loaded training dataset is empty.")
            return False

        if stage == 'fit':
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            print(f"Total samples - Training: {train_size}, Validation: {val_size}")
            
            self.train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
            self.val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
            
        elif stage == 'test':
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

    def _setup_model(self):
        """Initializes the Lightning model."""
        print(f"Initializing model type: {self.cfg.model_type}")
        
        if self.cfg.model_type == 'exPress':
            self.model = exPressModel(self.model_cfg, self.optimizer_cfg)
        else:
            raise ValueError(f"Model type '{self.cfg.model_type}' setup not implemented.")

    
