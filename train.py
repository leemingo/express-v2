import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

# Import from other project files
from datasets import PressingSequenceDataset, SoccerMapInputDataset
from components import press


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration."""
    
    # Set random seed
    pl.seed_everything(cfg.seed)
    
    # Initialize wandb if enabled
    if cfg.logging.use_wandb:
        if cfg.logging.wandb.entity is None:
            # Try to get entity from environment variable
            cfg.logging.wandb.entity = os.getenv("WANDB_ENTITY")
        
        # Log configuration to wandb
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"Initialized Weights & Biases logging for project: {cfg.logging.wandb.project}")
    else:
        print("Weights & Biases logging disabled")
    
    # Create component based on model type
    component_dict = {
        "soccermap": press.SoccerMapComponent,
        "exPress": press.exPressComponent,
        "xgboost": press.XGBoostComponent,
    }
    
    if cfg.model_type not in component_dict:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
    
    # Create component with configuration
    exp = component_dict[cfg.model_type](cfg)
    
    # Run training or testing
    if cfg.mode == 'train':
        exp.train()
        print("\nAttempting to test the best model after training...")
        exp.test(exp.checkpoint_cb.best_model_path)
    elif cfg.mode == 'test':
        if not cfg.ckpt_path:
            print("Error: ckpt_path must be provided when running in 'test' mode.")
        else:
            exp.test(cfg.ckpt_path)
    else:
        print(f"Error: Unknown mode '{cfg.mode}'. Choose 'train' or 'test'.")
    
    # Close wandb if enabled
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()