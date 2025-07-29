import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import getpass

# Import from other project files
from datasets import PressingSequenceDataset, SoccerMapInputDataset
from components import press


def setup_wandb(cfg: DictConfig):
    """Wandb setup and login process"""
    
    if not cfg.logging.use_wandb:
        print("Weights & Biases logging disabled")
        return False
    
    # Check and input API key
    api_key = cfg.logging.wandb.api_key
    if not api_key:
        # Check from environment variable
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            print("Wandb API key is not set.")
            print("Please enter the API key in config file logging.wandb.api_key")
            print("or set the environment variable WANDB_API_KEY.")
            print("Or enter it here directly (https://wandb.ai/settings):")
            api_key = getpass.getpass("API Key: ").strip()
            
            if not api_key:
                print("API key was not entered. Wandb logging will be disabled.")
                cfg.logging.use_wandb = False
                return False
            
            # Save to config
            cfg.logging.wandb.api_key = api_key
            print("API key has been saved to config.")
        else:
            print(f"API key loaded from environment variable: {api_key[:8]}...")
    else:
        print(f"API key loaded from config: {api_key[:8]}...")
    
    # Check and input entity
    if cfg.logging.wandb.entity is None:
        entity = os.getenv("WANDB_ENTITY")
        if not entity:
            print("Please enter Wandb entity (username/team name):")
            entity = input("Entity: ").strip()
            if entity:
                cfg.logging.wandb.entity = entity
            else:
                print("Entity was not entered. Using default value.")
                cfg.logging.wandb.entity = None
        else:
            cfg.logging.wandb.entity = entity
    
    # Try Wandb login
    try:
        # Call wandb.login()
        wandb.login(key=api_key)
        print("✅ Wandb login successful!")
        
        # Log configuration to wandb
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"✅ Weights & Biases logging initialized.")
        print(f"   Project: {cfg.logging.wandb.project}")
        print(f"   Entity: {cfg.logging.wandb.entity}")
        print(f"   Experiment name: {cfg.exp_name}")
        return True
        
    except Exception as e:
        print(f"❌ Wandb initialization failed: {e}")
        print("Wandb logging will be disabled.")
        cfg.logging.use_wandb = False
        return False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration."""
    
    # Set random seed
    pl.seed_everything(cfg.seed)
    
    # Setup wandb
    wandb_enabled = setup_wandb(cfg)
    
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
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()