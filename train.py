import argparse

# Import from other project files
from datasets import PressingSequenceDataset, SoccerMapInputDataset
from components import press

if __name__ == "__main__":
    # Setup argument parser to accept checkpoint path from command line
    parser = argparse.ArgumentParser(description="Train a pressing evaluation model.")
    # parser.add_argument("--model_type", type=str, default="soccermap", choices=['soccermap', 'xgboost', 'exPress'], help="Path to the model checkpoint (.ckpt) file saved during training.")
    # parser.add_argument("--root_path", type=str, default="/data/MHL/pressing-intensity", help="Path to the data file.")
    parser.add_argument("--model_type", type=str, default="exPress", choices=['soccermap', 'xgboost', 'exPress'], help="Path to the model checkpoint (.ckpt) file saved during training.")
    parser.add_argument("--root_path", type=str, default="/data/MHL/pressing-intensity-0.9", help="Path to the data file.")
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'], help="Mode: 'train' or 'test'.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file (Required for 'test' mode).")
    parser.add_argument("--params_path", type=str, default="params.json", help="Path to the JSON containing configurations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed number.")
    parser.add_argument("--exp_name", type=str, default="vat_one_frame_0.7_w/ovel", help="Experient name.")
    
    args = parser.parse_args()
    component_dict = {
                    "soccermap": press.SoccerMapComponent,
                    "exPress": press.exPressComponent,
                }

    exp = component_dict[args.model_type](args)
    
    if args.mode == 'train':
        exp.train()
        print("\nAttempting to test the best model after training...")
        exp.test(exp.checkpoint_cb.best_model_path)
    elif args.mode == 'test':
        if not args.ckpt_path:
             # In standalone test mode, checkpoint path is mandatory
             print("Error: --ckpt_path must be provided when running in 'test' mode.")
        else:
            exp.test(args.ckpt_path)
    else:
        print(f"Error: Unknown mode '{args.mode}'. Choose 'train' or 'test'.")