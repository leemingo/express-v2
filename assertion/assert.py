import os
import warnings
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from pathlib import Path
base_path = Path(__file__).resolve().parent.parent
print(f"Base path: {base_path}")
data_path = Path(base_path, "data", "bepro", "processed")
yaml_file = Path(base_path, "assertion", "transitions.yaml")

from bepro import convert_to_actions
from validator import Validator
import assertion.config as lsdpconfig

def load_and_save_data():
    match_id_lst = [id for id in os.listdir(data_path) if "DS" not in id] # DF_Stores: acOS에서 Finder가 해당 폴더의 메타데이터를 저장하기 위해 자동으로 생성하는 숨김 파일
    
    for match_id in tqdm(match_id_lst[23:], desc="Loading games"):  
        with open(f"{data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
            match_dict = pickle.load(f)
            events = match_dict['event_df']
            teams_dict = match_dict['teams']
            metadata = match_dict['meta_data']

        teams_df = pd.concat([teams_dict["Home"], teams_dict["Away"]], ignore_index=True)
        player_name_to_player_id = {row.player: int(row.pID) for row in teams_df.itertuples()}
        player_id_to_team_id = {int(row.pID): int(row.tID) for row in teams_df.itertuples()}
        player_id_to_team = {int(row.pID): row.team for row in teams_df.itertuples()}
        
        events["game_id"] = match_id
        events["event_id"] = range(len(events))
        events["player_id"] = events["player_name"].map(player_name_to_player_id).astype(int)
        events["team_id"] = events["player_id"].map(player_id_to_team_id).astype(int)
        events["team"] = events["player_id"].map(player_id_to_team)

        actions = convert_to_actions(events)
      
        validator = Validator(actions, yaml_file)
        validator.validate_sequence()
        valid_events = validator.df_events

        validator.df_errors.to_csv(Path(data_path / match_id / f"error.csv"), index=False) 
        valid_events.to_csv(Path(data_path / f"{match_id}/valid_events.csv"), index=False)

        valid_events["type_name"] = valid_events["type_name"].map(lsdpconfig.versa_to_spadl_dict)
        valid_events = valid_events[valid_events["type_name"] != "non_action"].reset_index(drop=True)
        valid_events["type_id"] = valid_events["type_name"].apply(lambda t: lsdpconfig.spadl_actiontypes.index(t))
        valid_events.to_csv(Path(data_path / f"{match_id}/valid_events_filtered.csv"), index=False)

if __name__ == "__main__":
    load_and_save_data()
