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

PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0


def load_and_save_data():
    match_id_lst = [id for id in os.listdir(data_path) if "DS" not in id] # DF_Stores: acOS에서 Finder가 해당 폴더의 메타데이터를 저장하기 위해 자동으로 생성하는 숨김 파일
    
    for match_id in tqdm(match_id_lst[:3], desc="Loading games"):  
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

        # validation거친 VERSA데이터셋
        validator.df_errors.to_csv(Path(data_path / match_id / f"error.csv"), index=False) 
        valid_events.to_csv(Path(data_path / f"{match_id}/valid_events.csv"), index=False)

        # MLSA에서 사용하기로 한 데이터셋: 민호님께서 구현하신 트래킹 데이터랑 위치 스키마 통일
        # x=(-52.5, 52.5), y=(-34, 34)로 통일

        for col in ["start_x", "end_x"]:
            valid_events[col] += PITCH_X_MIN
        for col in ["start_y", "end_y"]:
            valid_events[col] += PITCH_Y_MIN
       
        valid_events["type_name"] = valid_events["type_name"].map(lsdpconfig.versa_to_spadl_dict)
        valid_events = valid_events[valid_events["type_name"] != "non_action"].reset_index(drop=True)
        valid_events["type_id"] = valid_events["type_name"].apply(lambda t: lsdpconfig.spadl_actiontypes.index(t))
        valid_events.to_csv(Path(data_path / f"{match_id}/valid_events_filtered.csv"), index=False)

if __name__ == "__main__":
    load_and_save_data()
