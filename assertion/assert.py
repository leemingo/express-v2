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

from soccerparser.constants import Constants
from soccerparser.kleague_parser import KLeagueParser

def parse_kleague_data(data_path, match_id):
    """
    Parse K-League data and save it to the specified directory.
    """
    with open(f"{data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
        match_dict = pickle.load(f)
        events = match_dict['event_df']
        teams_dict = match_dict['teams']
        metadata = match_dict['meta_data']
        tracking = match_dict['tracking_df']

    team_sheets = pd.concat([teams_dict["Home"], teams_dict["Away"]], ignore_index=True)
    player_code_to_player_id = {f"{row.team[0]}{int(row.jID):02d}": row.pID for row in team_sheets.itertuples()}
    player_id_to_player_name = {int(row.pID): row.player for row in team_sheets.itertuples()}
    player_name_to_player_id = {row.player: int(row.pID) for row in team_sheets.itertuples()}
    player_id_to_jersey_number = {int(row.pID): int(row.jID) for row in team_sheets.itertuples()}

    parser = KLeagueParser(match_id, data_dir=f"{base_path}/data/bepro/processed")
    parser.parse_events()
    parser.parse_ball_xy()

    parser.combine_events_and_ball_xy()
    parser.parse_player_xy()
    parser.combine_player_and_ball_xy()

    parse_events = parser.events
    parse_events["player_id"] = parse_events["player_code"].map(player_code_to_player_id)
    parse_events = parse_events[["event_id", "player_id"]]

    raw_events = events.reset_index(drop=True)
    raw_events["event_id"] = range(len(events))
    raw_events["player_id"] = raw_events["player_name"].map(player_name_to_player_id).astype(int)
    # player정보는 재매핑
    raw_events = raw_events[['event_id', 'period_type', 'period_name', 'period_order', 
                            'period_duration', 'period_start_time', 'event_time', 'player_id',
                            'team_name', 'events', 'x', 'y', 'to_x', 'to_y', 'attack_direction']]
    
    # parse에서도 event_id를 처음에 생성하므로 event_id를 기준으로 병합 가능
    merged = raw_events.merge(parse_events, on="event_id", how="left", suffixes=("", "_parse"))
    merged["player_id"] = merged["player_id_parse"].combine_first(merged["player_id"]).astype(int)
    merged = merged.drop(columns=["player_id_parse"])

    merged["player_name"] = merged["player_id"].map(player_id_to_player_name)
    merged["jersey_number"] = merged["player_id"].map(player_id_to_jersey_number)

    return merged

def load_and_save_data():
    match_id_lst = [id for id in os.listdir(data_path) if "DS" not in id] # DF_Stores: acOS에서 Finder가 해당 폴더의 메타데이터를 저장하기 위해 자동으로 생성하는 숨김 파일
    
    for match_id in tqdm(match_id_lst, desc="Loading games"):  
        with open(f"{data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
            match_dict = pickle.load(f)
            events = match_dict['event_df']
            teams_dict = match_dict['teams']
            _ = match_dict['meta_data']
            _ = match_dict['tracking_df']

        # K리그 데이터셋은 팀이름이 'Home', 'Away'로 되어있음          
        events = parse_kleague_data(data_path, match_id)

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
