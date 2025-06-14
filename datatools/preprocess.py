import os
import numpy as np
import pandas as pd
import pickle
import torch
os.chdir('/home/exPress/express-v2/')
from matplotlib import animation
import sys
sys.path.append(os.getcwd())
import config as C
from tqdm import tqdm

def data_collect(match_id):
    match_path = f"{processed_data_path}/{match_id}"
    with open(f"{match_path}/{match_id}_processed_dict.pkl", "rb") as f:
        match_dict = pickle.load(f)

    tracking_df = match_dict['tracking_df']
    event_df = pd.read_csv(os.path.join(match_path, 'valid_events.csv'))
    teams_dict = match_dict['teams']
    return tracking_df, event_df, teams_dict

def merge(tracking_df, event_df, teams_dict):
    teams_df = pd.concat([teams_dict['Home'], teams_dict['Away']])
    teams_df.reset_index(drop=True, inplace=True)
    teams_df['player_code'] = teams_df.apply(lambda row: row['team'][0] + str(row['xID']).zfill(2), axis=1)
    
    event_df = event_df.dropna(subset=['player_id']) 
    event_df['player_id'] = event_df['player_id'].astype(int).astype(str)
    event_df = event_df.merge(teams_df, how='left', left_on='player_id', right_on='pID')
    events = event_df.drop(['player_id', 'team_id'], axis=1)
    events["pID"] = events["pID"].astype(str)

    tracking_df['period_id'] = tracking_df['period_id'].astype('int')
    tracking_df = tracking_df.merge(teams_df[['pID', 'player_code']], how='left', left_on='id', right_on='pID')
    tracking_df.drop(['pID'], axis=1, inplace=True)

    # Ball 처리
    nan_mask = pd.isna(tracking_df['player_code'])
    tracking_df.loc[nan_mask, 'player_code'] = tracking_df.loc[nan_mask, 'id']

    wide_tracking_df = tracking_df.pivot_table(
        index=['period_id', 'timestamp', 'frame_id'],
        columns='player_code',
        values=['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a']
    )
    wide_tracking_df.columns = [f'{player_code}_{value}' for value, player_code in wide_tracking_df.columns]
    wide_tracking_df.reset_index(inplace=True)

    return events, wide_tracking_df, teams_df

def time_set(events, wide_tracking_df):
    # Event 시간 조정
    first_half_end = events.loc[events['period_id'] == 1, 'time_seconds'].max()
    events.loc[events['period_id'] == 2, 'time_seconds'] += first_half_end

    # Tracking 시간 조정
    converted_timestamps = pd.to_timedelta(wide_tracking_df['timestamp'], errors='coerce')
    seconds = converted_timestamps.dt.total_seconds()
    first_half_end = seconds[wide_tracking_df['period_id'] == 1].max()
    second_half_start = seconds[wide_tracking_df['period_id'] == 2].min()
    offset = first_half_end - second_half_start

    seconds_adjusted = seconds.copy()
    seconds_adjusted[wide_tracking_df['period_id'] == 2] += offset
    wide_tracking_df['time_seconds'] = seconds_adjusted

    # 병합
    traces_df = pd.merge_asof(
        events.sort_values("time_seconds"),
        wide_tracking_df.sort_values("time_seconds"),
        by='period_id',
        on='time_seconds',
        direction='nearest'
    )

    return traces_df

def left_right_inversion(traces_df):
    x_cols = [col for col in traces_df.columns if col.endswith("_x")]
    y_cols = [col for col in traces_df.columns if col.endswith("_y")]
    home_x_cols = [col for col in x_cols if col.startswith("H")]
    away_x_cols = [col for col in x_cols if col.startswith("A")]

    for period_id in traces_df["period_id"].unique():
        home_x = np.nanmean(traces_df.loc[traces_df["period_id"] == period_id, home_x_cols].iloc[0])
        away_x = np.nanmean(traces_df.loc[traces_df["period_id"] == period_id, away_x_cols].iloc[0])

        if away_x < home_x:
            traces_df.loc[traces_df["period_id"] == period_id, x_cols] *= -1
            traces_df.loc[traces_df["period_id"] == period_id, y_cols] *= -1
    return traces_df

# === 실행 ===
if __name__ == "__main__":
    raw_data_path = "/home/exPress/PlayerImputer/data/BEPRO/2024"
    processed_data_path = "/home/exPress/express-v2/data/bepro/processed"
    os.makedirs(processed_data_path, exist_ok=True)

    match_ids = sorted(os.listdir(raw_data_path))
    for match_id in tqdm(match_ids):
        try:
            print(f"▶ Processing match: {match_id}")
            tracking_df, event_df, teams_dict = data_collect(match_id)
            events, wide_tracking_df, teams_df = merge(tracking_df, event_df, teams_dict)
            traces_df = time_set(events, wide_tracking_df)
            traces_df = left_right_inversion(traces_df)
            teams_df.to_csv(f"{processed_data_path}/{match_id}/{match_id}_teams.csv", index=False)
            wide_tracking_df.to_csv(f"{processed_data_path}/{match_id}/{match_id}_traces.csv", index=False)
            traces_df.to_csv(f"{processed_data_path}/{match_id}/{match_id}_merged.csv", index=False)
            print(f"✔ Saved: {match_id}")
        except Exception as e:
            print(f"✖ Failed to process {match_id}: {e}")
