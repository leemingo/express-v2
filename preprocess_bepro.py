import json
import numpy as np
import os
import pandas as pd
import pickle
from scipy.signal import savgol_filter
from tqdm import tqdm

import config as C
from config import Constant, Column, Group

from utils_data import infer_ball_carrier

def load_single_json(file_path):
    """
    Loads and parses a single JSON file from the specified path.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict or list or None: The parsed Python object on success.
                              Returns None if the file is not found or
                              is not valid JSON.
    """
    try:
        # Open the file in read mode ('r') with UTF-8 encoding.
        # Using 'with' ensures the file is automatically closed after use.
        with open(file_path, 'r', encoding='utf-8') as f:
            # json.load() parses the JSON data from the file object.
            data = json.load(f)
            # print(f"Successfully loaded file: '{file_path}'")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return None

def load_jsonl(file_path):
    """
    Loads data from a JSON Lines (.jsonl) file.

    Each line in the file is expected to be a valid JSON object.
    Lines that are empty or cannot be parsed as JSON will be skipped with a warning.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list containing the Python objects parsed from each valid JSON line.
              Returns an empty list if the file is not found or contains no valid JSON lines.
    """
    data = [] # To store the parsed JSON objects from each line
    try:
        # Open the file in read mode ('r') with UTF-8 encoding.
        # 'with' ensures the file is closed automatically.
        with open(file_path, 'r', encoding='utf-8') as f:
            # Iterate through each line in the file.
            # enumerate adds line numbers (starting from 1) for better error reporting.
            for line_number, line in enumerate(f, 1):
                # Remove leading/trailing whitespace (including the newline character \n)
                processed_line = line.strip()

                # Skip empty lines
                if not processed_line:
                    continue

                try:
                    # Parse the current line (which is a string) into a Python object.
                    # Use json.loads() for parsing a string, not json.load().
                    parsed_object = json.loads(processed_line)
                    data.append(parsed_object)
                except json.JSONDecodeError:
                    # Handle lines that are not valid JSON.
                    print(f"Warning: Skipping line {line_number} in '{file_path}' due to JSON decoding error.")
                    # Optional: Print the problematic line for debugging
                    # print(f"         Problematic line content: {processed_line[:100]}...")
                except Exception as e:
                    # Handle any other unexpected errors during line processing
                    print(f"Warning: An unexpected error occurred processing line {line_number} in '{file_path}': {e}. Skipping line.")

    except FileNotFoundError:
        # Handle the case where the file itself doesn't exist.
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        # Handle other potential errors during file opening or reading (outside the line loop).
        print(f"An error occurred while reading the file '{file_path}': {e}")

    # Return the list of successfully parsed objects.
    return data

def create_team_dataframe(home_team_info, away_team_info):
    home_team_rows = []
    base_info = {
            'player': None,
            'position': None,
            'team': 'Home',
            'jID': None,
            'pID': None,
            'tID': home_team_info.get('team_id'),
            'xID': None   
        }
    for idx, player_data in enumerate(home_team_info['players']):
        player_info = base_info.copy()
        player_info['player'] = player_data['full_name_en']
        player_info['position'] = player_data['initial_position_name']
        player_info['jID'] = player_data['shirt_number']
        player_info['pID'] = player_data['player_id']
        player_info['xID'] = idx
        home_team_rows.append(player_info)

    home_df = pd.DataFrame(home_team_rows)
    home_df['pID'] = home_df['pID'].astype(str)
    home_df['tID'] = home_df['tID'].astype(str)

    away_team_rows = []
    base_info['team'] = 'Away'
    base_info['tID'] = away_team_info.get('team_id')
    for idx, player_data in enumerate(away_team_info['players']):
        player_info = base_info.copy()
        player_info['player'] = player_data['full_name_en']
        player_info['position'] = player_data['initial_position_name']
        player_info['jID'] = player_data['shirt_number']
        player_info['pID'] = player_data['player_id']
        player_info['xID'] = idx
        away_team_rows.append(player_info)

    away_df = pd.DataFrame(away_team_rows)
    away_df['pID'] = away_df['pID'].astype(str)
    away_df['tID'] = away_df['tID'].astype(str)


    return {'Home': home_df, 'Away': away_df}

def create_event_dataframe(match_path):
    first_half_event_path =  next((f for f in os.listdir(match_path) if "1_event" in f), None)
    second_half_event_path =  next((f for f in os.listdir(match_path) if "2_event" in f), None)

    first_half_event_data = load_single_json(f"{match_path}/{first_half_event_path}")
    second_half_event_data = load_single_json(f"{match_path}/{second_half_event_path}")

    first_half_event_df = pd.DataFrame(first_half_event_data['data'])
    second_half_event_df = pd.DataFrame(second_half_event_data['data'])
    total_event_df = pd.concat([first_half_event_df, second_half_event_df], axis=0)
    return total_event_df

def _calculate_kinematics(df: pd.DataFrame, smoothing_params: dict, max_speed: float, max_acceleration: float, is_ball: bool = False):
    """Calculates velocity and acceleration for a single agent over periods."""
    df_out = pd.DataFrame()
    required_cols = ['x', 'y', 'z', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns in input dataframe. Found: {df.columns.tolist()}")
        return df_out # Return empty if essential columns missing

    for period_id in df['period_id'].unique():
        period_df = df[df['period_id'] == period_id].copy()
        period_df = period_df.sort_values(by='timestamp') # Ensure order for diff

        # Replacing Nan values with linear interpolation.
        period_df['x'] = period_df['x'].interpolate().copy()
        period_df['y'] = period_df['y'].interpolate().copy()

        # Calculate time difference (dt) safely
        dt = period_df['timestamp'].diff().dt.total_seconds()
        # Avoid division by zero or large values for the first frame
        # dt.iloc[0] = dt.median() # Use median dt for the first frame or a typical dt (e.g., 0.04)
        # dt = dt.replace(0, np.nan).ffill().bfill() # Replace 0s, forward/backward fill NaNs

        # Calculate velocities
        period_df['vx'] = period_df['x'].diff() / dt
        period_df['vy'] = period_df['y'].diff() / dt
        period_df['vz'] = period_df['z'].diff() / dt if is_ball else 0.0

        # Smooth velocities (handle potential NaNs from diff)
        vel_cols = ['vx', 'vy', 'vz'] if is_ball else ['vx', 'vy']
        for col in vel_cols:
            data_to_smooth = period_df[col].fillna(0) # Fill NaNs before smoothing
             # Ensure window length is odd and <= data length
            window_length = min(smoothing_params['window_length'], len(data_to_smooth))
            if window_length % 2 == 0: window_length -= 1 # Make odd
            if window_length >= smoothing_params['polyorder'] + 1 and window_length > 0: # Basic check
                period_df[col] = savgol_filter(data_to_smooth,
                                              window_length=window_length,
                                              polyorder=smoothing_params['polyorder'])
            else: # Not enough data or invalid params, skip smoothing
                period_df[col] = data_to_smooth

        # Calculate accelerations
        period_df['ax'] = period_df['vx'].diff() / dt
        period_df['ay'] = period_df['vy'].diff() / dt
        period_df['az'] = period_df['vz'].diff() / dt if is_ball else 0.0

        # Fill NaN accelerations (occur at start and where dt was invalid)
        accel_cols = ['ax', 'ay', 'az']
        for col in accel_cols:
            period_df[col] = period_df[col].fillna(0)

        # Calculate Speed and Acceleration Magnitude & Apply Caps
        period_df['v'] = np.sqrt(period_df['vx']**2 + period_df['vy']**2 + period_df['vz']**2)
        period_df['a'] = np.sqrt(period_df['ax']**2 + period_df['ay']**2 + period_df['az']**2)

        period_df['v'] = np.minimum(period_df['v'], max_speed)
        period_df['a'] = np.minimum(period_df['a'], max_acceleration)

        df_out = pd.concat([df_out, period_df], ignore_index=True)

    return df_out


def resample_tracking_dataframe(tracking_df, target_hz):
    resample_freq_ms = int(1000 / target_hz)
    resample_freq_str = f'{resample_freq_ms}ms'
    
    period_list = []
    for period_id in tracking_df['period_id'].unique():
        period_df = tracking_df[tracking_df['period_id'] == period_id]

        min_timestamp = period_df['timestamp'].min()
        max_timestamp = period_df['timestamp'].max()
        global_original_index = pd.to_timedelta(sorted(period_df['timestamp'].unique()))
        global_target_index = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=resample_freq_str)

        grouped = period_df.groupby('id')
        resampled_list = []

        for agent_id, agent_group in grouped:
            group_df = agent_group.copy().set_index('timestamp')
            if group_df.index.has_duplicates:
                group_df = group_df.loc[~group_df.index.duplicated(keep='first')] # Some data contains duplicated coordinates for each players in the same frame.

            union_index = global_original_index.union(global_target_index)
            reindexed_group = group_df.reindex(union_index)

            # Interpolation
            interpolation_cols = ['x', 'y', 'speed']
            reindexed_group[interpolation_cols] = reindexed_group[interpolation_cols].interpolate(method='cubic', limit_area='inside')
            
            # 5. 최종 결과 필터링: 보간된 결과에서 25Hz 시간대의 데이터만 선택
            final_group = reindexed_group.reindex(global_target_index)

            # 6. 범주형 데이터 채우기
            final_group['id'] = agent_id
            ffill_cols = [col for col in group_df.columns if col not in interpolation_cols and col != 'id']
            final_group[ffill_cols] = final_group[ffill_cols].ffill()
            final_group = final_group.dropna(subset=['x', 'y'])
            resampled_list.append(final_group)

        period_list += resampled_list

    total_resampled_df = pd.concat(period_list).reset_index().rename(columns={'index': 'timestamp'})
    total_resampled_df['frame_id'] = (total_resampled_df['timestamp'].astype(np.int64) // (10**9 / target_hz)).astype(int)
    total_resampled_df = total_resampled_df.sort_values(['game_id', 'timestamp', 'period_id', 'frame_id', 'id'])

    return total_resampled_df



def rescale_pitch(tracking_df, meta_data):
    
    x_ori_min, x_ori_max = 0.0, meta_data['ground_width']
    y_ori_min, y_ori_max = 0.0, meta_data['ground_height']

    x_new_min, x_new_max = C.PITCH_X_MIN, C.PITCH_X_MAX
    y_new_min, y_new_max = C.PITCH_Y_MIN, C.PITCH_Y_MAX

    # 스케일링 팩터 계산
    scale_x = (x_new_max - x_new_min) / (x_ori_max - x_ori_min)  # 105.0 / 110.0
    scale_y = (y_new_max - y_new_min) / (y_ori_max - y_ori_min)  # 68.0 / 65.0

    tracking_df['x'] =  x_new_min + (tracking_df['x'] - x_ori_min) * scale_x
    tracking_df['y'] = y_new_min + (tracking_df['y'] - y_ori_min) * scale_y
    return tracking_df
    


def create_tracking_dataframe(match_path, meta_data, teams_dict):
    teams_df = pd.concat([teams_dict['Home'], teams_dict['Away']], axis=0)
    player_lookup = teams_df.set_index('pID')
    home_tid = teams_dict['Home']['tID'].iloc[0]
    away_tid = teams_dict['Away']['tID'].iloc[0]
    player_smoothing_params= C.DEFAULT_PLAYER_SMOOTHING_PARAMS
    ball_smoothing_params= C.DEFAULT_BALL_SMOOTHING_PARAMS
    max_player_speed=  C.MAX_PLAYER_SPEED
    max_player_acceleration=  C.MAX_PLAYER_ACCELERATION
    max_ball_speed=  C.MAX_BALL_SPEED
    max_ball_acceleration=  C.MAX_BALL_ACCELERATION

    match_id = match_path.split("/")[-1]
    first_half_tracking_data = load_jsonl(f"{match_path}/{match_id}_1_frame_data.jsonl")
    second_half_tracking_data = load_jsonl(f"{match_path}/{match_id}_2_frame_data.jsonl")

    all_object_rows = []

    for half_tracking_data in [first_half_tracking_data, second_half_tracking_data]:
        for frame_data in half_tracking_data:
            # Check ball state
            ball_state = frame_data.get('ball_state')
            if ball_state is None or ball_state == 'out':
                new_ball_state = 'dead'
                ball_owning_team_id = None
            else:
                new_ball_state = 'alive'
                if ball_state == 'home':
                    ball_owning_team_id = home_tid
                elif ball_state == 'away':
                    ball_owning_team_id = away_tid
                else:
                    ball_owning_team_id = ball_state

            # 2. Extract current frames base information.
            frame_info = {
                'game_id': match_id,
                'period_id': frame_data.get('period_order') + 1,
                'timestamp': frame_data.get('match_time'),
                'frame_id': frame_data.get('frame_index'),
                'ball_state': new_ball_state,
                'ball_owning_team_id': ball_owning_team_id,
            }

            for object in ['players', 'balls']:
                object_list = frame_data.get(object, [])
                if object_list:
                    for object_data in object_list:
                        row_data = frame_info.copy()
                        row_data.update(object_data)
                        if object == 'balls':
                            row_data['id'] = 'ball'
                            row_data['team_id'] = 'ball'
                            row_data['position_name'] = 'ball'
                        else:
                            # row_data['id'] = str(row_data['player_id'])
                            # teams_df['team_id'] = teams_df[teams_df['pID'] == row_data['id']]['position'].iloc[0]
                            # teams_df['position_name'] = teams_df[teams_df['pID'] == row_data['id']]['tID'].iloc[0]
                            player_pID = str(object_data.get('player_id'))
                            row_data['id'] = player_pID
                            if player_pID in player_lookup.index:
                                row_data['team_id'] = player_lookup.loc[player_pID, 'tID'] if 'tID' in player_lookup.columns else None
                                row_data['position_name'] = player_lookup.loc[player_pID, 'position'] if 'position' in player_lookup.columns else None
                            else:
                                row_data['team_id'] = None
                                row_data['position_name'] = None
                        row_data.pop('object')
                        row_data.pop('player_id')
                        all_object_rows.append(row_data)

    tracking_df = pd.DataFrame(all_object_rows)
    tracking_df['timestamp'] = pd.to_timedelta(tracking_df['timestamp'], unit='ms')
    # Rescale
    tracking_df = rescale_pitch(tracking_df, meta_data)
    # Resample (30Hz -> 25Hz)
    tracking_df = resample_tracking_dataframe(tracking_df, target_hz=25)
    
    # Calculate kinematics
    agent_ids_present = tracking_df['id'].unique()
    total_tracking_list = []
    for agent_id in agent_ids_present:
        is_ball = (agent_id == 'ball')

        current_agent_df = tracking_df[tracking_df['id'] == agent_id].copy()
        current_agent_df['z'] = 0.0 # bepro data doesn't have z information.

        # Drop rows with NaN coordinates for players
        if not is_ball:
            current_agent_df = current_agent_df.dropna(subset=['x', 'y'])

        # Calculate Kinematics using helper function
        smoothing = ball_smoothing_params if is_ball else player_smoothing_params
        max_v = max_ball_speed if is_ball else max_player_speed
        max_a = max_ball_acceleration if is_ball else max_player_acceleration

        kinematics_df = _calculate_kinematics(current_agent_df, smoothing, max_v, max_a, is_ball)
        total_tracking_list.append(kinematics_df)    
    total_tracking_df = pd.concat(total_tracking_list, axis=0, ignore_index=True)
    
    # Sort final DataFrame
    total_tracking_df = total_tracking_df.sort_values(
        by=["period_id", "timestamp", "frame_id", "id"], kind="mergesort"
    ).reset_index(drop=True)

    # Define final column order (example)
    final_cols_order = [
        'game_id', 'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'v', 'ax', 'ay', 'az', 'a',
        'id', 'team_id', 'position_name',
         #'is_ball_carrier'
    ]
    total_tracking_df = total_tracking_df[[col for col in final_cols_order if col in total_tracking_df.columns]]
    total_tracking_df = infer_ball_carrier(total_tracking_df)

    # Convert datatype
    total_tracking_df['game_id'] = total_tracking_df['game_id'].astype(str) 
    total_tracking_df['ball_owning_team_id'] = total_tracking_df['ball_owning_team_id'].astype(str) 
    total_tracking_df['ori_ball_owning_team_id'] = total_tracking_df['ori_ball_owning_team_id'].astype(str) 
    
    return total_tracking_df

if __name__ == "__main__":
    root_path = os.path.abspath("..")
    data_path = "/data/MHL/bepro/raw"

    match_id_lst = os.listdir(data_path)
    total_dict = {match_id : {} for match_id in match_id_lst}
    for match_id in match_id_lst:
        print(f"Preprocessing Match ID {match_id}: Converting data into kloppy format...")
        match_dict = {}
        # if not os.path.exists(os.path.join(os.path.dirname(data_path), "processed", f"{match_id}_processed_dict.pkl")):
        match_path = f"{data_path}/{match_id}"
        # Get Meta Data
        meta_data_path = f"{match_path}/{match_id}_metadata.json"
        meta_data = load_single_json(meta_data_path)

        # Get Team Info
        teams_dict = create_team_dataframe(meta_data['home_team'], meta_data['away_team']) # Return: teams_dict['Home'], teams_dict['Away']

        # Get Event Data
        event_df = create_event_dataframe(match_path)

        # Get Tracking Data
        tracking_df = create_tracking_dataframe(match_path, meta_data, teams_dict)
        total_dict[match_id]['tracking_df'] = tracking_df
        total_dict[match_id]['event_df'] = event_df
        total_dict[match_id]['teams'] = teams_dict
        total_dict[match_id]['meta_data'] = meta_data
        save_dir = os.path.join(os.path.dirname(data_path), "processed", match_id, f"{match_id}_processed_dict.pkl")
        with open(save_dir, "wb") as f:
            pickle.dump(total_dict[match_id], f)
        print(f"Preprocessing Match ID {match_id} Done. Saved location: {save_dir}")
        # else:
        #     with open(os.path.join(os.path.dirname(data_path), "processed", f"{match_id}_processed_dict.pkl"), "rb") as f:
        #         total_dict[match_id] = pickle.load(f)

            
                  

        


        

