import json
import numpy as np
import os
import pandas as pd
import pickle
from scipy.signal import savgol_filter
from tqdm import tqdm

import config as C
from config import Constant, Column, Group

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


def rescale_pitch(tracking_df, meta_data):\
    
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
    

def infer_ball_carrier(tracking_df, source='bepro'):
    # --- Helper Functions with corrected indentation and English comments ---
    def _determine_ball_owning_team_for_frame(frame_df, threshold, col_team_id, col_botid):
        # Check for existing BOTID in the frame
        current_botids = frame_df[col_botid].dropna()
        if not current_botids.empty and current_botids.iloc[0] != 'neutral':
            return current_botids.iloc[0]  # Use the first valid value
        
        # If existing BOTID is 'neutral' or absent, infer
        # Check if 'ball_dist' column has any valid (non-NaN) values
        if 'ball_dist' in frame_df.columns and frame_df['ball_dist'].notna().any():
            # Exclude NaN values and find the minimum value and its index
            valid_ball_dist_series = frame_df['ball_dist'].dropna()
            # Since .notna().any() was checked above, valid_ball_dist_series should not be empty
            # (if all values were NaN, .any() would be False)
            # However, it might be safer to double-check for emptiness after dropna()
            if not valid_ball_dist_series.empty:
                min_dist_idx = valid_ball_dist_series.idxmin()  # Index from the original frame_df
                min_dist = valid_ball_dist_series.min()

                if min_dist < threshold:
                    return frame_df.loc[min_dist_idx, col_team_id]
        return pd.NA  # or np.nan

    def _determine_ball_owning_player_for_frame(frame_df, threshold, col_object_id, col_team_id, col_bopid):
        # This function assumes that the frame's BALL_OWNING_TEAM_ID has already been determined
        # and is present in the 'final_botid_for_frame' column.
        frame_botid_value = frame_df['final_botid_for_frame'].iloc[0]  # Value applied uniformly to all rows within the frame.

        # Check for existing BOPID in the frame (among players of the determined owning team).
        if pd.notna(frame_botid_value):
            # If 'neutral', check for existing BOPID based on all players.
            if frame_botid_value == 'neutral':
                players_considered_for_existing_bopid = frame_df.copy()
            else:
                players_considered_for_existing_bopid = frame_df[frame_df[col_team_id] == frame_botid_value]
            
            if not players_considered_for_existing_bopid.empty: # Check if there are any players to consider.
                current_bopids_on_team = players_considered_for_existing_bopid[col_bopid].dropna()
                # Check if the existing BOPID is a valid value (not 'neutral').
                if not current_bopids_on_team.empty and current_bopids_on_team.iloc[0] != 'neutral': 
                    return current_bopids_on_team.iloc[0]
        
        # If no valid existing BOPID, or if the owning team is unknown (NA), try to infer (owning team must be known for inference).
        if pd.isna(frame_botid_value): # If owning team is NA, player inference is not possible.
            return pd.NA

        # Infer player based on the owning team.
        if frame_botid_value == 'neutral': # If 'neutral', calculate ball distance based on all players.
            players_to_search_in = frame_df.copy()
        else:
            players_to_search_in = frame_df[frame_df[col_team_id] == frame_botid_value]
        
        if not players_to_search_in.empty and \
           'ball_dist' in players_to_search_in.columns and \
           players_to_search_in['ball_dist'].notna().any():
            
            valid_ball_dist_series_on_team = players_to_search_in['ball_dist'].dropna()
            if not valid_ball_dist_series_on_team.empty:
                min_dist_on_team_idx = valid_ball_dist_series_on_team.idxmin()
                min_dist_on_team = valid_ball_dist_series_on_team.min()
                if min_dist_on_team < threshold:
                    return players_to_search_in.loc[min_dist_on_team_idx, col_object_id]
        
        return pd.NA

    # --- Main function logic with corrected indentation and English comments ---
    # 0. Initialize BALL_OWNING_PLAYER_ID and BALL_OWNING_TEAM_ID if they don't exist
    obj_id_dtype = tracking_df[Column.OBJECT_ID].dtype if Column.OBJECT_ID in tracking_df.columns else 'object'
    team_id_dtype = tracking_df[Column.TEAM_ID].dtype if Column.TEAM_ID in tracking_df.columns else 'object'

    if Column.BALL_OWNING_PLAYER_ID not in tracking_df.columns:
        tracking_df[Column.BALL_OWNING_PLAYER_ID] = pd.Series(dtype=obj_id_dtype, index=tracking_df.index)
    if Column.BALL_OWNING_TEAM_ID not in tracking_df.columns:
        tracking_df[Column.BALL_OWNING_TEAM_ID] = pd.Series(dtype=team_id_dtype, index=tracking_df.index)

    # 1. Separate ball and player data
    # Since the 'id' column is used, either Column.OBJECT_ID should point to 'id',
    # or 'id' should be used directly instead of Column.OBJECT_ID.
    # Here, 'id' is used as per the provided code.
    ball_df = tracking_df[tracking_df['id'] == 'ball'].copy() # Added .copy()
    players_df = tracking_df[tracking_df['id'] != 'ball'].copy() # Added .copy()

    # Defensive code for cases where data is missing
    if ball_df.empty or players_df.empty:
        tracking_df[Column.IS_BALL_CARRIER] = False
        # If BALL_OWNING_TEAM_ID doesn't exist, dropna might remove all rows, so check if the column exists.
        if Column.BALL_OWNING_TEAM_ID in tracking_df.columns:
            tracking_df = tracking_df.dropna(subset=[Column.BALL_OWNING_TEAM_ID])
        return tracking_df.reset_index(drop=True)


    # 2. Prepare ball positions: one row per frame, with ball's x, y, z
    # For bepro, just using x, y
    ball_pos_per_frame = ball_df.groupby(Group.BY_FRAME, as_index=False).first()  # Assumes one ball entry per frame
    ball_pos_per_frame = ball_pos_per_frame[Group.BY_FRAME + [Column.X, Column.Y]].rename(
        columns={Column.X: "ball_x", Column.Y: "ball_y"}
    )

    # Merge ball positions to player data
    players_df_with_ball_pos = pd.merge(players_df, ball_pos_per_frame, on=Group.BY_FRAME, how="left")

    # 3. Calculate distance to ball for each player
    # Ensure coordinates are numeric and handle potential NaNs (especially for Z)
    if source == 'bepro':
        coord_cols_to_numeric = ["ball_x", "ball_y", Column.X, Column.Y]

        for col in coord_cols_to_numeric:
            if col in players_df_with_ball_pos.columns: # Check if column exists
                players_df_with_ball_pos[col] = pd.to_numeric(players_df_with_ball_pos[col], errors='coerce')
            else: # To handle cases where columns might not be created due to merge, etc.
                players_df_with_ball_pos[col] = pd.NA 

        # If ball_x, ball_y are NA (ball position for the frame was not available in ball_pos_per_frame), distance calculation is not possible.
        # In this case, dist_sq will be NA, and np.sqrt(NA) will also be NA.
        dist_sq = (
            (players_df_with_ball_pos[Column.X] - players_df_with_ball_pos["ball_x"]) ** 2 +
            (players_df_with_ball_pos[Column.Y] - players_df_with_ball_pos["ball_y"]) ** 2
        )
    elif source == 'sportec':  # Using Z
        coord_cols_to_numeric = ["ball_x", "ball_y", Column.X, Column.Y]
        if Column.Z in players_df_with_ball_pos.columns: coord_cols_to_numeric.append(Column.Z)
        if "ball_z" in players_df_with_ball_pos.columns: coord_cols_to_numeric.append("ball_z")

        for col in coord_cols_to_numeric:
            if col in players_df_with_ball_pos.columns:
                players_df_with_ball_pos[col] = pd.to_numeric(players_df_with_ball_pos[col], errors='coerce')
            else:
                players_df_with_ball_pos[col] = pd.NA


        # Fill Z with 0.0 if missing or NaN (common for 2D data or if ball Z isn't always present)
        for col_z in [Column.Z, "ball_z"]:
            if col_z not in players_df_with_ball_pos.columns: # If Z column itself doesn't exist, create it with 0
                players_df_with_ball_pos[col_z] = 0.0
            players_df_with_ball_pos[col_z] = players_df_with_ball_pos[col_z].fillna(0.0)

        dist_sq = (
            (players_df_with_ball_pos[Column.X] - players_df_with_ball_pos["ball_x"]) ** 2 +
            (players_df_with_ball_pos[Column.Y] - players_df_with_ball_pos["ball_y"]) ** 2 +
            (players_df_with_ball_pos[Column.Z] - players_df_with_ball_pos["ball_z"]) ** 2
        )
    else: # Unknown source
        raise ValueError(f"Unknown source: {source}. Must be 'bepro' or 'sportec'.")

    players_df_with_ball_pos["ball_dist"] = np.sqrt(dist_sq)

    # 4. Determine BALL_OWNING_TEAM_ID per frame (Use original column)
    # BALL_CARRIER_THRESHOLD must be defined within the scope of this function.
    # If it's a class member, it would be self.BALL_CARRIER_THRESHOLD or self._ball_carrier_threshold.
    # Here, it's assumed to be a local or global variable.
    botid_series = players_df_with_ball_pos.groupby(Group.BY_FRAME, group_keys=True).apply(
        _determine_ball_owning_team_for_frame,
        threshold=C.BALL_CARRIER_THRESHOLD, 
        col_team_id=Column.TEAM_ID,
        col_botid=Column.BALL_OWNING_TEAM_ID # Use original BOTID column
    )

    # Ensure botid_series has a name for merging if it's not empty
    if not botid_series.empty:
        botid_series = botid_series.rename("final_botid_for_frame")
        # Merge determined BOTID back to player data for next step
        players_df_with_ball_pos = pd.merge(
            players_df_with_ball_pos,
            botid_series,
            on=Group.BY_FRAME,
            how="left"
        )
    else: # No player data or groups, create column with NaNs
        players_df_with_ball_pos["final_botid_for_frame"] = pd.NA
        
    # 5. Determine BALL_OWNING_PLAYER_ID per frame
    bopid_series = players_df_with_ball_pos.groupby(Group.BY_FRAME, group_keys=True).apply(
        _determine_ball_owning_player_for_frame,
        threshold=C.BALL_CARRIER_THRESHOLD,
        col_object_id=Column.OBJECT_ID,
        col_team_id=Column.TEAM_ID,
        col_bopid=Column.BALL_OWNING_PLAYER_ID # Use original BOPID column
    )

    if not bopid_series.empty:
        bopid_series = bopid_series.rename("final_bopid_for_frame")
    
    # 6. Consolidate frame-level inferences (BOTID, BOPID)
    frame_summary_components = []
    # Add to frame_summary_components only if botid_series and bopid_series are not None and are non-empty Series.
    if isinstance(botid_series, pd.Series) and not botid_series.empty:
        frame_summary_components.append(botid_series)
    if isinstance(bopid_series, pd.Series) and not bopid_series.empty:
        frame_summary_components.append(bopid_series)

    if not frame_summary_components: # If players_df was empty or grouping resulted in no groups
        # Create an empty frame_summary including Group.BY_FRAME columns
        frame_summary = pd.DataFrame(columns=Group.BY_FRAME + [Column.BALL_OWNING_TEAM_ID, Column.BALL_OWNING_PLAYER_ID])
    else:
        frame_summary = pd.concat(frame_summary_components, axis=1).reset_index()
        rename_map = {}
        if "final_botid_for_frame" in frame_summary.columns:
            rename_map["final_botid_for_frame"] = Column.BALL_OWNING_TEAM_ID
        if "final_bopid_for_frame" in frame_summary.columns:
            rename_map["final_bopid_for_frame"] = Column.BALL_OWNING_PLAYER_ID
        if rename_map:
            frame_summary = frame_summary.rename(columns=rename_map)

    # 7. Merge frame summary back to the original full DataFrame (tracking_df)
    # Original tracking_df's BALL_OWNING_PLAYER_ID is dropped, and BALL_OWNING_TEAM_ID is backed up.
    output_df = tracking_df.drop(columns=[Column.BALL_OWNING_PLAYER_ID], errors='ignore')
    # Check if the column exists before prefixing with 'ori_'
    if Column.BALL_OWNING_TEAM_ID in output_df.columns:
        output_df = output_df.rename(columns={Column.BALL_OWNING_TEAM_ID: "ori_" + Column.BALL_OWNING_TEAM_ID})
    
    # Check if Group.BY_FRAME key columns exist in frame_summary and merge
    # (If frame_summary is empty but has columns, merge will fill with NA)
    # (Defensive code for when frame_summary is completely empty or key columns are missing)
    all_keys_present_in_summary = all(key in frame_summary.columns for key in Group.BY_FRAME)
    
    if not frame_summary.empty and all_keys_present_in_summary:
        output_df = pd.merge(output_df, frame_summary, on=Group.BY_FRAME, how="left")
    else: # If frame_summary is unsuitable for merge, fill target columns with NA
        if Column.BALL_OWNING_TEAM_ID not in output_df.columns:
            output_df[Column.BALL_OWNING_TEAM_ID] = pd.NA
        if Column.BALL_OWNING_PLAYER_ID not in output_df.columns:
            output_df[Column.BALL_OWNING_PLAYER_ID] = pd.NA

    # 8. Set IS_BALL_CARRIER column
    # True if OBJECT_ID matches BALL_OWNING_PLAYER_ID and BOPID is not NA
    # Exclude 'neutral' state
    output_df[Column.IS_BALL_CARRIER] = \
        (output_df[Column.OBJECT_ID] == output_df[Column.BALL_OWNING_PLAYER_ID]) & \
        (output_df[Column.BALL_OWNING_PLAYER_ID].notna()) & \
        (output_df[Column.BALL_OWNING_PLAYER_ID] != 'neutral') 


    # 9. Drop rows where the (newly determined) BALL_OWNING_TEAM_ID is NA
    # If you want to keep 'neutral' values, fill with 'neutral' before dropna,
    # and then either don't dropna or modify the condition.
    # Currently, only NA is removed. If 'neutral' also needs to be removed, an additional condition is needed.
    if Column.BALL_OWNING_TEAM_ID in output_df.columns:
        output_df = output_df.dropna(subset=[Column.BALL_OWNING_TEAM_ID])
    
    # Part that finally deletes the BALL_OWNING_PLAYER_ID column (was in previous code).
    # If you want to keep this column, remove/comment out this line.
    # Actual column name needs verification. Using Column.BALL_OWNING_PLAYER_ID is recommended.
    # Assuming Column.BALL_OWNING_PLAYER_ID is the string "ball_owning_player_id" for this specific line from original code
    if 'ball_owning_player_id' in output_df.columns: 
         output_df = output_df.drop(columns=['ball_owning_player_id']) 

    return output_df.reset_index(drop=True) # Reset index


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
    
    # Calculate kinematics
    agent_ids_present = tracking_df['id'].unique()
    total_tracking_df = pd.DataFrame()
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
        total_tracking_df = pd.concat([total_tracking_df, kinematics_df], axis=0, ignore_index=True)
    
    # Sort final DataFrame
    total_tracking_df = total_tracking_df.sort_values(
        by=["period_id", "timestamp", "id"], kind="mergesort"
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
        match_dict = {}
        if not os.path.exists(os.path.join(os.path.dirname(data_path), "processed", f"{match_id}_processed_dict.pkl")):
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
            with open(os.path.join(os.path.dirname(data_path), "processed", f"{match_id}_processed_dict.pkl"), "wb") as f:
                pickle.dump(total_dict[match_id], f)
        
        else:
            with open(os.path.join(os.path.dirname(data_path), "processed", f"{match_id}_processed_dict.pkl"), "rb") as f:
                total_dict[match_id] = pickle.load(f)

            
                  

        


        

