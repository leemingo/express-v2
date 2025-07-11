import json
import numpy as np
import os
import pandas as pd
import pickle
import argparse
from scipy.signal import savgol_filter
from tqdm import tqdm
from typing import Dict, Tuple, Union, Optional, List

import config as C
from config import Constant, Column, Group

from utils_data import infer_ball_carrier

def load_single_json(file_path: str) -> Optional[Union[Dict, List]]:
    """Loads and parses a single JSON file from the specified path.
    
    This function provides a robust way to load JSON files with comprehensive error
    handling. It supports both dictionary and list JSON structures and returns None
    if the file cannot be loaded or parsed.
    
    Args:
        file_path: The path to the JSON file to load.
        
    Returns:
        The parsed Python object (dict or list) on success, None if the file 
        is not found or contains invalid JSON.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        
    Example:
        >>> data = load_single_json("config.json")
        >>> if data is not None:
        ...     print("Successfully loaded JSON data")
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return None

def load_jsonl(file_path: str) -> List[Dict]:
    """Loads data from a JSON Lines (.jsonl) file.
    
    Each line in the file is expected to be a valid JSON object. Lines that are
    empty or cannot be parsed as JSON will be skipped with a warning. This function
    is particularly useful for processing large datasets stored in JSONL format.
    
    Args:
        file_path: The path to the .jsonl file.
        
    Returns:
        A list containing the Python objects parsed from each valid JSON line.
        Returns an empty list if the file is not found or contains no valid JSON lines.
        
    Example:
        >>> data = load_jsonl("tracking_data.jsonl")
        >>> print(f"Loaded {len(data)} records")
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                processed_line = line.strip()
                if not processed_line:
                    continue
                try:
                    data.append(json.loads(processed_line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {line_number} in '{file_path}' due to JSON decoding error.")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred processing line {line_number} in '{file_path}': {e}. Skipping line.")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}': {e}")
    return data

def create_team_dataframe(home_team_info: Dict, away_team_info: Dict) -> Dict[str, pd.DataFrame]:
    """Creates team dataframes for both home and away teams.
    
    This function processes team information from BePro data format and creates
    structured DataFrames containing player information for both teams. It handles
    player metadata including names, positions, IDs, and team assignments.
    
    Args:
        home_team_info: Dictionary containing home team information including players.
        away_team_info: Dictionary containing away team information including players.
        
    Returns:
        Dictionary with 'Home' and 'Away' keys, each containing a pandas DataFrame
        with player information including player names, positions, IDs, etc.
        
    Example:
        >>> teams_dict = create_team_dataframe(home_info, away_info)
        >>> home_df = teams_dict['Home']
        >>> away_df = teams_dict['Away']
        >>> print(f"Home team has {len(home_df)} players")
    """
    def create_team_rows(team_info: Dict, team_type: str) -> pd.DataFrame:
        """Helper function to create team rows for a single team."""
        base_info = {
            'player': None, 'position': None, 'team': team_type,
            'jID': None, 'pID': None, 'tID': team_info.get('team_id'), 'xID': None
        }
        rows = []
        for idx, player_data in enumerate(team_info['players']):
            player_info = base_info.copy()
            player_info.update({
                'player': player_data['full_name_en'],
                'position': player_data['initial_position_name'],
                'jID': player_data['shirt_number'],
                'pID': str(player_data['player_id']),
                'xID': idx
            })
            rows.append(player_info)
        return pd.DataFrame(rows)
    
    home_df = create_team_rows(home_team_info, 'Home')
    away_df = create_team_rows(away_team_info, 'Away')
    
    # Convert IDs to string type
    for df in [home_df, away_df]:
        df['pID'] = df['pID'].astype(str)
        df['tID'] = df['tID'].astype(str)
    
    return {'Home': home_df, 'Away': away_df}

def create_event_dataframe(match_path: str) -> pd.DataFrame:
    """Creates event dataframe from match data.
    
    Loads first and second half event data files and combines them into a single
    DataFrame. This function handles the BePro event data format and ensures
    proper concatenation of data from both halves of the match.
    
    Args:
        match_path: Path to the match directory containing event data files.
        
    Returns:
        Combined pandas DataFrame containing all event data from both halves.
        
    Raises:
        ValueError: If event data files cannot be loaded.
        
    Example:
        >>> event_df = create_event_dataframe("/path/to/match")
        >>> print(f"Total events: {len(event_df)}")
    """
    first_half_event_path = next((f for f in os.listdir(match_path) if "1_event" in f), None)
    second_half_event_path = next((f for f in os.listdir(match_path) if "2_event" in f), None)

    first_half_event_data = load_single_json(f"{match_path}/{first_half_event_path}")
    second_half_event_data = load_single_json(f"{match_path}/{second_half_event_path}")

    if first_half_event_data is None or second_half_event_data is None:
        raise ValueError("Failed to load event data files")

    first_half_event_df = pd.DataFrame(first_half_event_data['data'])
    second_half_event_df = pd.DataFrame(second_half_event_data['data'])
    return pd.concat([first_half_event_df, second_half_event_df], axis=0)

def _apply_smoothing_and_outlier_removal(period_df: pd.DataFrame, col: str, 
                                       is_outlier: pd.Series, smoothing_params: Dict) -> pd.DataFrame:
    """Helper function to apply smoothing and outlier removal to a column.
    
    This function implements a comprehensive smoothing and outlier removal pipeline
    using Savitzky-Golay filtering. It first masks outliers, interpolates missing
    values, and then applies smoothing with appropriate parameter validation.
    
    Args:
        period_df: DataFrame containing the period data.
        col: Column name to apply smoothing to.
        is_outlier: Boolean Series indicating outlier values.
        smoothing_params: Dictionary containing smoothing parameters including
                         'window_length' and 'polyorder'.
        
    Returns:
        DataFrame with smoothed column values.
        
    Example:
        >>> smoothed_df = _apply_smoothing_and_outlier_removal(
        ...     period_df, 'vx', is_outlier, {'window_length': 11, 'polyorder': 3}
        ... )
    """
    period_df[col] = period_df[col].mask(is_outlier)
    period_df[col] = period_df[col].interpolate(limit_direction='both')
    
    data_to_smooth = period_df[col].fillna(0)
    window_length = min(smoothing_params['window_length'], len(data_to_smooth))
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length >= smoothing_params['polyorder'] + 1 and window_length > 0:
        period_df[col] = savgol_filter(data_to_smooth, window_length=window_length, polyorder=smoothing_params['polyorder'])
    else:
        period_df[col] = data_to_smooth
    
    return period_df

def _calculate_kinematics(df: pd.DataFrame, smoothing_params: Dict, max_speed: float, 
                         max_acceleration: float, is_ball: bool = False) -> pd.DataFrame:
    """Calculates velocity and acceleration for a single agent over periods.
    
    This function implements a comprehensive kinematics calculation pipeline for
    tracking data. It processes data period by period, calculating velocities and
    accelerations while applying outlier detection and smoothing techniques.
    
    Args:
        df: DataFrame containing tracking data with columns ['x', 'y', 'z', 'timestamp', 'period_id'].
        smoothing_params: Dictionary containing smoothing parameters for Savitzky-Golay filter.
        max_speed: Maximum allowed speed value for outlier detection.
        max_acceleration: Maximum allowed acceleration value for outlier detection.
        is_ball: Boolean indicating if the agent is a ball (affects z-coordinate handling).
        
    Returns:
        DataFrame with calculated kinematics including velocity (vx, vy, vz, v) and 
        acceleration (ax, ay, az, a) columns.
        
    Example:
        >>> kinematics_df = _calculate_kinematics(
        ...     tracking_df, smoothing_params, max_speed=12.0, max_acceleration=15.0, is_ball=False
        ... )
    """
    df_out = pd.DataFrame()
    required_cols = ['x', 'y', 'z', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns in input dataframe. Found: {df.columns.tolist()}")
        return df_out

    for period_id in df['period_id'].unique():
        period_df = df[df['period_id'] == period_id].copy()
        period_df = period_df.sort_values(by='timestamp').reset_index(drop=True)

        # Interpolate coordinates
        period_df['x'] = period_df['x'].interpolate()
        period_df['y'] = period_df['y'].interpolate()

        # Calculate time difference
        dt = period_df['timestamp'].diff().dt.total_seconds()

        # Calculate velocities
        vel_cols = ['vx', 'vy', 'vz'] if is_ball else ['vx', 'vy']
        coord_cols = ['x', 'y', 'z'] if is_ball else ['x', 'y']
        
        for vel_col, coord_col in zip(vel_cols, coord_cols):
            period_df[vel_col] = period_df[coord_col].diff() / dt
            if not is_ball and vel_col == 'vz':
                period_df[vel_col] = 0.0

        # Calculate speed and apply outlier removal
        period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
        is_speed_outlier = period_df['v'] > max_speed
        
        for col in vel_cols:
            period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_speed_outlier, smoothing_params)
        
        # Recalculate speed after smoothing
        period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
        
        # Calculate accelerations
        accel_cols = ['ax', 'ay', 'az'] if is_ball else ['ax', 'ay']
        for accel_col, vel_col in zip(accel_cols, vel_cols):
            period_df[accel_col] = period_df[vel_col].diff() / dt
            if not is_ball and accel_col == 'az':
                period_df[accel_col] = 0.0

        # Calculate acceleration magnitude and apply outlier removal
        period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
        is_accel_outlier = period_df['a'] > max_acceleration
        
        for col in accel_cols:
            period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_accel_outlier, smoothing_params)
        
        # Recalculate acceleration after smoothing
        period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
        
        # Limit speed and acceleration
        period_df['v'] = np.minimum(period_df['v'], max_speed)
        period_df['a'] = np.minimum(period_df['a'], max_acceleration)

        df_out = pd.concat([df_out, period_df], ignore_index=True)

    return df_out

def resample_tracking_dataframe(tracking_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Resamples tracking data to target frequency.
    
    This function resamples tracking data from its original frequency to a target
    frequency using interpolation techniques. It handles both forward and backward
    filling for different types of data columns and ensures proper time alignment.
    
    Args:
        tracking_df: DataFrame containing tracking data with timestamp index.
        target_hz: Target frequency in Hz for resampling.
        
    Returns:
        Resampled DataFrame with data at the target frequency.
        
    Example:
        >>> resampled_df = resample_tracking_dataframe(tracking_df, target_hz=25)
    """
    resample_freq_ms = int(1000 / target_hz)
    resample_freq_str = f'{resample_freq_ms}ms'
    
    period_list = []
    for period_id in tracking_df['period_id'].unique():
        period_df = tracking_df[tracking_df['period_id'] == period_id]

        min_timestamp = pd.Timedelta(0)
        max_timestamp = period_df['timestamp'].max()
        global_original_index = pd.to_timedelta(sorted(period_df['timestamp'].unique()))
        global_target_index = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=resample_freq_str)

        grouped = period_df.groupby('id')
        resampled_list = []

        for agent_id, agent_group in grouped:
            group_df = agent_group.copy().set_index('timestamp')
            if group_df.index.has_duplicates:
                group_df = group_df.loc[~group_df.index.duplicated(keep='first')]

            union_index = global_original_index.union(global_target_index)
            reindexed_group = group_df.reindex(union_index)

            # Interpolation
            interpolation_cols = ['x', 'y', 'speed']
            reindexed_group[interpolation_cols] = reindexed_group[interpolation_cols].interpolate(method='pchip', limit_area='inside')
            
            # Forward fill other columns
            ffill_cols = [col for col in group_df.columns if col not in interpolation_cols and col != 'id']
            reindexed_group[ffill_cols] = reindexed_group[ffill_cols].ffill()
            final_group = reindexed_group.reindex(global_target_index)

            # Fill categorical data
            final_group['id'] = agent_id
            final_group = final_group.dropna(subset=['x', 'y'])
            resampled_list.append(final_group)

        period_list += resampled_list

    total_resampled_df = pd.concat(period_list).reset_index().rename(columns={'index': 'timestamp'})
    total_resampled_df['frame_id'] = (total_resampled_df['timestamp'].astype(np.int64) // (10**9 / target_hz)).astype(int)
    total_resampled_df = total_resampled_df.sort_values(['game_id', 'timestamp', 'period_id', 'frame_id', 'id'])

    return total_resampled_df

def rescale_pitch(tracking_df: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
    """Rescales pitch coordinates to standard dimensions.
    
    This function transforms pitch coordinates from the original coordinate system
    to a standardized pitch coordinate system. It handles both x and y coordinates
    and applies appropriate scaling factors based on the pitch metadata.
    
    Args:
        tracking_df: DataFrame containing tracking data with x, y coordinates.
        meta_data: Dictionary containing pitch metadata including ground_width and ground_height.
        
    Returns:
        DataFrame with rescaled x, y coordinates to standard pitch dimensions.
        
    Example:
        >>> rescaled_df = rescale_pitch(tracking_df, meta_data)
    """
    x_ori_min, x_ori_max = 0.0, meta_data['ground_width']
    y_ori_min, y_ori_max = 0.0, meta_data['ground_height']

    x_new_min, x_new_max = C.PITCH_X_MIN, C.PITCH_X_MAX
    y_new_min, y_new_max = C.PITCH_Y_MIN, C.PITCH_Y_MAX

    scale_x = (x_new_max - x_new_min) / (x_ori_max - x_ori_min)
    scale_y = (y_new_max - y_new_min) / (y_ori_max - y_ori_min)

    tracking_df['x'] = x_new_min + (tracking_df['x'] - x_ori_min) * scale_x
    tracking_df['y'] = y_new_min + (tracking_df['y'] - y_ori_min) * scale_y
    return tracking_df

def create_tracking_dataframe(match_path: str, meta_data: Dict, teams_dict: Dict) -> pd.DataFrame:
    """Creates tracking dataframe from match data.
    
    This function processes raw tracking data, applies coordinate transformations,
    resamples to target frequency, and calculates kinematics for all agents. It
    handles the complete pipeline from raw BePro data to processed tracking data
    with kinematics calculations.
    
    Args:
        match_path: Path to the match directory containing tracking data files.
        meta_data: Dictionary containing match metadata including pitch dimensions.
        teams_dict: Dictionary containing team information for player lookup.
        
    Returns:
        DataFrame containing processed tracking data with kinematics calculations
        for all players and the ball.
        
    Example:
        >>> tracking_df = create_tracking_dataframe(match_path, meta_data, teams_dict)
        >>> print(f"Processed {len(tracking_df)} tracking records")
    """
    teams_df = pd.concat([teams_dict['Home'], teams_dict['Away']], axis=0)
    player_lookup = teams_df.set_index('pID')
    home_tid = teams_dict['Home']['tID'].iloc[0]
    away_tid = teams_dict['Away']['tID'].iloc[0]
    
    # Configuration
    player_smoothing_params = C.DEFAULT_PLAYER_SMOOTHING_PARAMS
    ball_smoothing_params = C.DEFAULT_BALL_SMOOTHING_PARAMS
    max_player_speed = C.MAX_PLAYER_SPEED
    max_player_acceleration = C.MAX_PLAYER_ACCELERATION
    max_ball_speed = C.MAX_BALL_SPEED
    max_ball_acceleration = C.MAX_BALL_ACCELERATION

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
                ball_owning_team_id = home_tid if ball_state == 'home' else (away_tid if ball_state == 'away' else ball_state)

            # Extract frame information
            frame_info = {
                'game_id': match_id,
                'period_id': frame_data.get('period_order') + 1,
                'timestamp': frame_data.get('match_time'),
                'frame_id': frame_data.get('frame_index'),
                'ball_state': new_ball_state,
                'ball_owning_team_id': ball_owning_team_id,
            }

            for object_type in ['players', 'balls']:
                object_list = frame_data.get(object_type, [])
                if object_list:
                    for object_data in object_list:
                        row_data = frame_info.copy()
                        row_data.update(object_data)
                        
                        if object_type == 'balls':
                            row_data.update({
                                'id': 'ball',
                                'team_id': 'ball',
                                'position_name': 'ball'
                            })
                        else:
                            player_pID = str(object_data.get('player_id'))
                            row_data['id'] = player_pID
                            if player_pID in player_lookup.index:
                                row_data['team_id'] = player_lookup.loc[player_pID, 'tID']
                                row_data['position_name'] = player_lookup.loc[player_pID, 'position']
                            else:
                                row_data['team_id'] = None
                                row_data['position_name'] = None
                        
                        # Remove unnecessary columns
                        row_data.pop('object', None)
                        row_data.pop('player_id', None)
                        all_object_rows.append(row_data)

    tracking_df = pd.DataFrame(all_object_rows)
    tracking_df['timestamp'] = pd.to_timedelta(tracking_df['timestamp'], unit='ms')
    # Rescale pitch coordinates
    tracking_df = rescale_pitch(tracking_df, meta_data)
    tracking_df = resample_tracking_dataframe(tracking_df, target_hz=25)
    
    # Calculate kinematics for each agent
    total_tracking_list = []
    for agent_id in tracking_df['id'].unique():
        is_ball = (agent_id == 'ball')
        current_agent_df = tracking_df[tracking_df['id'] == agent_id].copy()
        current_agent_df['z'] = 0.0  # bepro data doesn't have z information

        # Drop rows with NaN coordinates for players
        if not is_ball:
            current_agent_df = current_agent_df.dropna(subset=['x', 'y']).copy()

        # Calculate kinematics
        smoothing = ball_smoothing_params if is_ball else player_smoothing_params
        max_v = max_ball_speed if is_ball else max_player_speed
        max_a = max_ball_acceleration if is_ball else max_player_acceleration

        kinematics_df = _calculate_kinematics(current_agent_df, smoothing, max_v, max_a, is_ball)
        total_tracking_list.append(kinematics_df)
    
    total_tracking_df = pd.concat(total_tracking_list, axis=0, ignore_index=True)
    
    # Sort and format final DataFrame
    total_tracking_df = total_tracking_df.sort_values(
        by=["period_id", "timestamp", "frame_id", "id"], kind="mergesort"
    ).reset_index(drop=True)

    # Define final column order
    final_cols_order = [
        'game_id', 'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'v', 'ax', 'ay', 'az', 'a',
        'id', 'team_id', 'position_name'
    ]
    total_tracking_df = total_tracking_df[[col for col in final_cols_order if col in total_tracking_df.columns]]
    total_tracking_df = infer_ball_carrier(total_tracking_df)

    # Convert datatypes
    total_tracking_df['game_id'] = total_tracking_df['game_id'].astype(str)
    total_tracking_df['ball_owning_team_id'] = total_tracking_df['ball_owning_team_id'].astype(str)
    total_tracking_df['ori_ball_owning_team_id'] = total_tracking_df['ori_ball_owning_team_id'].astype(str)
    
    return total_tracking_df

def load_all_data(data_path: str) -> None:
    """Loads and processes all BePro data in the specified directory.
    
    This function processes all match directories in the given data path, creating
    tracking dataframes, event dataframes, and team information for each match.
    The processed data is saved as pickle files in a 'processed' subdirectory.
    
    Args:
        data_path: Path to the directory containing match data folders.
        
    Example:
        >>> load_all_data("/path/to/bepro/data")
    """
    match_id_lst = os.listdir(data_path)
    total_dict = {match_id: {} for match_id in match_id_lst}
    processed_path = os.path.join(os.path.dirname(data_path), "processed")
    
    for match_id in match_id_lst:
        print(f"Preprocessing Match ID {match_id}: Converting data into kloppy format...")
        match_path = f"{data_path}/{match_id}"
        
        # Get Meta Data
        meta_data_path = f"{match_path}/{match_id}_metadata.json"
        meta_data = load_single_json(meta_data_path)
        
        if meta_data is None:
            print(f"Error: Could not load metadata for match {match_id}. Skipping.")
            continue

        # Get Team Info
        teams_dict = create_team_dataframe(meta_data['home_team'], meta_data['away_team'])

        # Get Event Data
        try:
            event_df = create_event_dataframe(match_path)
        except Exception as e:
            print(f"Error: Could not load event data for match {match_id}: {e}")
            event_df = None

        # Get Tracking Data
        try:
            tracking_df = create_tracking_dataframe(match_path, meta_data, teams_dict)
        except Exception as e:
            print(f"Error: Could not load tracking data for match {match_id}: {e}")
            tracking_df = None
        
        # Save processed data
        total_dict[match_id] = {
            'tracking_df': tracking_df,
            'event_df': event_df,
            'teams': teams_dict,
            'meta_data': meta_data
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(f"{processed_path}/{match_id}/{match_id}_processed_dict.pkl"), exist_ok=True)
        with open(f"{processed_path}/{match_id}/{match_id}_processed_dict.pkl", "wb") as f:
            pickle.dump(total_dict[match_id], f)
        print(f"Preprocessing Match ID {match_id} Done. Saved location: {processed_path}/{match_id}/{match_id}_processed_dict.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BePro raw tracking data.")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to raw BePro data directory")
    
    args = parser.parse_args()
    data_path = args.data_path
    load_all_data(data_path)
    print('Done')