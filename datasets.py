import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from config import *
# --- Constants and Configuration ---
NUM_AGENTS = 23
NUM_TEAM_PLAYERS = 11
H, W = 68, 104  # Grid dimensions for SoccerMap (Height, Width)
NUM_FEATURE_CHANNELS = 17 #13 # Number of input channels for models.


class PressingSequenceDataset(Dataset):
    """Dataset for pressing sequences in football matches.
    
    This dataset loads tracking data, event data, and pressing intensity data to create
    sequences for analyzing pressing situations in football matches.
    
    Attributes:
        data_path (str): Path to the directory containing match data or preprocessed files.
        match_id_lst (list): List of match IDs to load.
        sequence_length (int): Length of the input sequence in frames.
        feature_cols (list): List of feature columns to use for model input.
        cols_to_flip (list): List of coordinate columns to flip for direction normalization.
        features_seqs (list): List of feature tensors for each sample.
        pressintensity_seqs (list): List of pressing intensity tensors for each sample.
        labels (list): List of binary labels for each sample.
        pressed_ids (list): List of pressed player IDs for each sample.
        presser_ids (list): List of pressing player IDs for each sample.
        agent_orders (list): List of agent orderings for each sample.
        match_infos (list): List of match information strings for each sample.
        data (list): List of dictionaries containing all sample data.
    """
    
    def __init__(self, data_path, match_id_lst=None, num_frames_to_sample=10, feature_cols=None, highpress_only=False, press_threshold=0.9):
        """Initialize the PressingSequenceDataset.
        
        Args:
            data_path (str): Path to the directory containing match data or preprocessed files.
            match_id_lst (list, optional): List of match IDs to load. If None, loads all matches
                in the data_path directory. Defaults to None.
            sequence_length (int, optional): Length of the input sequence in frames. 
                Defaults to 150.
            feature_cols (list, optional): List of feature columns to use. If None, infers 
                default kinematic features. Defaults to None.
            highpress_only (bool, optional): Whether to only consider highpress situations.
                Defaults to False.
            num_frames_to_sample (int, optional): Number of frames to sample for each sequence.
                Defaults to 10.
        """
        self.data_path = data_path
        self.match_id_lst = match_id_lst if match_id_lst is not None else os.listdir(self.data_path)
        self.num_frames_to_sample = num_frames_to_sample
        self.feature_cols = feature_cols if feature_cols else self._infer_feature_cols()
        self.cols_to_flip = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
        self.highpress_only = highpress_only
        self.press_threshold = press_threshold
        self._load_data()
    
    def _infer_feature_cols(self):
        """Infer the default feature columns for model input.
        
        Returns:
            list: List of feature column names to use for model training.
                Includes kinematic features (position, velocity, acceleration),
                binary features (teammate, goalkeeper), distance and angle features,
                and event type features.
        """
        return [
            'x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a',  # Kinematic features
            'is_teammate', 'is_goalkeeper',  # Binary features
            'distance_to_goal', 'sin_angle_to_goal', 'cos_angle_to_goal',  # Goal features
            'distance_to_ball', 'sin_angle_to_ball', 'cos_angle_to_ball',  # Ball features
            'cos_velocity_angle', 'sin_velocity_angle',  # Velocity angle features
            'type_id'  # Event type feature
        ]
    
    
    def _normalize_coordinate_direction(self, df, home_team_id):
        """Normalize coordinate system to ensure consistent attack direction.
        
        This function normalizes the coordinate system within the DataFrame to ensure
        the home team always attacks in a consistent direction (left-to-right).
        It handles both period-based flipping and initial orientation correction.
        
        Args:
            df (pd.DataFrame): The tracking DataFrame to process. Must contain
                'period_id', 'frame_id', 'x', 'y', 'team_id' columns. Will also
                flip 'vx', 'vy', 'ax', 'ay' if they exist.
            home_team_id (str): The team ID of the home team.
            
        Returns:
            pd.DataFrame: A DataFrame with normalized coordinate directions.
                Does not modify the input df directly, works on a copy.
                
        Raises:
            ValueError: If minimum x value cannot be determined for orientation check.
        """
        df_normalized = df.copy()
        
        # Step 1: Unify direction for the second half
        second_half_mask = df_normalized['period_id'] == 2
        
        # Flip coordinates and related vectors for second-half data
        for col in self.cols_to_flip:
            if col in df_normalized.columns and df_normalized[col].dtype != 'object':
                df_normalized.loc[second_half_mask, col] = -df_normalized.loc[second_half_mask, col]
        
        # Step 2: Check home team attack direction and flip all if necessary
        first_period_frames = df_normalized[df_normalized['period_id'] == 1]['frame_id']
        if not first_period_frames.empty:
            first_frame_idx = first_period_frames.unique()[0]
            first_frame_df = df_normalized[
                (df_normalized['period_id'] == 1) & 
                (df_normalized['frame_id'] == first_frame_idx) & 
                (df_normalized['team_id'] != 'ball')
            ].copy()

            if not first_frame_df.empty and not first_frame_df['x'].isna().all():
                try:
                    min_x_team_id = first_frame_df.loc[first_frame_df['x'].idxmin(), 'team_id']
                    
                    # If leftmost entity is not home team, flip all coordinates
                    if min_x_team_id != home_team_id:
                        for col in self.cols_to_flip:
                            if col in df_normalized.columns and df_normalized[col].dtype != 'object':
                                df_normalized.loc[:, col] = -df_normalized.loc[:, col]
                except ValueError:
                    print(f"Warning: Could not determine minimum 'x' value for frame {first_frame_idx}. Skipping orientation check.")
            else:
                print(f"Warning: No valid data found for the first frame ({first_frame_idx}) of period 1. Skipping main orientation check.")
        else:
            print("Warning: No data found for period 1. Skipping main orientation check.")
            
        return df_normalized

    def _preprocess_event_df(self, event_df, teams_df):
        """Preprocess event dataframe for analysis.
        
        This function normalizes time data, converts player IDs to consistent format,
        creates player codes, and merges team information with event data.
        
        Args:
            event_df (pd.DataFrame): Raw event dataframe containing match events.
            teams_df (pd.DataFrame): Team information dataframe containing player details.
            
        Returns:
            pd.DataFrame: Preprocessed event dataframe with normalized time data,
                consistent player IDs, and merged team information.
        """
        # Normalize time data to 0.04 second intervals
        event_df['time_seconds'] = (event_df['time_seconds'] / 0.04).round() * 0.04
        event_df['relative_time_seconds'] = (event_df['relative_time_seconds'] / 0.04).round() * 0.04
        
        # Convert to timedelta objects
        event_df['time_seconds'] = pd.to_timedelta(event_df['time_seconds'], unit='s')
        event_df['relative_time_seconds'] = pd.to_timedelta(event_df['relative_time_seconds'], unit='s')
        
        # Convert player_id to consistent string format
        event_df['player_id'] = event_df['player_id'].astype(int).astype(str)
        
        # Reset team dataframe index and create player codes
        teams_df.reset_index(drop=True, inplace=True)
        teams_df['player_code'] = teams_df.apply(
            lambda row: row['team'][0] + str(row['xID']).zfill(2), axis=1
        )

        # Merge event data with team information
        event_df = event_df.merge(
            teams_df,
            how='left',
            left_on='player_id',
            right_on='pID'
        )

        return event_df
    
    def _merge_tracking_pressing_df(self, tracking_df, pressing_df, teams_df):
        """Merge tracking data with pressing intensity data and team information.
        
        This function combines tracking data, pressing intensity data, and team information
        into a single dataframe. It filters for alive ball states, normalizes timestamps,
        and handles ball identification.
        
        Args:
            tracking_df (pd.DataFrame): Tracking data containing player and ball positions.
            pressing_df (pd.DataFrame): Pressing intensity data for each frame.
            teams_df (pd.DataFrame): Team information containing player codes.
            
        Returns:
            pd.DataFrame: Merged dataframe with tracking, pressing, and team data.
            
        Raises:
            ValueError: If unexpected IDs are found where player_code is NaN.
        """
        # Merge tracking and pressing data
        total_df = pd.merge(
            tracking_df, pressing_df, 
            on=['game_id', 'period_id', 'timestamp', 'frame_id'], 
            how='left'
        )
        
        # Filter for alive ball states only
        total_df = total_df[total_df['ball_state'] == 'alive']
        
        # Normalize second half timestamps
        total_df.loc[total_df['period_id'] == 2, 'timestamp'] -= pd.Timedelta(minutes=45)
        
        # Merge with team information
        total_df = total_df.merge(
            teams_df[['pID', 'player_code']],
            how='left',
            left_on='id',
            right_on='pID'
        )
        total_df.drop(['pID'], axis=1, inplace=True)

        # Handle ball identification
        nan_mask = pd.isna(total_df['player_code'])
        nan_ids = set(total_df.loc[nan_mask, 'id'].unique())
        expected_ids_ball = {'ball'}
        expected_ids_empty = set()

        if not (nan_ids == expected_ids_ball or nan_ids == expected_ids_empty):
            raise ValueError(
                f"Found unexpected IDs in rows where player_code is NaN. "
                f"Expected: {{'ball'}} or {{}}, but found: {nan_ids}"
            )
        
        # Set ball ID as player_code for ball rows
        total_df.loc[nan_mask, 'player_code'] = total_df.loc[nan_mask, 'id']
        return total_df

    # Function to determine whether the situation is highpress
    def is_highpress(self, row, total_df, teams_df):
        """
        Function to determine if the given frame is a highpress situation
        
        Args:
            row: A row from first_frames_df
            total_df: DataFrame containing tracking data
            teams_df: DataFrame containing team information
        
        Returns:
            bool: True if highpress situation, False otherwise
        """

        pitch_first_third = PITCH_X_MIN + (PITCH_X_MAX - PITCH_X_MIN) * 1 / 3
        pitch_second_third = PITCH_X_MIN + (PITCH_X_MAX - PITCH_X_MIN) * 2 / 3

        period_id = row['period_id']
        frame_id = row['frame_id']
        team_id = row['team_id']
        player_id = row['pressing_player']
        
        # Get the position information of the pressing player
        player_df = total_df[
            (total_df['period_id'] == period_id) &
            (total_df['frame_id'] == frame_id) &
            (total_df['id'] == player_id)
        ]
        
        if player_df.empty:
            return False
        
        player_x = player_df['x'].iloc[0]
        team_name = teams_df[teams_df['tID'] == team_id]['team'].iloc[0]
        
        # If the home team is pressing: pressing in the opponent's area (after 2/3 point)
        if team_name == 'Home':
            return player_x >= pitch_second_third
        # If the away team is pressing: pressing in the opponent's area (before 1/3 point)
        else:
            return player_x <= pitch_first_third

    def _get_event_name(self, x):
        """Extract event names from event data.
        
        Args:
            x: List of event dictionaries containing event information.
            
        Returns:
            list: List of event names extracted from the event data.
        """
        event_name_lst = []
        for val in x:
            if "event_name" in val:
                event_name_lst.append(val['event_name'])
            elif "name" in val:
                event_name_lst.append(val['name'])
        return event_name_lst

    def _check_pressing_success(self, row, event_df, teams_dict):
        """Check if pressing was successful based on subsequent events.
        
        This function analyzes events that occur within 5 seconds after a pressing
        situation to determine if the pressing team successfully gained possession.
        
        Args:
            row (pd.Series): Row containing pressing situation information.
            event_df (pd.DataFrame): Event dataframe containing match events.
            teams_dict (dict): Dictionary containing team information.
            
        Returns:
            bool: True if pressing was successful (possession gained), False otherwise.
        """
        # Events that indicate successful possession gain
        possession_gained_events = [
            'pass', 'dribble', 'recovery', 'interception', 'cross', 'throw_in', 
            'take_on', 'shot', 'freekick_crossed', 'corner_crossed', 'goalkick'
        ] 
        
        # Determine pressing team based on ball carrier's team
        if row['team_id'] == teams_dict['Home']['tID'].unique()[0]:
            pressing_team = teams_dict['Away']['tID'].unique()[0]
        elif row['team_id'] == teams_dict['Away']['tID'].unique()[0]:
            pressing_team = teams_dict['Home']['tID'].unique()[0]
        else:
            return False

        # Check events within 5 seconds after pressing
        check_timegap = pd.Timedelta(seconds=5)
        window_events = event_df[
            (event_df['period_id'] == row['period_id']) &
            (event_df['time_seconds'] >= row['timestamp']) &
            (event_df['time_seconds'] <= row['timestamp'] + check_timegap)
        ]
        event_teams = window_events['tID'].unique()

        if pressing_team in event_teams:
            pressing_team_events = window_events[window_events['tID'] == pressing_team]
            for _, event_row in pressing_team_events.iterrows():
                if event_row['type_name'] in possession_gained_events:
                    # Always successful events
                    if event_row['type_name'] in ['dribble', 'recovery', 'interception', 'cross', 
                                                 'throw_in', 'shot', 'freekick_crossed', 'corner_crossed', 'goalkick']:
                        return True
                    # Events that need success check
                    elif event_row['type_name'] in ['pass', 'take_on']:
                        if event_row['result_name'] == 'Successful':
                            return True
            return False
        else:
            return False

    def _generate_features(self, frame_df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features from tracking data for a single frame.
        
        This function calculates various features including binary attributes,
        distance and angle features relative to goal and ball, and velocity
        angle features for each player in the frame.
        
        Args:
            frame_df (pd.DataFrame): Player and ball data for a single frame.
                Must contain columns: 'id', 'team_id', 'position_name', 'is_ball_carrier',
                'x', 'y', 'vx', 'vy', etc.
                
        Returns:
            pd.DataFrame: DataFrame with new features added including:
                - is_teammate: Binary indicator if player is on same team as ball carrier
                - is_goalkeeper: Binary indicator if player is goalkeeper
                - distance_to_goal: Euclidean distance to goal
                - sin_angle_to_goal, cos_angle_to_goal: Trigonometric angles to goal
                - distance_to_ball: Euclidean distance to ball
                - sin_angle_to_ball, cos_angle_to_ball: Trigonometric angles to ball
                - cos_velocity_angle, sin_velocity_angle: Velocity angle relative to ball carrier
        """
        df = frame_df.copy()

        # Step 1: Identify main objects
        ball_row = df[df['id'] == 'ball']
        ball_carrier_row = df[df['is_ball_carrier'] == True]

        # Handle cases without ball or ball carrier
        if ball_row.empty or ball_carrier_row.empty:
            feature_cols_to_add = [
                'is_teammate', 'is_goalkeeper', 'distance_to_goal', 'sin_angle_to_goal',
                'cos_angle_to_goal', 'distance_to_ball', 'sin_angle_to_ball',
                'cos_angle_to_ball', 'cos_velocity_angle', 'sin_velocity_angle'
            ]
            for col in feature_cols_to_add:
                df[col] = 0.0
            return df

        ball_carrier = ball_carrier_row.iloc[0]
        ball = ball_row.iloc[0]
        
        # Goal position (left to right attack direction)
        goal_pos = np.array([52.5, 0.0])

        # Step 2: Calculate binary attributes
        df['is_teammate'] = (df['team_id'] == ball_carrier['team_id']).astype(float)
        df['is_goalkeeper'] = (df['position_name'] == 'GK').astype(float)

        # Step 3: Prepare data for vector calculations
        player_positions = df[['x', 'y']].values.astype(np.float64)
        player_velocities = df[['vx', 'vy']].values.astype(np.float64)
        ball_position = ball[['x', 'y']].values.astype(np.float64)
        carrier_velocity = ball_carrier[['vx', 'vy']].values.astype(np.float64)

        # Step 4: Features relative to goal
        vector_to_goal = goal_pos - player_positions
        df['distance_to_goal'] = np.linalg.norm(vector_to_goal, axis=1)
        angle_to_goal_rad = np.arctan2(vector_to_goal[:, 1], vector_to_goal[:, 0])
        df['sin_angle_to_goal'] = np.sin(angle_to_goal_rad)
        df['cos_angle_to_goal'] = np.cos(angle_to_goal_rad)

        # Step 5: Features relative to ball
        vector_to_ball = ball_position - player_positions
        df['distance_to_ball'] = np.linalg.norm(vector_to_ball, axis=1)
        angle_to_ball_rad = np.arctan2(vector_to_ball[:, 1], vector_to_ball[:, 0])
        df['sin_angle_to_ball'] = np.sin(angle_to_ball_rad)
        df['cos_angle_to_ball'] = np.cos(angle_to_ball_rad)

        # Step 6: Velocity angle features
        dot_product = np.sum(player_velocities * carrier_velocity, axis=1)
        norm_player = np.linalg.norm(player_velocities, axis=1)
        norm_carrier = np.linalg.norm(carrier_velocity)
        
        # Prevent division by zero
        denominator = (norm_player * norm_carrier) + 1e-8
        
        df['cos_velocity_angle'] = np.clip(dot_product / denominator, -1.0, 1.0)
        
        # Calculate sine using 2D vector cross product
        cross_product = (player_velocities[:, 0] * carrier_velocity[1] - 
                        player_velocities[:, 1] * carrier_velocity[0])
        df['sin_velocity_angle'] = np.clip(cross_product / denominator, -1.0, 1.0)

        return df
       
    def _load_data(self):
        """Load and process all match data to create pressing sequence samples.
        
        This function loads tracking data, event data, and pressing intensity data
        for all matches, processes them to identify pressing situations, and creates
        training samples with features and labels.
        """
        first_frames_list = []

        # Initialize lists to store all samples
        all_features_seqs = []
        all_pressintensity_seqs = []
        all_labels = []
        all_pressed_ids = []
        all_presser_ids = []
        all_agent_orders = []
        all_match_infos = []
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
            
        total_dict = {match_id: {} for match_id in self.match_id_lst}
        for match_id in tqdm(self.match_id_lst, desc=f"Loading {self.data_path} data"):
            print(f"Load match_id: {match_id}")
            total_dict[match_id] = {}
            
            # Load processed match data
            try:
                with open(f"{self.data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
                    match_dict = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Processed data file not found for match {match_id}. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading processed data for match {match_id}: {e}")
                continue

            # Extract and prepare tracking data
            tracking_df = match_dict['tracking_df'].copy()
            teams_dict = match_dict['teams'].copy()
            home_team = teams_dict['Home'].copy()
            away_team = teams_dict['Away'].copy()
            teams_df = pd.concat([home_team, away_team])
            meta_data = match_dict['meta_data']

            # Sort and normalize tracking data
            tracking_df = tracking_df.sort_values(['game_id', 'period_id', 'frame_id']).reset_index(drop=True)
            tracking_df = self._normalize_coordinate_direction(tracking_df, teams_dict['Home']['tID'].iloc[0])
            
            # Load pressing intensity data
            try:
                with open(f"{self.data_path}/{match_id}/{match_id}_pressing_intensity.pkl", "rb") as f:
                    pressing_df = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Pressing intensity file not found for match {match_id}. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading pressing intensity for match {match_id}: {e}")
                continue

            # Load and preprocess event data
            try:
                event_df = pd.read_csv(f"{self.data_path}/{match_id}/valid_events_filtered2.csv")
                event_df = self._preprocess_event_df(event_df, teams_df)
            except FileNotFoundError:
                print(f"Warning: Event file not found for match {match_id}. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading event data for match {match_id}: {e}")
                continue
            
            # Merge all data and store in total_dict
            total_df = self._merge_tracking_pressing_df(tracking_df, pressing_df, teams_df)
            total_dict[match_id].update({
                'tracking_df': total_df,
                'event_df': event_df,
                'meta_data': meta_data,
                'Home': match_dict['teams']['Home'],
                'Away': match_dict['teams']['Away']
            })

            # Construct pressed_df if pressing intensity is greater than 0.9 for ball carrier.
            # ball_carrier_df: schema
            # "row": home team player_id, "column": away team player_id
            # "probability_to_intercept"(len(row), len(column)): matrix of shape (home players, away players) representing pressing intensity each player exerts on opponent
            pressed_dict = {}
            ball_carrier_df = total_df[total_df['is_ball_carrier'] == True].copy() # Extract only frames with ball carrier: for detecting pressing situations (intensity, speed)
            for idx, row in tqdm(ball_carrier_df.iterrows(), desc= "Get Pressing Intensity", miniters=len(ball_carrier_df)//10):                    
                if len(np.where(row['rows'] == row['id'])[0]) != 0: # If ball carrier is in home team
                    pressed_axis = 'rows'
                    presser_axis = 'columns'
                    id_loc = np.where(row[pressed_axis] == row['id'])[0]  # Index of ball carrier in the row
                    # Since it can be a nested list, apply tolist() twice
                    pressing_values = row['probability_to_intercept'][id_loc].tolist()[0].tolist() # If ball carrier is in home team, pressing intensity for away players
                elif len(np.where(row['columns'] == row['id'])[0]) != 0: # If ball carrier is in away team
                    pressed_axis = 'columns'
                    presser_axis = 'rows'
                    id_loc = np.where(row[pressed_axis] == row['id'])[0] # Index of ball carrier in the column
                    pressing_values = [x[id_loc] for x in row['probability_to_intercept']] # If ball carrier is in away team, pressing intensity for home players
                else:
                    continue
                if max(pressing_values) > self.press_threshold:
                    pressed_dict[idx] = {}
                    pressed_dict[idx]['pressing_value'] = max(pressing_values)
                    max_idx = pressing_values.index(max(pressing_values))
                    pressed_dict[idx]['pressing_player'] = row[presser_axis][max_idx]
            pressed_df = ball_carrier_df.loc[list(pressed_dict.keys())].copy()
            pressed_df['pressing_values'] = [d['pressing_value'] for d in pressed_dict.values()]
            pressed_df['pressing_player'] = [d.get('pressing_player') for d in pressed_dict.values()]
            
            # Split pressing sequence
            period_list = []
            for period_id in pressed_df['period_id'].unique():
                period_df = pressed_df[pressed_df['period_id']==period_id].copy()
                # If frame_id difference is greater than 125 frames, consider as a new sequence (i.e., not consecutive pressed rows)
                period_df['frame_diff'] = period_df['frame_id'].diff()
                period_df['sequence_id'] = (period_df['frame_diff'] > 125).cumsum()

                # For each sequence, get the first frame info to set X and Y
                first_frames = period_df.groupby('sequence_id', as_index=False)[['timestamp', 'period_id', 'frame_id', 'id', 'team_id', 'pressing_player']].first()

                # Extract only required columns from total_df
                lookup = total_df[['period_id', 'frame_id', 'id', 'v']]

                # pressing_player column matches total_df.id,
                # so align column names and merge
                first_frames = first_frames.merge(
                    lookup.rename(columns={'id': 'pressing_player'}),
                    on=['period_id', 'frame_id', 'pressing_player'],
                    how='left'
                )

                # Only consider as pressing if the pressing player's speed is at least 2.0 m/s
                first_frames = first_frames[first_frames['v'] >= 2.0]
                period_list.append(first_frames)

            first_frames_df = pd.concat(period_list, axis=0, ignore_index=True)    
            first_frames_df['ball_ownership_changed'] = first_frames_df.apply(self._check_pressing_success, axis=1, event_df=event_df, teams_dict=teams_dict) # window=150 is an example    
            
            # Use when analyzing highpress only
            if self.highpress_only:
                first_frames_df['is_highpress'] = first_frames_df.apply(self.is_highpress, axis=1, total_df=total_df, teams_df=teams_df)
                first_frames_df = first_frames_df[first_frames_df['is_highpress'] == True]
            first_frames_list.append(first_frames_df)

            # Create dictionary: extract and store by period_id (lookup table: optimize search space)
            events_by_period = {period: df for period, df in event_df.groupby('period_id')}
            tracking_by_period = {period: df for period, df in total_df.groupby('period_id')}

            print(f"Match ID: {match_id} | Total Frames: {len(total_df)} | First Frames: {len(first_frames_df)}\n")
            if self.highpress_only:
                print(f"Highpress Count: {first_frames_df[first_frames_df['is_highpress'] == True].shape[0]}")
                print(f"Highpress Success Count: {first_frames_df[(first_frames_df['is_highpress'] == True) & (first_frames_df['ball_ownership_changed'] == 1)].shape[0]}")
            
            # Construct samples
            for _, row in tqdm(first_frames_df.iterrows(), desc= "Get Samples"):#, miniters=len(first_frames)//10):
                try:
                    period_id = row['period_id']
                    frame_id = row['frame_id']
                    timestamp = row['timestamp']
                    label = int(row['ball_ownership_changed'])
                    pressed_player = row['id']
                    pressing_player = row['pressing_player']

                    # Events that occurred in the 5 seconds before the pressing started
                    event_period_df = events_by_period.get(period_id)
                    window_event_df = event_period_df[
                        (event_period_df['time_seconds'] >= timestamp - pd.Timedelta(seconds=5)) &
                        (event_period_df['time_seconds'] <= timestamp)
                    ]

                    # Tracking data within 5 seconds before pressing started
                    trace_period_df = tracking_by_period.get(period_id)
                    window_trace_df = trace_period_df[
                        (trace_period_df['timestamp'] >= timestamp - pd.Timedelta(seconds=5)) &
                        (trace_period_df['timestamp'] <= timestamp)
                    ].copy()

                    counts_per_timestamp = window_trace_df.groupby('timestamp').size()
                    pressing_frame_count = counts_per_timestamp.get(timestamp)
                    valid_counts = counts_per_timestamp[counts_per_timestamp == pressing_frame_count]
                    available_timestamps = valid_counts.index.tolist()

                    # Always include the last frame (pressing moment), so temporarily remove it from the sampling pool: will add later
                    if timestamp in available_timestamps:
                        available_timestamps.remove(timestamp)

                    # Sample (num_frames_to_sample-1) from (all-1) available timestamps
                    if len(available_timestamps) >= (self.num_frames_to_sample - 1):
                        # sampled_timestamps = random.sample(available_timestamps, num_frames_to_sample - 1)
                        stride = len(available_timestamps) / (self.num_frames_to_sample - 1)
                        sampled_timestamps = [available_timestamps[int(i * stride)] for i in range(self.num_frames_to_sample - 1)]
                    else:
                        # If less than num_frames_to_sample, use all available
                        print(f"Warning : {match_id}-{period_id}-{frame_id} doesn't have {self.num_frames_to_sample} windows({len(available_timestamps)}).")
                        sampled_timestamps = available_timestamps

                    final_timestamps = sampled_timestamps + [timestamp]
                    final_timestamps.sort()
                    X_slice = window_trace_df[window_trace_df['timestamp'].isin(final_timestamps)].copy()

                    # Always press left -> right
                    # If pressed player's team is home, flip: if carrier is in home team, mirror left-right (pressing team always attacks left to right)
                    if X_slice.loc[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['team_id'].iloc[0] == match_dict['teams']['Home']['tID'].iloc[0]:
                        for col in self.cols_to_flip:
                            X_slice.loc[:, col] = -X_slice.loc[:, col]
                        
                    # Get Features
                    # Generate kinematic features and event features per frame/agent
                    window_event_df = window_event_df.copy()
                    window_event_df["type_id"] += 1
                    X_slice = pd.merge_asof(X_slice, window_event_df[["time_seconds", "type_id"]], left_on="timestamp", right_on="time_seconds", direction="forward")
                    X_slice["type_id"] = X_slice["type_id"].ffill().fillna(0).astype(int)
                    X_slice = X_slice.set_index('frame_id').groupby('frame_id', group_keys=False).apply(self._generate_features)
                    
                    X_slice.reset_index(inplace=True)

                    # Fill players if there are less than 22 players in the frame
                    agents_rows = X_slice[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['rows'].values[0].tolist() # Home team
                    agents_cols = X_slice[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['columns'].values[0].tolist() # Away team
                    missing_cnt = 0
                    num_missing_rows = NUM_TEAM_PLAYERS - len(agents_rows)
                    if num_missing_rows > 0:
                        for i in range(num_missing_rows):
                            agents_rows.append(f"Missing_{missing_cnt}")
                            missing_cnt += 1

                    num_missing_cols = NUM_TEAM_PLAYERS - len(agents_cols)
                    if num_missing_cols > 0:
                        for i in range(num_missing_cols):
                            agents_cols.append(f"Missing_{missing_cnt}")
                            missing_cnt += 1
                    agents_order = agents_rows + agents_cols
                    
                    # Ensure the player IDs are consistent and match num_agents (22 players + 1 ball)
                    all_known_agents = set(X_slice['id'].unique())
                    missing_agent_ids = [agent for agent in agents_order if agent not in all_known_agents and 'Missing' in agent]

                    frame_lst = X_slice['frame_id'].unique()
                    if missing_agent_ids:
                        # Add missing player rows with zero values for each frame in X_slice
                        missing_rows = []
                        for missing_agent_id in missing_agent_ids:
                            for frame in frame_lst:
                                missing_row = {col: 0 for col in X_slice.columns}  # Fill all columns with 0
                                missing_row['id'] = missing_agent_id  # Set the 'id' to the missing player's id
                                missing_row['frame_id'] = frame  # Set the frame_id for the current frame in the sequence
                                missing_rows.append(missing_row)
                        if missing_rows:
                            # Create a DataFrame for the missing rows and append to the slice
                            missing_df = pd.DataFrame(missing_rows)
                            X_slice = pd.concat([X_slice, missing_df], ignore_index=True)
                    
                    agents_order.append('ball')

                    #X_slice.loc[:, 'id'] = pd.Categorical(X_slice['id'], categories=agents_order, ordered=True)
                    X_slice['id'] = pd.Categorical(X_slice['id'], categories=agents_order, ordered=True)

                    # Sort the players by their ID to maintain a consistent order
                    X_slice = X_slice.sort_values(by=['frame_id', 'id'])
            
                    # Get the features
                    x_tensor = torch.tensor(X_slice[self.feature_cols].values, dtype=torch.float32)

                    X_slice_pressing = X_slice[X_slice['is_ball_carrier']==True]['probability_to_intercept']
                    X_slice_pressing = X_slice_pressing.dropna()
                    pressing_intensity_tensor = torch.tensor(np.stack(X_slice_pressing.map(lambda x: np.stack(x)).values), dtype=torch.float32)
                    _, h, w = pressing_intensity_tensor.shape
                    pad_h = NUM_TEAM_PLAYERS - h
                    pad_w = NUM_TEAM_PLAYERS - w
                    pressing_intensity_tensor = F.pad(pressing_intensity_tensor, (0, pad_w, 0, pad_h), "constant", 0)

                    x_tensor = x_tensor.reshape(-1, NUM_AGENTS, len(self.feature_cols))
                    y_tensor = torch.tensor(label, dtype=torch.long)

                    # Debug 
                    if x_tensor.isnan().any():
                        print("Nan found", match_id, period_id, frame_id)
                        continue
                    match_info = f"{match_id}-{period_id}-{frame_id}"

                    all_features_seqs.append(x_tensor)
                    all_pressintensity_seqs.append(pressing_intensity_tensor)
                    all_labels.append(y_tensor)
                    all_pressed_ids.append(pressed_player)
                    all_presser_ids.append(pressing_player)    
                    all_agent_orders.append(agents_order)
                    all_match_infos.append(match_info)
                
                except (FileNotFoundError, ValueError, KeyError) as e:
                    error_type = type(e).__name__
                    period_id = row.get('period_id', 'unknown')
                    frame_id = row.get('frame_id', 'unknown')
                    print(f"{error_type} in match {match_id}, period {period_id}, frame {frame_id}: {e}")
                    continue
                except Exception as e:
                    period_id = row.get('period_id', 'unknown')
                    frame_id = row.get('frame_id', 'unknown')
                    print(f"Unexpected error in match {match_id}, period {period_id}, frame {frame_id}: {e}")
                    continue
        # Store all processed data
        self.features_seqs = all_features_seqs
        self.pressintensity_seqs = all_pressintensity_seqs
        self.labels = all_labels
        self.pressed_ids = all_pressed_ids
        self.presser_ids = all_presser_ids
        self.agent_orders = all_agent_orders
        self.match_infos = all_match_infos
        
        # Create data list for compatibility with other dataset classes
        self.data = [
            {
                'features': self.features_seqs[i],
                'pressing_intensity': self.pressintensity_seqs[i],
                'label': self.labels[i],
                'pressed_id': self.pressed_ids[i],
                'presser_id': self.presser_ids[i],
                'agent_order': self.agent_orders[i],
                'match_info': self.match_infos[i]
            }
            for i in range(len(self.features_seqs))
        ]

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing the sample data with keys:
                - features: Feature tensor of shape [SeqLen, Agents, Features]
                - pressing_intensity: Pressing intensity tensor of shape [SeqLen, 11, 11]
                - label: Binary label tensor (0 or 1)
                - pressed_id: String ID of the pressed player
                - presser_id: String ID of the pressing player
                - agent_order: List of agent IDs in order
                - match_info: String containing match information
        """
        return {
            'features': self.features_seqs[idx],       # Shape: [SeqLen, Agents, Features]
            'pressing_intensity': self.pressintensity_seqs[idx],         # Shape: [SeqLen, ?, ?] (Adjust shape based on data)
            'label': self.labels[idx],                   # Shape: [1] or scalar
            'pressed_id': self.pressed_ids[idx],         # String (Player ID)
            'presser_id': self.presser_ids[idx],         # String (Player ID)
            'agent_order': self.agent_orders[idx],        # List of Strings (Agent IDs in order)
            'match_info': self.match_infos[idx]
        }
        
    def __len__(self):
        """Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.features_seqs)


class ToSoccerMapTensor:
    """Convert tracking data to spatial representation for soccer analysis.
    
    This class transforms tracking data into a spatial grid representation
    suitable for convolutional neural networks, creating feature maps that
    encode player positions, velocities, and other spatial relationships.
    
    Attributes:
        y_bins (int): Number of bins in the y-direction (height).
        x_bins (int): Number of bins in the x-direction (width).
        num_features (int): Number of feature channels in the output.
        seq_len (int): Length of the sequence to process.
        device (torch.device): Device to use for tensor operations.
        goal_coo (torch.Tensor): Goal coordinates in the spatial representation.
    """

    def __init__(self, dim=(68, 104), num_features=NUM_FEATURE_CHANNELS, num_frames_to_sample=3):
        """Initialize the ToSoccerMapTensor transformer.
        
        Args:
            dim (tuple, optional): The dimensions of the pitch in the spatial representation.
                The original pitch dimensions are 105x68, but even numbers are easier
                to work with. Defaults to (68, 104).
            num_features (int, optional): Number of input channels for models.
                Defaults to NUM_FEATURE_CHANNELS.
            sequence_length (int, optional): Length of the sequence to process.
                Defaults to 3.
        """
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self.num_features = num_features
        self.seq_len = num_frames_to_sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Goal coordinates in center-origin coordinates
        self.goal_coo = torch.tensor([[CENTER_X, 0.0]], device=self.device)

    def _get_cell_indexes(self, x: torch.Tensor, y: torch.Tensor):
        """Convert continuous coordinates to grid cell indices.
        
        This function transforms continuous pitch coordinates (center-origin) to
        discrete grid cell indices suitable for spatial feature maps.
        
        Args:
            x (torch.Tensor): X-coordinates in center-origin system [-52.5, 52.5].
            y (torch.Tensor): Y-coordinates in center-origin system [-34.0, 34.0].
            
        Returns:
            tuple: (x_bin, y_bin) tensors containing grid cell indices.
                x_bin: X-direction bin indices [0, x_bins-1].
                y_bin: Y-direction bin indices [0, y_bins-1].
        """
        x = x.float()
        y = y.float()
        
        # Shift origin from pitch center to bottom-left corner
        # x: [-52.5, 52.5] -> [0, 105], y: [-34.0, 34.0] -> [0, 68]
        x_shifted = x + 52.5
        y_shifted = y + 34.0

        # Normalize to [0, 1] range
        x_norm = x_shifted / PITCH_LENGTH
        y_norm = y_shifted / PITCH_WIDTH

        # Scale to bin coordinates
        x_cont = x_norm * self.x_bins
        y_cont = y_norm * self.y_bins
        
        # Clamp to valid index range [0, bins-1]
        x_clamped = torch.clamp(x_cont, min=0, max=self.x_bins - 1)
        y_clamped = torch.clamp(self.y_bins - y_cont, min=0, max=self.y_bins - 1)

        # Convert to integer bin indices
        x_bin = x_clamped.to(torch.long)
        y_bin = y_clamped.to(torch.long)
    
        return x_bin, y_bin
    
    def __call__(self, sample: dict):
        """Process sample data to generate spatial map representation.
        
        This method transforms tracking data into a spatial grid representation
        with multiple feature channels encoding player positions, velocities,
        and spatial relationships.
        
        Args:
            sample (dict): Dictionary containing sample data with keys:
                - features: Tensor of shape [T, N, F] with kinematic features
                - pressing_intensity: Tensor of shape [T, 11, 11] with pressing data
                - label: Binary label tensor (0 or 1)
                - presser_id: String ID of the pressing player
                - agent_order: List of agent IDs in order
                
        Returns:
            tuple: (spatial_map, label) where:
                - spatial_map: Tensor of shape [num_features * seq_len, y_bins, x_bins]
                - label: Tensor containing the binary label
        """

        # Output
        home_indices = slice(0, 11)
        away_indices = slice(11, 22)
        goal_coo = torch.tensor([[52.5, 0]]) # x-axis: -52.5 to 52.5, y-axis: -34 to 34

        # features: ['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a']
        #x_tensor = sample['features'][-3:, ...] # (T, N, F) -> Get the last frame in the sequence(T)

        x = sample['features']
        T, N, F = x.shape
        if T >= self.seq_len:
            # Only the last 3 frames
            x_tensor = x[-self.seq_len:]                  # shape: (seq_len, N, F)
        else:
            # Pad with zeros for missing frames
            pad_frames = self.seq_len - T
            pad = torch.zeros((pad_frames, N, F),
                            device=x.device,
                            dtype=x.dtype)
            # Pad at the front, then actual frames at the back
            x_tensor = torch.cat([pad, x], dim=0)  # shape: (seq_len, N, F)
            
        # num_seqs = x_tensor.shape[0]
        num_seqs = 1
        
        matrix = np.zeros((self.num_features * num_seqs, self.y_bins, self.x_bins))

        target = sample['label']
        presser_id = sample['presser_id']
        agents_order = sample['agent_order']
        presser_idx = agents_order.index(presser_id)
        home_team_ids = agents_order[:11]
        away_team_ids = agents_order[11:22]

        for i in range(num_seqs):
            # x_feat = x_tensor[i]
            x_feat = x_tensor[-1]

            if presser_id in home_team_ids:
                pressing_indices = home_indices
                pressed_indices = away_indices
                x_bin_pressing, y_bin_pressing = self._get_cell_indexes(x_feat[pressing_indices, :1], x_feat[pressing_indices, 1:2])
                x_bin_pressed, y_bin_pressed = self._get_cell_indexes(x_feat[pressed_indices, :1], x_feat[pressed_indices, 1:2])
            elif presser_id in away_team_ids:
                pressed_indices = home_indices
                pressing_indices = away_indices
                x_bin_pressed, y_bin_pressed= self._get_cell_indexes(x_feat[pressed_indices, :1], x_feat[pressed_indices, 1:2])
                x_bin_pressing, y_bin_pressing = self._get_cell_indexes(x_feat[pressing_indices, :1], x_feat[pressing_indices, 1:2])
            else:
                raise ValueError(f"Invalid presser_id: {presser_id}")
            
            # Ch 1: Locations of pressing teams
            matrix[0 + i * self.num_features, y_bin_pressing, x_bin_pressing] = 1

            # Ch 2: vx of pressing teams
            matrix[1 + i * self.num_features, y_bin_pressing, x_bin_pressing] = x_feat[pressing_indices, 2:3]

            # Ch 3: vy of pressing teams
            matrix[2 + i * self.num_features, y_bin_pressing, x_bin_pressing] = x_feat[pressing_indices, 3:4]
            
            # CH 4: ax of pressing teams
            matrix[3 + i * self.num_features, y_bin_pressing, x_bin_pressing] = x_feat[pressing_indices, 5:6]

            # CH 5: ay of pressing teams
            matrix[4 + i * self.num_features, y_bin_pressing, x_bin_pressing] = x_feat[pressing_indices, 6:7]

            # CH 6: Locations of pressed teams
            matrix[5 + i * self.num_features, y_bin_pressed, x_bin_pressed] = 1

            # Ch 7: vx of pressed teams
            matrix[6 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 2:3]

            # Ch 8: vy of pressed teams
            matrix[7 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 3:4]

            # CH 9: ax of pressed teams
            matrix[8 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 5:6]

            # CH 10: ay of pressed teams
            matrix[9 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 6:7]

            # CH 11: Distance to ball
            y_coords = torch.arange(0.5, self.y_bins, device=x_feat.device) # Shape: [y_bins]
            x_coords = torch.arange(0.5, self.x_bins, device=x_feat.device)

            # Create 2D grid coordinate tensors
            # 'ij' indexing: yy changes along dim 0, xx changes along dim 1
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij') # yy shape: [y_bins, x_bins], xx shape: [y_bins, x_bins]

            x0_ball, y0_ball = self._get_cell_indexes(x_feat[22, :1], x_feat[22, 1:2])
            x0_ball_center = x0_ball.float() + 0.5
            y0_ball_center = y0_ball.float() + 0.5 #x0_ball.float() + 0.5

            matrix[10 + i * self.num_features, : , :] = torch.sqrt((xx - x0_ball_center)**2 + (yy - y0_ball_center)**2).numpy()

            # CH 8: Distance to goal
            x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
            x0_goal_center = x0_goal.float() + 0.5
            y0_goal_center = y0_goal.float() + 0.5

            matrix[11 + i * self.num_features, : , :] = torch.sqrt((xx - x0_goal_center)**2 + (yy - y0_goal_center)**2).numpy()

            # CH 9: Cosine of the angle between the ball and goal
            coords = torch.stack([xx, yy], dim=-1) # Shape: [H, W, 2]
            goal_coo_bin = torch.tensor([x0_goal_center, y0_goal_center], dtype=torch.float32, device=x_feat.device) # Shape: [2]
            ball_coo_bin = torch.tensor([x0_ball_center, y0_ball_center], dtype=torch.float32, device=x_feat.device) # Shape: [2]
            a = goal_coo_bin - coords # Shape: [H, W, 2], vector from cell (i,j) to goal
            b = ball_coo_bin - coords # Shape: [H, W, 2], vector from cell (i,j) to ball

            norm_a = torch.linalg.norm(a, dim=-1) # Shape: [H, W]
            norm_b = torch.linalg.norm(b, dim=-1) # Shape: [H, W]
            denominator = norm_a * norm_b + 1e-8 # Add epsilon here
            cosine_angle = torch.sum(a * b, dim=-1) / (norm_a + norm_b + 1e-8)
            cosine_angle = torch.clamp(cosine_angle, min=-1.0, max=1.0) # Shape: [H, W]
            matrix[12 + i * self.num_features, : , :] = cosine_angle.numpy()

            # CH 10: Sine of the angle between the ball and goal
            sine_angle = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (norm_a + norm_b + 1e-8)
            sine_angle = torch.clamp(sine_angle, min=-1.0, max=1.0)
            matrix[13 + i * self.num_features, : , :] = sine_angle.numpy()

            # CH 11: Angle (in radians) to the goal location
            vector_y = y0_goal_center - coords[..., 1] # Shape: [H, W]
            vector_x = x0_goal_center - coords[..., 0] # Shape: [H, W]
            angle_rad = torch.abs(torch.arctan2(vector_y, vector_x))
            matrix[14 + i * self.num_features, : , :] = angle_rad.numpy()

            # CH 12, 13: Cosine, sine angle to the teammates from pressers
            teammates_idx = [i for i in range(pressing_indices.start, pressing_indices.stop) if i != presser_idx]

            pos_presser = x_feat[presser_idx, 0:2]           # Ball carrier position [x, y], Shape: [2]
            v_presser = x_feat[presser_idx, 2:4]            # Ball carrier velocity [vx, vy], Shape: [2] -> ASSUMPTION about indices 2, 3
            pos_teammates = x_feat[teammates_idx, 0:2]

            vec_to_teammate = pos_teammates - pos_presser # Shape: [10, 2]
            norm_v_carrier = torch.linalg.norm(v_presser)        # Scalar magnitude |u|
            norm_vec_to_tm = torch.linalg.norm(vec_to_teammate, dim=1) # Magnitude |v| for each teammate, Shape: [10]

            cosine_angle = torch.sum(v_presser.unsqueeze(0) * vec_to_teammate, dim=1) / (norm_v_carrier * norm_vec_to_tm + 1e-8) # Shape: [10]
            sine_angle = (v_presser[0] * vec_to_teammate[:, 1] - v_presser[1] * vec_to_teammate[:, 0]) / (norm_v_carrier * norm_vec_to_tm + 1e-8)   # Shape: [10]
            cosine_angle = torch.clamp(cosine_angle, min=-1.0, max=1.0)
            sine_angle = torch.clamp(sine_angle, min=-1.0, max=1.0)

            x_bin_tm, y_bin_tm = self._get_cell_indexes(pos_teammates[:, 0], pos_teammates[:, 1])
            matrix[15 + i * self.num_features, y_bin_tm, x_bin_tm] = cosine_angle.numpy()
            matrix[16 + i * self.num_features, y_bin_tm, x_bin_tm] = sine_angle.numpy()

            # CH 7: Location of the presser
            # x_bin_presser, y_bin_presser = self._get_cell_indexes(x_feat[presser_idx:presser_idx+1, :1], x_feat[presser_idx:presser_idx+1, 1:2])
            # matrix[13 + i * self.num_features, y_bin_presser, x_bin_presser] = 1

            # CH8: Location of the ball
            # x_bin_ball, y_bin_ball = self._get_cell_indexes(x_feat[22:23, :1], x_feat[22:23, 1:2])
            # matrix[14 + i * self.num_features, y_bin_ball, x_bin_ball] = 1

            # # CH9: vx of the ball
            # matrix[15 + i * self.num_features, y_bin_ball, x_bin_ball] = x_feat[22:23, 2:3]

            # # CH10: vy of the ball
            # matrix[16 + i * self.num_features, y_bin_ball, x_bin_ball] = x_feat[22:23, 3:4]

        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.tensor([target]).float(),
            )
        else:
            # simulated features: not exist labels
            return (
                torch.from_numpy(matrix).float(),
                None,
        )


# --- Base Dataset Class ---
class BaseDataset(Dataset):
    """Base dataset class with common functionality.
    
    This class provides common functionality for loading dataset data from
    pickle files or creating new datasets from raw data.
    
    Attributes:
        loaded_data (list): List of sample dictionaries containing all data.
    """
    
    def _load_dataset_data(self, pickled_dataset_path, data_path, match_id_lst, num_frames_to_sample, feature_cols, highpress_only=False, press_threshold=0.9):
        """Load dataset data from pickle file or create new dataset.
        
        Args:
            pickled_dataset_path (str, optional): Path to pickled dataset file.
            data_path (str, optional): Path to raw data directory.
            match_id_lst (list, optional): List of match IDs.
            num_frames_to_sample (int): Number of frames to sample for each sequence.
            feature_cols (list, optional): Feature columns to use.
            highpress_only (bool): Only consider highpress situations.
            press_threshold (float): Threshold for pressing intensity.
        """
        if pickled_dataset_path and os.path.exists(pickled_dataset_path):
            print(f"Loading dataset from {pickled_dataset_path}...")
            try:
                with open(pickled_dataset_path, "rb") as f:
                    self.loaded_data = pickle.load(f)
                print(f"Successfully loaded {len(self.loaded_data)} samples from pickle file.")
            except FileNotFoundError:
                print(f"Error: Dataset file not found at {pickled_dataset_path}")
                raise
            except Exception as e:
                print(f"Error loading pickled dataset: {e}")
                raise
        else:
            raise ValueError("Pickled_dataset_path is not provided. Please generate the dataset first.")
            
    def __len__(self):
        """Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.loaded_data)


# --- Dataset Class (Loads Pickled Sequence Data, Applies Transform) ---
class SoccerMapInputDataset(BaseDataset):
    """Dataset for soccer map input generation.
    
    This dataset loads data from pickled files or creates new datasets from raw data,
    then applies spatial transformations to generate soccer map representations.
    
    Attributes:
        loaded_data (list): List of sample dictionaries containing all data.
        transform (ToSoccerMapTensor): Transformer to convert data to spatial maps.
    """
    
    def __init__(self, args, pickled_dataset_path=None, data_path=None, match_id_lst=None, num_frames_to_sample=10, feature_cols=None):
        """Initialize the SoccerMapInputDataset.
        
        Args:
            args (argparse.Namespace): Arguments object containing configuration settings.
            pickled_dataset_path (str, optional): Path to pickled dataset file.
                If provided and exists, loads data from this file. Defaults to None.
            data_path (str, optional): Path to raw data directory. Used if pickled_dataset_path
                is None. Defaults to None.
            match_id_lst (list, optional): List of match IDs. Used if pickled_dataset_path
                is None. Defaults to None.
            num_frames_to_sample (int, optional): Number of frames to sample for each sequence.
                Used if pickled_dataset_path is None. Defaults to 10.
            feature_cols (list, optional): Feature columns to use. Used if pickled_dataset_path
                is None. Defaults to None.
                
        Raises:
            ValueError: If neither pickled_dataset_path nor data_path/match_id_lst are provided.
            FileNotFoundError: If pickled dataset file is not found.
        """
        self._load_dataset_data(pickled_dataset_path, data_path, match_id_lst, num_frames_to_sample, feature_cols)

        # Initialize transformer here, used in __getitem__
        self.transform = ToSoccerMapTensor()

    def __getitem__(self, idx):
        """Get a sample and transform it to spatial map representation.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (spatial_map, label) where:
                - spatial_map: Tensor of shape [num_features * seq_len, y_bins, x_bins]
                - label: Tensor containing the binary label
                
        Raises:
            IndexError: If index is out of bounds.
        """
        if idx >= len(self.loaded_data):
            raise IndexError("Index out of bounds")

        spatial_map, label = self.transform(self.loaded_data[idx])
        return spatial_map, label

class exPressInputDataset(BaseDataset):
    """Dataset for express input generation with feature normalization.
    
    This dataset loads data from pickled files or creates new datasets from raw data,
    applies feature normalization, and returns data suitable for express models.
    
    Attributes:
        loaded_data (list): List of sample dictionaries containing all data.
        feature_min_vals (torch.Tensor): Minimum values for feature normalization.
        feature_max_vals (torch.Tensor): Maximum values for feature normalization.
        min_vals (torch.Tensor): Reshaped minimum values for broadcasting.
        max_vals (torch.Tensor): Reshaped maximum values for broadcasting.
        feature_ranges (torch.Tensor): Feature ranges for normalization.
    """
    
    def __init__(self, pickled_dataset_path=None, data_path=None, match_id_lst=None, num_frames_to_sample=10, feature_cols=None, wo_vel=False):
        """Initialize the exPressInputDataset.
        
        Args:
            pickled_dataset_path (str, optional): Path to pickled dataset file.
                If provided and exists, loads data from this file. Defaults to None.
            data_path (str, optional): Path to raw data directory. Used if pickled_dataset_path
                is None. Defaults to None.
            match_id_lst (list, optional): List of match IDs. Used if pickled_dataset_path
                is None. Defaults to None.
            num_frames_to_sample (int, optional): Number of frames to sample for each sequence.
                Used if pickled_dataset_path is None. Defaults to 10.
            feature_cols (list, optional): Feature columns to use. Used if pickled_dataset_path
                is None. Defaults to None.
            wo_vel (bool, optional): Whether to exclude velocity features. Defaults to False.
                
        Raises:
            ValueError: If neither pickled_dataset_path nor data_path/match_id_lst are provided.
            FileNotFoundError: If pickled dataset file is not found.
        """
        self.wo_vel = wo_vel
        self._load_dataset_data(pickled_dataset_path, data_path, match_id_lst, num_frames_to_sample, feature_cols)
        
        # Define categorical features (these should not be normalized)
        self.categorical_features = ['is_teammate', 'is_goalkeeper', 'type_id']

        # Get feature columns from the dataset
        if hasattr(self, 'feature_cols'):
            feature_cols = self.feature_cols
        else:
            # Default feature columns if not available
            feature_cols = [
                'x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a',  # Kinematic features
                'is_teammate', 'is_goalkeeper',  # Binary features
                'distance_to_goal', 'sin_angle_to_goal', 'cos_angle_to_goal',  # Goal features
                'distance_to_ball', 'sin_angle_to_ball', 'cos_angle_to_ball',  # Ball features
                'cos_velocity_angle', 'sin_velocity_angle',  # Velocity angle features
                'type_id'  # Event type feature
            ]
        feature_dim = len(feature_cols)
         # Create mask for continuous features (features that should be normalized)
        self.continuous_feature_mask = torch.tensor([
            feature not in self.categorical_features for feature in feature_cols
        ], dtype=torch.bool)

        # Define velocity and acceleration feature indices
        self.velocity_features = ['vx', 'vy', 'v']  # indices: 2, 3, 4
        self.acceleration_features = ['ax', 'ay', 'a']  # indices: 5, 6, 7

        # Get indices for velocity and acceleration features
        self.velocity_indices = [i for i, feat in enumerate(feature_cols) if feat in self.velocity_features]
        self.acceleration_indices = [i for i, feat in enumerate(feature_cols) if feat in self.acceleration_features]

         # Create separate normalization parameters for player and ball
        # Player normalization (using MAX_PLAYER_SPEED and MAX_PLAYER_ACCELERATION)
        self.player_velocity_min = torch.tensor([-MAX_PLAYER_SPEED, -MAX_PLAYER_SPEED, 0.0])
        self.player_velocity_max = torch.tensor([MAX_PLAYER_SPEED, MAX_PLAYER_SPEED, MAX_PLAYER_SPEED])
        self.player_acceleration_min = torch.tensor([-MAX_PLAYER_ACCELERATION, -MAX_PLAYER_ACCELERATION, 0.0])
        self.player_acceleration_max = torch.tensor([MAX_PLAYER_ACCELERATION, MAX_PLAYER_ACCELERATION, MAX_PLAYER_ACCELERATION])
        
        # Ball normalization (using MAX_BALL_SPEED and MAX_BALL_ACCELERATION)
        self.ball_velocity_min = torch.tensor([-MAX_BALL_SPEED, -MAX_BALL_SPEED, 0.0])
        self.ball_velocity_max = torch.tensor([MAX_BALL_SPEED, MAX_BALL_SPEED, MAX_BALL_SPEED])
        self.ball_acceleration_min = torch.tensor([-MAX_BALL_ACCELERATION, -MAX_BALL_ACCELERATION, 0.0])
        self.ball_acceleration_max = torch.tensor([MAX_BALL_ACCELERATION, MAX_BALL_ACCELERATION, MAX_BALL_ACCELERATION])
        
        # Initialize normalization parameters for other continuous features
        other_continuous_indices = [i for i in range(feature_dim) if self.continuous_feature_mask[i] and 
                                  i not in self.velocity_indices and i not in self.acceleration_indices]

        if other_continuous_indices:
            other_continuous_min = torch.tensor([FEAT_MIN[i] for i in other_continuous_indices])
            other_continuous_max = torch.tensor([FEAT_MAX[i] for i in other_continuous_indices])
            self.other_min_vals = other_continuous_min.reshape(1, 1, -1)
            self.other_max_vals = other_continuous_max.reshape(1, 1, -1)
            self.other_feature_ranges = self.other_max_vals - self.other_min_vals
            self.other_feature_ranges[self.other_feature_ranges == 0] = 1.0
        else:
            self.other_min_vals = None
            self.other_max_vals = None
            self.other_feature_ranges = None 
    
    def _normalize_features(self, features, is_ball=False):
        """Normalize features with different parameters for player and ball.
        
        Args:
            features (torch.Tensor): Feature tensor of shape [T, A, F]
            is_ball (bool): Whether these are ball features (True) or player features (False)
            
        Returns:
            torch.Tensor: Normalized feature tensor
        """
        normalized_features = features.clone()
        
        # Normalize velocity features
        if self.velocity_indices:
            if is_ball:
                vel_min = self.ball_velocity_min.reshape(1, 1, -1)
                vel_max = self.ball_velocity_max.reshape(1, 1, -1)
            else:
                vel_min = self.player_velocity_min.reshape(1, 1, -1)
                vel_max = self.player_velocity_max.reshape(1, 1, -1)
            
            vel_range = vel_max - vel_min
            vel_range[vel_range == 0] = 1.0
            
            for i, feat_idx in enumerate(self.velocity_indices):
                normalized_features[..., feat_idx] = (features[..., feat_idx] - vel_min[..., i]) / vel_range[..., i]
        
        # Normalize acceleration features
        if self.acceleration_indices:
            if is_ball:
                acc_min = self.ball_acceleration_min.reshape(1, 1, -1)
                acc_max = self.ball_acceleration_max.reshape(1, 1, -1)
            else:
                acc_min = self.player_acceleration_min.reshape(1, 1, -1)
                acc_max = self.player_acceleration_max.reshape(1, 1, -1)
            
            acc_range = acc_max - acc_min
            acc_range[acc_range == 0] = 1.0
            
            for i, feat_idx in enumerate(self.acceleration_indices):
                normalized_features[..., feat_idx] = (features[..., feat_idx] - acc_min[..., i]) / acc_range[..., i]
        
        # Normalize other continuous features
        if self.other_feature_ranges is not None:
            other_continuous_indices = [i for i in range(19) if self.continuous_feature_mask[i] and 
                                      i not in self.velocity_indices and i not in self.acceleration_indices]
            
            for i, feat_idx in enumerate(other_continuous_indices):
                normalized_features[..., feat_idx] = (features[..., feat_idx] - self.other_min_vals[..., i]) / self.other_feature_ranges[..., i]
        
        return normalized_features
    
    def __getitem__(self, idx):
        """Get a sample with normalized features.
        
        This method retrieves a sample, applies min-max normalization to features,
        and returns the processed data suitable for express models.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing the sample data with keys:
                - features: Normalized feature tensor of shape [T, A, F]
                - pressing_intensity: Pressing intensity tensor of shape [T, 11, 11]
                - label: Binary label tensor (0 or 1)
                - pressed_id: String ID of the pressed player
                - presser_id: String ID of the pressing player
                - agent_order: List of agent IDs in order
                - match_info: String containing match information
                
        Raises:
            IndexError: If index is out of bounds or data is not loaded.
        """
        if self.loaded_data is None or idx >= len(self.loaded_data):
            raise IndexError("Index out of bounds or data not loaded")
            
        # Extract features and apply normalization
        features = self.loaded_data[idx]['features']  # (T, A, F)
        # Separate players and ball (last agent is ball)
        player_features = features[:, :-1, :]  # (T, 22, F) - all players
        ball_features = features[:, -1:, :]    # (T, 1, F) - ball only

         # Normalize player features
        normalized_player_features = self._normalize_features(player_features, is_ball=False)
        
        # Normalize ball features
        normalized_ball_features = self._normalize_features(ball_features, is_ball=True)
        
        # Combine player and ball features
        features = torch.cat([normalized_player_features, normalized_ball_features], dim=1)
        # Extract other data
        pressing_intensity = self.loaded_data[idx]['pressing_intensity']  # (T, 11, 11)
        label = self.loaded_data[idx]['label']
        pressed_id = self.loaded_data[idx]['pressed_id']
        presser_id = self.loaded_data[idx]['presser_id']
        agent_order = self.loaded_data[idx]['agent_order']
        match_info = self.loaded_data[idx]['match_info']

        # Ablation study for checking performance w/o velocity features or w/o event type features
        if self.wo_vel:
            used_feature_ids = [i for i in range(19) if i not in [2, 3, 4, 5, 6, 7, 16, 17]]
            features = features[:, :, used_feature_ids]

        return {
                'features': features.float(),         # Shape: [T, A, F] e.g., [125, 23, 8]
                'pressing_intensity': pressing_intensity.float(), # Shape: [T, 11, 11]
                'label': label.float(),             # Shape: [1] or scalar
                'pressed_id': pressed_id,           # String
                'presser_id': presser_id,           # String
                'agent_order': agent_order,        # List
                'match_info': match_info
            }


def generate_datasets(args):
    """Generate training, validation, and test datasets based on arguments"""
    # Get match ID list
    match_id_lst = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    match_id_lst.sort()
    
    # Filter out excluded matches
    match_id_lst = [match_id for match_id in match_id_lst if match_id not in args.exclude_matches]
    print(f"Total matches found: {len(match_id_lst)}")
    
    def create_and_save_dataset(args, match_ids, save_path, dataset_name):
        """Helper function to create and save a dataset"""
        print(f"Creating {dataset_name} dataset...")
        dataset = PressingSequenceDataset(
            args.data_path, 
            match_id_lst=match_ids, 
            num_frames_to_sample=args.num_frames_to_sample,
            highpress_only=args.high_only,
            press_threshold=args.press_threshold
        )
        
        with open(f"{save_path}/{dataset_name}.pkl", "wb") as f:
            pickle.dump(dataset, f)
        print(f"{dataset_name} dataset saved: {len(dataset)} samples")
        return dataset
    
    if args.cross_validation:
        # Cross validation version
        print(f"Generating {args.n_folds}-fold cross validation datasets...")
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(match_id_lst)):
            print(f"\n=== Fold {fold_idx + 1}/{args.n_folds} ===")
            
            # Split train_val into train and validation
            train_val_match_ids = [match_id_lst[i] for i in train_val_indices]
            test_match_ids = [match_id_lst[i] for i in test_indices]
            
            # Further split train_val into train and validation (80% train, 20% validation)
            train_val_kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_indices, val_indices = next(train_val_kf.split(train_val_match_ids))
            
            train_match_ids = [train_val_match_ids[i] for i in train_indices]
            val_match_ids = [train_val_match_ids[i] for i in val_indices]
            
            print(f"Train: {len(train_match_ids)}, Valid: {len(val_match_ids)}, Test: {len(test_match_ids)}")
            
            # Create fold directory
            fold_save_path = f"{args.save_path}/fold_{fold_idx + 1}"
            os.makedirs(fold_save_path, exist_ok=True)
            
            # Create and save datasets
            create_and_save_dataset(args, train_match_ids, fold_save_path, f"train_dataset_{args.press_threshold}")
            create_and_save_dataset(args, val_match_ids, fold_save_path, f"valid_dataset_{args.press_threshold}")
            create_and_save_dataset(args, test_match_ids, fold_save_path, f"test_dataset_{args.press_threshold}")
            
            # Save fold info
            fold_info = {
                'train_match_ids': train_match_ids,
                'val_match_ids': val_match_ids,
                'test_match_ids': test_match_ids,
                'fold_idx': fold_idx + 1,
                'total_folds': args.n_folds
            }
            with open(f"{fold_save_path}/fold_info.pkl", "wb") as f:
                pickle.dump(fold_info, f)
            
            print(f"Fold {fold_idx + 1} completed successfully!")
    
    else:
        # Simple train/val/test split version
        print("Generating simple train/validation/test split datasets...")
        
        # Check and normalize ratios
        total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: Ratios sum to {total_ratio}, normalizing to 1.0...")
            args.train_ratio /= total_ratio
            args.valid_ratio /= total_ratio
            args.test_ratio /= total_ratio
        
        # Split dataset
        total_matches = len(match_id_lst)
        train_end = int(total_matches * args.train_ratio)
        valid_end = train_end + int(total_matches * args.valid_ratio)
        
        train_match_ids = match_id_lst[:train_end]
        valid_match_ids = match_id_lst[train_end:valid_end]
        test_match_ids = match_id_lst[valid_end:]
        
        print(f"Train: {len(train_match_ids)}, Valid: {len(valid_match_ids)}, Test: {len(test_match_ids)}")
        
        # Create and save datasets
        create_and_save_dataset(args, train_match_ids, args.save_path, f"train_dataset")
        create_and_save_dataset(args, valid_match_ids, args.save_path, f"valid_dataset")
        create_and_save_dataset(args, test_match_ids, args.save_path, f"test_dataset")
        
        print("Dataset generation completed successfully!")


if __name__ == "__main__":
    import argparse
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate pressing intensity datasets from processed match data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed match data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated datasets")
    parser.add_argument("--exclude_matches", type=str, nargs="+", default=["126319", "153381", "153390", "126285"], help="List of match IDs to exclude from processing")
    parser.add_argument("--high_only", action="store_true", help="Only consider highpress situations")
    parser.add_argument("--num_frames_to_sample", type=int, default=10, help="Number of frames to sample for each sequence")
    parser.add_argument("--cross_validation", action="store_true", help="Use cross validation instead of simple train/val/test split")
    parser.add_argument("--n_folds", type=int, default=6, help="Number of folds for cross validation")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of matches to use for training (when not using CV)")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Ratio of matches to use for validation (when not using CV)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of matches to use for testing (when not using CV)")
    parser.add_argument("--press_threshold", type=float, default=0.9, help="Threshold for pressing intensity")
    
    args = parser.parse_args()
    
    # Check path validity
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist.")
        exit(1)
    
    # Create save path
    os.makedirs(args.save_path, exist_ok=True)
    
    generate_datasets(args)