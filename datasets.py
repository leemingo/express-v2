from kloppy import sportec
from unravel.soccer import KloppyPolarsDataset

import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from pressing_intensity import CustomPressingIntensity
from config import *

coordinates = "secondspectrum"
num_agents = 23
# --- Constants and Configuration ---
H, W = 68, 104  # Grid dimensions (Height, Width)
NUM_FEATURE_CHANNELS = 13 # Number of input channels for SoccerMap
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2 # 52.5
CENTER_Y = PITCH_WIDTH / 2  # 34.0


class PressingSequenceDataset(Dataset):
    def __init__(self, data_path, match_id_lst=None, sequence_length=150, feature_cols=None):
        """
        Initializes the dataset for pressing sequences.

        Args:
            data_path (str): Path to the directory containing match data or preprocessed files.
            match_id_lst (list, optional): List of match IDs to load. Defaults to a predefined list.
            sequence_length (int, optional): Length of the input sequence (X) in frames. Defaults to 125 (5 seconds).
            feature_cols (list, optional): List of feature columns to use. If None, infers default kinematic features.
        """
        self.data_path = data_path
        self.match_id_lst = match_id_lst if match_id_lst is not None else os.listdir(self.data_path)
        self.sequence_length = sequence_length
        # Determine feature columns to use
        self.feature_cols = feature_cols if feature_cols else self._infer_feature_cols()
        # Load and process data to create samples
        self.cols_to_flip = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
        self._load_data()
    
    def _normalize_coordinate_direction(self, df, home_team_id):
        """
        Normalizes the coordinate system within the DataFrame to ensure the home team
        always attacks in a consistent direction (e.g., left-to-right).
        Modifies the DataFrame based on period and initial home team orientation.

        Args:
            df (pd.DataFrame): The tracking DataFrame to process. Must contain
                               'period_id', 'frame_id', 'x', 'y', 'team_id' columns.
                               Will also flip 'vx', 'vy', 'ax', 'ay' if they exist.
            home_team_id (str): The team ID of the home team.

        Returns:
            pd.DataFrame: A DataFrame with normalized coordinate directions.
                          (Does not modify the input df directly, works on a copy).
        """

        df_normalized = df.copy()
        # --- 1. Unify direction for the second half ---
        # Create a mask for the second half (period_id == 2)
        second_half_mask = df_normalized['period_id'] == 2

        # List of kinematic columns (vectors) to flip along with coordinates
        

        # Flip the sign of coordinates and related vectors for second-half data
        for col in self.cols_to_flip:
            if col in df_normalized.columns:
                # Only attempt sign flip if the column is not of object type
                if df_normalized[col].dtype != 'object':
                    df_normalized.loc[second_half_mask, col] = -df_normalized.loc[second_half_mask, col]
                else:
                    # Handle object type columns (e.g., do nothing)
                    pass
        
        # --- 2. Check home team attack direction and flip all if necessary ---
        # Find the first frame ID in the first period (period_id == 1)
        first_period_frames = df_normalized[df_normalized['period_id'] == 1]['frame_id']
        if not first_period_frames.empty:
            first_frame_idx = first_period_frames.unique()[0]
            first_frame_df = df_normalized[(df_normalized['period_id'] == 1) & (df_normalized['frame_id'] == first_frame_idx) & (df_normalized['team_id'] != 'ball')].copy()

            if not first_frame_df.empty and not first_frame_df['x'].isna().all():
                # Find the team ID of the entity with the minimum 'x' value (usually leftmost player/ball)
                try:
                    min_x_team_id = first_frame_df.loc[first_frame_df['x'].idxmin(), 'team_id']

                    # If the leftmost entity is not the home team, assume home team is attacking right-to-left
                    # In this case, flip all coordinates and vectors for the entire match.
                    if min_x_team_id != home_team_id:
                        for col in self.cols_to_flip:
                            if col in df_normalized.columns:
                                if df_normalized[col].dtype != 'object':
                                    # Use .loc to modify the DataFrame slice directly
                                    df_normalized.loc[:, col] = -df_normalized.loc[:, col]
                                else:
                                    pass
                except ValueError:
                    # Handle cases where idxmin() fails (e.g., all 'x' values are NaN)
                    print(f"Warning: Could not determine minimum 'x' value for frame {first_frame_idx}. Skipping orientation check.")
            else:
                print(f"Warning: No valid data found for the first frame ({first_frame_idx}) of period 1. Skipping main orientation check.")
        else:
            print("Warning: No data found for period 1. Skipping main orientation check.")
        return df_normalized

    def _get_event_name(self, x):
        event_name_lst = []
        for val in x:
            if "event_name" in val:
                event_name_lst.append(val['event_name'])
            elif "name" in val:
                event_name_lst.append(val['name'])
        return event_name_lst

    # 각 마지막 frame에 대해, 이후 125/150 frame 내에 ball_owning_team의 값이 변경되었는지 체크하여 label(Y)을 설정
    def _check_change(row, total_df, window=150):
        current_period_id = row['period_id']
        current_frame_id = row['frame_id']

        condition = (
            (total_df['period_id'] == current_period_id) &
            (total_df['frame_id'] >= current_frame_id) &
            (total_df['frame_id'] < current_frame_id + window)
        )
        subset = total_df[condition]

        if subset.empty:
            raise ValueError(
                f"Subset for period_id {current_period_id}, frame_id {current_frame_id} "
                f"(looking ahead {window} frames) is empty. "
                "Cannot determine ball ownership change."
            )
            
        return subset['ball_owning_team_id'].nunique() > 1            
    
    def _check_pressing_success(self, row, event_df, teams_dict):
        possession_gained_events = ['pass', 'dribble', 'recovery', 'interception',
                                    'cross', 'throw_in', 'take_on', 'shot',
                                    'freekick_crossed', 'corner_crossed', 'goalkick'] 
        
        if row['team_id'] == teams_dict['Home']['tID'].unique()[0]:
            pressing_team = teams_dict['Away']['tID'].unique()[0]

        elif row['team_id'] == teams_dict['Away']['tID'].unique()[0]:
            pressing_team = teams_dict['Home']['tID'].unique()[0]

        check_timegap = pd.Timedelta(seconds=5)
        window_events = event_df[
            (event_df['period_id'] == row['period_id']) &
            (event_df['time_seconds'] >= row['timestamp']) &
            (event_df['time_seconds'] <= row['timestamp'] + check_timegap)
        ]
        event_teams = window_events['tID'].unique()

        if pressing_team in event_teams:
            pressing_team_events = window_events[window_events['tID'] == pressing_team]
            for _, row in pressing_team_events.iterrows():
                if row['type_name'] in possession_gained_events:
                    if row['type_name'] in ['dribble', 'recovery', 'interception', 'cross', 'throw_in', 'shot', 'freekick_crossed', 'corner_crossed', 'goalkick']:
                        return True
                    else: # [Pass, take_on]
                        result = row['result_name']
                        if result == 'Successful':
                            return True
            return False
        else:
            return False

    
    def _preprocess_event_df(self, event_df, teams_df):
        event_df['time_seconds'] = (event_df['time_seconds'] / 0.04).round() * 0.04
        event_df['relative_time_seconds'] = (event_df['relative_time_seconds'] / 0.04).round() * 0.04
        event_df['time_seconds'] = pd.to_timedelta(event_df['time_seconds'], unit='s')
        event_df['relative_time_seconds'] = pd.to_timedelta(event_df['relative_time_seconds'], unit='s')
        event_df['player_id'] = event_df['player_id'].astype(str)
        
        teams_df.reset_index(drop=True, inplace=True)
        teams_df['player_code'] = teams_df.apply(lambda row : row['team'][0] + str(row['xID']).zfill(2), axis=1)
        event_df = event_df.merge(
            teams_df,
            how='left',
            left_on='player_id',
            right_on='pID'
        )
        return event_df
    
    def _merge_tracking_pressing_df(self, tracking_df, pressing_df, teams_df):
        total_df = pd.merge(tracking_df, pressing_df, on=['game_id', 'period_id', 'timestamp', 'frame_id'], how='left')
        total_df = total_df[total_df['ball_state'] == 'alive']
        total_df.loc[total_df['period_id'] == 2, 'timestamp'] -= pd.Timedelta(minutes=45)
        total_df = total_df.merge(
            teams_df[['pID', 'player_code']],
            how='left',
            left_on = 'id',
            right_on = 'pID'
        )
        total_df.drop(['pID'], axis=1, inplace=True)

        # Ball id
        nan_mask = pd.isna(total_df['player_code'])
        nan_ids = set(total_df.loc[nan_mask, 'id'].unique())
        expected_ids_ball = {'ball'}
        expected_ids_empty = set()

        if not (nan_ids == expected_ids_ball or nan_ids == expected_ids_empty):
            raise ValueError(f"Found unexpected IDs in rows where player_code is NaN. "
                    f"Expected: {{'ball'}} or {{}}, but found: {nan_ids}"
            )
        
        total_df.loc[nan_mask, 'player_code'] = total_df.loc[nan_mask, 'id']
        return total_df

    
    
    def _load_data(self):
        total_dfs = []
        first_frames_list = []

        all_features_seqs = []
        all_pressintensity_seqs = []
        all_labels = []
        all_presser_ids = []
        all_agent_orders = []
        if os.path.exists(self.data_path):  
            total_dict = {match_id : {} for match_id in self.match_id_lst}
            for match_id in self.match_id_lst:
                
                print(f"Load match_id : {match_id}")
                total_dict[match_id] = {}
                with open(f"{data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
                    match_dict = pickle.load(f)
                tracking_df = match_dict['tracking_df'].copy()
                # Make the direction unified.
                teams_dict = match_dict['teams'].copy()
                home_team = teams_dict['Home'].copy()
                away_team = teams_dict['Away'].copy()
                teams_df = pd.concat([home_team, away_team])
                meta_data = match_dict['meta_data']
                # Make the direction unified.
                
                tracking_df = self._normalize_coordinate_direction(tracking_df, teams_dict['Home']['pID'].iloc[0])
                
                with open(f"{data_path}/{match_id}/{match_id}_presing_intensity.pkl", "rb") as f:
                    pressing_df = pickle.load(f)

                event_df = pd.read_csv(f"{data_path}/{match_id}/valid_events_filtered.csv")                
                # Preprocessing event data.
                event_df = self._preprocess_event_df(event_df, teams_df)
            
                total_df = self._merge_tracking_pressing_df(tracking_df, pressing_df, teams_df)

                total_dict[match_id]['tracking_df'] = total_df
                total_dict[match_id]['event_df'] = event_df
                total_dict[match_id]['meta_data'] = meta_data
                total_dict[match_id]['Home'] = match_dict['teams']['Home']
                total_dict[match_id]['Away'] = match_dict['teams']['Away']

                # ball carrier에 대해 pressing intensity가 0.9보다 큰 경우 pressed_df 구성
                pressed_dict = {}
                ball_carrier_df = total_df[total_df['is_ball_carrier'] == True].copy()
                ball_carrier_df.sort_values(['period_id', 'frame_id'], inplace=True)
                for idx, row in tqdm(ball_carrier_df.iterrows(), desc= "Get Pressing Intensity", miniters=len(ball_carrier_df)//10):
                    if len(np.where(row['rows'] == row['id'])[0]) != 0:
                        pressed_axis = 'rows'
                        presser_axis = 'columns'
                        id_loc = np.where(row[pressed_axis] == row['id'])[0]
                        # 다중 list nested 구조로 되어 있을 수 있으므로 tolist()를 두 번 적용
                        pressing_values = row['probability_to_intercept'][id_loc].tolist()[0].tolist()
                    elif len(np.where(row['columns'] == row['id'])[0]) != 0:
                        pressed_axis = 'columns'
                        presser_axis = 'rows'
                        id_loc = np.where(row[pressed_axis] == row['id'])[0]
                        pressing_values = [x[id_loc] for x in row['probability_to_intercept']]
                    else:
                        continue
                    if max(pressing_values) > 0.9:
                        pressed_dict[idx] = {}
                        pressed_dict[idx]['pressing_value'] = max(pressing_values)
                        max_idx = pressing_values.index(max(pressing_values))
                        pressed_dict[idx]['pressing_player'] = row[presser_axis][max_idx]
                pressed_df = ball_carrier_df.loc[list(pressed_dict.keys())].copy()
                pressed_df['pressing_values'] = [d['pressing_value'] for d in pressed_dict.values()]
                pressed_df['pressing_player'] = [d.get('pressing_player') for d in pressed_dict.values()]
                
                # Period 나눠서
                period_list = []
                for period_id in pressed_df['period_id'].unique():
                    period_df = pressed_df[pressed_df['period_id']==period_id].copy()
                    # frame_id 차이가 125 프레임 이상인 경우 새로운 시퀀스로 판단 (즉, 연속된 pressed 행이 아닐 경우)
                    period_df['frame_diff'] = period_df['frame_id'].diff()
                    period_df['sequence_id'] = (period_df['frame_diff'] > 125).cumsum()

                    # 각 시퀀스별 첫 번째 frame을 기준으로 X와 Y를 설정하기 위해 첫번째 frame 정보를 구함
                    first_frames = period_df.groupby('sequence_id', as_index=False)[['timestamp', 'period_id', 'frame_id', 'id', 'team_id', 'pressing_player']].first()

                    # total_df 에서 필요한 컬럼만 추출
                    lookup = total_df[['period_id', 'frame_id', 'id', 'v']]

                    # pressing_player 컬럼은 total_df.id 와 매칭되어 있으므로,
                    # 컬럼명을 맞춰서 merge
                    first_frames = first_frames.merge(
                        lookup.rename(columns={'id': 'pressing_player'}),
                        on=['period_id', 'frame_id', 'pressing_player'],
                        how='left'
                    )

                    # 압박하는 선수의 속도가 2.0 m/s 인 경우만 압박으로 간주
                    first_frames = first_frames[first_frames['v'] >= 2.0]
                    period_list.append(first_frames)

                first_frames_df = pd.concat(period_list, axis=0, ignore_index=True)    
                first_frames_df['ball_ownership_changed'] = first_frames_df.apply(self._check_pressing_success, axis=1, event_df=event_df, teams_dict=teams_dict) # window=150은 예시    

                first_frames_list.append(first_frames_df)

                for _, row in tqdm(first_frames_df.iterrows(), desc= "Get Samples", miniters=len(first_frames)//10):
                    try:
                        period_id = row['period_id']
                        frame_id = row['frame_id']
                        timestamp = row['timestamp']
                        label = int(row['ball_ownership_changed'])
                        pressing_player = row['pressing_player']
                        window_event_df = event_df[
                            (event_df['period_id'] == period_id) &
                            (event_df['time_seconds'] >= timestamp - pd.Timedelta(seconds=5)) &
                            (event_df['time_seconds'] <= timestamp)
                        ]
                        
                        timestamps_list = window_event_df['time_seconds'].unique().tolist() + [timestamp]
                        X_slice = total_df[(total_df['period_id'] == period_id) & (total_df['timestamp'].isin(timestamps_list))].copy()
                        
                        # If slice data doesn't have immediately preceding data, continue. (Data Problem)
                        # Always press left -> right
                        # If pressed players' team is home, flip
                        if X_slice.loc[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['team_id'].iloc[0] == match_dict['teams']['Home']['tID'].iloc[0]:
                            for col in self.cols_to_flip:
                                X_slice.loc[:, col] = -X_slice.loc[:, col]

                        agents_rows = X_slice[pd.isna(X_slice['rows'])==False]['rows'].values[0] # Home team
                        agents_cols = X_slice[pd.isna(X_slice['columns'])==False]['columns'].values[0] #Away team
                        agents_order = agents_rows.tolist() + agents_cols.tolist()
                        
                        # Ensure the player IDs are consistent and match num_agents (23 players)
                        num_unique_agents = X_slice['id'].nunique()
                        frame_lst = X_slice['frame_id'].unique()
                        if num_unique_agents < num_agents:
                            # Find the missing players
                            num_missing_agents = num_agents - num_unique_agents
                        
                            # Add missing player rows with zero values for each frame in X_slice
                            missing_rows = []
                            for missing_player in range(num_missing_agents):
                                for frame in frame_lst:
                                    missing_row = {col: 0 for col in X_slice.columns}  # Fill all columns with 0
                                    missing_row['id'] = f"Missing_{missing_player}"  # Set the 'id' to the missing player's id
                                    missing_row['frame_id'] = frame  # Set the frame_id for the current frame in the sequence
                                    missing_rows.append(missing_row)
                                agents_order.append(f"Missing_{missing_player}") 

                            # Create a DataFrame for the missing rows and append to the slice
                            missing_df = pd.DataFrame(missing_rows)
                            X_slice = pd.concat([X_slice, missing_df], ignore_index=True)
                        
                        agents_order.append('ball')

                        X_slice.loc[:, 'id'] = pd.Categorical(X_slice['id'], categories=agents_order, ordered=True)
                        # Sort the players by their ID to maintain a consistent order
                        X_slice = X_slice.sort_values(by=['frame_id', 'id'])
                
                        # Get the features
                        x_tensor = torch.tensor(X_slice[self.feature_cols].values, dtype=torch.float32)
                        X_slice_pressing = X_slice[X_slice['is_ball_carrier']==True]['probability_to_intercept']
                        X_slice_pressing = X_slice_pressing[pd.isna(X_slice_pressing)==False]
                        pressing_intensity_tensor = torch.tensor(np.stack(X_slice_pressing.map(lambda x: np.stack(x)).values), dtype=torch.float32)
                        x_tensor = x_tensor.reshape(-1, num_agents, len(self.feature_cols))
                        y_tensor = torch.tensor(label, dtype=torch.long)
                        
                        # Debug 
                        if x_tensor.isnan().any():
                            print("Find Nan", match_id, period_id, frame_id)
                            continue
                        
                        all_features_seqs.append(x_tensor)
                        all_pressintensity_seqs.append(pressing_intensity_tensor)
                        all_labels.append(y_tensor)
                        all_presser_ids.append(pressing_player)    
                        all_agent_orders.append(agents_order)
                    
                    except Exception as e:
                        period_id = row['period_id']
                        frame_id = row['frame_id']
                        print(f"Error | Match ID : {match_id} | Period {period_id} | Frame ID {frame_id} : {e}")
                        continue
                self.features_seqs = all_features_seqs
                
                self.pressintensity_seqs = all_pressintensity_seqs
                self.labels = all_labels
                self.presser_ids = all_presser_ids
                self.agent_orders = all_agent_orders

    def _infer_feature_cols(self):
        ignore = ['game_id', 'period_id', 'timestamp', 'ball_owning_team_id']
        # return [col for col in self.total_df.columns if col not in ignore and self.total_df[col].dtype != 'O']
        return ['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a']

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for the sample at the given index.
        """
        # Retrieve data from the stored lists using the index
        return {
            'features': self.features_seqs[idx],       # Shape: [SeqLen, Agents, Features]
            'pressing_intensity': self.pressintensity_seqs[idx],         # Shape: [SeqLen, ?, ?] (Adjust shape based on data)
            'label': self.labels[idx],                   # Shape: [1] or scalar
            'presser_id': self.presser_ids[idx],         # String (Player ID)
            'agent_order': self.agent_orders[idx]        # List of Strings (Agent IDs in order)
        }
    
    def __len__(self):
        # 생성된 샘플의 개수 반환
        return len(self.features_seqs)


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104), num_features=NUM_FEATURE_CHANNELS):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define goal coordinates (adjust if necessary based on normalized space)
        # Example: Opponent's goal at (+52.5, 0) in center-origin coordinates
        self.goal_coo = torch.tensor([[CENTER_X, 0.0]], device=self.device)

    def _get_cell_indexes(self, x: torch.Tensor, y: torch.Tensor):
        """Converts continuous coordinates (center-origin) to grid cell indices."""
        x = x.float()
        y = y.float()
        # Shift the origin from the pitch center to the bottom-left corner:
        #   x: [-52.5, 52.5] -> [0, 105]
        #   y: [-34.0,  34.0] -> [0,  68]
        x_shifted = x + 52.5
        y_shifted = y + 34.0

        # Normalize into [0, 1]
        x_norm = x_shifted / PITCH_LENGTH
        y_norm = y_shifted /  PITCH_WIDTH

        # Scale to bin coordinates
        x_cont = x_norm * self.x_bins
        y_cont = y_norm * self.y_bins
        
        # Clamp into valid index range [0, bins-1]
        x_clamped = torch.clamp(x_cont, min=0, max=self.x_bins - 1)
        y_clamped = torch.clamp(y_cont, min=0, max=self.y_bins - 1)

        # Convert to integer bin indices
        # x_bin = x_clamped.to(torch.int64).to(torch.uint8)
        # y_bin = y_clamped.to(torch.int64).to(torch.uint8)
        x_bin = x_clamped.to(torch.long)
        y_bin = y_clamped.to(torch.long)
    
        return x_bin, y_bin
    
    def __call__(self, sample):
        """
        Processes data for frames to generate SoccerMap input.

        Args:
            frame_data (dict): Dictionary containing data for one frame, including:
                'x_feat': Tensor [23, 8] with kinematic features.
                'agents_order': List of 23 agent IDs.
                'presser_id': ID of the pressing player.
                'home_team_ids': Set/List of home player IDs.
                'away_team_ids': Set/List of away player IDs.
                'pressing_outcome': Binary label (1 for success, 0 for failure) - YOU NEED TO DEFINE THIS.
                'relevant_location': Tuple (x, y) of the location for the loss mask - YOU NEED TO DEFINE THIS.

        Returns:
            tuple: (spatial_map, mask, label) ready for the model, or None if processing fails.
        """

        # Output
        home_indices = slice(0, 11)
        away_indices = slice(11, 22)
        goal_coo = torch.tensor([[52.5, 0]])

        x_tensor = sample['features'][-1:, ...]
        num_seqs = x_tensor.shape[0]
        matrix = np.zeros((self.num_features * num_seqs, self.y_bins, self.x_bins))
        press_intensity = sample['pressing_intensity']
        target = sample['label']
        presser_id = sample['presser_id']
        agents_order = sample['agent_order']
        presser_idx = agents_order.index(presser_id)
        home_team_ids = agents_order[:11]
        away_team_ids = agents_order[11:22]

        for i in range(num_seqs):
            x_feat = x_tensor[i]
            # x_feat = x_tensor[-1]

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

            # CH 4: Locations of pressed teams
            matrix[3 + i * self.num_features, y_bin_pressed, x_bin_pressed] = 1

            # Ch 5: vx of pressed teams
            matrix[4 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 2:3]

            # Ch 6: vy of pressed teams
            matrix[5 + i * self.num_features, y_bin_pressed, x_bin_pressed] = x_feat[pressed_indices, 3:4]

            # CH 7: Distance to ball
            y_coords = torch.arange(0.5, self.y_bins, device=x_feat.device) # Shape: [y_bins]
            x_coords = torch.arange(0.5, self.x_bins, device=x_feat.device)

            # Create 2D grid coordinate tensors
            # 'ij' indexing: yy changes along dim 0, xx changes along dim 1
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            # yy shape: [y_bins, x_bins], xx shape: [y_bins, x_bins]

            x0_ball, y0_ball = self._get_cell_indexes(x_feat[22, :1], x_feat[22, 1:2])
            x0_ball_center = x0_ball.float() + 0.5
            y0_ball_center = x0_ball.float() + 0.5

            ball_distance = torch.sqrt((xx - x0_ball_center)**2 + (yy - y0_ball_center)**2)
            matrix[6 + i * self.num_features, : , :] = ball_distance.numpy()

            # CH 8: Distance to goal
            x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
            x0_goal_center = x0_goal.float() + 0.5
            y0_goal_center = y0_goal.float() + 0.5

            goal_distance = torch.sqrt((xx - x0_goal_center)**2 + (yy - y0_goal_center)**2)
            matrix[7 + i * self.num_features, : , :] = goal_distance.numpy()

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
            matrix[8 + i * self.num_features, : , :] = cosine_angle.numpy()

            # CH 10: Sine of the angle between the ball and goal
            sine_angle = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (norm_a + norm_b + 1e-8)
            sine_angle = torch.clamp(sine_angle, min=-1.0, max=1.0)
            matrix[9 + i * self.num_features, : , :] = sine_angle.numpy()

            # CH 11: Angle (in radians) to the goal location
            vector_y = y0_goal_center - coords[..., 1] # Shape: [H, W]
            vector_x = x0_goal_center - coords[..., 0] # Shape: [H, W]
            angle_rad = torch.abs(torch.arctan2(vector_y, vector_x))
            matrix[10 + i * self.num_features, : , :] = angle_rad.numpy()

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
            matrix[11 + i * self.num_features, y_bin_tm, x_bin_tm] = cosine_angle.numpy()
            matrix[12 + i * self.num_features, y_bin_tm, x_bin_tm] = sine_angle.numpy()

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


# --- Dataset Class (Loads Pickled Sequence Data, Applies Transform) ---
class SoccerMapInputDataset(Dataset): # Renamed from PressingFrameDataset
    def __init__(self, pickled_dataset_path):
        """Loads data from the pickled PressingSequenceDataset object."""
        print(f"Loading dataset from {pickled_dataset_path}...")
        try:
            with open(pickled_dataset_path, "rb") as f:
                # Load the dictionary saved by PressingSequenceDataset
                self.loaded_data = pickle.load(f)

            # Initialize transformer here, used in __getitem__
            self.transform = ToSoccerMapTensor()
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {pickled_dataset_path}")
        except Exception as e:
            print(f"Error loading pickled dataset: {e}")


    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        """Retrieves sequence info and transforms it into a spatial map & label."""
        if idx >= len(self.loaded_data):
            raise IndexError("Index out of bounds")
        
        spatial_map, label = self.transform(self.loaded_data[idx])
        return spatial_map, label

class exPressInputDataset(Dataset):
    def __init__(self, pickled_dataset_path):
        """Loads data from the pickled PressingSequenceDataset object."""
        print(f"Loading dataset from {pickled_dataset_path}...")
        try:
            with open(pickled_dataset_path, "rb") as f:
                # Load the dictionary saved by PressingSequenceDataset
                self.loaded_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {pickled_dataset_path}")
        except Exception as e:
            print(f"Error loading pickled dataset: {e}")

        # Normalization
        self.feature_min_vals = torch.Tensor(FEAT_MIN)
        self.feature_max_vals = torch.Tensor(FEAT_MAX)
        
        # __getitem__에서 브로드캐스팅을 위해 shape 변경: [1, 1, NumFeatures]
        self.min_vals = self.feature_min_vals.reshape(1, 1, -1)
        self.max_vals = self.feature_max_vals.reshape(1, 1, -1)
        
        self.feature_ranges = self.max_vals - self.min_vals
         # Prevent division by zero for constant features
        self.feature_ranges[self.feature_ranges == 0] = 1.0 

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.loaded_data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing sequence data for the sample at the given index.
        Graph construction and transformation happen inside the model.
        """
        if self.loaded_data is None or idx >= len(self.loaded_data):
            raise IndexError("Index out of bounds or data not loaded")
        features = self.loaded_data[idx]['features'][-1:, ...]
        # Normalize
        features = (features - self.min_vals) / self.feature_ranges

        pressing_intensity = self.loaded_data[idx]['pressing_intensity'][-1:, ...]
        _, cur_players1, cur_players2 = pressing_intensity.shape
        num_team_players = 11

        if cur_players1 != num_team_players: #11
            pad_d1 = num_team_players - cur_players1
            # Pad second to last dimension (bottom)
            pressing_intensity = F.pad(pressing_intensity, (0, 0, 0, pad_d1), mode='constant', value=0)
        if cur_players2 != num_team_players:
            pad_d2 = num_team_players - cur_players2
            # Pad last dimension (right side)
            pressing_intensity = F.pad(pressing_intensity, (0, pad_d2), mode='constant', value=0)

        #  # --- 'features'에 Min-Max 정규화 적용 ---
        # normalized_features = features.float() # 기본값은 원본 (정규화 파라미터가 없는 경우)
        # if self.feature_min_vals is not None and self.feature_max_vals is not None:
        #     # 정규화: (X - X_min) / (X_max - X_min)
        #     # self.feature_ranges가 0인 경우 1.0으로 설정했으므로, 해당 피처는 (X - X_min) / 1.0 = 0 이 됩니다 (X==X_min 가정).
        #     normalized_features = (features.float() - self.min_vals_bcast) / (self.feature_ranges + 1e-8) # 1e-8은 매우 작은 범위에 대한 안정성 추가
        #     # 값의 범위를 [0, 1]로 클램핑할 수도 있습니다 (정밀도 문제 등으로 약간 벗어날 수 있음).
        #     # normalized_features = torch.clamp(normalized_features, 0, 1)
        # # --- 정규화 끝 ---
        
        label = self.loaded_data[idx]['label']
        presser_id = self.loaded_data[idx]['presser_id']
        agent_order = self.loaded_data[idx]['agent_order']

        return {
                'features': features.float(),         # Shape: [T, A, F] e.g., [125, 23, 8]
                'pressing_intensity': pressing_intensity.float(), # Shape: [T, 11, 11]
                'label': label.float(),             # Shape: [1] or scalar
                'presser_id': presser_id,           # String
                'agent_order': agent_order          # List
            }


if __name__ == "__main__":
    data_path = "/data/MHL/bepro/processed"
    save_path = "/data/MHL/pressing-intensity"
    os.makedirs(save_path, exist_ok=True)
    match_id_lst = os.listdir(data_path)
    train_match_id_lst = match_id_lst[:-4]
    test_match_id_lst = match_id_lst[-4:]
    train_dataset = PressingSequenceDataset(data_path, match_id_lst=train_match_id_lst)
    test_dataset = PressingSequenceDataset(data_path, match_id_lst=test_match_id_lst)
    with open(f"{save_path}/train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(f"{save_path}/test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    print("Done")
    