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

import scipy.signal as signal
from scipy.ndimage import shift

coordinates = "secondspectrum"
num_agents = 23
num_team_players = 11
# --- Constants and Configuration ---
H, W = 68, 104  # Grid dimensions (Height, Width)
NUM_FEATURE_CHANNELS = 13 # Number of input channels for SoccerMap
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2 # 52.5
CENTER_Y = PITCH_WIDTH / 2  # 34.0

def calc_single_agent_velocities(traces: pd.DataFrame, remove_outliers=True, smoothing=True):
    if remove_outliers:
        MAX_SPEED = 12
        MAX_ACCEL = 8

    if smoothing:
        W_LEN = 9
        P_ORDER = 2

    x = traces["x"]
    y = traces["y"]

    if smoothing:
        x = pd.Series(signal.savgol_filter(x, window_length=21, polyorder=P_ORDER), index=x.index)
        y = pd.Series(signal.savgol_filter(y, window_length=21, polyorder=P_ORDER), index=y.index)

    fps = 25
    vx = np.diff(x.values, prepend=x.iloc[0]) / (1 / fps)
    vy = np.diff(y.values, prepend=y.iloc[0]) / (1 / fps)

    if remove_outliers:
        speeds = np.sqrt(vx**2 + vy**2)
        is_speed_outlier = speeds > MAX_SPEED
        is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / (1 / fps)) > MAX_ACCEL
        is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
        vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
        vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

    if smoothing:
        vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
        vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

    speeds = np.sqrt(vx**2 + vy**2)

    ax = np.diff(vx, prepend=vx[0]) / (1 / fps)
    ay = np.diff(vy, prepend=vy[0]) / (1 / fps)
    
    if remove_outliers:
        accel = np.sqrt(ax**2 + ay**2)
        is_accel_outlier = accel > MAX_ACCEL
        is_outlier = is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
        ax = pd.Series(np.where(is_outlier, np.nan, ax)).interpolate(limit_direction="both").values
        ay = pd.Series(np.where(is_outlier, np.nan, ay)).interpolate(limit_direction="both").values

    if smoothing:
        ax = signal.savgol_filter(ax, window_length=W_LEN, polyorder=P_ORDER)
        ay = signal.savgol_filter(ay, window_length=W_LEN, polyorder=P_ORDER)

    accels = np.sqrt(ax**2 + ay**2)

    feature_cols = ['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a']
    traces.loc[x.index, feature_cols] = np.stack([x, y, vx, vy, speeds, ax, ay, accels]).round(6).T

    return traces


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

    def _generate_features(self, frame_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get additional features from tracking data

        Args:
            frame_df (pd.DataFrame): 단일 프레임에 대한 선수 및 공 데이터.
                                    'id', 'team_id', 'position_name', 'is_ball_carrier',
                                    'x', 'y', 'vx', 'vy' 등의 컬럼이 있어야 합니다.

        Returns:
            pd.DataFrame: 새로운 특징들이 추가된 데이터프레임.
        """
        # 원본 수정을 방지하기 위해 복사본 사용
        df = frame_df.copy()

        # --- 1. 주요 객체 식별 ---
        ball_row = df[df['id'] == 'ball']
        ball_carrier_row = df[df['is_ball_carrier'] == True]

        # 공 또는 공 소유자가 없는 프레임은 처리하지 않음
        if ball_row.empty or ball_carrier_row.empty:
            # 필요한 모든 특징 컬럼을 0으로 채워서 반환 (에러 방지)
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
        
        # 공격 방향을 왼쪽에서 오른쪽으로 가정하고, 골대 위치를 (52.5, 0)으로 설정
        # (사전 단계에서 좌표계 통일이 되었다고 가정)
        goal_pos = np.array([52.5, 0.0])

        # --- 2. 특징 계산 ---

        # A. 이진 속성 (Binary attributes)
        # 'is_ball_carrier'는 이미 존재한다고 가정
        df['is_teammate'] = (df['team_id'] == ball_carrier['team_id']).astype(float)
        df['is_goalkeeper'] = (df['position_name'] == 'GK').astype(float)
        # is_goalpost는 실제 노드가 아니므로 특징으로 추가하지 않음

        # C. 벡터 계산을 위한 데이터 준비 (Numpy 배열)
        player_positions = df[['x', 'y']].values.astype(np.float64)
        player_velocities = df[['vx', 'vy']].values.astype(np.float64)
        ball_position = ball[['x', 'y']].values.astype(np.float64)
        carrier_velocity = ball_carrier[['vx', 'vy']].values.astype(np.float64)

        # D. 골대 기준 상대적 특징 (Relative to goalpost)
        vector_to_goal = goal_pos - player_positions
        df['distance_to_goal'] = np.linalg.norm(vector_to_goal, axis=1)
        angle_to_goal_rad = np.arctan2(vector_to_goal[:, 1], vector_to_goal[:, 0])
        df['sin_angle_to_goal'] = np.sin(angle_to_goal_rad)
        df['cos_angle_to_goal'] = np.cos(angle_to_goal_rad)

        # E. 공 기준 상대적 특징 (Relative to ball)
        vector_to_ball = ball_position - player_positions
        df['distance_to_ball'] = np.linalg.norm(vector_to_ball, axis=1)
        angle_to_ball_rad = np.arctan2(vector_to_ball[:, 1], vector_to_ball[:, 0])
        df['sin_angle_to_ball'] = np.sin(angle_to_ball_rad)
        df['cos_angle_to_ball'] = np.cos(angle_to_ball_rad)

        # F. 선수 속도 벡터와 공 소유자 속도 벡터 간의 각도
        dot_product = np.sum(player_velocities * carrier_velocity, axis=1)
        norm_player = np.linalg.norm(player_velocities, axis=1)
        norm_carrier = np.linalg.norm(carrier_velocity)
        
        # 0으로 나누는 것을 방지하기 위한 작은 값(epsilon)
        denominator = (norm_player * norm_carrier) + 1e-8
        
        df['cos_velocity_angle'] = np.clip(dot_product / denominator, -1.0, 1.0)
        
        # 2D 벡터의 외적(cross product)을 이용한 사인 값 계산
        cross_product = player_velocities[:, 0] * carrier_velocity[1] - player_velocities[:, 1] * carrier_velocity[0]
        df['sin_velocity_angle'] = np.clip(cross_product / denominator, -1.0, 1.0)
        
        return df
    
    def _load_data(self):
        total_dfs = []
        first_frames_list = []

        all_features_seqs = []
        all_pressintensity_seqs = []
        all_labels = []
        all_pressed_ids = []
        all_presser_ids = []
        all_agent_orders = []
        all_match_infos = []
        if os.path.exists(self.data_path):  
            total_dict = {match_id : {} for match_id in self.match_id_lst}
            for match_id in self.match_id_lst:
                
                print(f"Load match_id : {match_id}")
                total_dict[match_id] = {}
                with open(f"{data_path}/{match_id}/{match_id}_processed_dict.pkl", "rb") as f:
                    match_dict = pickle.load(f)
                tracking_df = match_dict['tracking_df'].copy()                
                teams_dict = match_dict['teams'].copy()
                home_team = teams_dict['Home'].copy()
                away_team = teams_dict['Away'].copy()
                teams_df = pd.concat([home_team, away_team])
                meta_data = match_dict['meta_data']
                # Make the direction unified.
                
                tracking_df = tracking_df.drop(columns=["vx", "vy", "ax", "ay", "v", "a"])         
                tracking_df = tracking_df.groupby(['game_id', 'period_id', 'id']).apply(
                    calc_single_agent_velocities, include_groups=False
                ).reset_index(drop=False).drop(columns=['level_3'])
                tracking_df = tracking_df.sort_values(['game_id', 'period_id', 'frame_id']).reset_index(drop=True)

                tracking_df = self._normalize_coordinate_direction(tracking_df, teams_dict['Home']['tID'].iloc[0])
                
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
                        pressed_player = row['id']
                        pressing_player = row['pressing_player']
                        window_event_df = event_df[
                            (event_df['period_id'] == period_id) &
                            (event_df['time_seconds'] >= timestamp - pd.Timedelta(seconds=5)) &
                            (event_df['time_seconds'] <= timestamp)
                        ]
                        
                        timestamps_list = window_event_df['time_seconds'].unique().tolist() + [timestamp]
                        X_slice = total_df[(total_df['period_id'] == period_id) & (total_df['timestamp'].isin(timestamps_list))].copy()
                        
                        # Always press left -> right
                        # If pressed players' team is home, flip
                        if X_slice.loc[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['team_id'].iloc[0] == match_dict['teams']['Home']['tID'].iloc[0]:
                            for col in self.cols_to_flip:
                                X_slice.loc[:, col] = -X_slice.loc[:, col]

                        # Get Features
                        X_slice = X_slice.set_index('frame_id').groupby('frame_id', group_keys=False).apply(self._generate_features)
                        X_slice.reset_index(inplace=True)

                        agents_rows = X_slice[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['rows'].values[0].tolist() # Home team
                        agents_cols = X_slice[(X_slice['frame_id']==frame_id) & (X_slice['is_ball_carrier']==True)]['columns'].values[0].tolist() #Away team
                        missing_cnt = 0
                        num_missing_rows = num_team_players - len(agents_rows)
                        if num_missing_rows > 0:
                            for i in range(num_missing_rows):
                                agents_rows.append(f"Missing_{missing_cnt}")
                                missing_cnt += 1

                        num_missing_cols = num_team_players - len(agents_cols)
                        if num_missing_cols > 0:
                            for i in range(num_missing_cols):
                                agents_cols.append(f"Missing_{missing_cnt}")
                                missing_cnt += 1
                        agents_order = agents_rows + agents_cols
                        
                        # Ensure the player IDs are consistent and match num_agents (22 players + 1 ball)
                        num_unique_agents = X_slice['id'].nunique()
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

                        X_slice.loc[:, 'id'] = pd.Categorical(X_slice['id'], categories=agents_order, ordered=True)
                        # Sort the players by their ID to maintain a consistent order
                        X_slice = X_slice.sort_values(by=['frame_id', 'id'])
                
                        # Get the features
                        x_tensor = torch.tensor(X_slice[self.feature_cols].values, dtype=torch.float32)
                        X_slice_pressing = X_slice[X_slice['is_ball_carrier']==True]['probability_to_intercept']
                        X_slice_pressing = X_slice_pressing.dropna()
                        pressing_intensity_tensor = torch.tensor(np.stack(X_slice_pressing.map(lambda x: np.stack(x)).values), dtype=torch.float32)
                        _, h, w = pressing_intensity_tensor.shape
                        pad_h = num_team_players - h
                        pad_w = num_team_players - w
                        pressing_intensity_tensor = F.pad(pressing_intensity_tensor, (0, pad_w, 0, pad_h), "constant", 0)

                        x_tensor = x_tensor.reshape(-1, num_agents, len(self.feature_cols))
                        y_tensor = torch.tensor(label, dtype=torch.long)
                        
                        # Debug 
                        if x_tensor.isnan().any():
                            print("Find Nan", match_id, period_id, frame_id)
                            continue
                        match_info = f"{match_id}-{period_id}-{frame_id}"

                        all_features_seqs.append(x_tensor)
                        all_pressintensity_seqs.append(pressing_intensity_tensor)
                        all_labels.append(y_tensor)
                        all_pressed_ids.append(pressed_player)
                        all_presser_ids.append(pressing_player)    
                        all_agent_orders.append(agents_order)
                        all_match_infos.append(match_info)
                    
                    except Exception as e:
                        period_id = row['period_id']
                        frame_id = row['frame_id']
                        print(f"Error | Match ID : {match_id} | Period {period_id} | Frame ID {frame_id} : {e}")
                        continue
                self.features_seqs = all_features_seqs
                
                self.pressintensity_seqs = all_pressintensity_seqs
                self.labels = all_labels
                self.pressed_ids = all_pressed_ids
                self.presser_ids = all_presser_ids
                self.agent_orders = all_agent_orders
                self.match_infos = all_match_infos

    def _infer_feature_cols(self):
        ignore = ['game_id', 'period_id', 'timestamp', 'ball_owning_team_id']
        # return [col for col in self.total_df.columns if col not in ignore and self.total_df[col].dtype != 'O']
        return ['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a', 
                'is_teammate', 'is_goalkeeper', 'distance_to_goal', 'sin_angle_to_goal',
                'cos_angle_to_goal', 'distance_to_ball', 'sin_angle_to_ball',
                'cos_angle_to_ball', 'cos_velocity_angle', 'sin_velocity_angle'
            ]

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for the sample at the given index.
        """
        # Retrieve data from the stored lists using the index
        return {
            'features': self.features_seqs[idx],       # Shape: [SeqLen, Agents, Features]
            'pressing_intensity': self.pressintensity_seqs[idx],         # Shape: [SeqLen, ?, ?] (Adjust shape based on data)
            'label': self.labels[idx],                   # Shape: [1] or scalar
            'pressed_id': self.pressed_ids[idx],         # String (Player ID)
            'presser_id': self.presser_ids[idx],         # String (Player ID)
            'agent_order': self.agent_orders[idx],        # List of Strings (Agent IDs in order)
            'match_info': self.match_infos[idx]
        }

    def __setitem__(self, idx, data):
        if idx < 0 or idx >= len(self.features_seqs):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.features_seqs)}")

        if 'features' in data:
            self.features_seqs[idx] = data['features']
            
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

        x_tensor = sample['features']
        num_seqs = x_tensor.shape[0]
        num_seqs = 30
        matrix = np.zeros((self.num_features * num_seqs, self.y_bins, self.x_bins))
        press_intensity = sample['pressing_intensity']
        target = sample['label']
        presser_id = sample['presser_id']
        agents_order = sample['agent_order']
        presser_idx = agents_order.index(presser_id)
        home_team_ids = agents_order[:11]
        away_team_ids = agents_order[11:22]

        for i in range(num_seqs):
            x_feat = x_tensor[120+i]
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
    def __init__(self, pickled_dataset_path, feature_min_vals=None, feature_max_vals=None):
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
        num_features = self.loaded_data[0]['features'].shape[2]
        if feature_min_vals is None and feature_max_vals is None:
             # Compute min/max across the entire dataset
            current_min_vals = torch.full((num_features,), float('inf'), dtype=torch.float32)
            current_max_vals = torch.full((num_features,), float('-inf'), dtype=torch.float32)
        
            for idx in range(len(self.loaded_data)):
                features_tensor = self.loaded_data[0]['features']
                min_in_tensor = torch.min(torch.min(features_tensor, dim=0).values, dim=0).values
                max_in_tensor = torch.max(torch.max(features_tensor, dim=0).values, dim=0).values

                current_min_vals = torch.minimum(current_min_vals, min_in_tensor)
                current_max_vals = torch.maximum(current_max_vals, max_in_tensor)

            self.feature_min_vals = current_min_vals
            self.feature_max_vals = current_max_vals
        else:
            self.feature_min_vals = feature_min_vals
            self.feature_max_vals = feature_max_vals
        
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
        features = self.loaded_data[idx]['features']
        # Normalize
        features = (features - self.min_vals) / self.feature_ranges

        pressing_intensity = self.loaded_data[idx]['pressing_intensity']
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
                # 'features': normalized_features.float(),         # Shape: [T, A, F] e.g., [125, 23, 8]
                'pressing_intensity': pressing_intensity.float(), # Shape: [T, 11, 11]
                'label': label.float(),             # Shape: [1] or scalar
                'presser_id': presser_id,           # String
                'agent_order': agent_order          # List
            }
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data/bepro/processed")
    save_path = os.path.join(current_dir, "data/bepro/pressing_intensity")
    os.makedirs(save_path, exist_ok=True)

    exclude_ids = ['126319', '153381', '153390', '126285']
    valid_ids = ['126476', '153364', '153373']
    test_ids = ['153379', '153385', '153387']

    match_id_lst = sorted([
        match_id for match_id in os.listdir(data_path)
        if match_id not in exclude_ids
    ])

    # 이후 로직은 동일하게 유지
    train_dataset = PressingSequenceDataset(data_path, match_id_lst=match_id_lst[:30])    
    with open(f"{save_path}/train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    valid_dataset = PressingSequenceDataset(data_path, match_id_lst=valid_ids)    
    with open(f"{save_path}/valid_dataset.pkl", "wb") as f:
        pickle.dump(valid_dataset, f)

    test_dataset = PressingSequenceDataset(data_path, match_id_lst=test_ids)
    with open(f"{save_path}/test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

    print("Done")
