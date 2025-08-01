
import datatools.pitch_control as pc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
import fire
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

#Visualization
os.chdir('..')
import config as C
current_dir = os.path.dirname(__file__)

FEATURE_FUNCTIONS = []
PARAMS = pc.default_model_params()

def register_feature(func):
    FEATURE_FUNCTIONS.append(func)
    return func

@register_feature
def add_distance_ball_to_center_of_goal(df):
    """
    (수정 버전) .map()을 사용하여 안정적으로 공의 위치를 모든 행에 전파합니다.
    """
    df = df.copy()
    
    # 1. 'frame_id'를 키로, 공의 'x', 'y' 좌표를 값으로 하는 맵(Series)을 만듭니다.
    ball_x_map = df.loc[df['id'] == 'ball'].set_index('frame_id')['x']
    ball_y_map = df.loc[df['id'] == 'ball'].set_index('frame_id')['y']

    # 2. map을 사용해 각 행의 frame_id에 맞는 공의 좌표를 새 임시 컬럼에 할당합니다.
    df['ball_x'] = df['frame_id'].map(ball_x_map)
    df['ball_y'] = df['frame_id'].map(ball_y_map)

    # 3. 벡터 연산을 통해 거리를 계산합니다.
    dx = C.PITCH_X_MIN - df['ball_x']
    dy = df['ball_y']
    df['distance_ball_to_goal'] = np.sqrt(dx**2 + dy**2)
    
    # 4. 사용이 끝난 임시 컬럼을 제거합니다.
    df = df.drop(columns=['ball_x', 'ball_y'])

    return df
    
@register_feature
def add_distance_ball_to_sideline(df):
    """
    각 행마다 공과 가장 가까운 사이드라인(측면 라인) 사이의 거리를 계산하여
    'distance_ball_sideline' 컬럼으로 추가합니다.

    Args:
        df (pd.DataFrame): 'ball_y' 컬럼이 포함된 DataFrame. (보통은 공 위치 병합 후 사용)
    Returns:
        pd.DataFrame: 'distance_ball_sideline' 컬럼이 추가된 DataFrame
    """
    ball_positions = df[df['id'] == 'ball'][['frame_id', 'x', 'y']]
    ball_positions = ball_positions.rename(columns={'x': 'ball_x', 'y': 'ball_y'})
    df_merged = pd.merge(df, ball_positions, on='frame_id', how='left')

    # 사이드라인까지의 최소 거리 계산
    ball_sideline = np.minimum(
        df_merged['ball_y'] - C.PITCH_Y_MIN,
        C.PITCH_Y_MAX - df_merged['ball_y']
    )

    df_merged['distance_ball_to_sideline'] = ball_sideline
    df_merged = df_merged.drop(columns=['ball_x', 'ball_y'])

    return df_merged

@register_feature
def add_distance_ball_goalline(df):
    """
    각 이벤트 시점의 공과 해당 팀의 골라인 사이의 거리를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'team', 'ball_x' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'distance_ball_goalline' 컬럼을 담은 DataFrame.
    """
    ball_positions = df[df['id'] == 'ball'][['frame_id', 'x', 'y']]
    ball_positions = ball_positions.rename(columns={'x': 'ball_x', 'y': 'ball_y'})
    
    df_merged = pd.merge(df, ball_positions, on='frame_id', how='left')
    
    df_merged['distance_ball_to_goalline'] = C.PITCH_X_MAX - df_merged['ball_x']
    
    df_merged = df_merged.drop(columns=['ball_x', 'ball_y'])


    return df_merged

@register_feature
def add_actor_speed(df):
    """
    각 이벤트 시점의 actor의 속도를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'player_code' 컬럼 및
                                  '{player_code}_vx', '{player_code}_vy' 형태의 속도 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'actor_speed' 컬럼을 담은 DataFrame.
    """
    actor_speed = df[df['is_ball_carrier'] == True][['frame_id', 'vx', 'vy']]
    actor_speed = actor_speed.rename(columns={'vx': 'actor_vx', 'vy': 'actor_vy'})
    df_merged = pd.merge(df, actor_speed, on='frame_id', how='left')
    
    df_merged['actor_speed'] = np.sqrt(df_merged['actor_vx'] ** 2 + df_merged['actor_vy'] ** 2)
    
    df_merged = df_merged.drop(columns=['actor_vx', 'actor_vy'])

    return df_merged

@register_feature
def add_angle_to_center_of_goal(df):
    """
    각 이벤트 시점의 공에서 해당 팀의 골대까지의 각도를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.
    """
    ball_positions = df[df['id'] == 'ball'][['frame_id', 'x', 'y']]
    ball_positions = ball_positions.rename(columns={'x': 'ball_x', 'y': 'ball_y'})
    
    df_merged = pd.merge(df, ball_positions, on='frame_id', how='left')
    
    df_merged['angle_to_center_of_goal'] = np.arctan2(df_merged['ball_y'], df_merged['ball_x'])

    df_merged = df_merged.drop(columns=['ball_y', 'ball_x'])

    return df_merged

@register_feature
def add_elapsed_time(df):
    """
    각 프레임의 경기 전체 기준 경과 시간을 계산하여 'elapsed_time' 컬럼을 추가합니다.
    전반전은 그대로, 후반전은 2700초를 더해 누적 시간으로 변환합니다.

    Args:
        df (pd.DataFrame): 'timestamp', 'period_id' 컬럼을 포함한 X_slice

    Returns:
        pd.DataFrame: 'elapsed_time' 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    # timedelta → 초로 변환
    df['elapsed_time'] = df['timestamp'].dt.total_seconds()
    
    # 후반전(period 2)은 2700초(=45분)를 더함
    df.loc[df['period_id'] == 2, 'elapsed_time'] += 2700    
    return df

@register_feature
def add_goal_diff(merged_df) -> pd.DataFrame:
    goal_diff_series = abs(merged_df['att_goal_count'] - merged_df['def_goal_count'])
    merged_df['goal_diff'] = goal_diff_series
    return merged_df

# --- 1. 최단 거리 수비수와의 거리 ---
# @register_feature
# def add_closest_defender_distance(df):
#     df = df.copy()
#     # 공통 함수를 한 번만 호출
#     results_by_frame = _get_frame_interaction_info(df)
    
#     # map을 사용해 각 행의 frame_id에 맞는 결과를 매핑하고, 'distance' 값만 추출
#     df['closest_defender_dist'] = df['frame_id'].map(results_by_frame).str.get('dist_def_to_actor')
#     return df
# --- 2. 최단 거리 수비수의 속도 ---
# @register_feature
# def add_closest_defender_speed(df):
#     df = df.copy()
#     results_by_frame = _get_frame_interaction_info(df)
    
#     # 'defender_speed' 값만 추출
#     df['closest_defender_speed'] = df['frame_id'].map(results_by_frame).str.get('speed_def_near_actor')
#     return df
# --- 3. 행위자와 최단거리 수비수의 속도 차이 ---
@register_feature
def add_speed_diff_actor_defender(df):
    df = df.copy()
    
    # 1. 각 프레임별 볼 캐리어의 정보를 추출합니다.
    carrier_df = df.loc[df['is_ball_carrier'] == True, ['frame_id', 'vx', 'vy']]
    
    # 2. 프레임별 볼 캐리어의 속도를 계산합니다.
    carrier_df['carrier_speed'] = np.sqrt(carrier_df['vx']**2 + carrier_df['vy']**2)
    
    # 3. 'frame_id'를 키로 하는 Series (딕셔너리처럼 사용)로 만듭니다.
    #    결과: {0: 5.2, 1: 5.4, ...} (frame_id: carrier_speed)
    carrier_speed_map = carrier_df.set_index('frame_id')['carrier_speed']
    
    # 4. map을 사용해 프레임별 캐리어 속도를 모든 행에 전파합니다.
    df['carrier_speed'] = df['frame_id'].map(carrier_speed_map)
    
    # 5. 속도 차이를 계산합니다.
    #    'speed_def_near_actor' 컬럼은 이미 df에 존재한다고 가정합니다.
    df['speed_diff_actor_defender'] = (df['speed_def_near_actor'] - df['carrier_speed']).abs()
    
    # 임시로 사용한 'carrier_speed' 컬럼은 제거해도 됩니다.
    df = df.drop(columns=['carrier_speed'])
    
    return df


# --- 4. 공과 가장 가까운 수비수와 사이드라인 사이의 거리 ---
@register_feature
def add_dist_def_near_ball_to_sideline(df):
    df = df.copy()
    df['dist_def_near_ball_to_sideline'] = np.minimum(df['y_def_near_ball'] - C.PITCH_Y_MIN,  C.PITCH_Y_MAX -df['y_def_near_ball']).abs()
    
    return df


# --- 5. 공과 가장 가까운 수비수와 골라인 사이의 거리 ---
@register_feature
def add_dist_def_near_ball_to_goaline(df):
    df = df.copy()
    df['dist_def_near_ball_to_goaline'] = (df['x_def_near_ball'] -  C.PITCH_X_MIN).abs()
    
    return df

# --- 6. (공에 가장 가까운 수비수와 골라인 사이의 거리, 공과 골라인 사이의 거리), 두 거리의 차이 ---
@register_feature
def add_diff_ball_defender_goalline(df):
    df = df.copy()
    
    ball_x_map = df.loc[df['id'] == 'ball'].set_index('frame_id')['x']

    df['ball_x'] = df['frame_id'].map(ball_x_map)
    

    ball_to_goaline = (df['ball_x'] - C.PITCH_X_MIN).abs()
    defender_to_goaline = (df['x_def_near_ball'] - C.PITCH_X_MIN).abs()
    df['diff_ball_defender_goalline'] = (ball_to_goaline - defender_to_goaline).abs()
    df.drop(columns=['ball_x'], inplace=True)

    return df

# --- 7. (공에 가장 가까운 수비수와 사이드라인 사이의 거리, 공과 사이드라인 사이의 거리), 두 거리의 차이 ---
@register_feature
def add_diff_ball_defender_sideline(df):
    df = df.copy()    
    ball_y_map = df.loc[df['id'] == 'ball'].set_index('frame_id')['y']

    df['ball_y'] = df['frame_id'].map(ball_y_map)    

    ball_to_sideline = np.minimum(
        (df['ball_y'] - C.PITCH_Y_MIN).abs(), 
        (C.PITCH_Y_MAX - df['ball_y']).abs()
    )
    defender_to_sideline = np.minimum(
        (df['y_def_near_ball'] - C.PITCH_Y_MIN).abs(), 
        (C.PITCH_Y_MAX - df['y_def_near_ball']).abs()
    )
    df['diff_ball_defender_sideline'] = (ball_to_sideline - defender_to_sideline).abs()
    df.drop(columns=['ball_y'], inplace=True)

    return df

def flatten_df(merged_df):    
    # 1. ball 위치를 따로 추출
    ball_df = merged_df[merged_df['id'] == 'ball'][['frame_id', 'x', 'y']].rename(columns={'x': 'ballx', 'y': 'bally'})


    # 2. 공 소유 선수를 제외한 나머지 행들만 필터링 (또는 모든 행을 쓸 수 있음)
    player_df = merged_df[merged_df['id'] != 'ball'].copy()

    # 3. 각 선수의 위치 정보에 해당 프레임의 공 위치 병합
    melted = pd.merge(player_df, ball_df, on='frame_id', how='left')

    # 4. 필요한 컬럼만 추려내기 (원하시면 순서 조정 가능)
    melted = melted[[
        'player_code',
        'period_id',
        'frame_id',
        'team_id',
        'ball_owning_team_id',
        'is_ball_carrier',
        'position_name',
        'x', 'y', 'vx', 'vy',
        'ballx', 'bally'
    ]].rename(columns={
        'frame_id': 'event_id',
        'player_code': 'key',
        'is_ball_carrier': 'player_on_ball',
        'position_name': 'position'
    })
    melted['team_id'] = melted['team_id'].astype(int)
    melted['ball_owning_team_id'] = melted['ball_owning_team_id'].astype(int)
    melted['team_on_ball'] = melted['team_id'] == melted['ball_owning_team_id']

    melted["x"] = melted["x"] + C.PITCH_X_MAX
    melted["y"] = melted["y"] + C.PITCH_Y_MAX
    melted["ballx"] = melted["ballx"] + C.PITCH_X_MAX
    melted["bally"] = melted["bally"] + C.PITCH_Y_MAX
    return melted

# 워커 함수: 이제 서브-데이터프레임을 직접 받습니다.
def compute_pitch_control_single(event_id, event_df_dict, params, radius_m=4.0):
    """
    이 함수는 이제 event_id와 해당 id에 해당하는 sub-DataFrame을 직접 인자로 받습니다.
    """
    try:
        event_df = pd.DataFrame(event_df_dict[event_id])

        PPCFa, xgrid, ygrid = pc.generate_pitch_control_for_event(
            event_df.iloc[0], event_df, params,
            field_dimen=(105, 68),
            n_grid_cells_x=52, n_grid_cells_y=34, offsides=False,
        )

        ball_x = event_df.iloc[0]['ballx']
        ball_y = event_df.iloc[0]['bally']

        X, Y = np.meshgrid(xgrid, ygrid)
        distance = np.sqrt((X - ball_x) ** 2 + (Y - ball_y) ** 2)
        final_mask = (distance <= radius_m) & (~np.isnan(PPCFa))

        summed = PPCFa[final_mask].sum() if np.any(final_mask) else 0.0
        return {"event_id": event_id, "sum_pitch_control": summed}
    
    except Exception as e:
        print(f"[event_id {event_id}] pitch control 계산 실패: {e}")
        return {"event_id": event_id, "sum_pitch_control": 0.0}

@register_feature
def add_pitch_control_parallel(df, radius_m=4.0, max_workers=20):
    params = pc.default_model_params()
    melted_df = flatten_df(df)
    results = []
    event_df_dict = defaultdict(list)
    for row in melted_df.to_dict(orient='records'):
        event_df_dict[row['event_id']].append(row)

    event_ids = list(event_df_dict.keys())
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # to_dict 대신 groupby 객체를 사용해 작업을 제출합니다.
        # grouped는 (event_id, sub_dataframe) 튜플을 생성합니다.
        futures = {
            executor.submit(compute_pitch_control_single, eid, event_df_dict, params, radius_m): eid
            for eid in event_ids
        }

        for future in as_completed(futures):
            results.append(future.result())      

    pc_df = pd.DataFrame(results, columns=['event_id', 'sum_pitch_control'])
    final_df = df.merge(pc_df, how='left', left_on='frame_id', right_on='event_id')

    return final_df

def run(radius_m=4.0):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    traces_df = pd.read_csv(os.path.join(current_dir, 'notebooks','126285_X_slice.csv'))
    # traces_df['order'] = traces_df.index

    params = pc.default_model_params()

    result = add_pitch_control_parallel(traces_df, params, radius_m=radius_m)

if __name__ == "__main__":


    fire.Fire(run)