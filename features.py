
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

def distance_ball_goal(merged_df):
    """
    각 이벤트 시점의 공과 해당 팀의 골대 사이의 거리를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'team', 'ball_x', 'ball_y' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'distance_ball_goal' 컬럼을 담은 DataFrame.
                      인덱스는 merged_df와 동일합니다.
    """
    goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0.0

    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    distance = np.sqrt(dx**2 + dy**2)

    # action_id, time_seconds와 함께 DataFrame으로 반환
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'distance_ball_goal': distance
    }, index=merged_df.index)
    
def distance_ball_sideline(merged_df):
    """
    각 이벤트 시점의 공과 가장 가까운 사이드라인(측면 라인) 사이의 거리를 계산합니다.

    Args:
        merged_df (pd.DataFrame): 'ball_y' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.Series: 'distance_ball_sideline' 피처를 담은 Series.
    """
    # 공의 Y 좌표가 상단 사이드라인 (PITCH_Y_MAX)과 하단 사이드라인 (PITCH_Y_MIN) 중 어디에 더 가까운지 계산
    dist_to_top_sideline = C.PITCH_Y_MAX - merged_df['ball_y']
    dist_to_bottom_sideline = merged_df['ball_y'] - C.PITCH_Y_MIN
    
    # 두 거리 중 더 작은 값을 선택
    ball_sideline = np.minimum(dist_to_top_sideline, dist_to_bottom_sideline)
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'], # action_id 포함
        'time_seconds': merged_df['time_seconds'], # time_seconds 포함
        'distance_ball_goal': distance
    }, index=merged_df.index)


def distance_ball_sideline(merged_df):
    """
    각 이벤트 시점의 공과 가장 가까운 사이드라인(측면 라인) 사이의 거리를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'ball_y' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'distance_ball_sideline' 컬럼을 담은 DataFrame.
    """
    ball_sideline = np.minimum(merged_df['ball_y'] - C.PITCH_Y_MIN, C.PITCH_Y_MAX - merged_df['ball_y'])   
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'], # action_id 포함
        'time_seconds': merged_df['time_seconds'], # time_seconds 포함
        'distance_ball_sideline': ball_sideline
    },index=merged_df.index)

def distance_ball_goalline(merged_df):
    """
    각 이벤트 시점의 공과 해당 팀의 골라인 사이의 거리를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'team', 'ball_x' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'distance_ball_goalline' 컬럼을 담은 DataFrame.
    """
    goaline_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MIN, C.PITCH_X_MAX)
    ball_goaline = np.abs(goaline_x - merged_df['ball_x'])
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'distance_ball_goalline': ball_goaline
    }, index=merged_df.index)

def actor_speed(merged_df):
    """
    각 이벤트 시점의 actor의 속도를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'player_code' 컬럼 및
                                  '{player_code}_vx', '{player_code}_vy' 형태의 속도 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'actor_speed' 컬럼을 담은 DataFrame.
    """
    actor_speeds = np.full(len(merged_df), np.nan)
    for i, (_, row) in enumerate(merged_df.iterrows()):
        xID = row.player_code
        vx, vy = row.get(f"{xID}_vx", np.nan), row.get(f"{xID}_vy", np.nan)  # ← 오타 수정

        if pd.isna(vx) or pd.isna(vy):
            continue
        speed = np.sqrt(vx ** 2 + vy ** 2)  # ← np.sqrt(x^2 + y^2)
        actor_speeds[i] = speed

    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'actor_speed': actor_speeds
    }, index=merged_df.index)

def angle_to_goal(merged_df):
    """
    각 이벤트 시점의 공에서 해당 팀의 골대까지의 각도를 계산합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds', 'team', 'ball_x', 'ball_y' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'angle_to_goal' 컬럼을 담은 DataFrame.
    """
    # 홈 팀의 목표 골대 X 좌표는 C.PITCH_X_MAX, 원정 팀은 C.PITCH_X_MIN

    goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0  # 골대는 항상 중앙 y좌표

    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    angle = np.degrees(np.arctan2(dy, dx))

    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'angle_to_goal': angle
    }, index=merged_df.index)

def elapsed_time(merged_df):
    """
    각 이벤트 시점의 경과 시간을 반환합니다.
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'elapsed_time' 컬럼을 담은 DataFrame.
    """
    result = merged_df['time_seconds']
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'elapsed_time': elapsed_time
    }, index=merged_df.index)

def time_since_last_opponent_action(merged_df):
    """
    각 이벤트 시점까지의 가장 최근 상대 팀 액션 이후 경과 시간을 계산합니다
    action_id, time_seconds 및 계산된 피처를 포함하는 DataFrame을 반환합니다.

    Args:
        merged_df (pd.DataFrame): 'action_id', 'time_seconds' 컬럼을 포함하는 DataFrame.

    Returns:
        pd.DataFrame: 'action_id', 'time_seconds', 'elapsed_time' 컬럼을 담은 DataFrame.
    """
    result = []

    for idx, row in merged_df.iterrows():
        current_time = row['time_seconds']
        current_team = row['tID']

        # 상대 팀 이벤트 중 현재 시간보다 이전 것만 필터링
        opponent_events = merged_df[(merged_df['tID'] != current_team) &
                                 (merged_df['time_seconds'] < current_time)]

        if not opponent_events.empty:
            last_opponent_time = opponent_events['time_seconds'].max()
            time_diff = current_time - last_opponent_time
        else:
            time_diff = 0.0

    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'time_since_last_opponent_action': time_diff
    }, index=merged_df.index)

def cumul_goal(merged_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = merged_df.copy()

    # 골 여부 판별
    merged_df['is_goal'] = (merged_df['type_name'] == "Shot") & (merged_df['result_name'] == "Goal")

    # 누적 골 초기화
    att_goal_count = np.zeros(len(merged_df), dtype=int)
    def_goal_count = np.zeros(len(merged_df), dtype=int)

    for team in merged_df['tID'].unique():
        # 공격/수비 여부 판단
        is_att = merged_df['tID'] == team
        is_def = ~is_att

        # 공격팀 입장에서 자기가 골 넣은 경우
        att_goal = merged_df['is_goal'] & is_att
        att_goal_cumsum = att_goal.cumsum().shift(fill_value=0)
        att_goal_count[is_att] = att_goal_cumsum[is_att]

        # 공격팀 입장에서 상대가 골 넣은 경우
        def_goal = merged_df['is_goal'] & is_def
        def_goal_cumsum = def_goal.cumsum().shift(fill_value=0)
        def_goal_count[is_att] = def_goal_cumsum[is_att]

    return pd.DataFrame({
        'att_goal_count': att_goal_count,
        'def_goal_count': def_goal_count
    }, index=merged_df.index)

def att_goal(merged_df: pd.DataFrame) -> pd.DataFrame:
    result = cumul_goal(merged_df)
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'att_goal': result['att_goal_count']
    }, index=merged_df.index)

def def_goal(merged_df: pd.DataFrame) -> pd.DataFrame:
    result = cumul_goal(merged_df)
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'def_goal': result['def_goal_count']
    }, index=merged_df.index)


def goaldiff(merged_df, events_df) -> pd.DataFrame:

    goal_results = cumul_goal(merged_df, full_events_df)   
    goal_diff_series = goal_results['att_goal_count'] - goal_results['def_goal_count']
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'goal_diff': goal_diff_series
    }, index=merged_df.index)

def get_closest_defender_info(merged_df):
    # 결과 저장 리스트
    ids, xs, ys, vxs, vys, speeds, dists = ([] for _ in range(7))

    for _, row in merged_df.iterrows():
        ball_x, ball_y = row['ball_x'], row['ball_y']
        team_prefix   = row['player_code'][0]          # 'H' 또는 'A'
        defender_pref = 'A' if team_prefix == 'H' else 'H'

        candidates = []  # (x, y, vx, vy, id)

        for i in range(20):
            x = row.get(f"{defender_pref}{i:02}_x")
            y = row.get(f"{defender_pref}{i:02}_y")
            vx = row.get(f"{defender_pref}{i:02}_vx")
            vy = row.get(f"{defender_pref}{i:02}_vy")
            if pd.notna(x) and pd.notna(y):
                candidates.append((x, y, vx, vy, f"{defender_pref}{i:02}"))

        if candidates:
            arr = np.array([(c[0], c[1]) for c in candidates])        # (N,2)
            diff = arr - np.array([ball_x, ball_y])
            dist = np.linalg.norm(diff, axis=1)
            k = dist.argmin()

            x, y, vx, vy, pid = candidates[k]
            ids.append(pid)
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)
            speeds.append(np.hypot(vx, vy))
            dists.append(dist[k])
        else:
            ids.append(np.nan); xs.append(np.nan); ys.append(np.nan)
            vxs.append(np.nan); vys.append(np.nan); speeds.append(np.nan); dists.append(np.nan)

    # 결과 DataFrame (merged_df와 같은 index 사용)
    return pd.DataFrame({
        'frame_id' : merged_df['frame_id'],
        'closest_defender_id'   : ids,
        'closest_defender_x'    : xs,
        'closest_defender_y'    : ys,
        'closest_defender_vx'   : vxs,
        'closest_defender_vy'   : vys,
        'closest_defender_speed': speeds,
        'closest_defender_dist' : dists,
    }, index=merged_df.index)

# --- 1. 최단 거리 수비수와의 거리 ---
def closest_defender_dist(merged_df):

    results_df = get_closest_defender_info(merged_df)
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'closest_defender_dist': results_df['closest_defender_dist']
    }, index=merged_df.index)

# --- 2. 최단 거리 수비수의 속도 ---
def closest_defender_speed(merged_df):
    results_df = get_closest_defender_info(merged_df) 

    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'closest_defender_speed': results_df['closest_defender_speed']
    }, index=merged_df.index)

# --- 3. 행위자와 최단거리 수비수의 속도 차이 ---
def speed_diff_actor_defender(merged_df):
    results_df = get_closest_defender_info(merged_df)    
    actor_speeds_series = actor_speed(merged_df)['actor_speed']

    # 속도 차이 계산, closest_defender_speed는 results_df에 있음.
    speed_diff = actor_speeds_series - results_df['closest_defender_speed']
    
    return pd.DataFrame({
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'speed_diff_actor_defender': speed_diff
    }, index=merged_df.index)

def nb_of_m_radius(merged_df, dist=3.0):
    """행위자 기준 반경 dist m 안에 있는 상대편 선수 수를 반환한다."""
    processed_df = merged_df.copy()

    home_cols = [(f"H{i:02}_x", f"H{i:02}_y") for i in range(20)
                 if f"H{i:02}_x" in merged_df.columns and f"H{i:02}_y" in merged_df.columns]
    away_cols = [(f"A{i:02}_x", f"A{i:02}_y") for i in range(20)
                 if f"A{i:02}_x" in merged_df.columns and f"A{i:02}_y" in merged_df.columns]

    counts = []

    for _, row in merged_df.iterrows():
        actor_team = row['player_code'][0]          # 'H' 또는 'A'
        opponent_cols = away_cols if actor_team == 'H' else home_cols

        ex, ey = row['ball_x'], row['ball_y']     # 이벤트 기준점

        cnt = 0
        for x_col, y_col in opponent_cols:
            px, py = row[x_col], row[y_col]
            if pd.notna(px) and pd.notna(py):
                if (px - ex) ** 2 + (py - ey) ** 2 <= dist ** 2:
                    cnt += 1
        counts.append(cnt)

    return pd.DataFrame({
        'action_id': processed_df['action_id'],
        'time_seconds': processed_df['time_seconds'],
         f'nb_of_{dist}m_radius': counts
    }, index=processed_df.index)

def nb_of_3m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 3)
    return results_df
                   
def nb_of_5m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 5)
    return results_df

def nb_of_10m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 10)
    return results_df

# --- 4. 공에 가장 가까운 수비수의 위치와 가장 가까운 터치라인 사이의 거리 ---
def dist_defender_to_sideline(merged_df):
    results_df = get_closest_defender_info(merged_df) # 최단거리 수비수 x, y 좌표 사용
       
    sideline_dist = np.minimum(results_df['closest_defender_y']-C.PITCH_Y_MIN, C.PITCH_Y_MAX - results_df['closest_defender_y'])
    return pd.DataFrame({        
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'dist_defender_to_sideline': sideline_dist
    }, index=merged_df.index)

# --- 5. 공에 가장 가까운 수비수와 골라인 사이의 거리 ---
def dist_defender_to_goaline(merged_df):
    results_df = get_closest_defender_info(merged_df) # 최단거리 수비수 x, y 좌표 사용
    def_goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goaline_dist = np.abs(def_goal_x - results_df['closest_defender_x'])

    return pd.DataFrame({        
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'dist_defender_to_goaline': goaline_dist
    }, index=merged_df.index)

# --- 6. (공에 가장 가까운 수비수와 사이드라인 사이의 거리, 공과 사이드라인 사이의 거리), 두 거리의 차이 ---
def diff_ball_defender_sideline(merged_df):
    results_df = get_closest_defender_info(merged_df)
    
    ball_y = merged_df['ball_y']
    defender_y = results_df['closest_defender_y']
    
    ball_to_sideline = np.minimum(ball_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - ball_y)
    defender_to_sideline = np.minimum(defender_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - defender_y)
    
    diff = ball_to_sideline - defender_to_sideline
    return pd.DataFrame({        
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'diff_ball_defender_sideline': diff
    }, index=merged_df.index)

def flatten_df(merged_df, teams_df):    
    team_sheets_lookup = {
        row["player_code"]: {
            "player_id": row["pID"],
            "team_id":   row["tID"],
            "position":  row["position"],
            "team": row["team"]
         }    for _, row in teams_df.iterrows()
    }    
    melted = []
    tracking_ids = [col[:-2] for col in merged_df.columns if re.fullmatch(r'[HA]\d{2}_x', col)]
    for i, row in enumerate(tqdm(merged_df.itertuples(), total=len(merged_df))):
        event_id = row.action_id
        period_id = row.period_id

        for base_id in tracking_ids:
            x = getattr(row, f"{base_id}_x", np.nan)
            y = getattr(row, f"{base_id}_y", np.nan)
            vx = getattr(row, f"{base_id}_vx", np.nan)
            vy = getattr(row, f"{base_id}_vy", np.nan)

            # if pd.isna(x) or pd.isna(y):
            #     continue

            melted.append({
                "key": base_id,
                "period_id": period_id,
                "event_id": event_id,
                "team_id": team_sheets_lookup[base_id]["team_id"],
                "team_on_ball": row.player_code[0] == base_id[0],
                "player_on_ball": row.player_code == base_id ,
                "position": team_sheets_lookup[base_id]['position'],
                "x": x + C.PITCH_X_MAX,
                "y": y + C.PITCH_Y_MAX,
                "vx":vx,
                "vy":vy,
                "ballx": getattr(row, "ball_x", np.nan) + C.PITCH_X_MAX,
                "bally": getattr(row, "ball_y", np.nan) + C.PITCH_Y_MAX,
            })
    melted_df = pd.DataFrame(melted)
    return melted_df


def single_event_pitch_control(event_id, event_df_dict, params, radius_m):
    try:
        event_df = pd.DataFrame(event_df_dict[event_id])
        PPCFa, xgrid, ygrid = pc.generate_pitch_control_for_event(
            event_df.iloc[0], event_df, params,
            field_dimen=(C.PITCH_X_MAX - C.PITCH_X_MIN, C.PITCH_Y_MAX - C.PITCH_Y_MIN),
            n_grid_cells_x=52, n_grid_cells_y=34, offsides=False,
        )
        ball_x = event_df.iloc[0]['ballx']
        ball_y = event_df.iloc[0]['bally']
        X, Y = np.meshgrid(xgrid, ygrid)
        distance = np.sqrt((X - ball_x)**2 + (Y - ball_y)**2)
        final_mask = (distance <= radius_m) & (~np.isnan(PPCFa))
        summed = PPCFa[final_mask].sum() if np.any(final_mask) else 0.0
        return {"event_id": event_id, "sum_pitch_control": summed}
    except Exception as e:
        print(f"[event_id {event_id}] pitch control 계산 실패: {e}")
        return {"event_id": event_id, "summed_pitch_control": 0.0}


def sum_pitch_control(merged_df, teams_df, radius_m=4.0):
    params = pc.default_model_params()
    melted_df = flatten_df(merged_df, teams_df)

    # event_id별로 분할해서 전달 가능한 dict로 만들어줌
    event_df_dict = defaultdict(list)
    for row in melted_df.to_dict(orient='records'):
        event_df_dict[row['event_id']].append(row)

    event_ids = list(event_df_dict.keys())
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(single_event_pitch_control, eid, event_df_dict, params, radius_m): eid
            for eid in event_ids
        }
        for future in as_completed(futures):
            results.append(future.result())
    results_df = pd.DataFrame(results).sort_values("event_id").reset_index(drop=True)

    return pd.DataFrame({        
        'action_id': merged_df['action_id'],
        'time_seconds': merged_df['time_seconds'],
        'sum_pitch_control': results_df['sum_pitch_control']
    }, index=merged_df.index)

def run(radius_m=4.0):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    traces_file = os.path.join(current_dir, 'notebook', 'traces_df.csv')
    teams_file = os.path.join(current_dir, 'notebook', 'teams.csv')

    traces_df = pd.read_csv(traces_file)
    traces_df['event_id'] = traces_df.index

    teams = pd.read_csv(teams_file)

    result = sum_pitch_control(traces_df.iloc[:100, :], teams, radius_m=radius_m)
    print(result)
    # result.to_csv('pitch_control_result.csv', index=False)

if __name__ == "__main__":
    fire.Fire(run)
