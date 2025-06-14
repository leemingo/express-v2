
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
    ```
    `
    goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0  # 양쪽 골대 y좌표는 동일
    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    distance = np.sqrt(dx**2 + dy**2)

    return pd.DataFrame({
        'distance_ball_goal': distance
    }, index=merged_df.index)

def distance_ball_sideline(merged_df):
    ball_sideline = np.minimum(merged_df['ball_y'] - C.PITCH_Y_MIN, C.PITCH_Y_MAX - merged_df['ball_y'])   
    
    return pd.DataFrame({
        'distance_ball_sideline' : ball_sideline 
    },index=merged_df.index)

def distance_ball_goalline(merged_df):
    goaline_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MIN, C.PITCH_X_MAX)
    ball_goaline = np.abs(goaline_x - merged_df['ball_x'])
    
    return pd.DataFrame({
        'distance_ball_goalline' : ball_goaline},
    index=merged_df.index)

def get_actor_speed(merged_df):
    actor_speeds = np.full(len(merged_df), np.nan)

    for i, (_, row) in enumerate(merged_df.iterrows()):
        xID = row.player_code
        vx, vy = row.get(f"{xID}_vx", np.nan), row.get(f"{xID}_vy", np.nan)  # ← 오타 수정

        if pd.isna(vx) or pd.isna(vy):
            continue

        speed = np.sqrt(vx ** 2 + vy ** 2)  # ← np.sqrt(x^2 + y^2)
        actor_speeds[i] = speed

    return pd.DataFrame({
        'actor_speed': actor_speeds},
        index=merged_df.index)

def angle_to_goal(merged_df):
    goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0  # 골대는 항상 중앙 y좌표

    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    angle = np.degrees(np.arctan2(dy, dx))

    return pd.DataFrame({
        'frame_id': merged_df['frame_id'],
        'angle_to_goal' : angle },
    index=merged_df.index)

def compute_time_elapsed(merged_df):
    # timestamp → timedelta 변환
    result = merged_df['time_seconds']
    return pd.DataFrame({
        'elapsed_time': result}, 
    index=merged_df.index)

def time_since_last_opponent_action(merged_df, event_df):
    time_since_opponent_action = pd.Series(0.0, index=merged_df.index)
    all_teams = pd.concat([merged_df['tID'], event_df['tID']]).unique()

    for current_team_id in all_teams:
        current_team_merged = merged_df[merged_df['tID'] == current_team_id].sort_values(by='time_seconds')
        opponent_events = event_df[event_df['tID'] != current_team_id].sort_values(by='time_seconds')

        if not current_team_merged.empty and not opponent_events.empty:
            # merge_asof를 사용하여 각 현재 팀 이벤트 시간 이전의 가장 최근 상대 팀 이벤트 시간을 찾습니다.
            # 'direction='backward''는 current_time보다 같거나 작은 가장 최근의 값을 찾으라는 의미입니다.
            merged_result = pd.merge_asof(
                current_team_merged,
                opponent_events[['time_seconds']].rename(columns={'time_seconds': 'last_opponent_time'}),
                left_on='time_seconds',
                right_on='last_opponent_time',
                direction='backward',
                suffixes=('', '_opponent') # 접미사를 사용하여 컬럼 이름 충돌 방지
            )

            # 계산된 시간 차이를 원래 merged_df의 해당 인덱스에 할당
            # NaN 값은 상대 팀 액션이 없었음을 의미하므로 0으로 처리 (또는 다른 기본값)
            time_diff = merged_result['time_seconds'] - merged_result['last_opponent_time'].fillna(merged_result['time_seconds'])
            time_since_opponent_action.loc[current_team_merged.index] = time_diff.values

    return pd.DataFrame({
        'time_since_last_opponent_action': time_since_opponent_action
    }, index=merged_df_original.index)


def _prepare_event_data_for_cumul_goals(event_df_original: pd.DataFrame) -> pd.DataFrame:

    event_df = event_df_original.copy()
    event_df['time_seconds'] = pd.to_numeric(event_df['time_seconds'], errors='coerce')
    event_df['is_goal'] = (
        (event_df['type_name'] == 'Shot') &
        (event_df['result_name'] == 'Goal')
    )
    return event_df.sort_values(by='time_seconds')

def cumul_goal_att(merged_df, events_df) -> pd.DataFrame:

    # past_events_for_point가 비어있으면 0으로 채워진 결과를 반환합니다.
    if merged_df.empty:
        # B를 알 수 없으므로, 길이가 1인 더미 DataFrame 반환 (호출부에서 처리하도록)
        return pd.DataFrame({'att_goal_count': [np.nan]}, index=[merged_df.index.min() if not merged_df.empty else 0])

    event_df_prepared = _prepare_event_data_for_cumul_goals(events_df) # 여기서 매번 호출하면 비효율적!
    
    current_point_tID = merged_df['tID'].iloc[0]
    current_point_time_seconds = merged_df['time_seconds'].max() # 해당 배치 내 가장 최근 시간

    # 현재 팀의 골 이벤트만 필터링합니다.
    current_team_goals = event_df_prepared[
        (event_df_prepared['tID'] == current_point_tID) & (event_df_prepared['is_goal'])
    ].copy()

    # 각 골 이벤트에 대해 시간 순서대로 누적 합계를 계산합니다.
    if not current_team_goals.empty:
        current_team_goals['cumulative_goals'] = 1
        current_team_goals['cumulative_goals'] = current_team_goals['cumulative_goals'].cumsum()
    else:
        # 골이 없다면 빈 DataFrame에 cumulative_goals 컬럼만 추가
        current_team_goals['cumulative_goals'] = pd.Series(dtype=int)

    temp_df = pd.DataFrame({'time_seconds': [current_point_time_seconds], 'tID': [current_point_tID]})
    merged_att_goals = pd.merge_asof(
        temp_df,
        current_team_goals[['time_seconds', 'cumulative_goals']],
        left_on='time_seconds',
        right_on='time_seconds',
        direction='backward'
    )

    att_count = merged_att_goals['cumulative_goals'].fillna(0).astype(int).iloc[0]
    
    B = len(merged_df) # `past_events`의 실제 B 값
    return pd.DataFrame({'att_goal_count': np.full((B,), att_count)}, index=merged_df.index)


def cumul_goal_def(merged_df, events_df) -> pd.DataFrame:
    
    event_df_prepared = _prepare_event_data_for_cumul_goals(events_df)

    current_point_tID = merged_df['tID'].iloc[0]
    current_point_time_seconds = merged_df['time_seconds'].max()

    all_tids_in_events = event_df_prepared['tID'].unique()
    opponent_team_ids = [tid for tid in all_tids_in_events if tid != current_point_tID]

    if not opponent_team_ids: # 상대 팀이 없는 경우
        B = len(merged_df)
        return pd.DataFrame({'def_goal_count': np.full((B,), 0)}, index=merged_df.index)

    # 상대 팀의 골 이벤트만 필터링합니다.
    opponent_goals = event_df_prepared[
        (event_df_prepared['tID'].isin(opponent_team_ids)) & (event_df_prepared['is_goal'])
    ].copy()

    if not opponent_goals.empty:
        opponent_goals['cumulative_goals'] = 1
        opponent_goals['cumulative_goals'] = opponent_goals['cumulative_goals'].cumsum()
    else:
        opponent_goals['cumulative_goals'] = pd.Series(dtype=int)

    temp_df = pd.DataFrame({'time_seconds': [current_point_time_seconds], 'tID': [current_point_tID]})
    merged_def_goals = pd.merge_asof(
        temp_df,
        opponent_goals[['time_seconds', 'cumulative_goals']],
        left_on='time_seconds',
        right_on='time_seconds',
        direction='backward'
    )

    def_count = merged_def_goals['cumulative_goals'].fillna(0).astype(int).iloc[0]
    
    B = len(merged_df)
    return pd.DataFrame({'def_goal_count': np.full((B,), def_count)}, index=merged_df.index)

def goaldiff(merged_df, events_df) -> pd.DataFrame:

    if merged_df.empty:
        # 비어있을 경우 NaN으로 채워진 결과 반환 (update_features에서 처리)
        return pd.DataFrame({'goal_diff': [np.nan]}, index=[merged_df.index.min() if not merged_df.empty else 0])

    att_goals_df = cumul_goal_att(merged_df, full_events_df)
    def_goals_df = cumul_goal_def(merged_df, full_events_df)

    goal_diff_series = att_goals_df['att_goal_count'] - def_goals_df['def_goal_count']
    
    # 결과를 단일 열 DataFrame으로 반환
    return pd.DataFrame({'goal_diff': goal_diff_series}, index=merged_df.index)


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
def get_min_defender_distance(merged_df):

    results_df = get_closest_defender_info(merged_df)
    
    return results_df[['closest_defender_dist']]

# --- 2. 최단 거리 수비수의 속도 ---
def get_closest_defender_speed(merged_df):
    results_df = get_closest_defender_info(merged_df) 
    return results_df[['closest_defender_speed']]

# --- 3. 행위자와 최단거리 수비수의 속도 차이 ---
def get_speed_diff_actor_defender(merged_df):
    results_df = get_closest_defender_info(merged_df)    
    actor_speeds_series = get_actor_speed(merged_df)['actor_speed']

    # 속도 차이 계산, closest_defender_speed는 results_df에 있음.
    speed_diff = actor_speeds_series - results_df['closest_defender_speed']
    
    return pd.DataFrame(speed_diff, columns=['speed_diff_actor_defender'], index=merged_df.index)


def nb_of_m_radius(merged_df, dist=3.0):
    """행위자 기준 반경 dist m 안에 있는 상대편 선수 수를 반환한다."""
    # 미리 홈/어웨이 좌표 컬럼 목록을 만들어두면 루프 안에서 재생성하지 않아도 됨
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

    return pd.DataFrame({f'nb_of_{dist}m_radius': counts}, index=merged_df.index)

def nb_of_3m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 3.0)
    return results_df
                   
def nb_of_5m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 5.0)
    return results_df

def nb_of_10m_radius(merged_df):
    results_df = nb_of_m_radius(merged_df, 10.0)
    return results_df

# --- 4. 공에 가장 가까운 수비수의 위치와 가장 가까운 터치라인 사이의 거리 ---
def get_dist_defender_to_sideline(merged_df):
    results_df = get_closest_defender_info(merged_df) # 최단거리 수비수 x, y 좌표 사용
       
    sideline_dist = np.minimum(results_df['closest_defender_y']-C.PITCH_Y_MIN, C.PITCH_Y_MAX - results_df['closest_defender_y'])
    return pd.DataFrame({        
        'dist_defender_to_sideline': sideline_dist
    }, index=merged_df.index)

# --- 5. 공에 가장 가까운 수비수와 골라인 사이의 거리 ---
def get_dist_defender_to_goaline(merged_df):
    results_df = get_closest_defender_info(merged_df) # 최단거리 수비수 x, y 좌표 사용
    def_goal_x = np.where(merged_df['team'] == 'Home', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goaline_dist = np.abs(def_goal_x - results_df['closest_defender_x'])

    return pd.DataFrame({
        'dist_defender_to_goaline': goaline_dist
    }, index=merged_df.index)

# --- 6. (공에 가장 가까운 수비수와 사이드라인 사이의 거리, 공과 사이드라인 사이의 거리), 두 거리의 차이 ---
def get_diff_ball_defender_sideline(merged_df):
    results_df = get_closest_defender_info(merged_df)
    
    ball_y = merged_df['ball_y']
    defender_y = results_df['closest_defender_y']
    
    ball_to_sideline = np.minimum(ball_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - ball_y)
    defender_to_sideline = np.minimum(defender_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - defender_y)
    
    diff = ball_to_sideline - defender_to_sideline
    
    return pd.DataFrame({
        'diff_ball_defender_sideline': diff
    }, index=merged_df.index)

def flatten_df(merged_df, teams_df):
    
    teams_df["tracking_id"] = teams_df.apply(
        lambda t: f'{t["team"][0]}{t["xID"]:02d}', 
        axis=1
    )
    team_sheets_lookup = {
        row["tracking_id"]: {
            "player_id": row["player_id"],
            "team_id":   row["team_id"],
            "position":  row["position"],
            "team": row["team"]
        }
        for _, row in teams_df.iterrows()
    }
    
    melted = []
    tracking_ids = [col[:-2] for col in merged_df.columns if re.fullmatch(r'[HA]\d{2}_x', col)]
    for i, row in enumerate(tqdm(merged_df.itertuples(), total=len(merged_df))):
        event_id = row.event_id
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
        return {"event_id": event_id, "summed_pitch_control": summed}
    except Exception as e:
        print(f"[event_id {event_id}] pitch control 계산 실패: {e}")
        return {"event_id": event_id, "summed_pitch_control": 0.0}


def sum_pitch_control(merged_df, teams_df, radius_m=4.0):
    params = pc.default_model_params()
    melted_df = flatten_df(merged_df, teams_df)

    # ➤ event_id별로 분할해서 전달 가능한 dict로 만들어줌
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
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    results_df = pd.DataFrame(results).sort_values("event_id").reset_index(drop=True)

    return results_df

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
