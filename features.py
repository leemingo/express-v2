
import datatools.pitch_control as pc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re

#Visualization
os.chdir('..')
import config as C
current_dir = os.path.dirname(__file__)

def distance_ball_goal(merged_df):
    goal_x = np.where(merged_df['attack_direction'] == 'RIGHT', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0  # 양쪽 골대 y좌표는 동일
    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    distance = np.sqrt(dx**2 + dy**2)
    return pd.DataFrame({'distance_ball_goal' : distance},index=merged_df.index)

def distance_ball_sideline(merged_df):
    ball_sideline = np.minimum(merged_df['ball_y'] - C.PITCH_Y_MIN, C.PITCH_Y_MAX - merged_df['ball_y'])   
    
    return pd.DataFrame({'distance_ball_sideline' : ball_sideline },index=merged_df.index)

def distance_ball_goalline(merged_df):
    goaline_x = np.where(merged_df['attack_direction'] == 'RIGHT', C.PITCH_X_MIN, C.PITCH_X_MAX)
    ball_goaline = np.abs(goaline_x - merged_df['ball_x'])
    
    return pd.DataFrame({'distance_ball_goalline' : ball_goaline},index=merged_df.index)

def get_actor_speed(merged_df):
    actor_speeds = np.full(len(merged_df), np.nan)

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        xID = row.player_code
        vx, vy = row.get(f"{xID}_vx", np.nan), row.get(f"{xID}_vy", np.nan)  # ← 오타 수정

        if pd.isna(vx) or pd.isna(vy):
            continue

        speed = np.sqrt(vx ** 2 + vy ** 2)  # ← np.sqrt(x^2 + y^2)
        actor_speeds[idx] = speed

    return pd.DataFrame({'actor_speed': actor_speeds}, index=merged_df.index)

def angle_to_goal(merged_df):
    goal_x = np.where(merged_df['attack_direction'] == 'RIGHT', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goal_y = 0  # 골대는 항상 중앙 y좌표

    dx = goal_x - merged_df['ball_x']
    dy = goal_y - merged_df['ball_y']
    angle = np.degrees(np.arctan2(dy, dx))

    return pd.DataFrame({'angle_to_goal' : angle },index=merged_df.index)


def safe_to_timedelta(series: pd.Series) -> pd.Series:
    """
    모든 값을 Timedelta로 변환한다.
    """
    def _convert(x):
        if isinstance(x, pd.Timedelta):          # 그대로
            return x
        elif isinstance(x, pd.Timestamp):        # 날짜 → 그날 00:00 기준 경과
            return x - x.normalize()
        elif isinstance(x, str):                 # '0 days …' 같은 문자열
            try:
                return pd.to_timedelta(x)
            except ValueError:
                return pd.NaT
        else:
            return pd.NaT                        # 기타(숫자 등)

    return series.apply(_convert)

def compute_time_elapsed(merged_df):
    # timestamp → timedelta 변환
    converted_timestamps = pd.to_timedelta(merged_df['timestamp'], errors='coerce')
    seconds = converted_timestamps.dt.total_seconds()

    # 전반 종료, 후반 시작 간 시간차 계산
    first_half_end = seconds[merged_df['period_id'] == 1].max()
    second_half_start = seconds[merged_df['period_id'] == 2].min()
    offset = first_half_end - second_half_start

    # 후반전만 offset 추가
    seconds_adjusted = seconds.copy()
    seconds_adjusted[merged_df['period_id'] == 2] += offset

    return pd.DataFrame({'elapsed_time': seconds_adjusted}, index=merged_df.index)

def time_since_last_opponent_action(merged_df):
    # 누적 시간 컬럼 계산
    merged_df = merged_df.copy()
    merged_df['elapsed_time'] = compute_time_elapsed(merged_df)

    result = []

    for idx, row in merged_df.iterrows():
        current_time = row['elapsed_time']
        current_team = row['tID']

        # 상대 팀 이벤트 중 현재 시간보다 이전 것만 필터링
        opponent_events = merged_df[(merged_df['tID'] != current_team) &
                                 (merged_df['elapsed_time'] < current_time)]

        if not opponent_events.empty:
            last_opponent_time = opponent_events['elapsed_time'].max()
            time_diff = current_time - last_opponent_time
        else:
            time_diff = 0.0

        result.append(time_diff)

    return pd.DataFrame({'time_since_last_opponent_action': result}, index=merged_df.index)


def is_goal(qual_list):
    for d in qual_list:
        if (
            d.get('event_name') == 'Shots & Goals'
            and d.get('property', {}).get('Outcome') == 'Goals'
        ):
            return True
    return False

def cumul_goal(merged_df) -> pd.DataFrame:
    merged_df = merged_df.copy()
    merged_df['events'] = merged_df['events'].apply(ast.literal_eval)

    merged_df['is_goal'] = merged_df['events'].apply(is_goal)

    # 누적 골 초기화
    def_goal_count = np.zeros(len(merged_df), dtype=int)
    att_goal_count = np.zeros(len(merged_df), dtype=int)

    # 팀별로 누적 골 인덱싱
    for team in merged_df['tID'].unique():
        is_att = merged_df['tID'] == team
        is_def = ~is_att

        att_goal = merged_df['is_goal'] & is_att
        def_goal = merged_df['is_goal'] & is_def

        att_cumsum = att_goal.cumsum()
        def_cumsum = def_goal.cumsum()

        att_goal_count[is_att] = att_cumsum[is_att].shift(fill_value=0)
        def_goal_count[is_att] = def_cumsum[is_att].shift(fill_value=0)

    return pd.DataFrame({
        'def_goal_count': def_goal_count,
        'att_goal_count': att_goal_count
    }, index=merged_df.index)

def goaldiff(merged_df):
    df = cumul_goal(merged_df)
    df['goal_diff'] = df['att_goal_count'] - df['def_goal_count']
    return df[['goal_diff']]


def get_closest_defender_info(merged_df):
    # 결과 저장 리스트
    ids, xs, ys, vxs, vys, speeds, dists = ([] for _ in range(7))

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
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

    # 속도 차이 계산, closest_defender_speed는 results_df에 있음음.
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

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
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
    return sideline_dist.to_frame(name='dist_defender_to_sideline')

# --- 5. 공에 가장 가까운 수비수와 골라인 사이의 거리 ---
def get_dist_defender_to_goaline(merged_df):
    results_df = get_closest_defender_info(merged_df) # 최단거리 수비수 x, y 좌표 사용
    def_goal_x = np.where(merged_df['attack_direction'] == 'RIGHT', C.PITCH_X_MAX, C.PITCH_X_MIN)
    goaline_dist = np.abs(def_goal_x - results_df['closest_defender_x'])
    return goaline_dist.to_frame(name='dist_defender_to_goaline')

# --- 6. (공에 가장 가까운 수비수와 사이드라인 사이의 거리, 공과 사이드라인 사이의 거리), 두 거리의 차이 ---
def get_diff_ball_defender_sideline(merged_df):
    results_df = get_closest_defender_info(merged_df)
    
    ball_y = merged_df['ball_y']
    defender_y = results_df['closest_defender_y']
    
    ball_to_sideline = np.minimum(ball_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - ball_y)
    defender_to_sideline = np.minimum(defender_y - C.PITCH_Y_MIN, C.PITCH_Y_MAX - defender_y)
    
    diff = ball_to_sideline - defender_to_sideline
    
    return pd.DataFrame(diff, columns=['diff_ball_defender_sideline'], index=merged_df.index)



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


def sum_pitch_control(merged_df, teams_df, radius_m = 4.0):
    params = pc.default_model_params()
    melted_df = flatten_df(merged_df, teams_df)
    results = []
    
    for event_id in tqdm(melted_df["event_id"].unique()):
        event_df = melted_df[melted_df["event_id"] == event_id].copy()
        summed = 0.0  # 기본값

        try:        
            PPCFa, xgrid, ygrid = pc.generate_pitch_control_for_event(
                event_df.iloc[0], event_df, params, 
                field_dimen=(C.PITCH_X_MAX-C.PITCH_X_MIN, C.PITCH_Y_MAX-C.PITCH_Y_MIN,), 
                n_grid_cells_x=52, n_grid_cells_y=34, offsides=False, #52.2
            )        
                
            ball_x = event_df.iloc[0]['ballx']
            ball_y = event_df.iloc[0]['bally']

            # 1. xgrid, ygrid로 meshgrid 만들기 → 전체 셀의 좌표 (2D)
            X, Y = np.meshgrid(xgrid, ygrid)  # shape = (rows, cols)

            # 2. 각 셀 중심과 공 위치 간 거리 계산
            distance = np.sqrt((X - ball_x)**2 + (Y - ball_y)**2)

            # 3. radius 안에 있는 셀 마스크
            in_radius_mask = distance <= radius_m

            # 4. 유효한 셀 마스크 (NaN 제외)
            valid_mask = ~np.isnan(PPCFa)

            # 5. 두 조건 모두 만족하는 셀만 합산
            final_mask = in_radius_mask & valid_mask
            summed = PPCFa[final_mask].sum()

        except Exception as e:
            print(f"[event_id {event_id}] pitch control 계산 실패: {e}")

        results.append({
            "summed_pitch_control": summed
        })
    return pd.DataFrame(results)
    


if __name__=="__main__":
    traces_file = os.path.join(current_dir, 'notebook', 'traces_df.csv')
    teams_file = os.path.join(current_dir, 'notebook', 'teams.csv')

    traces_df = pd.read_csv(traces_file)
    traces_df['event_id'] = traces_df.index
    
    teams = pd.read_csv(teams_file)

    
    # pitch control 값 구하기
    result = sum_pitch_control(traces_df.iloc[:100,:], teams)
    print(result)
