
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
    melted['team_on_ball'] = melted['team_id'] == melted['ball_owning_team_id']
    melted['team_id'] = melted['team_id'].astype(int)
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

# 메인 병렬 처리 함수: groupby를 사용합니다.
def add_pitch_control_parallel(df, params, radius_m=4.0, max_workers=4):
    params = pc.default_model_params()
    melted_df = flatten_df(df)
    print(melted_df)
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

    traces_df = pd.read_csv(os.path.join(current_dir,'final_x_slice.csv'))
    # traces_df['order'] = traces_df.index

    params = pc.default_model_params()

    result = add_pitch_control_parallel(traces_df, params, radius_m=radius_m)
    print(result)

if __name__ == "__main__":
    fire.Fire(run)