
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from datasets import PressingSequenceDataset, SoccerMapInputDataset
import config as C
import features as F
from bisect import bisect_right
from collections import defaultdict
from collections import defaultdict
import torch
import pandas as pd
import os

def transform_feature_dict(feature_dict):
    """
    주어진 feature_dict의 값을 [0, 1] 범위로 Min-Max 스케일링합니다.
    결측치(NaN)는 변환 과정에서 제외되며, 최종적으로는 0.0으로 처리됩니다.
    """
    vals = [v for v in feature_dict.values() if pd.notna(v)]
    
    if not vals: # 값이 없거나 모두 NaN인 경우
        return {k: 0.0 if pd.isna(v) else v for k, v in feature_dict.items()} # 모든 NaN을 0.0으로, 그 외는 그대로 둠

    min_val = min(vals)
    max_val = max(vals)
    
    # 0으로 나누는 오류 방지: max_val과 min_val이 같으면 (모든 유효한 값이 동일하면) 1.0으로 나눔
    range_val = max_val - min_val if max_val != min_val else 1.0 
    
    transformed_dict = {}
    for k, v in feature_dict.items():
        if pd.notna(v):
            transformed_dict[k] = (v - min_val) / range_val
        else:
            transformed_dict[k] = 0.0 # NaN 값은 0.0으로 처리
    return transformed_dict

def update_features(dataset, feature_list, processed_data_path, transform = True, cache_dir="feature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    game_sample_map = defaultdict(list)

    # 1. game_id 기준으로 샘플 인덱스와 frame_id를 모음
    for idx, sample in enumerate(dataset): # dataset.__len__에 의해 결정되는 길이만큼 루프
        game_id, period_id_str, frame_id_str = sample['match_info'].split('-')
        frame_id = int(frame_id_str)
        period_id = int(float(period_id_str))
        game_sample_map[game_id].append((idx, period_id, frame_id))


    # 2. game_id 단위로 모든 처리 (데이터 로드, timestamp 매핑, 피처 계산, 데이터셋 업데이트)
    for game_id, frame_infos_for_game in game_sample_map.items():
        game_feature_dir = os.path.join(cache_dir, game_id)
        os.makedirs(game_feature_dir, exist_ok=True)
        print(f"\n--- Processing Game ID: {game_id} ---")

        tracking_df = pd.read_csv(os.path.join(processed_data_path, game_id, f"{game_id}_traces.csv"))
        events_df = pd.read_csv(os.path.join(processed_data_path, game_id, f"{game_id}_merged.csv"))
        teams_df = pd.read_csv(os.path.join(processed_data_path, game_id, f"{game_id}_teams.csv"))
        events_df["time_seconds"] = pd.to_numeric(events_df["time_seconds"], errors="coerce")

        # 2-1. frame_id -> timestamp 매핑 (스코프가 이 루프 내부로 제한됨)
        frame_id_to_ts = tracking_df.set_index(['period_id', 'frame_id'])['time_seconds'].to_dict() 

        precomputed_game_features = {} 
        # 2-2. 일단 전체에 대해 계산 진행
        for feature_name in feature_list: # 각 피처 이름에 대해
            feature_cache_path = os.path.join(game_feature_dir, f"{feature_name}.pkl")
            if os.path.exists(feature_cache_path):
                with open(feature_cache_path, 'rb') as f:
                    feat_result_df = pickle.load(f)
            else:
                func = getattr(F, feature_name)
                game_feature_dir = os.path.join(cache_dir, game_id)
                os.makedirs(game_feature_dir, exist_ok=True)
                if feature_name == "sum_pitch_control":
                    feat_result_df  = func(events_df, teams_df)                              
                else:
                    feat_result_df = func(events_df)  # Series with index=action_id
                with open(feature_cache_path, 'wb') as f:
                    pickle.dump(feat_result_df, f)
            raw_dict = feat_result_df.set_index('action_id')[feature_name].to_dict() 
            
            if transform:
                precomputed_game_features[feature_name] = transform_feature_dict(raw_dict)
            else:
                precomputed_game_features[feature_name] = raw_dict
                
        # 2-3. 샘플별 과거 이벤트 B개에 대해 feature 추출
        for sample_idx, period_id, fid in frame_infos_for_game: 
            ts = frame_id_to_ts.get((period_id, fid)) 
            current_sample_data_dict = dataset[sample_idx] 
            original_feature_tensor = current_sample_data_dict['features'] 
            B, N, D = original_feature_tensor.shape # B는 이 샘플의 과거 이벤트 수 (context_length)
            past_events = events_df[events_df["time_seconds"] < ts].head(B)
            event_ids = past_events["action_id"].tolist()

            # 각 feature별로 (B, 1) 텐서를 생성하고 합치기
            new_feature_list = []
            for feature_name in feature_list:
                feat_dict = precomputed_game_features[feature_name]
                feat_vals = [feat_dict.get(eid, np.nan) for eid in event_ids]  # (B,)
                feat_tensor = torch.tensor(feat_vals, dtype=torch.float32).view(B, 1)  # (B, 1)
                expanded = feat_tensor.unsqueeze(1).expand(-1, N, -1)  # (B, N, 1)
                new_feature_list.append(expanded)

            # 모든 feature 통합
            new_concat = torch.cat(new_feature_list, dim=-1)  # (B, N, F')
            final_feature_tensor = torch.cat([original_feature_tensor, new_concat], dim=-1)
            current_sample_data_dict['features'] = final_feature_tensor
            dataset[sample_idx] = current_sample_data_dict  # 최종 저장
            


if __name__ =="__main__":
    
    feature_list = [
                # 'distance_ball_goal', 'distance_ball_sideline',
                # 'distance_ball_goalline', 'actor_speed', 'angle_to_goal', 
                # 'elapsed_time', 'time_since_last_opponent_action', 'def_goal', 'att_goal', 'goal_diff', 'closest_defender_dist', 
                # 'closest_defender_speed', 'speed_diff_actor_defender', 'nb_of_3m_radius', 'nb_of_5m_radius','nb_of_10m_radius', 
                # 'dist_defender_to_sideline','dist_defender_to_goaline', 'diff_ball_defender_goalline', 'diff_ball_defender_sideline'
                'sum_pitch_control'
                ]
    
    data_path = "/home/exPress/express-v2/data/bepro/pressing_intensity"

    with open(f"{data_path}/train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open(f"{data_path}/valid_dataset.pkl", "rb") as f:
        valid_dataset = pickle.load(f)

    with open(f"{data_path}/test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    processed_data_path = "/home/exPress/express-v2/data/bepro/processed"

    update_features(train_dataset, feature_list, processed_data_path, cache_dir="/home/exPress/express-v2/data/bepro/feature")
    update_features(valid_dataset, feature_list, processed_data_path, cache_dir="/home/exPress/express-v2/data/bepro/feature")
    update_features(test_dataset, feature_list, processed_data_path, cache_dir="/home/exPress/express-v2/data/bepro/feature")