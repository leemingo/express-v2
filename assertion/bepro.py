"""BePro data to LSDP converter.

This module converts BePro format soccer data to LSDP (Labelled Soccer Data Protocol)
format. It handles data cleaning, coordinate transformations, event parsing,
and various data corrections to ensure consistency and quality.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame
from scipy.optimize import linear_sum_assignment # Hungarian algorithm
from assertion.schema import LSDPSchema
import assertion.config as lsdpconfig
# from datatools import spadl
# import datatools.lsdp as lsdp

field_length = 105
field_width = 68

# Pitch dimensions for consistency with main project
PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0

HEIGHT_POST = 2.5
TOUCH_LINE_LENGTH = 105
GOAL_LINE_LENGTH = 68

LEFT_POST = 0.449 * GOAL_LINE_LENGTH # convert ratio to meter
RIGHT_POST = 0.551 * GOAL_LINE_LENGTH # convert ratio to meter
CENTER_POST = (LEFT_POST+RIGHT_POST) / 2

Eighteen_YARD = 16.4592 # 18yard = 16.4592meter

def convert_to_actions(events: pd.DataFrame) -> DataFrame[LSDPSchema]:
    """Convert K-league events to LSDP actions.
    
    This function transforms BePro format event data into LSDP format,
    performing data cleaning, coordinate transformations, and various
    corrections to ensure data quality and consistency.
    
    Args:
        events: DataFrame containing BePro format event data.
        
    Returns:
        DataFrame[LSDPSchema]: Converted events in LSDP format.
    """

    events["period_id"] = events["period_order"] + 1 
    events = events.rename(columns={"events": "event_types"})

    events = events.sort_values(["period_id", "event_time"], kind="mergesort").reset_index(drop=True) 
    # 분석에 활용하지 않은 이벤트 제거: 결측치 & 중복값
    events = _clean_events(events, remove_event_types=["HIR", "MAX_SPEED", "VHIR", "Assists", "Key Passes", "SPRINT", "Set Piece Defence"])

    # preprocess location: 이전에 일단 처리한 상태라서 주석 처리함
    events = _convert_locations(events)   

    # reactor: 이벤트에 반응하는 선수. Goal Conceded, Own Goal, Foul, Duel
    events["reactor_team_id"], events["reactor_player_id"] = -1, -1
    events["reactor_event_id"] = -1
    events = _find_duel_pairs(events) # Duel이벤트를 pair로 묶는 작업

    # 동일한 특성을 갖거나 세분화된 특성을 갖는 경위: foul(foul won), setpiece(corner, goalkick..), save(catch, parry)...
    events = _simplify(events)
    events = _parse_event(events)
    events = add_related_info(events)
    actions = pd.DataFrame()

    actions["game_id"] = events.game_id.astype(int)
    actions["original_event_id"] = events.event_id.astype(object)
    actions["action_id"] = range(len(actions))
    actions["period_id"] = events.period_id.astype(int)

    # convert minute to absolute second
    # event_time(bepro): Time When the event occurred during the match, in the format of milliseconds after the match started
    # First half kick-off: 0(ms), second half kick-off: 2,700,000(ms)
    # Value can be greater than 2,700,000(ms) in the first half bacause of the stoppage time(period로 구분)
    actions["time_seconds"] = (
        events["event_time"] * 0.001 #convert milliseconds to seconds
        - ((events.period_id > 1) * 45 * 60) # convert 45(minutes) to 45*60(seconds)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )
    actions["relative_time_seconds"] = (
        events["relative_event_time"] * 0.001 #convert milliseconds to seconds
        - ((events.period_id > 1) * 45 * 60) # convert 45(minutes) to 45*60(seconds)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )

    actions["team_id"] = events.team_id.fillna(-1).astype(int)
    actions["player_id"] = events.player_id.fillna(-1).astype(int)
    actions["relative_player_id"] = events.relative_player_id.fillna(-1).astype(int)
    actions["reactor_team_id"] = -1
    actions["reactor_player_id"] = -1

    actions["type_name"] = events.type_name.apply(lambda x: x.get("event_name"))
    actions["type_id"] = actions.type_name.apply(lambda x: lsdpconfig.actiontypes.index(x))
    
    actions["result_name"] = events.type_name.apply(lambda x: x.get("property", {}).get("Outcome", np.nan))
    actions["result_name"] = actions.result_name.map({
        "Succeeded": "Successful",
        "Failed": "Unsuccessful",
        "Tackle Succeeded: No Possession": "Successful",
        "Tackle Succeeded: Possession": "Successful",
        "Tackle Failed": "Unsuccessful",
        "Shots Off Target": "Off Target",
        "Blocked Shots": "Blocked",
        "Shots On Target": "On Target",
        "Goals": "Goal",
        "Keeper Rush-outs": "Keeper Rush-outs"

    })
    actions["relative_id"] = events.relative_id.fillna(-1).astype(int)
    actions["pair_id"] = events.pair_id.fillna(-1).astype(int)
    
    # K-league경기장 형태를 SPADL형태로 변환 : 68x105 -> 105x68로 변환
    actions["start_x"] = events.x
    actions["start_y"] = events.y
    actions["end_x"] = events.to_x
    actions["end_y"] = events.to_y

    actions = _fix_shot(actions, home_team_id=events[events["team"] == "Home"].team_id.iloc[0])
    actions = _fix_clearances(actions)
    actions = _fix_dribble(actions)
    actions = _fix_end_location(actions) # interpolate missing end_location

    actions["dx"] = actions.end_x - actions.start_x 
    actions["dy"] = actions.end_y - actions.start_y

    return cast(DataFrame[LSDPSchema], actions)

def _convert_locations(events: pd.DataFrame) -> pd.DataFrame:
    """Convert coordinate system to standard orientation.
    
    This function normalizes the coordinate system so that the home team
    always attacks from left to right, and the away team from right to left.
    
    Args:
        events: DataFrame containing event data with coordinates.
        
    Returns:
        pd.DataFrame: Events with normalized coordinate system.
    """

    # left_to_right: Home팀이 왼쪽에서 오른쪽으로 공격하는 방향
    is_home = events.team == "Home"
    flip_xy = (
        (is_home & (events['attack_direction'] == 'LEFT')) | # Home팀이 왼쪽으로 공격하고 있으면 flip
        (~is_home & (events['attack_direction'] == 'RIGHT')) # Home팀이 오른쪽으로 공격하고 있으면 flip
    )

    events.loc[flip_xy, ['x', 'to_x']] = 1 - events.loc[flip_xy, ['x', 'to_x']].values
    events.loc[flip_xy, ['y', 'to_y']] = 1 - events.loc[flip_xy, ['y', 'to_y']].values

    events[['x', 'to_x']] *= field_length
    events[['y', 'to_y']] *= field_width

    return events

def _clean_events(df_events: pd.DataFrame, remove_event_types) -> pd.DataFrame:
    """Clean event data by removing specified event types and duplicates.
    
    This function removes unwanted event types and handles missing/duplicate data.
    
    Args:
        df_events: DataFrame containing event data.
        remove_event_types: List of event types to remove.
        
    Returns:
        pd.DataFrame: Cleaned event data.
        
    Note:
        - Missing data conditions: Events with empty event_types or missing
          team_id/player_id information are removed.
        - Duplicate conditions: Events with duplicate event_id or duplicate
          data in other columns are removed.

    - 결측치 조건(missing_cond)
    1.24    [Duel]  [Aerial]    ... NaN NaN NaN NaN NaN NaN [{'eventType': 'Duel', 'subEventType': 'Aerial...   NaN NaN 35
    67  85848   94916404    1   4641.0  259769.0    163181  0.193699    0.5342 event_types이 기록되어 있지 않는 경우 -> parsing이 불가능함, 단순히 이전 정보만으로는 예측 불가능
    2. event_types이 기록되어 있는데, team_id & player_id정보가 기록되어 있지 않는 경우 -> 이전 정보로는 불가능하겠지만, 다음 정보로는 가능함.
    
    - 중복 데이터(duplicated_cond)
    1. event_id가 중복되는 경우 -> 제거
    2. event_id는 다른데 그 외 데이터가 중복되는 경우 -> 제거
    """

    # 해당 이벤트만 제거
    # ex) Pass + Control Under Pressure -> Pass 
    df_events["event_types"] = df_events["event_types"].apply(
        lambda event_list: [event for event in event_list if ("event_name" in event.keys()) and (event["event_name"] not in remove_event_types)]
    )

    # 처음부터 빈 리스트이거나 제거하므로써 빈 리스트가 된 행 제거
    missing_cond = (
        (df_events['event_types'].apply(len) == 0)
        # | (events["team_id"].isna())
        # | (events["player_id"].isna())
    )
    df_events = df_events[~missing_cond].reset_index(drop=True)

    # keep=fist: 중복된 데이터 중 첫번째 데이터만 남기고 나머지 제거(첫번째 데이터만 False)
    df = df_events.copy()
    duplicated_cond1 = df.duplicated(subset="event_id", keep="first") # 첫번째 데이터만 False -> ~False=True

    non_event_cols = list(filter(lambda col: col != "event_id", df.columns)) # event_id 이외 컬럼
    for col in non_event_cols:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) or isinstance(x, dict) else x) # duplicated함수는 list, dict를 지원하지 않음
    duplicated_cond2 = df.duplicated(subset=non_event_cols, keep="first") # 첫번째 데이터만 False -> ~False=True
    
    df_events = df_events[~(duplicated_cond1 | duplicated_cond2)].reset_index(drop=True)

    return df_events.reset_index(drop=True)

def add_related_info(events):
    events[['relative_player_id', 'relative_id', 'relative_event_time']] = np.nan
    for i in range(len(events) - 1):
        cur = events.iloc[i]
        next = events.iloc[i+1:i+10] # 최소 10개 내 receive 이벤트를 찾음


        # 성공한 패스의 경우 receive 이벤트를 찾아서 related_id에 추가
        success_pass = (
            (cur["type_name"].get("event_name", None) in ['Pass_Freekick', 'Pass_Corner', 'Throw-In', 'Goal Kick', 'Pass', 'Cross']) &
            (cur["type_name"].get("property", {}).get("Outcome", None) == 'Succeeded')
        )
        if success_pass:
            receival = next[next["type_name"].apply(lambda x: x.get("event_name") in ['Pass Received', 'Cross Received'])]
            if not receival.empty:
                # game_di = 126298, event_id = 2563: 정호연 선수가 패스한 공을 정호연 선수가 받은 경우 
                # 결론: passer = receiver인 경우는 사용X. event_xy != observe_xy이기 때문에 feature(observe_xy)로 잘못 들어갈 수 있음
                if receival.iloc[0]["player_id"] == cur["player_id"]:
                    print(f"passer = receiver -> game_id: {cur.game_id}, event_id: {cur.event_id}, passer: {cur.player_name}, receiver: {receival.iloc[0].player_name}")
                else:
                    events.at[i, 'relative_id'] = receival.iloc[0]['event_id']
                    events.at[i, 'relative_player_id'] = receival.iloc[0]['player_id']
                    events.at[i, 'relative_event_time'] = receival.iloc[0]['event_time']
                    if pd.notna(events.at[i, 'to_x']) and pd.notna(events.at[i, 'to_y']):
                        events.at[i, 'to_x'] = receival.iloc[0]['x']
                        events.at[i, 'to_y'] = receival.iloc[0]['y']
            else:
                print(f"No receival event found for pass: {cur.game_id}, event_id: {cur.event_id}, passer: {cur.player_name}, qualifier: {cur}")
        
    return events

def _fix_shot(df_actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Fix shot events by correcting end coordinates based on shot result.
    
    This function corrects the end coordinates of shot events based on
    their results (goal, blocked, off target, etc.) to ensure logical
    consistency in the data.
    
    Args:
        df_actions: DataFrame containing action data.
        home_team_id: ID of the home team for coordinate calculations.
        
    Returns:
        pd.DataFrame: Actions with corrected shot coordinates.
    """
    away_idx = df_actions["team_id"] != home_team_id

    shot_type = [lsdpconfig.actiontypes.index("Shot"), lsdpconfig.actiontypes.index("Shot_Freekick"), 
                lsdpconfig.actiontypes.index("Shot_Corner"), lsdpconfig.actiontypes.index("Penalty Kick")]
    shot = df_actions["type_id"].isin(shot_type)
    owngoal = df_actions["type_id"] == lsdpconfig.actiontypes.index("Own Goal")

    # 왼쪽 측면에서의 슛은 왼쪽 골 포스트 바깥으로 설정
    out_left_idx = (
        df_actions["start_x"] < (LEFT_POST - Eighteen_YARD)
    )
    df_actions.loc[shot & out_left_idx, "end_x"] = TOUCH_LINE_LENGTH
    df_actions.loc[shot & out_left_idx, "end_y"] = LEFT_POST - Eighteen_YARD
    df_actions.loc[shot & out_left_idx & away_idx, "end_x"] = lsdpconfig.field_length - df_actions.loc[shot & out_left_idx & away_idx, "end_x"].values   
    df_actions.loc[shot & out_left_idx & away_idx, "end_y"] = lsdpconfig.field_width - df_actions.loc[shot & out_left_idx & away_idx, "end_y"].values 

    # 오른쪽 측면에서의 슛은 오른쪽 골 포스트 바깥으로 설정
    out_right_idx = (
        df_actions["start_x"] > (RIGHT_POST + Eighteen_YARD)
    )
    df_actions.loc[shot & out_right_idx, "end_x"] = TOUCH_LINE_LENGTH 
    df_actions.loc[shot & out_right_idx, "end_y"] = RIGHT_POST + Eighteen_YARD
    df_actions.loc[shot & out_right_idx & away_idx, "end_x"] = lsdpconfig.field_length - df_actions.loc[shot & out_right_idx & away_idx, "end_x"].values
    df_actions.loc[shot & out_right_idx & away_idx, "end_y"] = lsdpconfig.field_width - df_actions.loc[shot & out_right_idx & away_idx, "end_y"].values

    # 중앙에서의 슛은 중앙 골 포스트 방향으로 설정
    out_center_idx = (
        (df_actions["start_x"] >= (LEFT_POST - Eighteen_YARD))
        & (df_actions["start_x"] <= (RIGHT_POST + Eighteen_YARD))
    )
    df_actions.loc[shot & out_center_idx, "end_x"] = TOUCH_LINE_LENGTH 
    df_actions.loc[shot & out_center_idx, "end_y"] = CENTER_POST
    df_actions.loc[shot & out_center_idx & away_idx, "end_x"] = lsdpconfig.field_length - df_actions.loc[shot & out_center_idx & away_idx, "end_x"].values
    df_actions.loc[shot & out_center_idx & away_idx, "end_y"] = lsdpconfig.field_width - df_actions.loc[shot & out_center_idx & away_idx, "end_y"].values

    # 자책골의 경우, 우리 팀 진영의 중앙 골 포스트로 설정
    df_actions.loc[owngoal, "end_x"] = CENTER_POST
    df_actions.loc[owngoal, "end_y"] = 0
    df_actions.loc[owngoal & away_idx, "end_x"] = lsdpconfig.field_length - df_actions.loc[owngoal & away_idx, "end_x"].values
    df_actions.loc[owngoal & away_idx, "end_y"] = lsdpconfig.field_width - df_actions.loc[owngoal & away_idx, "end_y"].values

    # 블로킹된 슛의 경우, 다음 블로킹 이벤트의 위치로 설정
    # 주의: 블로킹한 액션은 수비팀의 액션이기 때문에 위치는 수비 진영을 기준으로 기록되어있음
    # outcome: 'Shots Off Target', 'Shots On Target', 'Blocked Shots', 'Goals', 'Keeper Rush-outs'
    blocked_idx = shot & (df_actions["result_name"] == "Blocked")

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1) 
    df_actions.loc[blocked_idx, "end_x"] = df_actions_next.loc[blocked_idx, "start_x"].values
    df_actions.loc[blocked_idx, "end_y"] = df_actions_next.loc[blocked_idx, "start_y"].values

    goalkeeper_type = [
        lsdpconfig.actiontypes.index("Aerial Clearance"), 
        lsdpconfig.actiontypes.index("Defensive Line Support"), 
        lsdpconfig.actiontypes.index("Catch"),
        lsdpconfig.actiontypes.index("Parry")
    ]
    goalkeeper_idx = shot & df_actions_next["type_id"].isin(goalkeeper_type)
    df_actions.loc[goalkeeper_idx, "end_x"] = df_actions_next.loc[goalkeeper_idx, "start_x"].values
    df_actions.loc[goalkeeper_idx, "end_y"] = df_actions_next.loc[goalkeeper_idx, "start_y"].values

    return df_actions

def _fix_end_location(actions):
    """Fix missing end locations for various event types.
    
    This function interpolates missing end coordinates for events that
    don't have them defined, using various heuristics based on event type.
    
    Args:
        actions: DataFrame containing action data.
        
    Returns:
        pd.DataFrame: Actions with interpolated end locations.
    """
    for _, period_actions in actions.groupby("period_id"):
        for idx, row in period_actions.iterrows():
            # 끝 위치가 정의되어 있는 경우는 건너뛰기
            if row[["end_x", "end_y"]].notna().all():
                continue
                 
            if row.type_name in ["Pass_Freekick", "Shot_Freekick", "Pass_Corner", "Shot_Corner", "Penalty Kick", "Throw-In", "Goal Kick"]:
                # 대부분의 Set Pieces는 Pass or Shot과 동시에 기록되기 때문에 fix_shot, add_related_info에서 정의가 가능하다.
                # 그러니 일부 이벤트는 Pass or Shot을 동시에 기록하지 않아서 끝 위치가 존재함에도 대체되지가 않는 경우가 존재한다.
                if idx < len(period_actions) - 1:
                    period_actions.at[idx, "end_x"] = period_actions.at[idx+1, "start_x"]
                    period_actions.at[idx, "end_y"] = period_actions.at[idx+1, "start_y"]
                else:
                    period_actions.at[idx, "end_x"] = row.start_x
                    period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Catch", "Parry", "Aerial Clearance"]:
                # 위 3가지 이벤트는 끝 위치 = 시작 위치 동일
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Defensive Line Support"]:
                # 오프 더 볼 움직임으로 끝 위치 재정의할 수 있을 줄 알았지만 실제로 이전 or 이후 중 어느 위치로 이동하는지 알 수 없음
                # https://www.notion.so/Let-s-See-the-Unseen-9ac6866c529e4ee399b7bd7a5ebab254?pvs=4#4ee0eb3584614e01aacc4f3da1bacdb8
                # if idx < len(period_actions) - 1:
                #     period_actions.at[idx, "end_x"] = period_actions.at[idx+1, "start_x"]
                #     period_actions.at[idx, "end_y"] = period_actions.at[idx+1, "start_y"]
                # else:
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Own Goal", "Goal", "Goal Conceded"]:
                # 실책/득점이벤트는 끝 위치가 정의되지 않음
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Duel", "Tackle", "Intervention", "Interception", "Block"]:
                # 수비이벤트는 끝 위치가 정의되지 않음
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Pass Received", "Cross Received", "Ball Received", "Recovery"]:
                # Received이벤트는 끝 위치 = 시작위치 동일
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Foul Won", "Error", "Foul", "Handball_Foul", "Foul_Throw", "Deflection", "Offside", "Hit", "Pause"]:
                # 끝 위치가 정의되지 않음
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Take-On"]:
                # 상대방을 돌파하는 이벤트는 끝 위치가 정의되지 않음
                period_actions.at[idx, "end_x"] = row.start_x
                period_actions.at[idx, "end_y"] = row.start_y
            elif row.type_name in ["Carry", "Clearance", "Shot", "Pass", "Cross"]:
                # 별도 처리
                # Carry: _fix_dribble
                # Clearance: _fix_clearances
                # Shot, Pass, Cross: fix_shot, add_related_info
                pass
            else:
                raise ValueError(f"Unexpected value: {row.type_name}")
            
        actions.update(period_actions)

    return actions

def _find_duel_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """Find and pair duel events using Hungarian algorithm.
    
    This function identifies duel events and pairs them based on
    temporal proximity using the Hungarian algorithm for optimal matching.
    
    Args:
        events: DataFrame containing event data.
        
    Returns:
        pd.DataFrame: Events with paired duel information.
    """
    def _pair_duel_events(period_group: pd.DataFrame) -> pd.DataFrame:
        has_duel = period_group["event_types"].apply(lambda ets: any(e["event_name"] == "Duels" for e in ets))
        duel_events = period_group[has_duel].reset_index(drop=False) # drop=False: index를 기준으로 병합및 정렬
        other_events = period_group[~has_duel].reset_index(drop=False)
        
        if len(duel_events) < 2:
            return period_group

        # Create a cost matrix based on event_time differences: Hungarian algorithm
        cost_matrix = abs(duel_events["event_time"].values[:, None] - duel_events["event_time"].values).astype(np.float64)
        np.fill_diagonal(cost_matrix, np.inf) # diagonal: 자기 자신과의 거리는 무한대로 설정(not mapping to itself)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        paired_indices = [[i, j] for i, j in zip(row_ind, col_ind) if i < j]
        paired_indices.sort(key=lambda x: abs(duel_events.at[x[0], "event_time"] - duel_events.at[x[1], "event_time"])) 

        # 시간차이가 가장 작은 조합 순으로 정렬
        selected_pairs = []
        for i, j in paired_indices:
            if i not in selected_pairs and j not in selected_pairs:
                duel_events.at[i, "pair_id"] = duel_events.at[j, "event_id"]
                duel_events.at[j, "pair_id"] = duel_events.at[i, "event_id"]

                selected_pairs.extend([i, j])

        # Handle unselected duel events (if any)
        unselected_indices = set(duel_events.index) - set(selected_pairs)
        duel_events.loc[list(unselected_indices), "pair_id"] = np.nan

        # Update the original group with paired duel events
        # 주의 사항: 동일한 시점에 발생한 이벤트는 sort_values함수 사용시 순서가 변경될 수 있음
        period_group = pd.concat(
            [other_events, duel_events], 
            axis=0, ignore_index=True, sort=False
            ).sort_values("index").drop(columns=["index"]).reset_index(drop=True)

        return period_group
    
    events = events.groupby("period_id").apply(_pair_duel_events)

    return events.reset_index(drop=True)

def _simplify(events: pd.DataFrame) -> pd.DataFrame:
    """Simplify event types by mapping to VERSA action types.
    
    This function maps complex event types to simplified VERSA action types
    for consistency in the LSDP format.
    
    Args:
        events: DataFrame containing event data.
        
    Returns:
        pd.DataFrame: Events with simplified action types.
    """
    # VERSA에서 사용하는 actiontype으로 매핑
    actiontype_mapping = {
        #'Set Piece Defence': 'Clearance',
        'Turnover': 'Error',
        'Shots & Goals': 'Shot',
        'Step-in': 'Carry',
        'Aerial Control': 'Aerial Clearance',
        'Passes' : 'Pass',
        'Tackles' : 'Tackle',
        'Blocks' : 'Block',
        'Set Pieces' : 'Set-piece',
        'Clearances' : 'Clearance',
        'Recoveries' : 'Recovery',
        'Mistakes' : 'Error',
        'Duels' : 'Duel',
        'Fouls' : 'Foul',
        'Crosses' : 'Cross',
        'Interceptions' : 'Interception',
        'Offsides' : 'Offside',
        "Passes Received": "Pass Received",
        "Crosses Received": "Cross Received",
        "Goals Conceded": "Goal Conceded",
        "Take-on": "Take-On",
        "Saves": "Save",
        "Defensive Line Supports": "Defensive Line Support",
        "Own Goals": "Own Goal",
    }

    events = events.copy()
    # 액션 타입을 VERSA에서 사용하는 액션 타입으로 매핑
    events = events[events["event_types"].apply(lambda x: any("event_name" in r for r in x))].reset_index(drop=True)
 
    events['event_types'] = events['event_types'].apply(
        lambda x: [{**r, "event_name": actiontype_mapping[r["event_name"]]} for r in x]
    )

    # 1. set_piece simplify: set_piece과 동시에 발생한 이벤트 정보는 set_piece의 부가적인 특성(pass + mistake)이므로 통합
    # ex) pass + set_piece -> set_piece, pass + mistake + set_piece -> set_piece
    not_set_piece = events[events['event_types'].apply(lambda x: not any(r['event_name'] == 'Set-piece' for r in x))].copy()
    set_piece = events[events['event_types'].apply(lambda x: any(r['event_name'] == 'Set-piece' for r in x))].copy()

    throw_in_cond = set_piece['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Throw-Ins' for r in x if 'property' in r))
    corner_cond = set_piece['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Corners' for r in x if 'property' in r))
    freekick_cond = set_piece['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Freekicks' for r in x if 'property' in r))
    goalkick_cond = set_piece['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Goal Kicks' for r in x if 'property' in r))
    penalty_kick_cond = set_piece['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Penalty Kicks' for r in x if 'property' in r))

    pass_cond = set_piece['event_types'].apply(lambda x: any(r['event_name'] in ['Pass', 'Cross'] for r in x))
    shot_cond = set_piece['event_types'].apply(lambda x: any(r['event_name'] == 'Shot' for r in x))

    # 새로운 events 컬럼 생성
    set_piece.loc[throw_in_cond, 'event_types'] = set_piece.loc[throw_in_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Throw-In"} for r in x if r['event_name'] in ['Pass', 'Cross']])
    set_piece.loc[corner_cond & pass_cond, 'event_types'] = set_piece.loc[corner_cond & pass_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Pass_Corner"} for r in x if r['event_name'] in ['Pass', 'Cross']])
    set_piece.loc[corner_cond & shot_cond, 'event_types'] = set_piece.loc[corner_cond & shot_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Shot_Corner"} for r in x if r['event_name'] == 'Shot'])
    set_piece.loc[freekick_cond & pass_cond, 'event_types'] = set_piece.loc[freekick_cond & pass_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Pass_Freekick"} for r in x if r['event_name'] in ['Pass', 'Cross']])
    set_piece.loc[freekick_cond & shot_cond, 'event_types'] = set_piece.loc[freekick_cond & shot_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Shot_Freekick"} for r in x if r['event_name'] == 'Shot'])
    set_piece.loc[goalkick_cond, 'event_types'] = set_piece.loc[goalkick_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Goal Kick"} for r in x if r['event_name'] in ['Pass', 'Cross']])
    set_piece.loc[penalty_kick_cond, 'event_types'] = set_piece.loc[penalty_kick_cond, "event_types"].apply(lambda x: [{"property": r['property'], "event_name": "Penalty Kick"} for r in x if r['event_name'] == 'Shot'])
    
    events = pd.concat([not_set_piece, set_piece], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

    # 2. Error 제거: Error과 동시에 발생한 이벤트 정보는 그 외 이벤트(pass + mistake)의 부가적인 특성이 mistake이므로 mistake 제거
    # ex) pass + mistake -> pass
    not_mistake = events[events['event_types'].apply(lambda x: not any(r['event_name'] == 'Error' for r in x))].copy()
    mistake = events[events['event_types'].apply(lambda x: any(r['event_name'] == 'Error' for r in x))].copy()
    mistake['event_types'] = mistake['event_types'].apply(
        lambda x: x if len(x) == 1 else [r for r in x if r['event_name'] != 'Error']
    )
    events = pd.concat([not_mistake, mistake], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

    # 3. cross simplify: cross과 동시에 발생한 Pass 이벤트는 cross의 부가적인 특성(pass + cross)이므로 통합
    # ex) pass + cross -> cross
    non_cross = events[events['event_types'].apply(lambda x: not any(r['event_name'] == 'Cross' for r in x))].copy()
    cross = events[events['event_types'].apply(lambda x: any(r['event_name'] == 'Cross' for r in x))].copy()
    cross['event_types'] = cross['event_types'].apply(
        lambda x: [r for r in x if r['event_name'] != 'Pass']
    )
    events = pd.concat([non_cross, cross], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

    # 4. foul & foul won: foul의 부가적인 특성(foul + foul won)을 기준으로 분리
    # foul won: Fouls Won, Penalty Kick Won
    non_foul = events[events['event_types'].apply(lambda x: not any(r['event_name'] == 'Foul' for r in x))].copy()
    foul = events[events['event_types'].apply(lambda x: any(r['event_name'] == 'Foul' for r in x))].copy()
    foulwon_cond = foul['event_types'].apply(
        lambda x: any(
            (r['event_name'] == 'Foul') and
            ((r['property'].get('Type', None) == "Fouls Won") or
            (r['property'].get('Penalty Kick Won', 'False') == 'True')) # 'False', 'True'가 boolean이 아닌 string으로 되어있음
            for r in x
        )
    )
    foulwon = foul[foulwon_cond].copy()
    foul = foul[~foulwon_cond].copy()
    foulwon['event_types'] = foulwon['event_types'].apply(
        lambda x: [{**r, "event_name": "Foul Won"} if r['event_name'] == 'Foul' else r for r in x]
    )
    events = pd.concat([non_foul, foul, foulwon], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)
    
    # 5. saves -> catch or parry로 변환
    saves = events[events['event_types'].apply(lambda x: any(r['event_name'] == 'Save' for r in x))].copy()
    non_saves = events[events['event_types'].apply(lambda x: not any(r['event_name'] == 'Save' for r in x))].copy()
    catch_cond = saves['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Catches' for r in x))
    parry_cond = saves['event_types'].apply(lambda x: any(r['property'].get('Type') == 'Parries' for r in x))
    saves.loc[catch_cond, 'event_types'] = saves.loc[catch_cond, "event_types"].apply(
        lambda x: [{**r, "event_name": "Catch"} if r['property'].get('Type') == 'Catches' else r for r in x]
    )   
    saves.loc[parry_cond, 'event_types'] = saves.loc[parry_cond, "event_types"].apply(
        lambda x: [{**r, "event_name": "Parry"} if r['property'].get('Type') == 'Parries' else r for r in x]
    )
    events = pd.concat([non_saves, saves], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

    return events

# _fix_dribble : Update the end position of dribble events based on their success or failure.
# If the dribble failed, the end position is set to the position of the next event.
# If the dribble succeeded, the end position is set to the position of the next event that is not a tackle or interception.
def _fix_dribble(df_actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(df_actions, shift_value=-1)

    failed_tackle = (
        (next_actions['type_id'] == lsdpconfig.actiontypes.index('Tackle')) &
        (next_actions['result_name'] == "Unsuccessful")
    )
    failed_interception = (
        (next_actions['type_id'] == lsdpconfig.actiontypes.index('Interception')) &
        (next_actions['result_name'] == "Unsuccessful")
    )

    same_team = df_actions.team_id == next_actions.team_id
    failed_defensive = (failed_tackle | failed_interception) & ~same_team

    # next_actions : 실패한 수비가 아닌 이벤트의 위치를 드리블의 끝 위치로 활용하고자하는 목적
    next_actions = next_actions.mask(failed_defensive)[["start_x", "start_y"]].bfill()

    selector_dribble = df_actions.type_id == lsdpconfig.actiontypes.index("Carry")

    df_actions.loc[selector_dribble, "end_x"] = next_actions.loc[selector_dribble, "start_x"].values
    df_actions.loc[selector_dribble, "end_y"] = next_actions.loc[selector_dribble, "start_y"].values

    return df_actions

def _parse_event(df_events: pd.DataFrame) -> pd.DataFrame:
    '''
    여러가지 이벤트 데이터의 액션 타입을 통일하기 위한 작업
    ex) Passes, Key Passes -> Pass
    '''
    events = df_events.copy()
    priority_map = lsdpconfig.priority_map

    new_rows = []
    for _, row in events.iterrows():
        event_list = row["event_types"]
        
        sorted_event_list = sorted(event_list, key=lambda e: priority_map.get(e["event_name"]))

        # Debugging: 동일한 우선순위를 가진 이벤트가 있으면 오류
        priorities = [priority_map.get(e["event_name"]) for e in sorted_event_list]
        if len(priorities) != len(set(priorities)):
            # 1. Pass & Foul Won: 동시에 등장하는 시나리오가 예상은 가지만, 이전에 2021~2024시즌 데이터를 처리하면서 단 한번도 예외 경우를 발생시킨 적이 없음.
            # 즉, 이러한 경우가 기존에 한번도 없었는데, 이번에 발생한 사례
            print(row.event_id, event_list, " -> ", priorities)
            pass_cond = any(e["event_name"] in ["Pass", "Cross"] for e in event_list)
            foulwon_cond = any(e["event_name"] == "Foul Won" for e in event_list)
            if pass_cond and foulwon_cond:
                temp_priority_map = priority_map.copy()
                temp_priority_map["Foul Won"] = temp_priority_map["Pass"] + 1
                sorted_event_list = sorted(event_list, key=lambda e: temp_priority_map.get(e["event_name"]))
                priorities = [temp_priority_map.get(e["event_name"]) for e in sorted_event_list]
                print("임시 해결: ", sorted_event_list, " -> ", priorities)
            else:
                raise ValueError(f"Duplicate priority values found for events: {event_list}")
        
        new_rows.extend([
            {**row.to_dict(), "type_name": event}
            for event in sorted_event_list
        ])

    return pd.DataFrame(new_rows)

def shift_with_edge_fix(actions: pd.DataFrame, shift_value: int) -> pd.DataFrame:
    """
    Shift each group by a specified value and fill NaN only in the first or last row.
    Does not alter NaN values in the middle of the group to avoid affecting naturally missing data.
    """

    shift_action = actions.groupby("period_id").shift(shift_value)

    if shift_value < 0:
        # When shifting upwards, last row gets NaN, so fill it with the original last value
        fill_indices = actions.groupby("period_id").tail(abs(shift_value)).index
    else:
        # When shifting downwards, first row gets NaN, so fill it with the original first value
        fill_indices = actions.groupby("period_id").head(abs(shift_value)).index

    # Fill the NaN rows with the corresponding original values
    shift_action.loc[fill_indices] = actions.loc[fill_indices]
    shift_action["period_id"] = actions["period_id"]

    return shift_action

def _fix_clearances(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    clearance_idx = actions.type_id == lsdpconfig.actiontypes.index("Clearance")

    actions.loc[clearance_idx, "end_x"] = next_actions.loc[clearance_idx, "start_x"].values
    actions.loc[clearance_idx, "end_y"] = next_actions.loc[clearance_idx, "start_y"].values

    return actions