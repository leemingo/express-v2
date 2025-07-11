"""Configuration of the LSDP.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the LSDP.
results : list(str)
    The action results used in the LSDP.
actiontypes : list(str)
    The action types used in the LSDP.

"""

import pandas as pd  # type: ignore

field_length = 105
field_width = 68

# Pitch dimensions for consistency with main project
PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0

priority_map = {
    "Pass_Freekick": 0,
    "Shot_Freekick": 0,
    "Pass_Corner": 0,
    "Shot_Corner": 0,
    "Penalty Kick": 0,
    "Throw-In": 0,
    "Goal Kick": 0,

    "Catch": 0,
    "Parry": 0,

    "Own Goal": 0,
    "Goal": 0,
    "Goal Conceded": 0,

    # "Second_Yellow_Card": 0,
    # "Direct_Red_Card": 0,
    # "Yellow_Card": 0,

    "Out": 0,
    "Goal Post": 0,
    "Goal Miss": 0,

    "Defensive Line Support": 1,
    "Aerial Clearance": 1,

    "Duel": 2,

    "Tackle": 3,

    "Intervention": 4,
    "Interception": 4,
    "Block": 4,

    "Pass Received": 5,
    "Cross Received": 5,
    "Ball Received": 5,
    "Recovery": 5, 

    "Foul Won": 6,
    "Error": 6, # 상대의 압박을 받지 않는 상황에서 선수가 볼 소유권을 잃는 경우, 공을 통제하지 못하여 루즈볼 상황을 만드는 경우,또는 공과 접촉하려 시도했으나 접촉에 실패한 모든 경우
    "Carry": 6,
    "Take-On": 6,
    "Shot": 6,
    "Clearance": 6,

    "Foul": 6,
    "Foul_Throw": 6,
    "Handball_Foul": 6,

    "Deflection": 6,
    "Offside": 6,
    "Hit": 6, 
    "Pass": 6,
    "Cross": 6,

    "Pause": 7, # 공이 플레이 중지된 경우
}

actiontypes = [
    "Pass_Freekick", "Shot_Freekick", "Pass_Corner", "Shot_Corner", "Penalty Kick",
    "Throw-In", "Goal Kick", "Catch", "Parry", "Own Goal", "Goal", 
    "Out", "Goal Post", "Goal Miss", "Defensive Line Support", "Aerial Clearance",
    "Duel", "Tackle", "Intervention", "Interception", "Block", "Pass Received",
    "Cross Received", "Ball Received", "Recovery", "Carry", "Take-On", "Shot",
    "Clearance", "Error", "Foul", "Foul_Throw", "Handball_Foul",
    "Deflection", "Offside", "Hit", "Pass", "Cross", "Pause"
]

versa_to_spadl_dict = {
    "Pass_Freekick":    "freekick_crossed",  
    "Shot_Freekick":    "shot_freekick",
    "Pass_Corner":      "corner_crossed",    
    "Shot_Corner":      "corner_crossed",  # 코너슛은 spadl에서 변환할 수 없는 액션이기 때문에 코너킥으로 처리한다.
    "Penalty Kick":     "shot_penalty",
    "Throw-In":         "throw_in",
    "Goal Kick":        "goalkick",
    "Catch":            "keeper_save",    # keeper_save, keeper_clain 둘 다 의미적으로 가능하지만, save로 통일
    "Parry":            "keeper_punch",   
    "Own Goal":         "non_action",      
    "Goal":             "non_action",       
    "Out":              "non_action",
    "Goal Post":        "non_action",
    "Goal Miss":        "non_action",
    "Defensive Line Support": "non_action",
    "Aerial Clearance": "non_action",    
    "Duel":             "non_action",
    "Tackle":           "non_action",     # 태클도 제거
    "Intervention":     "non_action",     # 인터벤션은 조금 애매한데..일단 보류
    "Interception":     "interception",
    "Block":            "block",
    "Pass Received":    "non_action",     # 패스 리시브도 제거
    "Cross Received":   "non_action",
    "Ball Received":    "non_action",
    "Recovery":         "recovery",       # spadl에 없는 액션 추가
    "Carry":            "dribble",       
    "Take-On":          "take_on",
    "Shot":             "shot",
    "Clearance":        "clearance",
    "Error":            "non_action",
    "Foul":             "foul",
    "Foul_Throw":       "foul",
    "Handball_Foul":    "foul",
    "Deflection":       "non_action",
    "Offside":          "non_action",
    "Hit":              "non_action",
    "Pass":             "pass",
    "Cross":            "cross",
    "Pause":            "non_action",
}

# original_version
actiontypes: list[str] = list(priority_map.keys())
spadl_actiontypes: list[str] = list(set(list(versa_to_spadl_dict.values())))

def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])

def spadl_actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(spadl_actiontypes)), columns=["type_id", "type_name"])