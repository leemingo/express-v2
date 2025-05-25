DEFAULT_PLAYER_SMOOTHING_PARAMS = {"window_length": 7, "polyorder": 1}
DEFAULT_BALL_SMOOTHING_PARAMS = {"window_length": 3, "polyorder": 1}
MAX_PLAYER_SPEED: float = 12.0
MAX_BALL_SPEED: float = 28.0
MAX_PLAYER_ACCELERATION: float = 6.0
MAX_BALL_ACCELERATION: float = 13.5
BALL_CARRIER_THRESHOLD: float = 25.0

PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0

class Constant:
    BALL = "ball"


class Column:
    BALL_OWNING_TEAM_ID = "ball_owning_team_id"
    BALL_OWNING_PLAYER_ID = "ball_owning_player_id"
    IS_BALL_CARRIER = "is_ball_carrier"
    PERIOD_ID = "period_id"
    TIMESTAMP = "timestamp"
    BALL_STATE = "ball_state"
    FRAME_ID = "frame_id"
    GAME_ID = "game_id"
    TEAM_ID = "team_id"
    OBJECT_ID = "id"
    POSITION_NAME = "position_name"

    X = "x"
    Y = "y"
    Z = "z"

    SPEED = "v"
    VX = "vx"
    VY = "vy"
    VZ = "vz"

    ACCELERATION = "a"
    AX = "ax"
    AY = "ay"
    AZ = "az"


class Group:
    BY_FRAME = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID]
    BY_FRAME_TEAM = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID, Column.TEAM_ID]
    BY_OBJECT_PERIOD = [Column.OBJECT_ID, Column.PERIOD_ID]
    BY_TIMESTAMP = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID, Column.TIMESTAMP]

TEAM_NAME = {'FC서울' : 'FC Seoul',
    '강원FC' : 'Gangwon FC',
    '광주FC' : 'Gwangju FC',
    '김천 상무 프로축구단' : 'Gimcheon Sangmu',
    '대구FC' : 'Daegu FC',
    '대전 하나 시티즌' : 'Daejeon Hana Citizen',
    '수원FC' : 'Suwon FC',
    '울산 HD FC' : 'Ulsan HD FC',
    '인천 유나이티드' : 'Incheon United',
    '전북 현대 모터스' : 'Jeonbuk Hyundai Motors',
    '제주SK FC' : 'Jeju SK FC',
    '포항 스틸러스' : 'Pohang Steelers'
    }
