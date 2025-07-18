DEFAULT_PLAYER_SMOOTHING_PARAMS = {"window_length": 7, "polyorder": 1}
DEFAULT_BALL_SMOOTHING_PARAMS = {"window_length": 3, "polyorder": 1}
MAX_PLAYER_SPEED: float = 12.0
MAX_BALL_SPEED: float = 28.0
MAX_PLAYER_ACCELERATION: float = 6.0
MAX_BALL_ACCELERATION: float = 13.5
BALL_CARRIER_THRESHOLD: float = 25.0

PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2 # 52.5
CENTER_Y = PITCH_WIDTH / 2  # 34.0

# FEAT_MIN = [-26., -8., -2., -13., 0., -5., -8., 0.]
# FEAT_MAX = [54., 21., 6., 1., 14., 2., 3., 6.]
FEAT_MIN = [-52.5, -32., -4., -5., 0., -4., -4., 0., 0., 0., 15., -1., 0., 0., -1., -1., -1., -1.]
FEAT_MAX = [52.5, 32., 1., 3., 5., 5., 5., 6., 1., 1., 88., 1., 1., 52., 1., 1., 1., 1.]

# [x, y, vx, vy, v, a, ]
# a: -13 ~ 11
FEAT_MIN = [-52.5, -32., -12., -12., 0., -8.,-8, 0, 0., 0., 0., -1., -1, 0., -1., -1., -1., -1., 0]
FEAT_MAX = [52.5, 32., 12., 12., 12., 8, 8, 8, 1., 1., 113., 1., 1., 111., 1., 1., 1., 1., 0]

# FEAT_MIN = [-52.5, -32., -63., -54., 0., -970., -600., 0., 0., 0., 0., -1., -1, 0., -1., -1., -1., -1., 0]
# FEAT_MAX = [52.5, 32., 83., 153., 28., 1405., 625., 13.5, 1., 1., 113., 1., 1., 111., 1., 1., 1., 1., 0]

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

TEAMNAME2ID = {'Gwangju FC': '4648',
    'Pohang Steelers': '4639',
    'Gimcheon Sangmu': '2353',
    'Jeonbuk Hyundai Motors': '4640',
    'Jeju SK FC': '4641',
    'Daegu FC': '4644',
    'Incheon United': '4646',
    'FC Seoul': '316',
    'Daejeon Hana Citizen': '4657',
    'Suwon FC': '4220',
    'Gangwon FC': '4643',
    'Ulsan HD FC': '2354'
    }

TEAMID2NAME = {'4648': 'Gwangju FC',
    '4639': 'Pohang Steelers',
    '2353': 'Gimcheon Sangmu',
    '4640': 'Jeonbuk Hyundai Motors',
    '4641': 'Jeju SK FC',
    '4644': 'Daegu FC',
    '4646': 'Incheon United',
    '316': 'FC Seoul',
    '4657': 'Daejeon Hana Citizen',
    '4220': 'Suwon FC',
    '4643': 'Gangwon FC',
    '2354': 'Ulsan HD FC'
}