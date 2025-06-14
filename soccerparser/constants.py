class Constants:
    def __init__(self):
        self.PITCH_SIZE = (105, 68)

        self.EVENT_HEADER = ["event_id", "session", "time", "home_away", "player_code", "event_types"]

        # VALID_TYPES: 온 더 볼 플레이..
        # self.VALID_TYPES = {
        #     "cornerKick",
        #     "freeKick",
        #     "penaltyKick",
        #     "goalKickSucceeded",
        #     "goalKickFailed",
        #     "throwIn",
        #     "crossSucceeded",
        #     "crossFailed",
        #     "passSucceeded",
        #     "passFailed",
        #     "passReceived",
        #     "possession",
        #     "controlUnderPressure",
        #     "dribbleSucceeded",
        #     "dribbleFailed",
        #     "ballMissed",
        #     "intercept",
        #     "cutoff",
        #     "block",
        #     "hit",
        #     "clearance",
        #     "duelSucceeded",
        #     "groundDuelSucceeded",
        #     "looseBallDuelSucceeded",
        #     "aerialDuelSucceeded",
        #     "aerialClearanceSucceeded",
        #     "shot",
        #     "shotMissed",
        #     "shotBlocked",
        #     "shotOnTarget",
        #     "goal",
        #     "goalAgainst",
        #     "saveByCatching",
        #     "saveByPunching",
        #     "handballFoul",
        #     "foulCommitted",
        # }

        # 검사할 이벤트만 정의
        self.VALID_TYPES = {
            #"Aerial Control",
            "Blocks",
            "Clearances",
            "Crosses",
            "Crosses Received",
            #"Defensive Line Supports",
            #"Duels",
            "Fouls",
            #"Goals Conceded",
            #"HIR",
            #"MAX_SPEED",
            "Mistakes",
            #"Offsides",
            #"Own Goals",
            "Passes",
            "Passes Received",
            "Recoveries",
            #"SPRINT",
            "Saves",
            #"Set Piece Defence",
            "Set Pieces",
            "Shots & Goals",
            "Step-in",
            #"Tackles",
            "Take-on",
            #"Turnover",
            #"VHIR",
        }

        self.TEAM_DICT = {
            "강원FC": "강원",
            "광주FC": "광주",
            "김천 상무 프로축구단": "김천",
            "대구FC": "대구",
            "대전 하나 시티즌": "대전",
            "FC서울": "서울",
            "성남FC": "성남",
            "수원FC": "수원F",
            "수원삼성 블루윙즈": "수원",
            "울산현대축구단": "울산",
            "울산 HD FC": "울산", # added
            "인천 유나이티드": "인천",
            "전북 현대 모터스": "전북",
            "제주 유나이티드": "제주",
            "제주SK FC": "제주", # added
            "포항 스틸러스": "포항",
        }

        self.EN_TEAM_DICT = {
            "Gangwon FC": "강원",
            "Gwangju FC": "광주",
            "Gimcheon Sangmu": "김천",
            "Daegu FC": "대구",
            "Daejeon Hana Citizen": "대전",
            "FC Seoul": "서울",
            "Seongnam FC": "성남",
            "Suwon FC": "수원F",
            "Suwon Samsung Bluewings": "수원",
            "Ulsan HD FC": "울산",
            "Incheon United": "인천",
            "Jeonbuk Hyundai Motors": "전북",
            "Jeju SK FC": "제주",
            "Pohang Steelers": "포항",
        }

        self.OHCOACH_USERS_2022 = ["대전", "서울", "아산", "안산", "울산", "전북", "제주"]
        # self.OHCOACH_USERS_2023 = ["대전", "서울", "울산", "전북", "제주", "천안"]
        self.OHCOACH_USERS_2023 = ["광주", "대구", "대전", "서울", "울산", "인천", "전북", "제주"]