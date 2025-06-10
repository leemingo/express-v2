import yaml
import numpy as np
import pandas as pd
from transitions import Machine

from assertion.handler import ErrorHandler
from assertion.condition import Condition

cols = [
    "game_id", "original_event_id", "action_id", "period_id", 
    "time_seconds", "team_id", "player_id", 'start_x', 'start_y',
    "type_name", "result_name",
]
# 대전제: State Machine은 현재의 Current State와 그 Attribute로 다음 State를 명확하게 예측 및 검증할 수 있다.
class Validator:
    def __init__(self, df_events, yaml_file):
        """
        이벤트 시퀀스를 DataFrame에서 처리하는 Validator 클래스.

        매개변수:
        - df: 이벤트 데이터 DataFrame.
        - yaml_file: 상태 전이 규칙이 포함된 YAML 파일 경로.
        """
        self.df_events = df_events
        self.df_errors = []
        self.drop_idx = []
        self.window_size = 5

        self.state_machine = None
        self.error_handler = ErrorHandler()
        self.condition = Condition()
        self.state_machine = None

        # YAML에서 전이 규칙 로드
        with open(yaml_file, 'r') as file:
            self.rules = yaml.safe_load(file)

    def _initialize_state_machine(self):
        """YAML 규칙을 기반으로 상태 머신 초기화."""
        states = self.rules["states"]

        transitions = [
            {
                "trigger": t["trigger"],
                "source": t["source"],
                "dest": t["dest"],
                "conditions": getattr(self.condition, t.get("conditions", "always_true")) #getattr(self.condition, "always_true") 
            }     
            for t in self.rules["transitions"]
        ]

        initial_state = self.rules["initial_state"]

        self.state_machine = Machine(
            states=states,
            transitions=transitions,
            initial=initial_state,
            auto_transitions=False  # 의도하지 않은 전이 방지
        )

    def validate_sequence(self):
        """
        DataFrame의 이벤트 시퀀스를 검증.
        상태 전이 시 Attributes를 윈도우에 추가하고 조건을 검증합니다.
        """
        period_df = []
        for _, period_group in self.df_events.groupby('period_id'):
            self._initialize_state_machine()
            period_group = period_group.reset_index(drop=True)
            idx_stack = list(period_group.index)[::-1] 
            self.drop_idx = []
            while idx_stack:
                idx = idx_stack.pop()
                current_state = self.state_machine.state
                trigger = period_group.loc[idx, "type_name"]

                # create a window table
                min_window_idx, max_window_idx = max(0, idx-self.window_size), min(len(period_group), idx+self.window_size+1)
                window = period_group.loc[min_window_idx: max_window_idx]
                window, is_unified = self.is_unified_window(window, idx) # Goal, Own Goal, Duel, Foul, Foul Won 병합

                if is_unified:
                    period_group = pd.concat(
                        [period_group.loc[:min_window_idx-1], window, period_group.loc[max_window_idx+1:]], 
                        axis=0, ignore_index=False, sort=False
                    ).reset_index(drop=True)
     
                    idx_stack = list(period_group.index)[:idx-1:-1] # idx부터 다시 transition modeling
                else:
                    if self.start(trigger, window, idx):
                        pass
                        #print(f"State: {current_state} -> {self.state_machine.state}, Trigger: {trigger}")

                        #Added logic: dribble logic and location validation
                        if (idx < len(period_group)-1) and (self.state_machine.state == "in_possession"):
                            period_group = self._add_dribbles(period_group, window, idx)
                            idx_stack = list(period_group.index)[:idx:-1] 

                        if period_group.at[idx, "type_name"] in ["Pass", "Cross", "Carry", "Take-On", "Shot", "Clearance"]:
                            period_group = self._validate_location(period_group, idx)
                    else:
                        # 1. Detection module
                        # self.detect(trigger, window, idx)
                        # # 주의 사항(idx): Detect는 그냥 해당 trigger의 오류를 기록하고 다음 expected state로 전이
                        # idx_stack = list(period_group.index)[:idx:-1] 

                        #2. Error Handler module
                        #print(f"State: {current_state} -> {self.state_machine.state}, Trigger: {trigger} (Error Detected)")
                        window, trigger = self.handler(trigger=trigger, window=window, idx=idx)

                        period_group = pd.concat(
                            [period_group.loc[:min_window_idx-1], window, period_group.loc[max_window_idx+1:]], 
                            axis=0, ignore_index=False, sort=False
                        ).reset_index(drop=True)
                        # 주의 사항(idx-1사용 X): handler를 통해 생성한 후 바로 transition을 수행하므로
                        # 이미 idx번째 transition이 수행된 상태(idx+1부터 시작)
                        idx_stack = list(period_group.index)[:idx:-1] 

                        # revalidate a window table: 미래 이벤트과 swap되는 경우, 재검사
                        # ex) Block(idx) -> Shot(idx+1)이 스왑되면 Shot(idx) -> Block(idx+1)로 idx부터 재검사
                        # 모든 로직을 idx부터 재검사하면 되지 않나? -> 어려운 이유: expected state를 사전에 정의함 + transition libary에 의해 전이가 자동으로 이루어지는 것을 통제하기 어려움(추후 리펙토링 할 사항항)
                        min_window_idx, max_window_idx = max(0, idx-self.window_size), min(len(period_group), idx+self.window_size+1)
                        window = period_group.loc[min_window_idx: max_window_idx]
                        window, is_unified = self.is_unified_window(window, idx) # Goal, Own Goal, Duel, Foul, Foul Won 병합

                        if is_unified:
                            period_group = pd.concat(
                                [period_group.loc[:min_window_idx-1], window, period_group.loc[max_window_idx+1:]], 
                                axis=0, ignore_index=False, sort=False
                            ).reset_index(drop=True)
            
                            idx_stack = list(period_group.index)[:idx-1:-1] # idx부터 다시 transition modeling

            # TODO: Drop Error Data: 에러 데이터도 일다 사용
            #period_group = period_group[~period_group["original_event_id"].isin(self.drop_idx)]
            period_df.append(period_group)

        self.df_events = pd.concat(period_df).reset_index(drop=True)
        self.df_events["action_id"] = range(len(self.df_events))

        self.df_errors = pd.DataFrame(self.df_errors, columns=["game_id", "event_id", "current_state", "next_state", "event", "error"])

    def detect(self, trigger, window, idx, error_type="Impossible trigger"):
        prev, cur, next = self.error_handler._get_context(window, idx)

        game_id = window.at[idx, "game_id"]
        event_id = window.at[idx, "original_event_id"]
        current_state = self.state_machine.state

        in_possession_cond = [
            "Pass Received", "Recovery", "Interception", "Carry", "Take-On", "Cross Received", "Catch", "Ball Received"
        ]
        in_transition_cond = [
            "Pass", "Cross", "Clearance", "Pass_Corner", "Pass_Freekick", "Goal Kick", "Throw-In"
        ]
        in_ball_neutral_cond = [ 
            "Block", "Intervention", "Parry", "Hit", "Deflection", "Error", "Goal Post", "Goal Miss"
        ]
        in_setpiece_cond = [
            "Offside", "Foul", "Foul_Throw", "Handball_Foul", "Foul Won" # Foul Won: 병합이 안된는 이벤트가 발견되어도 detect로 처리
        ]
        in_post_shot_cond = [
            "Shot", "Shot_Freekick", "Penalty Kick", "Shot_Corner"
        ]
        in_kickoff_cond = [
            "Goal", "Goal Conceded", "Own Goal"
        ]   

        # 해당 state에서 trigger가 발생할 수 없기 때문에 해당 trigger가 작동했다고 가정하고 다음 state를 예측
        # ex) In-possession state에서 goal-kick이 발생하면 에러가 발생한다. 따라서 goal-kick이 발생했다고 가정하고 다음 state를 -> In-transition으로 이동
        if trigger in in_possession_cond:
            expected_state = "in_possession"
        elif trigger in in_transition_cond:
            expected_state = "in_transition"
        elif trigger in in_ball_neutral_cond:
            expected_state = "ball_neutral"
        elif trigger in in_setpiece_cond:
            expected_state = "set_piece"
        elif trigger in in_post_shot_cond:
            expected_state = "post_shot"
        elif trigger in in_kickoff_cond:
            expected_state = "kick_off"
        elif trigger in ["Duel", "Tackle", "Aerial Clearance", "Defensive Line Support"]:
            expected_state = current_state # Duel, Tackle
        else:
            raise ValueError(f"Unknown trigger: {trigger}")

        self.state_machine.state = expected_state
        self._record_error(game_id, event_id, current_state, trigger, expected_state, error_type)

    def handler(self, trigger, window, idx):
        """
        현재 상태와 이벤트를 기반으로 조건을 평가하고 다음 상태를 예측합니다.
        경우의 수
        1. 다음 state가 나오는 경우 -> transition
        2. 다음 state가 존재하지 않는 경우 중 생성이 필요한 경우
        3. 다음 state가 존재하지 않는 경우 중 순서 조정이 필요한 경우
        4. 애초에 오류 데이터 -> 일단 넘기자
        """

        current_state = self.state_machine.state
        method = getattr(self.error_handler, f"check_{trigger.replace('-', '_').replace(' ', '_')}", None)

        if method is None:
            raise ValueError(f"Verfication function '{trigger}' not found.")

        first_idx = window.index[0]         # 생성(or)조정되기 전 trigger
        window = method(current_state, window, idx)
        window.index = range(first_idx, first_idx+len(window)) # index 재조정

        trigger = window.loc[idx]["type_name"] # 생성(or 조정)된 trigger

        if not self.start(trigger, window, idx):
            next_state = self._predict_next_state(trigger)
            if next_state is None:
                self.detect(trigger, window, idx)   
            else:
                self.detect(trigger, window, idx, "Condition mismatch")  
            self.drop_idx.append(window.at[idx, "original_event_id"])
            #raise ValueError(f"Transition modeling failed: {current_state} -> {trigger}")
        else:
            self._record_error(window.at[idx, "game_id"], window.at[idx, "original_event_id"], current_state, trigger, self.state_machine.state, "Resolved")
            
        return window, trigger
    
    def start(self, trigger, window, idx):
        """
        상태 머신을 시작합니다.
        transition library에서 trigger를 호출시 상태 전이가 발생하면 True를 반환하고 실패하면 False를 반환합니다.
        기존 라이브러리는 실패 시 transitions.core.MachineError 반환하기 때문에 exception활용
        """

        try:
            return self.state_machine.trigger(trigger, window=window, idx=idx)
        except:
            return False
        
    def _predict_next_state(self, trigger):
        """
        트리거를 기반으로 다음 상태를 예측합니다.
        만약 예측할 수 있는 상태가 없다면 None을 반환합니다.
        """
        return next(
                    (transition['dest'] for transition in self.rules['transitions']
                    if transition['trigger'] == trigger and self.state_machine.state in transition['source']),
                    None
        )

    def _record_error(self, game_id, event_id, current_state, event, next_state, error):
        """오류 정보를 기록."""
        error_info = {
            "game_id": game_id,
            "event_id": event_id,
            "current_state": current_state,
            "next_state": next_state,
            "event": event,
            "error": error
        }

        self.df_errors.append(error_info)

    def _add_dribbles(self, period_group, window, idx):
        cur = window.loc[idx]
        next = window.loc[idx+1:]

        # Carry의 조건사이에 발생하는 실패한 태클은 제외
        goal_conceded_cond = next["type_name"] == "Goal Conceded"
        self_loop_cond1 = next["type_name"].isin(["Duel", "Aerial Clearance", "Defensive Line Support"])
        self_loop_cond2 = (next["type_name"] == "Tackle") & (next["result_name"] == "Unsuccessful")     
        next = next[~(goal_conceded_cond | self_loop_cond1 | self_loop_cond2)]

        if not next.empty:
            next = next.iloc[0]
        else:
            next = window.loc[idx+1]
            
        dx = next.start_x - cur.start_x
        dy = next.start_y - cur.start_y
        far_enough = dx**2 + dy**2 >= 3**2 # TODO: 3m 이상이어야 드리블로 판단

        # In-possession state중, 공을 소유하는 조건만 드리블 로직 추가
        # Carry, Catch, self-loop(Duel, Tackle, Aerial Clearance, Defensive Line Support)는 제외
        if (cur.type_name not in ["Carry", "Catch", "Duel", "Tackle", "Aerial Clearance", "Defensive Line Support"]) & far_enough:
            if window.at[idx+1, "type_name"] == "Carry": # 연속된 Carry가 발생하는 경우 합침
                kwargs = {
                    "game_id": cur.game_id,
                    "original_event_id": np.nan,
                    "action_id": np.nan,
                    "period_id": cur.period_id,
                    "time_seconds": cur.time_seconds + 1e-3, # Dribbles occur right after receiving the ball
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": next.end_x,
                    "end_y": next.end_y,
                    "dx": next.end_x - cur.start_x,
                    "dy": next.end_y - cur.start_y,
                }
                carry = self.error_handler._create_new_event(window, type_name="Carry", idx=idx, **kwargs)
                period_group = pd.concat(
                    [period_group.loc[:idx], carry, period_group.loc[idx+2:]],  # prev -> in_possession'state(idx) -> carry -> in_posession'state(idx+1) -> next 
                    axis=0, ignore_index=False, sort=False
                ).reset_index(drop=True)
            else:
                kwargs = {
                    "game_id": cur.game_id,
                    "original_event_id": np.nan,
                    "action_id": np.nan,
                    "period_id": cur.period_id,
                    "time_seconds": cur.time_seconds + 1e-3, # Dribbles occur right after receiving the ball
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": next.start_x,
                    "end_y": next.start_y,
                    "dx": dx,
                    "dy": dy,
                }
                carry = self.error_handler._create_new_event(window, type_name="Carry", idx=idx, **kwargs)
                period_group = pd.concat(
                    [period_group.loc[:idx], carry, period_group.loc[idx+1:]],  # prev -> in_possession'state(idx) -> carry -> in_posession'state(idx+1) -> next 
                    axis=0, ignore_index=False, sort=False
                ).reset_index(drop=True)

            self._record_error(cur.game_id, cur.original_event_id, self.state_machine.state, "Carry", self.state_machine.state, "Dribble Add")

        return period_group
    
    def _validate_location(self, period_group, idx):
        # idx -> self.state_machine.state -> idx+1
        event_before_state_idx = idx
        event_before_state = period_group.loc[event_before_state_idx]
        
        next = period_group.loc[idx+1:self.window_size+idx+1]

        # goal_conceded_cond도 추가한 이유
        # Clearance -> Goal Conceded -> Own Goal 순으로 발생하면, Clearance의 끝 위치가 잘못 보간되는 문제가 발생할 수 있음.
        # Goal Conceded가 Own Goal보다 먼저 발생하는 경우, 에러 핸들러를 통해 병합하지만 Clearance는 해당 오류로 인한 조정을 하기 전전 상태임.
        # 즉, Goal Conceded 문제를 해결하기 전에 해당 이벤트를 사전에 제외해야 함.
        # Goal Conceded와 Own Goal은 서로 다른 팀에서 발생하는 이벤트이므로, 위치 좌표가 대칭 관계가 되어 보간 과정에서 오류가 발생함.
        goal_conceded_cond = next["type_name"] == "Goal Conceded"
        self_loop_cond1 = next["type_name"].isin(["Duel", "Aerial Clearance", "Defensive Line Support"])
        self_loop_cond2 = (next["type_name"] == "Tackle") & (next["result_name"] == "Unsuccessful")
        event_after_state = next[~(goal_conceded_cond | self_loop_cond1 | self_loop_cond2)]

        if not event_after_state.empty:
            event_after_state_idx = event_after_state.index[0]
            event_after_state = event_after_state.loc[event_after_state_idx]
        else:
            # 경기가 종료되는 경우
            return period_group

        def is_within_3m(prev, cur):
            distance = ((prev.end_x - cur.start_x)**2 + (prev.end_y - cur.start_y)**2) ** 0.5
            return distance <= 3
        
        # "Pass", "Cross", "Clearance"
        if self.state_machine.state == "in_transition":
            if is_within_3m(event_before_state, event_after_state):
                pass
            else:
                period_group.at[event_before_state_idx, "end_x"] = event_after_state.start_x
                period_group.at[event_before_state_idx, "end_y"] = event_after_state.start_y
                period_group.at[event_before_state_idx, "dx"] = period_group.at[event_before_state_idx, "end_x"] - period_group.at[event_before_state_idx, "start_x"]
                period_group.at[event_before_state_idx, "dy"] = period_group.at[event_before_state_idx, "end_y"] - period_group.at[event_before_state_idx, "start_y"]
        # "Carry", "Take-On"
        elif self.state_machine.state == "in_possession":
            if is_within_3m(event_before_state, event_after_state):
                pass
            else:
                if event_before_state.type_name == "Carry":
                    period_group.at[event_before_state_idx, "end_x"] = event_after_state.start_x
                    period_group.at[event_before_state_idx, "end_y"] = event_after_state.start_y
                    period_group.at[event_before_state_idx, "dx"] = period_group.at[event_before_state_idx, "end_x"] - period_group.at[event_before_state_idx, "start_x"]
                    period_group.at[event_before_state_idx, "dy"] = period_group.at[event_before_state_idx, "end_y"] - period_group.at[event_before_state_idx, "start_y"]
                else:
                    pass
        # "Shot"
        elif self.state_machine.state == "post_shot":
            if is_within_3m(event_before_state, event_after_state):
                pass
            else:
                if event_after_state.type_name in ["Goal Kick", "Throw-In", "Pass_Corner", "Shot_Corner"]:
                    pass
                elif event_after_state.type_name in ["Deflection", "Hit", "Block", "Parry", "Catch", "Error"]:
                    period_group.at[event_before_state_idx, "end_x"] = event_after_state.start_x
                    period_group.at[event_before_state_idx, "end_y"] = event_after_state.start_y
                    period_group.at[event_before_state_idx, "dx"] = period_group.at[event_before_state_idx, "end_x"] - period_group.at[event_before_state_idx, "start_x"]
                    period_group.at[event_before_state_idx, "dy"] = period_group.at[event_before_state_idx, "end_y"] - period_group.at[event_before_state_idx, "start_y"]
                # Goal Post, Goal Miss가 생성되기 전 Shot이후 In-play가 발생하는 상황은 대부분 위치좌표가 이상하므로 이를 반영
                elif event_after_state.type_name in ["Clearance", "Recovery", "Interception", "Ball Received", 
                                                     "Offside", "Foul", "Intervention"]:
                    period_group.at[event_before_state_idx, "end_x"] = event_after_state.start_x
                    period_group.at[event_before_state_idx, "end_y"] = event_after_state.start_y
                    period_group.at[event_before_state_idx, "dx"] = period_group.at[event_before_state_idx, "end_x"] - period_group.at[event_before_state_idx, "start_x"]
                    period_group.at[event_before_state_idx, "dy"] = period_group.at[event_before_state_idx, "end_y"] - period_group.at[event_before_state_idx, "start_y"]
                else:
                    pass
        else:
            # 이벤트의 좌표 검증이 불가능한 경우
            pass
        
        return period_group
    
    def is_unified_window(self, window, idx):
        is_unified = False
        prev, cur, next = self.error_handler._get_context(window, idx)
        
        # Duel에는 condition이 없지만 impossible trigger를 통과하기 위한 병합해야하는지 조건만 검사(impossible trigger는 병합로직X)
        if cur.type_name in ["Shot_Corner", "Shot", "Shot_Freekick", "Penalty Kick"]: 
            # Shot으로 부터 Goal을 생성해야함: 득점이벤트가 생성되지 않았고 병합할 Goal Conceded가 존재하는 경우
            if (cur.result_name == "Goal") and ("Goal" not in next.type_name.values) and ("Goal Conceded" in next.type_name.values):
                is_unified = True
                goal_conceded_idx = next[next["type_name"] == "Goal Conceded"].index[0]
                kwargs = {
                    "game_id": next.at[goal_conceded_idx, "game_id"],
                    "original_event_id": np.nan,
                    "action_id": np.nan,
                    "period_id": next.at[goal_conceded_idx, "period_id"],
                    "time_seconds": next.at[goal_conceded_idx, "time_seconds"],
                    "team_id": cur.team_id, # 득점에 성공한 선수가 actor.
                    "player_id": cur.player_id,
                    "reactor_team_id": next.at[goal_conceded_idx, "team_id"],    # goal conceded 선수(goalkeeper)
                    "reactor_player_id": next.at[goal_conceded_idx, "player_id"],
                    "start_x": cur.end_x,
                    "start_y": cur.end_y,
                    "end_x": cur.end_x,
                    "end_y": cur.end_y,
                    "dx": 0,
                    "dy": 0,
                }
                goal = self.error_handler._create_new_event(window, type_name="Goal", idx=idx, **kwargs)   
                next = next.drop(index=goal_conceded_idx) # Goal에 병합된 후 제거

                window = pd.concat(
                    [prev, pd.DataFrame([cur]), next.loc[:goal_conceded_idx-1], goal, next.loc[goal_conceded_idx+1:]],
                    axis=0, ignore_index=False, sort=False
                )  
        elif cur.type_name == "Own Goal":
            if any(next.type_name == "Goal Conceded"):
                is_unified = True

                goal_conceded_idx = next[next["type_name"] == "Goal Conceded"].index[0]

                cur = cur.copy() # cur을 명시적으로 복사: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
                cur[["reactor_team_id", "reactor_player_id"]] = next.loc[goal_conceded_idx, ["team_id", "player_id"]].values
                next = next.drop(index=goal_conceded_idx) # Own Goal에 병합된 후 제거

                window = pd.concat(
                    [prev, pd.DataFrame([cur]), next],
                    axis=0, ignore_index=False, sort=False
                )
        elif (cur.type_name == "Foul") & (cur.reactor_player_id == -1): # 수비수가 파울(Foul)을 저지른 후에 심판이 파울(Foul Won)을 선언하는 경우
            # foul won기준으로 병합
            if any(next.type_name == "Foul Won"):  
                is_unified = True      
                foul_won_idx = next[next["type_name"] == "Foul Won"].index[0]

                # 사이 액션 중 항의(추가 경고)하는 foul만 살리고 나머지 제거
                inter_events = window.loc[idx+1: foul_won_idx-1]
                protest_after_foul = inter_events[inter_events["type_name"] == "Foul"] 

                # Foul기준으로 병합하지만, 발생한 시점 & 위치는 먼저 발생한 Foul Won기준으로 설정
                kwargs = {
                    "game_id": cur.game_id,
                    "original_event_id": cur.original_event_id,
                    "action_id": np.nan,
                    "period_id": cur.period_id,
                    "time_seconds": cur.time_seconds,
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,

                    "reactor_team_id": next.at[foul_won_idx, "team_id"],
                    "reactor_player_id": next.at[foul_won_idx, "player_id"],

                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx" : 0,
                    "dy" : 0,

                    "result_name": cur.result_name
                }

                foul = self.error_handler._create_new_event(window, type_name="Foul", idx=idx, **kwargs)
                
                window = pd.concat(
                    [prev, foul, protest_after_foul, window.loc[foul_won_idx+1:]], 
                    axis=0, ignore_index=False, sort=False
                )
        elif cur.type_name == "Foul Won": # 심판이 반칙 가해자를 확인하기 전, 반칙의 피해자를 먼저 확인하는 경우(로 추정..)
            # foul 기준으로 병합
            if any(next.type_name == "Foul"):  
                is_unified = True
                foul_idx = next[next["type_name"] == "Foul"].index[0]

                # 사이 액션 중 항의하는 foul만 살리고 나머지 제거
                inter_events = window.loc[idx+1: foul_idx-1]
                protest_after_foul = inter_events[inter_events["type_name"] == "Foul"] 

                # Foul기준으로 병합하지만, 발생한 시점 & 위치는 먼저 발생한 Foul Won기준으로 설정
                kwargs = {
                    "game_id": next.at[foul_idx, "game_id"],
                    "original_event_id": next.at[foul_idx, "original_event_id"],
                    "action_id": np.nan,
                    "period_id": cur.period_id,
                    "time_seconds": cur.time_seconds,
                    "team_id": next.at[foul_idx, "team_id"],
                    "player_id": next.at[foul_idx, "player_id"],

                    "reactor_team_id": cur.team_id,
                    "reactor_player_id": cur.player_id,

                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx" : 0,
                    "dy" : 0,

                    "result_name": next.at[foul_idx, "result_name"]
                }

                foul = self.error_handler._create_new_event(window, type_name="Foul", idx=idx, **kwargs)
                window = pd.concat(
                    [prev, foul, protest_after_foul, window.loc[foul_idx+1:]], 
                    axis=0, ignore_index=False, sort=False
                )
        elif cur.type_name == "Duel":
            # 병합해야할 Duel Pair가 존재하는 경우
            if (cur.type_name == "Duel") and any((next.type_name == "Duel") & (next.original_event_id == cur.pair_id)):
                is_unified = True
                # Duel사이에 발생하는 이벤트 재조정 및 통합(주의사항: 시퀀스 형태로 정렬된 방식도 고려해야함)
                duel_idx2 = next[(next.type_name == "Duel") & (next["original_event_id"] == cur.pair_id)].index[0]

                # Duel에서 승리한 선수를 기준으로 병합. actor=승리자, reactor=패자
                # 이유: 시퀀스 데이터에는 경합에서 승리한 Duel이벤트만 존재하기 때문임(패배한 Duel이벤트는 시퀀스 데이터에 포함하지 않음)
                if cur.result_name == "Successful":
                    winner, loser = cur, next.loc[duel_idx2]
                elif next.at[duel_idx2, "result_name"] == "Successful":
                    winner, loser = next.loc[duel_idx2], cur
                else:
                    # 경쟁에 승리한 선수가 없으면 첫번째 선수를 임의로 actor로 설정
                    winner, loser = cur, next.loc[duel_idx2]

                kwargs = {
                    "game_id": winner.game_id,
                    "original_event_id": winner.original_event_id,
                    "action_id": np.nan,
                    "period_id": winner.period_id,
                    "time_seconds": winner.time_seconds,
                    "team_id": winner.team_id,
                    "player_id": winner.player_id,

                    "reactor_team_id": loser.team_id,
                    "reactor_player_id": loser.player_id,

                    "start_x": winner.start_x,
                    "start_y": winner.start_y,
                    "end_x": winner.end_x,
                    "end_y": winner.end_y,
                    "dx" : winner.end_x - winner.start_x,
                    "dy" : winner.end_y - winner.start_y,

                    "result_name": winner.result_name 
                }
                duel = self.error_handler._create_new_event(window, type_name="Duel", idx=idx, **kwargs)

                inter_duel = window.loc[idx+1:duel_idx2-1] # Duel사이에 발생한 이벤트트
                # Duel과 함께 발생한 이벤트들은 Duel이후에 삽입함. 그 외 데이터는 시퀀스 데이터 기반으로 정렬했기 때문에 유지. 
                duel_related_events = inter_duel[inter_duel["original_event_id"].isin([winner.original_event_id, loser.original_event_id])]
                non_duel_related_events = inter_duel[~inter_duel["original_event_id"].isin([winner.original_event_id, loser.original_event_id])]
                

                # 전자 Duel에 병합되면, 그 외 시퀀스 데이터보다 먼저 발생
                # ex) Pass -> Duel(성공) -> Pass Received -> Duel(실패): Pass -> Duel(패스 도중 경합) -> Pass Received
                # ex) Pass -> Duel(실패) -> Pass Received -> Duel(성공): Pass -> Pass Received -> Duel(패스 받고 경합)
                if cur.result_name == "Successful":
                    window = pd.concat(
                        [prev, duel, duel_related_events, non_duel_related_events, window.loc[duel_idx2+1:]],
                        axis=0, ignore_index=False, sort=False
                    )
                elif next.at[duel_idx2, "result_name"] == "Successful":
                    window = pd.concat(
                        [prev, non_duel_related_events, duel, duel_related_events, window.loc[duel_idx2+1:]],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    window = pd.concat(
                        [prev, duel, duel_related_events, non_duel_related_events, window.loc[duel_idx2+1:]],
                        axis=0, ignore_index=False, sort=False
                    )
            return window, is_unified
        elif cur.type_name == "Offside":
            setpiece_like = [
                "Pass_Freekick",
                "Shot_Freekick",
                "Pass_Corner",
                "Shot_Corner"
                "Penalty Kick",
                "Throw-In",
                "Goal Kick",
            ]

            if any(next['type_name'].isin(setpiece_like)):
                offside_idx = idx
                setpiece_idx = next[next["type_name"].isin(setpiece_like)].index[0] 

                # 사이 액션 중 항의하는 foul만 살리고 나머지는 제거
                inter_events = window.loc[offside_idx+1: setpiece_idx-1]
                foul_events = inter_events[inter_events["type_name"] == "Foul"]

                # offside ~ set-piece 사이에 발생한 이벤트가 모두 Foul인 경우, 병합할 필요가 없음
                # 즉, 파울 이외에 발생한 이벤트를 제거할 때만, is_unified=True
                if not inter_events[inter_events["type_name"] != "Foul"].empty:
                    is_unified = True
                
                window = pd.concat(
                    [prev, window.loc[[offside_idx]], foul_events, window.loc[setpiece_idx:]], 
                    axis=0, ignore_index=False, sort=False
                )

        return window, is_unified