import numpy as np
import pandas as pd
import assertion.config as lsdpconfig

cols = [
    "game_id", "original_event_id", "action_id", "period_id", 
    "time_seconds", "team_id", "player_id", 'start_x', 'start_y',
    "type_name", "result_name",
]
# Define Transition Conditions
class ErrorHandler:
    def _get_context(self, window, idx):

        prev = window.loc[:idx-1]     # 이전 이벤트
        cur = window.loc[idx]     # 현재 이벤트(현재 state에서 발생할 수 없는 이벤트)
        next = window.loc[idx+1:] # 이벤트 시퀀스 검증을 위함

        return prev, cur, next
    
    def _create_new_event(self, window, idx, type_name, **kwargs):
        prev, cur, next = self._get_context(window, idx)

        new_event = {
            "game_id": cur.game_id,
            "original_event_id": np.nan,
            "action_id": np.nan,
            "period_id": cur.period_id,
            "time_seconds": cur.time_seconds,
            "team_id": cur.team_id,
            "player_id": cur.player_id,
            "reactor_team_id": -1,
            "reactor_player_id": -1,
            "type_name": type_name,
            "type_id": lsdpconfig.actiontypes.index(type_name),
            "start_x": cur.start_x,
            "start_y": cur.start_y,
            "end_x": cur.end_x,
            "end_y": cur.end_y,
            "dx": cur.dx,
            "dy": cur.dy,
        }

        # 추가적인 파라미터로 특정 키에 대한 값을 설정하는 경우
        for key, value in kwargs.items():
            new_event[key] = value

        return pd.DataFrame([new_event])

    def _predict_out_event_location(self, window, idx):
        '''
        set-piece 이벤트를 발생시키는 에러 처리 시, 이전 이벤트의 정보들을 활용하여 Out 이벤트를 생성할 수 있음.
        ex) Pass -> Out(relative_x, relative_y, relative_event_time) -> Corner
        그러나 끝 위치, 시간을 기록하지 않은 일부 이벤트(Block, Deflection 등)의 경우, 휴리스틱 방식으로 값을 지정해야 함.
        ex) Block -> Corner: Block -> Out(Block의 끝 위치??) -> Corner

        처리 방식:
        - Throw-In: Throw-In의 시작 위치를 이용해 Out 이벤트의 시작 위치를 정의할 수 있으므로 추가 처리가 필요X.
        - Corner, Goal Kick: 이전 이벤트의 끝 위치를 활용할 수 없는 경우, 휴리스틱 방식으로 Out 이벤트의 위치, 시간을 보간함.
        '''
        prev, cur, next = self._get_context(window, idx)
        out_start_x, out_start_y, out_time = prev.iloc[-1].end_x, prev.iloc[-1].end_y, prev.iloc[-1].relative_time_seconds

        # 패스는 어차피 끝 위치가 모두 정의되어 있기 때문에 문제가 없음
        if pd.isna(out_start_x) or pd.isna(out_start_y):
            # TODO: Corner(or Goal Kick)가 발생한 위치 기준으로 보간
            if cur.start_x < lsdpconfig.config.field_length / 2:  
                out_start_x, out_start_y = 0, prev.iloc[-1].start_y
            else:
                out_start_x, out_start_y = lsdpconfig.config.field_length, prev.iloc[-1].start_y

        if pd.isna(out_time):
            # set-piece이벤트가 지연되는 경우(prev.time <<<< cur.time), 이전 이벤트의 시간을 기준(prev.time + 3s)으로 보간함
            # TODO: 3s는 휴리스틱한 값으로 설정함
            out_time = min(prev.iloc[-1].time_seconds + 3, (prev.iloc[-1].time_seconds + cur.time_seconds) / 2)  

        return out_start_x, out_start_y, out_time
    
    def _get_prev_pass_like(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        pass_like = [
            "Pass",
            "Cross",
            "Throw-In",
            "Pass_Freekick",
            "Pass_Corner",
            "Clearance",
            "Goal Kick",
        ]

        return prev[prev["type_name"].isin(pass_like)]
    
    def _get_prev_shot_like(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        shot_like = [
            "Shot",
            "Shot_Freekick",
            "Penalty Kick",
        ]

        return prev[prev["type_name"].isin(shot_like)]

    def _get_on_the_ball_before(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        on_the_ball = [
            "Pass",
            "Cross",
            "Carry",
            "Take-On",
            "Clearance",
            "Shot",
            "Pass Received",
            "Cross Received",
            "Ball Received",
            "Recovery", 
        ]

        return prev[prev['type_name'].isin(on_the_ball)]
    
    def _get_on_the_ball_after(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        on_the_ball = [
            "Pass",
            "Cross",
            "Carry",
            "Take-On",
            "Clearance",
            "Shot",
            "Pass Received",
            "Cross Received",
            "Ball Received",
            "Recovery", 
        ]

        return next[next['type_name'].isin(on_the_ball)]

    def check_Pass_Freekick(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window
                                                 
    def check_Shot_Freekick(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Pass_Corner(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state in ["in_transition", "in_possession", "ball_neutral", "post_shot"]: # 생성: post_shot상태에서 경기장 밖으로 나갔기 때문에 set_piece 상태로 이동
            start_x, start_y, time_seconds = self._predict_out_event_location(window, idx)
            kwargs = {
                "time_seconds": time_seconds,
                "team_id": prev.iloc[-1].team_id,
                "player_id": prev.iloc[-1].player_id,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": start_x,
                "end_y": start_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Out", idx=idx, **kwargs)

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )
        else:
            return window
            raise ValueError("Corner event is not in the post_shot state:" + current_state)
        
        return window

    def check_Shot_Corner(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state in ["in_transition", "in_possession", "ball_neutral", "post_shot"]: # 생성: post_shot상태에서 경기장 밖으로 나갔기 때문에 set_piece 상태로 이동
            start_x, start_y, time_seconds = self._predict_out_event_location(window, idx)
            kwargs = {
                "time_seconds": time_seconds,
                "team_id": prev.iloc[-1].team_id,
                "player_id": prev.iloc[-1].player_id,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": start_x,
                "end_y": start_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Out", idx=idx, **kwargs)

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )
        else:
            return window
            raise ValueError("Corner event is not in the post_shot state:" + current_state)
        
        return window
    
    def check_Penalty_Kick(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Throw_In(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state in ["in_transition", "in_possession", "ball_neutral", "post_shot"]:
            _, _, time_seconds = self._predict_out_event_location(window, idx)
            kwargs = {
                "time_seconds": time_seconds,
                "team_id": prev.iloc[-1].team_id,
                "player_id": prev.iloc[-1].player_id,
                "start_x": cur.start_x,      # Throw-In이전에 발생한 Out이벤트는 Throw-In이벤트의 start_x, start_y로 설정가능
                "start_y": cur.start_y,
                "end_x": cur.start_x,
                "end_y": cur.start_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Out", idx=idx, **kwargs)     

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )
        else:
            return window
            raise ValueError(f"Throw-In event is not in the ball_neutral or in_transition state :{current_state})")
        
        return window

    def check_Goal_Kick(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state in ["in_transition", "in_possession", "ball_neutral", "post_shot"]:  # 누구도 공을 못잡아서 골킥이 발생함
            start_x, start_y, time_seconds = self._predict_out_event_location(window, idx)
            kwargs = {
                "time_seconds": time_seconds,
                "team_id": prev.iloc[-1].team_id,
                "player_id": prev.iloc[-1].player_id,
                "start_x": start_x,  # TODO: 끝 위치가 정의되지 않는 block, defection등의 경우 휴리스틱하게 값을 지정해야함
                "start_y": start_y,
                "end_x": start_x,
                "end_y": start_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Out", idx=idx, **kwargs)    

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )

        return window

    def check_Catch(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        return window

    def check_Parry(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Own_Goal(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        
        return window

    def check_Goal(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Goal_Conceded(self, current_state, window, idx):
        '''
        Goal Conceded는 Goal(or Own Goal)과 병합된 이벤트이기 때문에 발생하지 않는데, error handler가 작동함
        1. Goal Conceded 이후에 Goal이 존재하는 경우, Goal Conceded과 병합
        2. Goal Conceded 이후에 Own Goal이 존재하는 경우, Goal Conceded과 병합
        3. Goal Conceded 이후에 Goal, Own Goal이 존재하지 않는 경우. ex) game_id=76055, event_id=74437410
        그러나 에러가 발생한 경우, Goal(Own Goal)이전에 Shot, Goal이 존재하지 않는 것이기 때문에 Shot, Goal이벤트를 생성해야할 필요가 있음.
        But, 원본 이벤트 데이터를 보면 Goal Conceded 이후에 득점으로 이어진 Shot이 존재하고 있었음.
        TODO: 선수 or 위치를 예측할 수 있을까? 자살골일 수도 있고?
        '''
        prev, cur, next = self._get_context(window, idx)    

        shot = next[(next["type_name"] == "Shot") & (next["result_name"] == "Goal")]
        owngoal = next[next["type_name"] == "Own Goal"]

        if not shot.empty:
            # Shot가 Goal Conceded이전에 발생하지 않아서 병합되지 않은 경우(순서문제제)
            shot_idx = shot.index[0]
            
            kwargs = {
                "game_id": cur.game_id,
                "original_event_id": np.nan,
                "action_id": np.nan,
                "period_id":cur.period_id,
                "time_seconds": cur.time_seconds,
                "team_id": shot.at[shot_idx, "team_id"],
                "player_id": shot.at[shot_idx, "player_id"],
                "reactor_team_id": cur.team_id,    # goal conceded 선수(goalkeeper)
                "reactor_player_id": cur.player_id,
                "start_x": shot.at[shot_idx, "end_x"],
                "start_y": shot.at[shot_idx, "end_y"],
                "end_x": shot.at[shot_idx, "end_x"],
                "end_y": shot.at[shot_idx, "end_y"],
                "dx": 0,
                "dy": 0,
            }
            goal = self._create_new_event(window, type_name="Goal", idx=idx, **kwargs) 
            window = pd.concat(
                [prev, next.loc[:shot_idx], goal, next.loc[shot_idx+1:]],
                axis=0, ignore_index=False, sort=False
            )
        elif not owngoal.empty:
            # Own Goal이 Goal Conceded이전에 발생하지 않아서 병합되지 않은 경우 
            owngoal_idx = owngoal.index[0]
            owngoal.at[owngoal_idx, "reactor_team_id"] = cur.team_id
            owngoal.at[owngoal_idx, "reactor_player_id"] = cur.player_id

            window = pd.concat(
                [prev, next.loc[:owngoal_idx-1], owngoal.loc[[owngoal_idx]], next.loc[owngoal_idx+1:]],
                axis=0, ignore_index=False, sort=False
            )
        else: # 득점한 이벤트가 존재하지 않는 경우
            # IN-transition state에서는 일단 In-possession state로 가는 trigger(액션)을 생성 or 처리해야함
            if current_state == "in_transition":
                pass_like = self._get_prev_pass_like(window, idx)

                if pass_like.empty:
                    return window
                    raise ValueError(f"In_transition에서 Pass가 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
                else:
                    pass_like = pass_like.iloc[-1]

                if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                    if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                        received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                        new_df = next.loc[[received_idx]]
                    else: # received가 존재하지 않는 경우
                        kwargs = {
                            "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                            "team_id": pass_like.team_id,
                            "player_id": pass_like.relative_player_id,
                            "start_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                # TODO: 예상되는 이벤트데이터의 위치 예측?
                            "start_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "end_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                  # shot의 위치랑 동일하게 설정
                            "end_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "dx": 0,
                            "dy": 0,
                        }
                        new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)                          
                    # 주의사항: pass -> goal conceded가 발생하는 경우
                    # 1. passer(A)와 goal conceded(A) 선수가 동일한 팀인 경우(=): 상대팀에게 뺏긴 경우
                    # 2. passer(A와 goal conceded(B) 선수가 다른른 팀인 경우(=): 상대팀에게 뺏기지 않고 득점을 수행한 경우           
                    interception_cond = (
                        (pass_like.team_id == cur.team_id)  # action.team_id == goalconceded.team_id: 패스한 팀이 득점을 허용한 경우(누군가에게 공을 뺏겼기 때문임임)
                    )
                    recovery_cond = (
                        (pass_like.team_id != cur.team_id)  # 패스한 팀과 다른 팀이 득점을 허용한 경우(실패한 패스를 recovery하고 득점으로 이어짐)
                    )
                    if interception_cond:
                        opponent = window[window.team_id != pass_like.team_id]
                        if opponent.empty:
                            # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                            print(f"Goal Conceded가 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                            return window  
                        
                        kwargs = {
                            "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2,   
                            "team_id": opponent.iloc[-1].team_id,
                            "player_id": np.nan,
                            "start_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                # TODO: 예상되는 이벤트데이터의 위치 예측?
                            "start_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "end_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                  # shot의 위치랑 동일하게 설정
                            "end_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "dx": 0,
                            "dy": 0,
                        }
                        new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)
                    elif recovery_cond:
                        kwargs = {
                            "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2,   
                            "team_id": pass_like.iloc[-1].team_id,
                            "player_id": np.nan,
                            "start_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                # TODO: 예상되는 이벤트데이터의 위치 예측?
                            "start_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "end_x": (prev.iloc[-1].start_x + cur.start_x) / 2,                  # shot의 위치랑 동일하게 설정
                            "end_y": (prev.iloc[-1].start_y + cur.start_y) / 2,
                            "dx": 0,
                            "dy": 0,
                        }
                        new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)
                    else:
                        return window
                        raise ValueError(f"Goal Conceded Error Handler의 나타나지 않는 경우: {window}")
                
                window = pd.concat(
                    [prev, new_df, pd.DataFrame([cur]), next], # Goal Conceded제거
                    axis=0, ignore_index=False, sort=False
                )     
            elif current_state == "in_possession":
                on_the_ball_before = self._get_on_the_ball_before(window, idx)
                if on_the_ball_before.empty:
                    print(f"In_possession에서 Goal Concede가 발생한 경우, on-ball이 존재해야 합니다: \n{window}")
                    return window
                    raise ValueError(f"In_possession에서 Goal Concede가 발생한 경우, on-ball이 존재해야 합니다: {window}")       

                kawrgs = {
                    "time_seconds": on_the_ball_before.iloc[-1].time_seconds,
                    "team_id": on_the_ball_before.iloc[-1].team_id,
                    "player_id": on_the_ball_before.iloc[-1].player_id,
                    "start_x": on_the_ball_before.iloc[-1].start_x,                # TODO: 예상되는 이벤트데이터의 위치 예측?
                    "start_y": on_the_ball_before.iloc[-1].start_y,
                    "end_x": cur.start_x,   # goal conceded location            
                    "end_y": cur.start_y,   # goal conceded location
                    "dx": cur.start_x - on_the_ball_before.iloc[-1].start_x,
                    "dy": cur.start_y - on_the_ball_before.iloc[-1].start_y,
                }
                shot = self._create_new_event(window, type_name="Shot", idx=idx, **kawrgs)

                kawrgs = {
                    "time_seconds": on_the_ball_before.iloc[-1].time_seconds,
                    "team_id": on_the_ball_before.iloc[-1].team_id,
                    "player_id": on_the_ball_before.iloc[-1].player_id,
                    "reactor_team_id": cur.team_id,
                    "reactor_player_id": cur.player_id,
                    "start_x": cur.start_x,                
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                goal = self._create_new_event(window, type_name="Goal", idx=idx, **kawrgs)

                kwargs = {
                    "time_seconds": on_the_ball_before.iloc[-1].time_seconds,
                    "team_id": on_the_ball_before.iloc[-1].team_id,
                    "player_id": on_the_ball_before.iloc[-1].player_id,
                    "reactor_team_id": cur.team_id,
                    "reactor_player_id": cur.player_id,
                    "start_x": on_the_ball_before.iloc[-1].start_x,                
                    "start_y": on_the_ball_before.iloc[-1].start_y,
                    "end_x": on_the_ball_before.iloc[-1].start_x,                       
                    "end_y": on_the_ball_before.iloc[-1].start_y,
                    "dx": 0,
                    "dy": 0.
                }
                owngoal = self._create_new_event(window, type_name="Own Goal", idx=idx, **kwargs)
    
                window = pd.concat(
                    [prev, shot, goal, next], # Goal Conceded제거
                    axis=0, ignore_index=False, sort=False
                )               
                #예외 경우: 자책골의 경우
                if on_the_ball_before.iloc[-1].team_id == cur.team_id:         
                    window = pd.concat(
                        [prev, owngoal, next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    window = pd.concat(
                        [prev, shot, goal, next], # Goal Conceded제거
                        axis=0, ignore_index=False, sort=False
                    )
            else:
                raise ValueError(f"Goal Conceded 이후에 득점한 이벤트가 존재하지 않는 경우, Shot, Goal이벤트를 생성해야 합니다: {current_state} \n{window}")

        return window
    
    def check_Defensive_Line_Support(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == 'in_transition':
            on_the_ball = [
                "Pass", "Cross", "Clearance",
                "Pass_Freekick", "Pass_Corner", "Throw-In", "Goal Kick"
            ]
        elif current_state == 'in_possession':
            on_the_ball = [
                "Pass Received", "Ball Received", "Interception", "Recovery"
                "Catch", "Carry", "Take-On"
            ]
        elif current_state == 'ball_neutral':
            on_the_ball = [
                "Defkectuib", "Hit", "Block", "Parry",
                "Error", "Goal Post", "Goal Miss",
                "Intervention"
            ]
        elif current_state == 'post_shot':
            on_the_ball = ["shot"]

        on_the_ball_after_aerial = next[next['type_name'].isin(on_the_ball)]

        catch_cond = (
            (cur.result_name == "Successful") and
            (not on_the_ball_after_aerial.empty) and
            (cur.player_id == on_the_ball_after_aerial.iloc[0].player_id)
        )
        parry_cond = (
            (cur.result_name == "Successful") and
            (not catch_cond)
        )
        if catch_cond:
            kwargs = {
                "time_seconds": cur.time_seconds, # aerial clearance이후 연속된 골키퍼 액션의 결과 생성
                "team_id": cur.team_id,
                "player_id": cur.player_id,
            }
            new_df = self._create_new_event(window, type_name="Catch", idx=idx, **kwargs)
        elif parry_cond:
            kwargs = {
                "time_seconds": cur.time_seconds, # aerial clearance이후 연속된 골키퍼 액션의 결과 생성
                "team_id": cur.team_id,
                "player_id": cur.player_id,
            }
            new_df = self._create_new_event(window, type_name="Parry", idx=idx, **kwargs)
        else:
            return window
        
        window = pd.concat(
            [prev, pd.DataFrame([cur]), new_df, next],
            axis=0, ignore_index=False, sort=False
        )

        return window

    def check_Aerial_Clearance(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        
        if current_state == 'in_transition':
            on_the_ball = [
                "Pass", "Cross", "Clearance", "Recovery"
            ]
        elif current_state == 'in_possession':
            on_the_ball = [
                "Pass Received", "Ball Received", "Interception", "Recovery"
                "Catch", "Carry", "Take-On"
            ]
        elif current_state == 'ball_neutral':
            on_the_ball = [
                "Defkectuib", "Hit", "Block", "Parry",
                "Error", "Goal Post", "Goal Miss",
                "Intervention"
            ]
        elif current_state == 'post_shot':
            on_the_ball = ["shot"]


        on_the_ball_after_aerial = next[next['type_name'].isin(on_the_ball)]

        catch_cond = (
            (cur.result_name == "Successful") and
            (not on_the_ball_after_aerial.empty) and
            (on_the_ball_after_aerial.iloc[0].type_name != "Recovery") and
            (cur.player_id == on_the_ball_after_aerial.iloc[0].player_id)
        )
        parry_cond = (
            (cur.result_name == "Successful") and
            (not catch_cond)
        )

        if catch_cond:
            kwargs = {
                "time_seconds": cur.time_seconds, # aerial clearance이후 연속된 골키퍼 액션의 결과 생성
                "team_id": cur.team_id,
                "player_id": cur.player_id,
            }
            new_df = self._create_new_event(window, type_name="Catch", idx=idx, **kwargs)
        elif parry_cond:
            kwargs = {
                "time_seconds": cur.time_seconds, # aerial clearance이후 연속된 골키퍼 액션의 결과 생성
                "team_id": cur.team_id,
                "player_id": cur.player_id
            }
            new_df = self._create_new_event(window, type_name="Parry", idx=idx, **kwargs)
        else:
            return window
        
        window = pd.concat(
            [prev, pd.DataFrame([cur]), new_df, next],
            axis=0, ignore_index=False, sort=False
        )

        return window

    def check_Duel(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)  
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )   
        else:
            return window
        
        return window

    def check_Tackle(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        pass_like = self._get_prev_pass_like(window, idx)
        on_the_ball_after_tackle = self._get_on_the_ball_after(window, idx)

        if current_state == "in_transition":
            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Tackle이 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            interception_cond = (
                (cur.result_name == "Successful") and 
                (not on_the_ball_after_tackle.empty) and
                (pass_like.team_id != cur.team_id) and
                (cur.player_id == on_the_ball_after_tackle.iloc[0].player_id)  # regain possession
            )

            intervention_cond = (
                (cur.result_name == "Successful") and 
                (not interception_cond)  # creating loose ball situation
            )
            if interception_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)   
            elif intervention_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Intervention", idx=idx, **kwargs)   
            else:
                return window
                raise ValueError(f"In_transtion에서 조건(is_tackle_in_transition)이 만족하지 않는 경우, Interception 또는 Intervention이 생성해야 합니다: {window}")
      
            window = pd.concat(
                [prev, pd.DataFrame([cur]), new_df, next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "in_possession":
            interception_cond = (
                (cur.result_name == "Successful") and 
                (not on_the_ball_after_tackle.empty) and
                (cur.player_id == on_the_ball_after_tackle.iloc[0].player_id)  # regain possession
            )
            intervention_cond = (
                (cur.result_name == "Successful") and 
                (not interception_cond) # creating loose ball situation
            )

            if interception_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)   
            elif intervention_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Intervention", idx=idx, **kwargs)   
            else:
                return window
                raise ValueError(f"In_possession에서 조건(is_tackle_in_possession)이 만족하지 않는 경우, Interception 또는 Intervention이 생성해야 합니다: {window}")
           
            window = pd.concat(
                [prev, pd.DataFrame([cur]), new_df, next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "ball_neutral":
            interception_cond = (
                (cur.result_name == "Successful") and 
                (not on_the_ball_after_tackle.empty) and
                (cur.player_id == on_the_ball_after_tackle.iloc[0].player_id)  # regain possession
            )
            intervention_cond = (
                (cur.result_name == "Successful") and 
                (not interception_cond) # creating loose ball situation
            )

            if interception_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)   
            elif intervention_cond:
                kwargs = {
                    "time_seconds": cur.time_seconds, # tackle이후 연속된 수비 액션의 결과 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Intervention", idx=idx, **kwargs)
            else:
                return window
                #raise ValueError(f"{current_state}에서 Tackle이 발생한 경우, Interception이 발생해야 합니다: {window}") 

            window = pd.concat(
                [prev, pd.DataFrame([cur]), new_df, next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)  
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )  
        else:
            return window
            #raise ValueError(f"Tackle event is not in the in_transition, in_possession, ball_neutral state: {current_state}") # TODO: 실제로 발견했지만 에러

        window = pd.concat(
            [prev, pd.DataFrame([cur]), new_df, next],
            axis=0, ignore_index=False, sort=False
        )
        return window

    def check_Intervention(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)    
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
                window = pd.concat(
                    [prev, new_df, pd.DataFrame([cur]), next], 
                    axis=0, ignore_index=False, sort=False
                )
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )   
        else:
            return window
        
        return window

    def check_Interception(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)    

                window = pd.concat(
                    [prev, new_df, pd.DataFrame([cur]), next], 
                    axis=0, ignore_index=False, sort=False
                )  
            elif not shot.empty and shot.iloc[-1].result_name == "On Target":
                # On-target 슛을 blocking하는 경우 Interception으로 기록하는데, 이는 일관성이 맞지 않으므로 Block으로 rename
                cur = cur.copy()
                cur.type_name = "Block"

                window = pd.concat(
                    [prev, cur, next], 
                    axis=0, ignore_index=False, sort=False
                )  
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
                window = pd.concat(
                    [prev, new_df, pd.DataFrame([cur]), next], 
                    axis=0, ignore_index=False, sort=False
                )
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
                window = pd.concat(
                    [prev, new_df, pd.DataFrame([cur]), next], 
                    axis=0, ignore_index=False, sort=False
                )  
            else:
                return window 
        else:
            return window

        return window

    def check_Block(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "in_possession":
            # 이전에 pass를 받은 경우로, bepro에서는 시퀀스를 receival -> block으로 표현함
            pass_like = [
                "Pass",
                "Cross",
                "Throw-In",
                "Pass_Freekick",
                "Pass_Corner",
                "Clearance",
                "Goal Kick",
            ]
            shot_like = [
                "Shot",
                "Shot_Freekick",
                "Penalty Kick",
            ]
            # current event <-> next event: swap
            prev_pass_or_shot = prev[prev.type_name.isin(pass_like + shot_like)]
            next_pass_or_shot = next[next.type_name.isin(pass_like + shot_like)]

            if prev_pass_or_shot.empty and next_pass_or_shot.empty:
                return window

            if not prev_pass_or_shot.empty and not next_pass_or_shot.empty:
                prev_diff_time = abs(prev_pass_or_shot.iloc[-1].time_seconds - cur.time_seconds)
                next_diff_time = abs(next_pass_or_shot.iloc[0].time_seconds - cur.time_seconds)

                # Block을 swap할 때는 이전/이후 둘 다 가능성이 존재하므로 시간차이를 고려하여 swap한다
                if prev_diff_time < next_diff_time:
                    swap = prev_pass_or_shot.iloc[-1]
                    swap_idx = swap.name

                    # prev_idx+1: pass, shot이후에 block를 삽입
                    window = pd.concat(
                        [prev.loc[:swap_idx], pd.DataFrame([cur]), prev.loc[swap_idx+1:], next], 
                        axis=0, ignore_index=False, sort=False
                    )
                elif prev_diff_time > next_diff_time:
                    swap = next_pass_or_shot.iloc[0]
                    swap_idx = swap.name
                    # next_idx+1: pass, shot이후에 block를 삽입
                    window = pd.concat(
                        [prev, next.loc[:swap_idx], pd.DataFrame([cur]), next.loc[swap_idx+1:]],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    raise ValueError("Block 이벤트의 이전, 이후 이벤트의 시간차이가 동일한 경우, 이벤트를 선택할 수 없습니다.")
            elif prev_pass_or_shot.empty and not next_pass_or_shot.empty:
                swap = next_pass_or_shot.iloc[0]
                swap_idx = swap.name
                window = pd.concat(
                    [prev, next.loc[:swap_idx], pd.DataFrame([cur]), next.loc[swap_idx+1:]],
                    axis=0, ignore_index=False, sort=False
                )
            elif not prev.empty and next_pass_or_shot.empty:
                swap = prev_pass_or_shot.iloc[-1]
                swap_idx = swap.name
                # prev_idx+1: pass, shot이후에 block를 삽입
                window = pd.concat(
                    [prev.loc[:swap_idx], pd.DataFrame([cur]), prev.loc[swap_idx+1:], next], 
                    axis=0, ignore_index=False, sort=False
                )
            else:
                return window

        return window

    def check_Pass_Received(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "in_transition":
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                raise ValueError(f"Pass Received전에는 항상 Pass가 존재한다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful":
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    # 현재까지(bepro, statsbomb) 존재하지는 않는 경우
                    print(f"성공한 패스의 Intended Receiver가 정확하게 존재해지 않는다\n: {window}")
            else:
                window = pd.concat(
                    [prev, next],
                    axis=0, ignore_index=False, sort=False
                )
                print(f"삭제({idx}): Pass Received이전에 항상 성공한 패스가 존재해야한다\n: {window}")
                # raise ValueError("Interded Received가 존재하지 않는다.")
        else:
            return window

        return window

    def check_Cross_Received(self, current_state, window, idx):
        return self.check_Pass_Received(current_state, window, idx)

    def check_Ball_Received(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)     
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:   
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )    
        else:
            return window
        
        return window

    def check_Recovery(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "in_possession":
            kwargs = {
                "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # prev이벤트를 수행한 선수가 loose ball을 만들어냄
                "team_id": prev.iloc[-1].team_id,
                "player_id": prev.iloc[-1].player_id,
                "start_x": prev.iloc[-1].end_x,
                "start_y": prev.iloc[-1].end_y,
                "end_x": prev.iloc[-1].end_x,
                "end_y": prev.iloc[-1].end_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Intervention", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != cur.team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)     
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )     
        else:
            return window
            raise ValueError("Recovery가 에러가 발생한 경우, loose ball을 만들어내는 Intervention이 발생해야 합니다: " + current_state)

        return window

    def check_Foul(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Foul_Throw(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window
    
    def check_Handball_Foul(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window
    
    def check_Foul_Won(self, current_state, window, idx):
        '''
        데이터를 잘못 기록한 문제로 제거되어야 정상적인 시퀀스가 됨
        '''
        prev, cur, next = self._get_context(window, idx)

        window = pd.concat(
            [prev, next],
            axis=0, ignore_index=False, sort=False
        )
        print(f"삭제({idx}): Foul Won은 Foul과 사전에 병합이 되어야하므로 존재해서는 안되는 이벤트이다\n: {window}")
        return window

    def check_Error(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot":
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)     
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )
        else:
            return window
            
        return window

    def check_Carry(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state == "ball_neutral":
            if prev.iloc[-1].team_id != cur.team_id: # recovery condition
                kwargs = {
                    "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # carry이전에 recovery이벤트 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Recovery이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   
            else: # Ball Received: Taking possession from a teammate other than receiving a pass        
                kwargs = {
                    "time_seconds": cur.time_seconds, # ball received이후 연속된 드리블 액션
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Ball Received", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "in_transition":
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Tackle이 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], pd.DataFrame([cur]), next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else: # received가 존재하지 않는 경우
                    kwargs = {
                        "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                        "team_id": pass_like.team_id,
                        "player_id": pass_like.relative_player_id,
                        "start_x": pass_like.end_x, 
                        "start_y": pass_like.end_y,
                        "end_x": pass_like.end_x,
                        "end_y": pass_like.end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
            else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)
                interception_cond = (
                    (pass_like.team_id != cur.team_id)  # 실패한 패스 이후 clearance가 발생하면 interception이 발생
                )
                recovery_cond = (
                    (pass_like.team_id == cur.team_id)  # 실패한 패스를 같은 팀원이 받은 경우 
                )
                if interception_cond:
                    kwargs = {
                        "time_seconds": cur.time_seconds, # interception이후 연속된 드리블 액션
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next], 
                        axis=0, ignore_index=False, sort=False
                    )
                elif recovery_cond:
                    kwargs = {
                        "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # clearance이전 recovery 결과 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,

                    }
                    new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    return window
                    raise ValueError(f"Carry가 에러가 발생한 경우, Interception 또는 Recovery이 발생해야 합니다: {window}")
        else:
            return window
            raise ValueError(f"TODO: Carry가 에러가 발생한 경우, 다른 state에서는 등장하는 사례가 없었다: {current_state}")
        
        return window

    def check_Take_On(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
        if current_state == "ball_neutral":
            if prev.iloc[-1].team_id != cur.team_id: # recovery condition
                kwargs = {
                    "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # carry이전에 recovery이벤트 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   
            else: # Ball Received: Taking possession from a teammate other than receiving a pass        
                kwargs = {
                    "time_seconds": cur.time_seconds, # ball received이후 연속된 드리블 액션
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Ball Received", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "in_transition":
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Tackle이 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], pd.DataFrame([cur]), next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else: # received가 존재하지 않는 경우
                    kwargs = {
                        "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                        "team_id": pass_like.team_id,
                        "player_id": pass_like.relative_player_id,
                        "start_x": pass_like.end_x, 
                        "start_y": pass_like.end_y,
                        "end_x": pass_like.end_x,
                        "end_y": pass_like.end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
            else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)
                interception_cond = (
                    (pass_like.team_id != cur.team_id)  # 실패한 패스 이후 clearance가 발생하면 interception이 발생
                )
                recovery_cond = (
                    (pass_like.team_id == cur.team_id)  # 실패한 패스를 같은 팀원이 받은 경우 
                )
                if interception_cond:
                    kwargs = {
                        "time_seconds": cur.time_seconds, # interception이후 연속된 드리블 액션
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,  
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next], 
                        axis=0, ignore_index=False, sort=False
                    )
                elif recovery_cond:
                    kwargs = {
                        "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # clearance이전 recovery 결과 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    return window
                    raise ValueError(f"Carry가 에러가 발생한 경우, Interception 또는 Recovery이 발생해야 합니다: {window}")
    
        return window
    
    def check_Shot(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "post_shot": # 생성: post_shot상태에서 슛이 또 발생한 경우, 누군가 공을 잡은 경우
            kwargs = {
                "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                "team_id": cur.team_id,
                "player_id": cur.player_id,
                "start_x": cur.start_x,
                "start_y": cur.start_y,
                "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                "end_y": cur.start_y,
                "dx": 0,
                "dy": 0,
            }
            new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "in_transition": # 누군가는 transtion상태의 공을 잡아야 전이가 가능
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Shot이 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], pd.DataFrame([cur]), next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else: # received가 존재하지 않는 경우
                    kwargs = {
                        "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                        "team_id": pass_like.team_id,
                        "player_id": pass_like.relative_player_id,
                        "start_x": pass_like.end_x, 
                        "start_y": pass_like.end_y,
                        "end_x": pass_like.end_x,
                        "end_y": pass_like.end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
            else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)
                interception_cond = (
                    (pass_like.team_id != cur.team_id)  # 실패한 패스 이후 shot이 발생하면 interception이 발생
                )
                recovery_cond = (
                    (pass_like.team_id == cur.team_id)  # 실패한 패스를 같은 팀원이 받은 경우
                )

                if interception_cond:
                    kwargs = {
                        "time_seconds": cur.time_seconds, # interception이후 연속된 슛팅팅 액션
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                elif recovery_cond:
                    kwargs = {
                        "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전 recovery 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    return window
                    raise ValueError(f"In_transition에서 shot이 발생한 에러인 경우, 공을 소유하는 이벤트를 생성해야 한다: {window}")
        elif current_state == "ball_neutral":
            if prev.iloc[-1].team_id != cur.team_id: # recovery condition
                kwargs = {
                    "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # carry이전에 recovery이벤트 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   
            else: # Ball Received: Taking possession from a teammate other than receiving a pass        
                kwargs = {
                    "time_seconds": cur.time_seconds, # ball received이후 연속된 드리블 액션
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Ball Received", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        else:
            return window
            raise ValueError("이외에 오류는 없다고 가정한다: " + current_state)
        
        return window

    def check_Clearance(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        if current_state == "in_transition":
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Clearance가가 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], pd.DataFrame([cur]), next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else: # received가 존재하지 않는 경우
                    kwargs = {
                        "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                        "team_id": pass_like.team_id,
                        "player_id": pass_like.relative_player_id,
                        "start_x": pass_like.end_x, 
                        "start_y": pass_like.end_y,
                        "end_x": pass_like.end_x,
                        "end_y": pass_like.end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
            else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)
                interception_cond = (
                    (pass_like.team_id != cur.team_id)  # 실패한 패스 이후 clearance가 발생하면 interception이 발생
                )
                recovery_cond = (
                    (pass_like.team_id == cur.team_id)  # 실패한 패스를 같은 팀원이 받은 경우 
                )
                if interception_cond:
                    kwargs = {
                        "time_seconds": cur.time_seconds, # interception이후 연속된 clearance 액션
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Interception", idx=idx,**kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=FileExistsError, sort=False
                )
                elif recovery_cond:
                    kwargs = {
                        "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # clearance이전 recovery 결과 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Recovery", idx=idx,**kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    return window
                    raise ValueError(f"Clearance가 실패하면, Interception이 발생해야 합니다: {window}")
        elif current_state == "ball_neutral":
            if prev.iloc[-1].team_id != cur.team_id: # recovery condition
                kwargs = {
                    "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # carry이전에 recovery이벤트 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   
            else: # Ball Received: Taking possession from a teammate other than receiving a pass        
                kwargs = {
                    "time_seconds": cur.time_seconds, # ball received이후 연속된 드리블 액션
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Ball Received", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "post_shot":  # shot -> clearnace?: Low Quality Shot의 경우 간섭없이 loose ball를 발생시키기도 함
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)   
            elif not shot.empty and shot.iloc[-1].result_name == "On Target":
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)           
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window
            
            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )      

        else:
            return window
            raise ValueError(f"TODO: Clearance가 에러가 발생한 경우, 다른 state에서는 등장하는 사례가 없었다: {current_state}")
        return window

    def check_Deflection(self, current_state, window, idx):
        '''
        Deflection: When a shot hits an outfield player's body and changes direction
        handler: Shot과의 스트림 문제를 해결하기 위해, Shot 이전에 발생한 Deflection을 처리한다.
        '''
        return self.check_Block(current_state, window, idx)

    def check_Offside(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Hit(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return window

    def check_Pass(self, current_state, window, idx):
        prev, cur, next = self._get_context(window, idx)
 
        if current_state == "in_transition":
            pass_like = self._get_prev_pass_like(window, idx)

            if pass_like.empty:
                return window
                raise ValueError(f"In_transition에서 Pass가 발생한 경우, Pass Like가 존재해야 합니다: {window}")       
            else:
                pass_like = pass_like.iloc[-1]

            if pass_like.result_name == "Successful": # 패스가 성공했기 때문에 received -> clearance
                if pass_like.relative_id in next.original_event_id.values: # 순서 조정
                    received_idx = next[next["original_event_id"] == pass_like.relative_id].index[0]
                    window = pd.concat(
                        [prev, next.loc[[received_idx]], pd.DataFrame([cur]), next.drop(index=received_idx)],
                        axis=0, ignore_index=False, sort=False
                    )
                else: # received가 존재하지 않는 경우
                    kwargs = {
                        "time_seconds": pass_like.relative_time_seconds, # pass received이후 연속된 clearance 액션
                        "team_id": pass_like.team_id,
                        "player_id": pass_like.relative_player_id,
                        "start_x": pass_like.end_x, 
                        "start_y": pass_like.end_y,
                        "end_x": pass_like.end_x,
                        "end_y": pass_like.end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Pass Received", idx=idx,**kwargs) # TODO: Cross Received도 가능은 함 
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
            else: # 패스가 실패했기 대문에 interception(or recovery) -> clearance(on-the-ball event)
                interception_cond = (
                    (pass_like.team_id != cur.team_id)  # 실패한 패스 이후 clearance가 발생하면 interception이 발생
                )
                recovery_cond = (
                    (pass_like.team_id == cur.team_id)  # 실패한 패스를 같은 팀원이 받은 경우
                )
                if interception_cond:
                    kwargs = {
                        "time_seconds": cur.time_seconds, # interception이후 연속된 pass 액션
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Interception", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                elif recovery_cond:
                    kwargs = {
                        "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # clearance이전 recovery 결과 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": cur.start_x,
                        "start_y": cur.start_y,
                        "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                        "end_y": cur.start_y,
                        "dx": 0,
                        "dy": 0,
                    }
                    new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)
                    window = pd.concat(
                        [prev, new_df, pd.DataFrame([cur]), next],
                        axis=0, ignore_index=False, sort=False
                    )
                else:
                    return window
                    raise ValueError(f"Clearance가 실패하면, Interception이 발생해야 합니다: {window}")
        elif current_state == "ball_neutral":
            if prev.iloc[-1].team_id != cur.team_id: # recovery condition
                kwargs = {
                    "time_seconds": (prev.iloc[-1].time_seconds + cur.time_seconds) / 2, # carry이전에 recovery이벤트 생성
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Recovery", idx=idx, **kwargs)   
            else: # Ball Received: Taking possession from a teammate other than receiving a pass        
                kwargs = {
                    "time_seconds": cur.time_seconds, # ball received이후 연속된 드리블 액션
                    "team_id": cur.team_id,
                    "player_id": cur.player_id,
                    "start_x": cur.start_x,
                    "start_y": cur.start_y,
                    "end_x": cur.start_x,   # Interception이후 Carry이벤트가 발생생
                    "end_y": cur.start_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Ball Received", idx=idx, **kwargs)   

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next],
                axis=0, ignore_index=False, sort=False
            )
        elif current_state == "post_shot":  # shot -> clearnace?: Low Quality Shot의 경우 간섭없이 loose ball를 발생시키기도 함
            shot = self._get_prev_shot_like(window, idx)

            if not shot.empty and shot.iloc[-1].result_name == "Blocked": 
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)   
            elif not shot.empty and shot.iloc[-1].result_name == "On Target":
                # 실제로 기록되지 않은 경우: Block은 무조건 슛을 막은은 상대팀 선수한테 기록하기 때문임
                if shot.iloc[-1].team_id == cur.team_id:
                    opponent = window[window.team_id != shot.iloc[-1].team_id]
                    if opponent.empty:
                        # 슛을 막은 팀의 id정보는 보간을 해야함(팀 정보가 없으면 확률값을 이상하게 예측함)
                        print(f"Block이 발생한 경우, 상대팀 선수가 존재해야 합니다: \n{window}")
                        return window
                    
                    kwargs = {
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": opponent.iloc[-1].team_id, # blocking한 선수의 주체는 모르지만 슛한 선수의 상대팀이 했다는 것은 확실함
                        "player_id": np.nan,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                else:
                    kwargs = {
                        "period_id": cur.period_id,
                        "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이전에 received이벤트 생성
                        "team_id": cur.team_id,
                        "player_id": cur.player_id,
                        "start_x": shot.iloc[-1].end_x, # TODO: 예상되는 이벤트데이터의 위치 예측?
                        "start_y": shot.iloc[-1].end_y,
                        "end_x": shot.iloc[-1].end_x,
                        "end_y": shot.iloc[-1].end_y,
                        "dx": 0,
                        "dy": 0,
                    }
                new_df = self._create_new_event(window, type_name="Block", idx=idx, **kwargs)     
            elif not shot.empty and shot.iloc[-1].result_name == "Off Target":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Post", idx=idx, **kwargs)
            elif not shot.empty and shot.iloc[-1].result_name == "Low Quality Shot":
                kwargs = {
                    "game_id": shot.iloc[-1].game_id,
                    "period_id": shot.iloc[-1].period_id,
                    "time_seconds": (shot.iloc[-1].time_seconds + cur.time_seconds) / 2, # shot이 goal post에 맞음
                    "team_id": shot.iloc[-1].team_id,
                    "player_id": shot.iloc[-1].player_id,
                    "start_x": shot.iloc[-1].end_x,
                    "start_y": shot.iloc[-1].end_y,
                    "end_x": shot.iloc[-1].end_x,
                    "end_y": shot.iloc[-1].end_y,
                    "dx": 0,
                    "dy": 0,
                }
                new_df = self._create_new_event(window, type_name="Goal Miss", idx=idx, **kwargs)
            else:
                return window

            window = pd.concat(
                [prev, new_df, pd.DataFrame([cur]), next], 
                axis=0, ignore_index=False, sort=False
            )      
        else:
            return window
            raise ValueError(f"TODO: Pass가 에러가 발생한 경우, 다른 state에서는 등장하는 사례가 없었다: {current_state}")
        
        return window

    def check_Cross(self, current_state, window, idx):
        return self.check_Pass(current_state, window, idx)