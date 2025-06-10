class Condition:
    def _get_context(self, window, idx):
        prev = window.loc[:idx-1]     # 이전 이벤트
        cur = window.loc[idx]     # 현재 이벤트(현재 state에서 발생할 수 없는 이벤트)
        next = window.loc[idx+1:] # 이벤트 시퀀스 검증을 위함

        return prev, cur, next
    
    # ----미래 이벤트 조건----
    def is_next_event(self, window, idx, type_names):
        prev, cur, next = self._get_context(window, idx)

        return next.empty or any(next.iloc[0]['type_name'] == t for t in type_names) # 경기가 종료되거나 다음 이벤트가 발생해야함

    def is_next_interception(self, window, idx):
        return self.is_next_event(window, idx, ["Interception"])

    def is_next_intervention(self, window, idx):
        return self.is_next_event(window, idx, ["Intervention"])

    def is_next_possession_gain(self, window, idx):
        return self.is_next_event(window, idx, ["Ball Received", "Recovery"])

    def is_next_loose_ball(self, window, idx):
        return self.is_next_event(window, idx, ["Deflection", "Block", "Hit"])

    def is_next_foul(self, window, idx):
        return self.is_next_event(window, idx, ["Foul", "Foul Won"])
    
    def is_next_setpiece(self, window, idx):
        setpiece_type = ["Pass_Freekick", "Pass_Corner", "Throw-in", "Shot_Freekick", 
                         "Shot_Corner", "Penalty Kick", "Goal Kick"]
        return self.is_next_event(window, idx, setpiece_type)
    
    def is_next_catch(self, window, idx):
        return self.is_next_event(window, idx, ["Catch"])
    
    def is_next_parry(self, window, idx):   
        return self.is_next_event(window, idx, ["Parry"])
    
    # ----현재 이벤트 조건----
    def is_failed_tackle(self, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return (cur.type_name == "Tackle") and (cur.result_name == "Unsuccessful")
    
    def is_prev_goal(self, window, idx): 
        """
        득점 이후 항의가 발생한 경우 예외 처리
        """
        prev, cur, next = self._get_context(window, idx)

        return "Goal" in prev["type_name"].values
    
    def is_blocked_shot(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        shot_like = [
            "Shot",
            "Shot_Freekick",
            "Penalty Kick",
        ]

        shot = prev[prev["type_name"].isin(shot_like)]
        return (not shot.empty) and (shot.result_name == "Blocked")

    def is_failed(self, window, idx):
        prev, cur, next = self._get_context(window, idx)

        return cur.result_name == "Unsuccessful"
    
    def is_goalkeeper_condition(self, window, idx):
        return (
            self.is_failed(window, idx) or
            self.is_next_setpiece(window, idx) or
            self.is_next_catch(window, idx) or
            self.is_next_parry(window, idx) or
            self.is_next_loose_ball(window, idx) or
            self.is_next_foul(window, idx) # 태클이 성공했지만 (태클한 선수가) 반칙을 당하는 경우
        )
    
    def is_intended_receiver(self, window, idx):
        prev, cur, next = self._get_context(window, idx)
        pass_like = [
            "Pass",
            "Cross",
            "Throw-In",
            "Pass_Freekick",
            "Pass_Corner",
            "Goal Kick",
        ]

        pass_like = prev[prev["type_name"].isin(pass_like)]

        return (
            (cur.result_name == "Unsuccessful") or 
            (pass_like.iloc[-1].relative_id == cur.original_event_id)
        )
    
    # ----tackle의 조건 세분화----
    def is_tackle_in_transition(self, window, idx):
        return (
            self.is_failed_tackle(window, idx) or
            self.is_next_interception(window, idx) or
            self.is_next_intervention(window, idx) or
            self.is_next_loose_ball(window, idx) or # 태클한 선수의 몸에 부딪히는 경우
            self.is_next_foul(window, idx) # 태클이 성공했지만 (태클한 선수가) 반칙을 당하는 경우
        )
    
    def is_tackle_in_posession(self, window, idx):
        return (
            self.is_failed_tackle(window, idx) or
            self.is_next_interception(window, idx) or
            self.is_next_intervention(window, idx) or
            self.is_next_foul(window, idx) # 태클이 성공했지만 (태클한 선수가) 반칙을 당하는 경우
        )
    
    def is_tackle_ball_neutral(self, window, idx):
        return (
            self.is_failed_tackle(window, idx) or
            self.is_next_interception(window, idx) or
            self.is_next_intervention(window, idx) or
            self.is_next_possession_gain(window, idx) or
            self.is_next_foul(window, idx) # 태클이 성공했지만 (태클한 선수가) 반칙을 당하는 경우
        )

    def is_tackle_post_shot(self, window, idx):
        return (
            self.is_failed_tackle(window, idx) or
            self.is_next_loose_ball(window, idx) or
            self.is_next_foul(window, idx) # 태클이 성공했지만 (태클한 선수가) 반칙을 당하는 경우
        )
    
    def always_true(self, window, idx):
        return True 

    # 절대 등장해서는 안되는 이벤트: Goal Conceded
    def always_false(self, window, idx):
        return False