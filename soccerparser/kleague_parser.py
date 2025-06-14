import json
import os
import sys
import pickle

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from soccerparser.constants import Constants

class KLeagueParser:
    def __init__(self, match_id: int, data_dir: str = "data/k1_2023"):
        self.match_id = match_id
        self.data_dir = data_dir
        self.constants = Constants()

        self.players = None
        self.events = None
        self.mislabels = None

        self.ball_xy = None
        self.player_xy = None
        self.traces = None

        self.fps = 25

    def parse_events(self):
        # event_path = f"{self.data_dir}/raw_data/{self.match_id}/{self.match_id}-player_nodes.json"
        # with open(event_path) as f:
        #     events = pd.DataFrame(json.load(f)["data"]).dropna(subset=["playerId"]).copy()

        # metadata_path = f"{self.data_dir}/raw_data/{self.match_id}/{self.match_id}-match_info.json"
        # with open(metadata_path) as f:
        #     metadata = json.load(f)
        #     home_team = self.constants.TEAM_DICT[metadata["data"]["homeTeam"]["name"]]
        #     away_team = self.constants.TEAM_DICT[metadata["data"]["awayTeam"]["name"]]

        with open(f"{self.data_dir}/{self.match_id}/{self.match_id}_processed_dict.pkl", "rb") as f:
            match_dict = pickle.load(f)
            events = match_dict['event_df']
            teams_dict = match_dict['teams']
            metadata = match_dict['meta_data']
            tracking = match_dict['tracking_df']

        player_name_to_id = {**teams_dict['Home'].set_index('player')['pID'].to_dict(),
                                  **teams_dict['Away'].set_index('player')['pID'].to_dict()}
        home_team = self.constants.TEAM_DICT[metadata["home_team"]["team_name"]]
        away_team = self.constants.TEAM_DICT[metadata["away_team"]["team_name"]]

        # events["session"] = events["eventPeriod"].replace({"FIRST_HALF": 1, "SECOND_HALF": 2})
        events["session"] = events["period_name"].replace({"1st Half": 1, "2nd Half": 2})
        events["time"] = np.where(
            events["session"] == 1,
            # events["eventTime"] / 1000,
            # events["eventTime"] / 1000 - 2700,
            events["event_time"] / 1000,
            events["event_time"] / 1000 - 2700,
        )
        # events["team_name"] = events["teamName"].replace(self.constants.TEAM_DICT)
        # events["player_id"] = events["playerId"].astype(int)
        # events["squad_num"] = events["backNumber"].astype(int)
        # events["player_name"] = events["playerLastName"] + events["playerName"] # 이미 존재함
        # events["event_types"] = events["filteredEventTypes"].apply(lambda x: " ".join(x)) # 애초에 포맷이 다른듯
        # events.rename(columns={"id": "event_id", "teamId": "team_id"}, inplace=True)
        events["team_name"] = events["team_name"].map(self.constants.EN_TEAM_DICT)
        events["player_id"] = events["player_name"].map(player_name_to_id).astype(int) # 이벤트 데이터에 player_id정보가 제공되어 있지 않음
        events["squad_num"] = events["player_shirt_number"].astype(int)
        events["event_types"] = events["events"].copy()
        
        events["event_id"] = range(len(events)) # 고유한 이벤트 ID가 존재하지 않기 때문에 중복이 없다고 가정하고 생성
        events["team_id"] = events["team_name"].map({home_team: metadata["home_team"]["team_id"], 
                                                     away_team: metadata["away_team"]["team_id"]})

        tracking["time"] = (tracking['timestamp'].dt.total_seconds()).astype(float)
        tracking["time"] = np.where(
            tracking["period_id"] == 1,
            tracking["time"],
            tracking["time"] - 2700
        )
        
        # updated_events = []
        # for (_, period_event_group), (_, period_trace_group) in zip(events.groupby("session"), tracking.groupby("period_id")):
        #     merged = pd.merge_asof(
        #         period_event_group,
        #         period_trace_group[["time", "frame_id"]],
        #         on="time",
        #         direction="nearest"
        #     ).rename(columns={"frame_id": "frame"})
        #     updated_events.append(merged)
        # events = pd.concat(updated_events, ignore_index=True)
      
        et = events["event_types"]
        valid_events = events[et.apply(self.is_valid_event)]
        
        player_cols = ["team_id", "player_id", "team_name", "squad_num", "player_name"]
        players = valid_events[player_cols].drop_duplicates().copy()
        players["home_away"] = players["team_name"].replace({home_team: "H", away_team: "A"})
        players["home_away"] = pd.Categorical(players["home_away"], ["H", "A"])
        players["player_code"] = players.apply(lambda x: f"{x['home_away']}{x['squad_num']:02d}", axis=1)
        self.players = players.sort_values(["home_away", "squad_num"], ignore_index=True)

        tracking_players = tracking.groupby(['id'], as_index=False).agg(
            t0=('time', 'first'), t1=('time', 'last'), 
            s0=('period_id', 'first'), s1=('period_id', 'last'))
        tracking_players = tracking_players[tracking_players["id"] != "ball"].reset_index(drop=True)
        tracking_players["player_id"] = tracking_players["id"].astype(int)
        self.players = self.players.merge(tracking_players, on="player_id", how="left")
        
        # changeIn, changOut정보가 이벤트 로그에 기록되어 있지 않기 때문에 대체
        # self.players["s0"] = 1
        # self.players["t0"] = 0
        # self.players["s1"] = 2
        # self.players["t1"] = np.nan

        # in_out = events[(et.str.contains("changeIn")) | (et.str.contains("changeOut")) | et.str.contains("red")]
        # events.loc[in_out.index, "time"] = events.loc[in_out.index, "time"].clip(0)

        # player_in = events[et.str.contains("changeIn")]
        # for i in player_in.index:
        #     player_id = player_in.at[i, "player_id"]
        #     self.players.loc[self.players["player_id"] == player_id, "s0"] = player_in.at[i, "session"]
        #     self.players.loc[self.players["player_id"] == player_id, "t0"] = round(player_in.at[i, "time"], 1)

        # player_out = events[et.str.contains("changeOut") | et.str.contains("redCard")]
        # for i in player_out.index:
        #     player_id = player_out.at[i, "player_id"]
        #     self.players.loc[self.players["player_id"] == player_id, "s1"] = player_out.at[i, "session"]
        #     self.players.loc[self.players["player_id"] == player_id, "t1"] = round(player_out.at[i, "time"], 1)
        events = pd.merge(events.groupby(["session", "time"], as_index=False).first(), self.players) # valid이벤트를 한번 도 수행하지 않는 선수는 제거
        self.events = events.sort_values(["session", "time"], ignore_index=True)[self.constants.EVENT_HEADER].copy()

    def parse_ball_xy(self):
        # ball_path = f"{self.data_dir}/raw_data/{self.match_id}/{self.match_id}-ball_positions.json"

        # with open(ball_path) as f:
        #     ball_json = json.load(f)["data"]
        with open(f"{self.data_dir}/{self.match_id}/{self.match_id}_processed_dict.pkl", "rb") as f:
            match_dict = pickle.load(f)
            tracking = match_dict['tracking_df']
        
        # ball_xy = []
        # session = 0
        # episode = 0

        # for session_json in ball_json.values():
        #     session += 1
        #     for episode_json in session_json:
        #         episode += 1
        #         episode_ball_xy = pd.DataFrame(episode_json)
        #         episode_ball_xy["session"] = session
        #         # episode_ball_xy["episode"] = episode
        #         ball_xy.append(episode_ball_xy)

        # ball_xy = pd.concat(ball_xy).groupby(["session", "eventTime"], as_index=False).first()    
        ball_xy = tracking[tracking["id"] == "ball"].reset_index(drop=True)
        ball_xy["session"] = ball_xy["period_id"].astype(int)
        ball_xy["frame"] = ball_xy["frame_id"].astype(int)

        ball_xy["time"] = (ball_xy['timestamp'].dt.total_seconds()).astype(float)
        ball_xy["time"] = np.where(
            ball_xy["period_id"] == 1,
            ball_xy["time"],
            ball_xy["time"] - 2700
        )

        self.ball_xy = ball_xy[["session", "frame", "time", "x", "y"]]

    def parse_player_xy(self):
        # try:
        #     trace_dir = f"{self.data_dir}/raw_data/{self.match_id}/{self.match_id}-scouting_view_data"
        #     trace_files = os.listdir(trace_dir)
        #     trace_files.sort()
        # except FileNotFoundError as e:
        #     print(e)
        #     return

        # traces = []
        # tqdm_desc = "Loading minute-wise trace files"
        # bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

        # for minute_file in tqdm(trace_files, desc=f"{tqdm_desc:33s}", bar_format=bar_format):
        #     with open(f"{trace_dir}/{minute_file}") as f:
        #         try:
        #             minute_json = json.load(f)
        #             traces.extend(minute_json)
        #         except json.decoder.JSONDecodeError as e:
        #             print(e)
        #             return

        with open(f"{self.data_dir}/{self.match_id}/{self.match_id}_processed_dict.pkl", "rb") as f:
            match_dict = pickle.load(f)
            tracking = match_dict['tracking_df']

        tracking = tracking[tracking["id"] != "ball"].reset_index(drop=True)
        tracking["player_id"] = tracking["id"].astype(int)
        
        traces = tracking.merge(self.players, on="player_id", how="left")
    
        # traces["session"] = traces["eventPeriod"].replace({"FIRST_HALF": 1, "SECOND_HALF": 2})
        # traces["time"] = np.where(traces["session"] == 1, traces["matchTime"] / 1000, traces["matchTime"] / 1000 - 2700)
        # traces["frame"] = ((traces["time"] * 6).round() * 5).astype(int)
        # traces = traces.rename(columns={"groundX": "x", "groundY": "y"})

        traces["session"] = traces["period_id"].astype(int)
        traces["frame"] = traces["frame_id"].astype(int)
        traces["time"] = (traces['timestamp'].dt.total_seconds()).astype(float)
        traces["time"] = np.where(
            traces["period_id"] == 1,
            traces["time"],
            traces["time"] - 2700
        )
        # for session in traces["session"].unique():
        #     session_traces = traces[traces["session"] == session]
        #     frames = ((session_traces["frame"] - session_traces["frame"].min()) // 5) * 5
        #     traces.loc[traces["session"] == session, "frame"] = frames
        
        # Reshape by horizontal stacking
        # pivot_table(coords, index=["session", "frame"], columns="player_code")
        # 각 행은 (session, frame)의 조합으로 구성되고, 각 열은 고유한 player_code를 나타낸다. 
        # 내부 셀에는 해당 session과 frame에서 해당 player_code에 해당하는 좌표 값이 들어간다.
        x = traces.pivot_table("x", ["session", "frame"], "player_code")
        y = (traces.pivot_table("y", ["session", "frame"], "player_code"))
     
        player_codes = x.columns.tolist()
        for p in self.players["player_code"]:
            if p not in player_codes:
                print(f"Player {p} not measured.")
                return

        x.columns = [f"{p}_x" for p in player_codes]
        y.columns = [f"{p}_y" for p in player_codes]

        xy_cols = np.stack([x.columns, y.columns]).ravel("F").tolist()
        xy_6fps = pd.concat([x, y], axis=1)[xy_cols].reset_index()
     
        # Resample from 6fps to 10fps
        # xy_10fps = []
        # for session in xy_6fps["session"].unique():
        #     session_xy_6fps = xy_6fps[xy_6fps["session"] == session].copy()

        #     frames = pd.DataFrame(np.arange(0, session_xy_6fps["frame"].max() + 1), columns=["frame"])
        #     frames["session"] = session
        #     session_xy_30fps = pd.merge(frames, session_xy_6fps, how="left")
        #     session_xy_30fps.interpolate(method="cubic", limit_area="inside", inplace=True)

        #     session_xy_10fps = session_xy_30fps[0::3].copy()
        #     session_xy_10fps["time"] = (np.arange(0, len(session_xy_10fps)) * 0.1 + 0.1).round(1)
        #     xy_10fps.append(session_xy_10fps)

        # self.player_xy = pd.concat(xy_10fps, ignore_index=True)
        frame_to_time_dict = {
            (session, frame): time for session, frame, time in traces.groupby(["session", "frame"])["time"].first().reset_index().values
        }
        self.player_xy = xy_6fps.copy()
        self.player_xy["time"] = self.player_xy.apply(lambda x: frame_to_time_dict[(x["session"], x["frame"])], axis=1)
        
        # Synchronize with player substitution records
        self.players["in_frame"] = 0
        self.players["out_frame"] = 0

        for player_idx in self.players.index:
            p = self.players.at[player_idx, "player_code"]

            s0 = self.players.at[player_idx, "s0"]
            t0 = self.players.at[player_idx, "t0"] #round(self.players.at[player_idx, "t0"] + 0.1, 1)
            i0 = self.player_xy[(self.player_xy["session"] == s0) & (self.player_xy["time"] == t0)].index[0]

            self.player_xy.loc[: i0 - 1, [f"{p}_x", f"{p}_y"]] = np.nan

            s1 = self.players.at[player_idx, "s1"]
            t1 = self.players.at[player_idx, "t1"]
            if t1 != t1:  # Played until the end of the game
                i1 = self.player_xy.index[-1]
            elif s1 > 1 and t1 == 0:  # Half-time substitution
                i1 = self.player_xy[self.player_xy["session"] == s1 - 1].index[-1]
                self.player_xy.loc[i1 + 1 :, [f"{p}_x", f"{p}_y"]] = np.nan
            else:  # Regular substitution
                try:
                    i1 = self.player_xy[(self.player_xy["session"] == s1) & (self.player_xy["time"] == t1)].index[0]
                    self.player_xy.loc[i1 + 1 :, [f"{p}_x", f"{p}_y"]] = np.nan
                except IndexError:
                    i1 = self.player_xy.index[-1]

            inplay_xy = self.player_xy.loc[i0:i1, [f"{p}_x", f"{p}_y"]]
            self.player_xy.loc[i0:i1, [f"{p}_x", f"{p}_y"]] = inplay_xy.interpolate(limit_direction="both")

            # in_frame, out_frame: # inplay tracking data index
            self.players.at[player_idx, "in_frame"] = i0
            self.players.at[player_idx, "out_frame"] = i1
        
        self.players.drop(["s0", "t0", "s1", "t1"], axis=1, inplace=True)

        # Label phases
        session_ends = set((self.player_xy.groupby("session")["time"].count().cumsum() - 1))
        player_outs = set(self.players["out_frame"].unique())
        phase_changes = np.sort(list(session_ends | player_outs))

        self.player_xy["phase"] = 0
        for i in range(len(phase_changes)):
            first_frame = phase_changes[i - 1] + 1 if i > 0 else 0
            last_frame = phase_changes[i]
            self.player_xy.loc[first_frame:last_frame, "phase"] = i + 1

        self.player_xy = self.player_xy[["session", "time", "phase"] + xy_cols]
        self.player_xy.index.name = "frame"

        return 1

    def is_valid_event(self, event_types):
        # valid type에 해당하는 이벤트가 존재하는지
        # return (event_types != event_types) or (len(set(event_types.split()) & self.constants.VALID_TYPES) > 0)
       
        return (event_types != event_types) or any(e.get("event_name", None) in self.constants.VALID_TYPES for e in event_types)

    def combine_events_and_ball_xy(self):
        # 현성님이 사용하신 ball_position데이터는 이벤트 시점에 공의 위치가 기록이 되기 때문에 이벤트 데이터의 time가 일치하지만,
        # 우리가 받은 트래킹 데이터는 ball_position이 없어서 트래킹 데이터의 공의 위치 기반으로 사용하기 때문에 구현 방식이 다르다.
        # self.events = pd.merge(self.events, self.ball_xy, how="right")
        updated_events = []
        for (_, period_event_group), (_, period_ball_group) in zip(self.events.groupby("session"), self.ball_xy.groupby("session")):
            merged = pd.merge_asof(
                period_event_group,
                period_ball_group[["time", "x", "y"]],
                on="time",
                direction="nearest"
            )
            updated_events.append(merged)
        self.events = pd.concat(updated_events, ignore_index=True)

        self.events = self.events[self.events["event_types"].apply(self.is_valid_event)].copy() 

        self.events["event_id"] = self.events["event_id"].fillna(0).astype(int)
        self.events["time"] = self.events["time"]

        return
        # 이후 로직은 사용하지 않음.
        # self.events["x"] = self.events["x"] * self.constants.PITCH_SIZE[0]
        # self.events["y"] = (1 - self.events["y"]) * self.constants.PITCH_SIZE[1]
        #self.events["event_types"] = self.events["event_types"].fillna("") # 발생하지 않음

        # Remove cooccurring events without annotation
        #self.events.drop_duplicates(inplace=True) # 발생하지 않음
        # cooccur_counts = self.events.groupby(["session", "time"])["event_id"].count()
        # cooccur_times = cooccur_counts[cooccur_counts > 1].reset_index().drop("event_id", axis=1)
        # cooccur_events = pd.merge(self.events.reset_index(), cooccur_times).set_index("index")
        # self.events = self.events.drop(cooccur_events[cooccur_events["event_types"] == ""].index)

        # Delay events of specific types that cooccurred with other events by 1 frame
        cooccur_counts = self.events.groupby(["session", "time"])["event_id"].count()
        cooccur_times = cooccur_counts[cooccur_counts > 1].reset_index().drop("event_id", axis=1)
        cooccur_events = pd.merge(self.events.reset_index(), cooccur_times).set_index("index")
        hits = cooccur_events[
            (cooccur_events["event_types"].str.contains("block"))
            | (cooccur_events["event_types"].str.contains("cutoff"))
            | (cooccur_events["event_types"].str.contains("hit"))
            | (cooccur_events["event_types"].str.contains("intercept"))
            | (cooccur_events["event_types"].str.contains("PassReceived"))
            | (cooccur_events["event_types"].str.contains("save"))
        ]
        self.events.loc[hits.index, "time"] = (self.events.loc[hits.index, "time"] + 0.1).round(1)
        self.events.sort_values(["session", "time"], ignore_index=True, inplace=True)

        cooccur_counts = self.events.groupby(["session", "time"])["event_id"].count()
        cooccur_times = cooccur_counts[cooccur_counts > 1].reset_index().drop("event_id", axis=1)
        for i in cooccur_times.index:
            session = cooccur_times.at[i, "session"]
            time = cooccur_times.at[i, "time"]
            event_pair = self.events[(self.events["session"] == session) & (self.events["time"] == time)]

            if len(event_pair["player_code"].unique()) == 1:
                # Merge cooccurring events performed by a single player
                self.events.at[event_pair.index[0], "event_types"] = " ".join(event_pair["event_types"])
                self.events = self.events.drop(event_pair.index[1:])

            else:
                # Rearrange event times so that they are not duplicated
                start_idx = event_pair.index[0]
                start_time = self.events.at[start_idx, "time"]
                time_diff = (self.events.loc[start_idx : start_idx + 10, "time"].diff()).round(1)

                if time_diff[time_diff > 0.3].empty:
                    end_idx = time_diff.index[-1]
                else:
                    end_idx = time_diff[time_diff > 0.3].index[0] - 1

                n_events = end_idx - start_idx + 1
                end_time = round(start_time + (n_events - 1) * 0.1, 1)
                self.events.loc[start_idx:end_idx, "time"] = np.linspace(start_time, end_time, n_events)

        self.events["time"] = self.events["time"].round(1)

        cooccur_counts = self.events.groupby(["session", "time"])["event_id"].count()
        cooccur_times = cooccur_counts[cooccur_counts > 1].reset_index().drop("event_id", axis=1)
        if cooccur_times.empty:
            print("All cooccurrences of event removed.")
            self.events.reset_index(drop=True, inplace=True)
        else:
            print("There remain some cooccurrences of event.")
            return pd.merge(self.events.reset_index(), cooccur_times).set_index("index")

    def correct_mislabels(self, thres_error=5):
        # Correct mislabeled players
        # self.events.set_index("frame", inplace=True)
        self.events.set_index(["frame", "event_id"], inplace=True)

        event_snapshots = self.traces[self.traces["player_code"].notna()].copy()
        event_snapshots["poss_x"] = event_snapshots.apply(lambda x: x[f"{x['player_code']}_x"], axis=1)
        event_snapshots["poss_y"] = event_snapshots.apply(lambda x: x[f"{x['player_code']}_y"], axis=1)
        
        error_x = event_snapshots["ball_x"] - event_snapshots["poss_x"]
        error_y = event_snapshots["ball_y"] - event_snapshots["poss_y"]
        event_snapshots["ball_error"] = np.sqrt(error_x**2 + error_y**2)
        # mislabels.index == frame
        mislabels = event_snapshots[~(event_snapshots["ball_error"] < thres_error)].copy()

        self.traces = self.traces.reset_index().set_index(["frame", "event_id"])
        mislabels = mislabels.reset_index().set_index(["frame", "event_id"])
        
        if not mislabels.empty:
            home_players = self.players[self.players["home_away"] == "H"]["player_code"]
            away_players = self.players[self.players["home_away"] == "A"]["player_code"]

            home_dist_x = mislabels[[f"{p}_x" for p in home_players]] - mislabels["ball_x"].values[:, np.newaxis]
            home_dist_y = mislabels[[f"{p}_y" for p in home_players]] - mislabels["ball_y"].values[:, np.newaxis]
            home_dist = np.sqrt(home_dist_x.values**2 + home_dist_y.values**2)
            home_nearest = pd.DataFrame(home_dist, columns=home_players).idxmin(axis=1)

            away_dist_x = mislabels[[f"{p}_x" for p in away_players]] - mislabels["ball_x"].values[:, np.newaxis]
            away_dist_y = mislabels[[f"{p}_y" for p in away_players]] - mislabels["ball_y"].values[:, np.newaxis]
            away_dist = np.sqrt(away_dist_x.values**2 + away_dist_y.values**2)
            away_nearest = pd.DataFrame(away_dist, columns=away_players).idxmin(axis=1)

            corrected = np.where(mislabels["player_code"].apply(lambda x: x[0] == "H"), home_nearest, away_nearest)
      
            self.events.loc[mislabels.index, "player_code"] = corrected
            self.traces.loc[mislabels.index, "player_code"] = corrected

            mislabels["annotated"] = mislabels["player_code"]
            mislabels["corrected"] = corrected
            self.mislabels = mislabels[["session", "time", "annotated", "corrected", "ball_error"]]

        self.events.reset_index(inplace=True)
        self.traces.reset_index(inplace=True)
        # Correct time-flipped event types
        # prev_players = self.events["player_code"].shift(1)
        # second_events = self.events[self.events["player_code"] == prev_players]
        # time_flipped = second_events[second_events["event_types"].str.contains("passReceived")]

        # for i in time_flipped.index:
        #     cur_types = self.events.at[i, "event_types"]
        #     prev_types = self.events.at[i - 1, "event_types"]
        #     self.events.at[i - 1, "event_types"] = cur_types
        #     self.events.at[i, "event_types"] = prev_types

    def combine_player_and_ball_xy(self):
        # pitch_x = self.constants.PITCH_SIZE[0]
        # pitch_y = self.constants.PITCH_SIZE[1]

        time_range = self.player_xy[["session", "time", "phase"]].reset_index()

        # 현성님은 fps=10으로 했기 때문에 time.round(1)로 하면, 시간 기반으로 이벤트 데이터랑 트래킹 데이터가 정확하게 병합이 가능하지만,
        # bepro는 fps=25로 했기 때문에 가장 가까운 시간대로 병합을 해야함.
        updated_events = []
        for (_, period_event_group), (_, period_trace_group) in zip(self.events.groupby("session"), time_range.groupby("session")):
            merged = pd.merge_asof(
                period_event_group,
                period_trace_group[["time", "phase", "frame"]],
                on="time",
                direction="nearest"
            )
            updated_events.append(merged)
        self.events = pd.concat(updated_events, ignore_index=True)

        # time컬럼 제외: 현성님 로직 처럼 time이 전부 일치하는 포맷이 아니기 때문에 병합할 때 제외.
        # 단, 이렇게 사용하면 매우 근접한 이벤트는 동일한 frame이 할당되기 때문에 추가적인 기준이 필요함 -> event_id 추가
        # event_cols = ["frame", "session", "time", "phase", "player_code", "event_types", "x", "y"]
        event_cols = ["frame", "session", "event_id", "phase", "player_code", "event_types", "x", "y"] 
        self.traces = pd.merge(self.events[event_cols], self.player_xy.reset_index("frame"), how="right")
        self.traces = self.traces.rename(columns={"x": "ball_x", "y": "ball_y"}).set_index("frame")

        # Detect and correct mislabels
        return self.correct_mislabels()
    
        # When the ball was possessed by a player
        poss_prev = self.traces["player_code"].fillna(method="ffill")
        poss_next = self.traces["player_code"].fillna(method="bfill")
        self.traces["player_code"] = poss_prev.where(poss_prev == poss_next, np.nan)

        poss_traces = self.traces[self.traces["player_code"].notna()]
        self.traces.loc[poss_traces.index, "ball_x"] = poss_traces.apply(lambda x: x[f"{x['player_code']}_x"], axis=1)
        self.traces.loc[poss_traces.index, "ball_y"] = poss_traces.apply(lambda x: x[f"{x['player_code']}_y"], axis=1)

        # When a team scored a goal
        goals = self.events[self.events["event_types"].str.contains("goalAgainst")]
        self.traces.loc[goals["frame"], "ball_x"] = np.where(goals["x"] < pitch_x / 2, 0, pitch_x)
        self.traces.loc[goals["frame"], "ball_y"] = pitch_y / 2

        # When the ball was out of play
        types_after_out = {"goalKickSucceeded", "goalKickFailed", "cornerKick", "throwIn"}
        next_event_types = self.events["event_types"].shift(-2).fillna("")
        ballout_events = self.events[next_event_types.apply(lambda x: len(set(x.split()) & types_after_out) > 0)]
        ballout_events = ballout_events[ballout_events["player_code"].notna()]

        for i in ballout_events.index:
            session = self.events.at[i, "session"]
            phase = self.events.at[i, "phase"]

            cur_frame = self.events.at[i, "frame"]
            cur_type = self.events.at[i, "event_types"].split()[0]
            cur_x = self.traces.at[cur_frame, "ball_x"]
            cur_y = self.traces.at[cur_frame, "ball_y"]

            next_frame = self.events.at[i + 2, "frame"]
            next_type = self.events.at[i + 2, "event_types"].split()[0]
            next_x = self.traces.at[next_frame, "ball_x"]
            next_y = self.traces.at[next_frame, "ball_y"]

            if "goalKick" in next_type or "cornerKick" in next_type:
                out_x = round(round(next_x / pitch_x, 0) * pitch_x, 0)
                if "shot" in cur_type or "cross" in cur_type:
                    out_y = int(cur_y < pitch_y / 2) * pitch_y * 0.2 + pitch_y * 0.4
                else:
                    out_y = cur_y
            else:  # "throwIn" in next_type
                out_x = next_x
                out_y = round(round(next_y / pitch_y, 0) * pitch_y, 0)

            time_diff = (np.linalg.norm([out_x - cur_x, out_y - cur_y]) / 15).round(1)
            out_frame = cur_frame + int(time_diff * 10)
            out_time = self.traces.at[out_frame, "time"]

            out_event = {
                "frame": out_frame,
                "session": session,
                "time": out_time,
                "phase": phase,
                "event_id": 0,
                "team_id": 0,
                "player_id": 0,
                "squad_num": 0,
                "event_types": "",
                "x": out_x,
                "y": out_y,
            }
            self.events.loc[len(self.events)] = out_event
            self.traces.loc[out_frame, ["ball_x", "ball_y"]] = [out_x, out_y]

        self.events.sort_values("frame", ignore_index=True, inplace=True)

        # When the ball was moving between players
        for session in self.traces["session"].unique():
            session_traces = self.traces[self.traces["session"] == session]
            session_ball_xy = session_traces[["ball_x", "ball_y"]].interpolate(limit_direction="both")
            self.traces.loc[self.traces["session"] == session, ["ball_x", "ball_y"]] = session_ball_xy

    def split_into_episodes(self, margin_frames=10):
        self.traces["episode"] = 0
        count = 0

        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_first_frame = phase_traces.index[0]
            phase_last_frame = phase_traces.index[-1]

            phase_events = phase_traces.dropna(subset=["event_types"])
            assert isinstance(phase_events, pd.DataFrame)

            time_diffs = phase_events["time"].diff().fillna(60)
            episodes = (time_diffs > 10).astype(int).cumsum() + count
            count = episodes.max() if not episodes.empty else count

            for episode in episodes.unique():
                episode_event_frames = episodes[episodes == episode]
                first_frame = max(phase_first_frame, episode_event_frames.index[0] - margin_frames)
                last_frame = min(phase_last_frame, episode_event_frames.index[-1] + margin_frames)
                self.traces.loc[first_frame:last_frame, "episode"] = episode

        header = ["session", "time", "phase", "episode", "player_code", "event_types"]
        xy_cols = [c for c in self.traces.columns if c[-2:] in ["_x", "_y"]]
        self.traces = self.traces[header + xy_cols]

    def calc_running_features(self, remove_outliers=True, smoothing=True):
        processor = TraceProcessor(self.traces)
        processor.calc_runnning_features(remove_outliers, smoothing)
        self.traces = processor.traces

    def save(self):
        player_dir = f"{self.data_dir}/player_info"
        event_dir = f"{self.data_dir}/parsed_events"
        trace_dir = f"{self.data_dir}/parsed_traces"

        if not os.path.exists(player_dir):
            os.makedirs(player_dir)
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        self.players.to_csv(f"{player_dir}/{self.match_id}.csv", index=False, encoding="utf-8-sig")
        self.events.to_csv(f"{event_dir}/{self.match_id}.csv", index=False, encoding="utf-8-sig")
        self.traces.to_csv(f"{trace_dir}/{self.match_id}.csv")


if __name__ == "__main__":
    season = "k1_2023"
    match_ids = np.sort([int(f) for f in os.listdir(f"data/{season}/raw_data") if f[0] != "."])
    constants = Constants()

    for i, match_id in enumerate(match_ids):
        # if os.path.exists(f"data/parsed_traces/{match_id}.csv"):
        #     continue

        metadata_path = f"data/{season}/raw_data/{match_id}/{match_id}-match_info.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            home_team = constants.TEAM_DICT[metadata["data"]["homeTeam"]["name"]]
            away_team = constants.TEAM_DICT[metadata["data"]["awayTeam"]["name"]]
            print(f"\n[{i}] {match_id}: {home_team} vs {away_team}")

            # users = constants.OHCOACH_USERS_2023 if season.endswith("2023") else constants.OHCOACH_USERS_2022
            # if home_team in users and away_team in users:
            #     print(f"\n[{i}] {match_id}: {home_team} vs {away_team}")
            # else:
            #     continue

        parser = KLeagueParser(match_id, data_dir=f"data/{season}")
        parser.parse_events()
        parser.parse_ball_xy()
        parser.combine_events_and_ball_xy()
        parser.parse_player_xy()

        if parser.player_xy is not None:
            parser.combine_player_and_ball_xy()
            if parser.mislabels is not None and parser.mislabels["ball_error"].isna().any():
                parser.split_into_episodes()
                parser.calc_running_features()
                parser.save()
                print("Successfully saved.")