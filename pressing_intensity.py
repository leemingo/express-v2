import numpy as np
from kloppy import sportec
from unravel.soccer import KloppyPolarsDataset, PressingIntensity
from typing import Literal, List, Union
import os 
import polars as pl
import pickle
import pandas as pd

from dataclasses import dataclass
from typing import Union, Any, Dict
import config as C
from config import Constant, Column, Group

@dataclass
class DefaultSettings:
    home_team_id: Union[str, int]
    away_team_id: Union[str, int]
    max_player_speed: float = 12.0
    max_ball_speed: float = 28.0
    max_player_acceleration: float = 6.0
    max_ball_acceleration: float = 13.5
    ball_carrier_threshold: float = 25.0

class PressingIntensityDataset(KloppyPolarsDataset):
    def __init__(self, tracking_df, settings):
        self.data = pl.from_pandas(tracking_df)
        self.settings = settings


class CustomPressingIntensity(PressingIntensity):
    @property
    def __exprs_variables(self):
        return [
            Column.X,
            Column.Y,
            Column.Z,
            Column.VX,
            Column.VY,
            Column.VZ,
            Column.SPEED,
            Column.TEAM_ID,
            Column.BALL_OWNING_TEAM_ID,
            Column.OBJECT_ID,
            Column.IS_BALL_CARRIER,
        ]

    def __repr__(self):
        n_frames = (
            self.output[Column.FRAME_ID].nunique() if hasattr(self, "output") else None
        )
        return f"PressingIntensity(n_frames={n_frames})"
    
    def fit(
        self,
        speed_threshold: float = None,
        reaction_time: float = 0.7,
        time_threshold: float = 1.5,
        sigma: float = 0.45,
        method: Literal["teams", "full"] = "teams",
        ball_method: Literal["include", "exclude", "max"] = "max",
        orient: Literal[
            "ball_owning", "pressing", "home_away", "away_home"
        ] = "ball_owning",
    ):
        """
        method: str ["teams", "full"]
            "teams" creates a 11x11 matrix, "full" creates a 22x22 matrix
        ball_method: str ["include", "exclude", "max"]
            "include" creates a 11x12 matrix
            "exclude" ignores ball
            "max" keeps 11x11 but ball carrier pressing intensity is now max(ball, ball_carrier)
        speed_threshold: float.
            Masks pressing intensity to only include players travelling above a certain speed
            threshold in meters per second.
        orient: str ["ball_owning", "pressing", "home_away", "away_home"]
            Pressing Intensity output as seen from the 'row' perspective.
            method and orient are in sync, meaning "full" and "away_home" sorts row and columns
            such that the away team players are displayed first
        """
        print("*****Custom Fit*****")
        # if period_id is not None and not isinstance(period_id, int):
        #     raise TypeError("period_id should be of type integer")
        if method not in ["teams", "full"]:
            raise ValueError("method should be 'teams' or 'full'")
        if ball_method not in ["include", "exclude", "max"]:
            raise ValueError("ball_method should be 'include', 'exclude' or 'max'")
        if orient not in ["ball_owning", "pressing", "home_away", "away_home"]:
            raise ValueError(
                "method should be 'ball_owning', 'pressing', 'home_away', 'away_home'"
            )
        if not isinstance(reaction_time, Union[float, int]):
            raise TypeError("reaction_time should be of type float")
        if speed_threshold is not None and not isinstance(
            speed_threshold, Union[float, int]
        ):
            raise TypeError("speed_threshold should be of type float (or None)")
        if not isinstance(time_threshold, Union[float, int]):
            raise TypeError("time_threshold should be of type float")
        if not isinstance(sigma, Union[float, int]):
            raise TypeError("sigma should be of type float")

        self._method = method
        self._ball_method = ball_method
        self._speed_threshold = speed_threshold
        self._reaction_time = reaction_time
        self._time_threshold = time_threshold
        self._sigma = sigma
        self._orient = orient
        
        period_id_lst = self.dataset['period_id'].unique().to_numpy()
        # self.result_df = pl.DataFrame()
        results_list = []
        for period_id in period_id_lst:
            period_df = self.dataset.filter(
                 (pl.col(Column.PERIOD_ID) == period_id)
            )
            frame_id_lst = period_df['frame_id'].unique().to_numpy()
            for frame_id in frame_id_lst:
                df = period_df.filter(
                    (pl.col(Column.FRAME_ID) == frame_id)
                )
                if df.filter(pl.col("is_ball_carrier") == True).shape[0] == 0 : continue

                sort_descending = [False] * len(Group.BY_TIMESTAMP)
                if self._orient in ["home_away", "away_home"]:
                    alias = "is_home"
                    sort_by = Group.BY_TIMESTAMP + [alias]
                    sort_descending = sort_descending + (
                        [True] if self._orient == "home_away" else [False]
                    )
                    with_columns = [
                        pl.when(pl.col(Column.TEAM_ID) == self.settings.home_team_id)
                        .then(True)
                        .when(pl.col(Column.TEAM_ID) == Constant.BALL)
                        .then(None)
                        .otherwise(False)
                        .alias(alias)
                    ]
                elif self._orient in ["ball_owning", "pressing"]:
                    alias = "is_ball_owning"
                    sort_by = Group.BY_TIMESTAMP + [alias]
                    sort_descending = sort_descending + (
                        [True] if self._orient == "ball_owning" else [False]
                    )
                    with_columns = [
                        pl.when(pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID))
                        .then(True)
                        .when(pl.col(Column.TEAM_ID) == Constant.BALL)
                        .then(None)
                        .otherwise(False)
                        .alias(alias)
                    ]
                try:
                    output_df = (
                        df.with_columns(with_columns)
                        .sort(by=sort_by, descending=sort_descending, nulls_last=True)
                        .group_by(Group.BY_TIMESTAMP, maintain_order=True)
                        .agg(
                            pl.map_groups(
                                exprs=self.__exprs_variables,
                                # function=self.__compute,
                                function=self._PressingIntensity__compute,
                            ).alias("results")
                        )
                        .unnest("results")
                    )
                except Exception as e:    
                    print(f"Error at index {frame_id}: {e}")
                    output_df = pl.DataFrame()  # 에러 발생 시 빈 DataFrame으로 대체
                # self.result_df = pl.concat([self.result_df, self.output])
                results_list.append(output_df.to_pandas())
        self.output = pd.concat(results_list, ignore_index=True)
        return self

# def load_dfl_spoho(data_path):

def load_bepro(data_path):
    data_path = "/data/MHL/bepro/processed"
    match_id_lst = os.listdir(data_path)
    total_dict = {match_id : {} for match_id in match_id_lst}

    for match_id in match_id_lst:
        if os.path.exists(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl")):
            with open(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl"), "rb") as f:
                total_dict[match_id] = pickle.load(f)
        else:
            print(f"Proceseed data {match_id} not exists.")
        
        tracking_df = total_dict[match_id]['tracking_df']
        teams_dict = total_dict[match_id]['teams']
        
        # Define orientation
        first_period = tracking_df[(tracking_df['period_id'] == 1)]
        first_frame = first_period[(first_period['frame_id'] == first_period['frame_id'].min())]
        left_tid = first_frame.loc[first_frame['x'].idxmin(), 'team_id']

        if left_tid == teams_dict['Home']['tID'].iloc[0]:
            orient = 'home_away'
        elif left_tid == teams_dict['Away']['tID'].iloc[0]:
            orient = 'away_home'
        else:
            print('error')

        settings = DefaultSettings(
            home_team_id=teams_dict['Home']['tID'].iloc[0],
            away_team_id=teams_dict['Away']['tID'].iloc[0],
            max_player_speed=C.MAX_PLAYER_SPEED,
            max_ball_speed=C.MAX_BALL_SPEED,
            max_player_acceleration=C.MAX_PLAYER_ACCELERATION,
            max_ball_acceleration=C.MAX_BALL_ACCELERATION,
            ball_carrier_threshold=C.BALL_CARRIER_THRESHOLD,
        )

        dataset = PressingIntensityDataset(tracking_df, settings)
        model = CustomPressingIntensity(dataset=dataset)
        print(f"Calcuate Pressing Intensity {match_id}")
        model.fit(
            method="teams",
            ball_method="max",
            orient=orient,
            speed_threshold=2.0,
        )
        with open(f"{data_path}/{match_id}/{match_id}_presing_intensity.pkl", "wb") as f:
                pickle.dump(model.output, f)



if __name__=="__main__":
    source = 'bepro'
    if source == 'dfl-spoho':
        coordinates = "secondspectrum"
        match_id_lst = [x.split("-")[-1] for x in os.listdir("/data/MHL/dfl-spoho/raw/")]
        os.makedirs("/data/MHL/pressing-intensity", exist_ok=True)
        print('test')
        for match_id in match_id_lst:
            print(match_id)
            kloppy_dataset = sportec.load_open_tracking_data(
                match_id=match_id, coordinates=coordinates
            )

            dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset, orient_ball_owning=False)
            home_team, away_team = kloppy_dataset.metadata.teams
            model = CustomPressingIntensity(dataset=dataset)
            model.fit(
                method="teams",
                ball_method="max",
                orient="home_away",
                speed_threshold=2.0,
            )

            # with open(f"/data/MHL/pressing-intensity/{match_id}.pkl", "wb") as f:
            #     pickle.dump(model.output, f)
            break
    elif source == 'bepro':
        load_bepro("/data/MHL/bepro/processed")
    print("Done")
