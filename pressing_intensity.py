import numpy as np
from kloppy import sportec
from unravel.soccer import KloppyPolarsDataset, PressingIntensity
from typing import Literal, List, Union, Optional, Dict, Any
import os 
import polars as pl
import pickle
import pandas as pd
import argparse

from dataclasses import dataclass
import config as C
from config import Constant, Column, Group

@dataclass
class DefaultSettings:
    """Configuration settings for pressing intensity calculations.
    
    This dataclass contains all the necessary parameters for calculating pressing
    intensity metrics, including team identifiers and various physical thresholds
    for player and ball movement.
    
    Attributes:
        home_team_id: Identifier for the home team.
        away_team_id: Identifier for the away team.
        max_player_speed: Maximum allowed player speed in m/s.
        max_ball_speed: Maximum allowed ball speed in m/s.
        max_player_acceleration: Maximum allowed player acceleration in m/s².
        max_ball_acceleration: Maximum allowed ball acceleration in m/s².
        ball_carrier_threshold: Distance threshold for determining ball carrier in meters.
    """
    home_team_id: Union[str, int]
    away_team_id: Union[str, int]
    max_player_speed: float = 12.0
    max_ball_speed: float = 28.0
    max_player_acceleration: float = 6.0
    max_ball_acceleration: float = 13.5
    ball_carrier_threshold: float = 25.0

class PressingIntensityDataset(KloppyPolarsDataset):
    """Custom dataset class for pressing intensity calculations.
    
    This class extends KloppyPolarsDataset to provide a specialized interface
    for pressing intensity analysis with custom settings and data handling.
    
    Attributes:
        data: Polars DataFrame containing tracking data.
        settings: DefaultSettings object containing calculation parameters.
    """
    
    def __init__(self, tracking_df: pd.DataFrame, settings: DefaultSettings) -> None:
        """Initialize the PressingIntensityDataset.
        
        Args:
            tracking_df: Pandas DataFrame containing tracking data.
            settings: DefaultSettings object with calculation parameters.
        """
        self.data = pl.from_pandas(tracking_df)
        self.settings = settings


class CustomPressingIntensity(PressingIntensity):
    """Custom implementation of pressing intensity calculations.
    
    This class extends the base PressingIntensity class to provide customized
    pressing intensity calculations with enhanced error handling and flexible
    parameter configurations.
    
    Attributes:
        output: DataFrame containing the calculated pressing intensity results.
    """
    
    @property
    def __exprs_variables(self) -> List[str]:
        """Get the list of expression variables used in calculations.
        
        Returns:
            List of column names required for pressing intensity calculations.
        """
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

    def __repr__(self) -> str:
        """String representation of the CustomPressingIntensity object.
        
        Returns:
            String containing the number of frames processed.
        """
        n_frames = (
            self.output[Column.FRAME_ID].nunique() if hasattr(self, "output") else None
        )
        return f"PressingIntensity(n_frames={n_frames})"
    
    def fit(
        self,
        speed_threshold: Optional[float] = None,
        reaction_time: float = 0.7,
        time_threshold: float = 1.5,
        sigma: float = 0.45,
        method: Literal["teams", "full"] = "teams",
        ball_method: Literal["include", "exclude", "max"] = "max",
        orient: Literal[
            "ball_owning", "pressing", "home_away", "away_home"
        ] = "ball_owning",
    ) -> "CustomPressingIntensity":
        """Fit the pressing intensity model with the given parameters.
        
        This method calculates pressing intensity for each frame in the dataset
        using the specified parameters and methods. It processes data period by
        period and frame by frame to generate comprehensive pressing intensity metrics.
        
        Args:
            speed_threshold: Masks pressing intensity to only include players travelling 
                           above a certain speed threshold in meters per second.
                           If None, no speed filtering is applied.
            reaction_time: Time delay for player reaction in seconds.
            time_threshold: Maximum time window for pressing intensity calculation.
            sigma: Standard deviation parameter for Gaussian smoothing.
            method: Matrix creation method.
                   - "teams": Creates a 11x11 matrix
                   - "full": Creates a 22x22 matrix
            ball_method: Method for handling ball in calculations.
                        - "include": Creates a 11x12 matrix
                        - "exclude": Ignores ball completely
                        - "max": Keeps 11x11 but ball carrier pressing intensity 
                                is max(ball, ball_carrier)
            orient: Orientation of the pressing intensity output.
                   - "ball_owning": From ball owning team perspective
                   - "pressing": From pressing team perspective
                   - "home_away": Home team first, then away team
                   - "away_home": Away team first, then home team
                   
        Returns:
            Self reference for method chaining.
            
        Raises:
            ValueError: If method, ball_method, or orient parameters are invalid.
            TypeError: If numeric parameters are not of correct type.
            
        Example:
            >>> model = CustomPressingIntensity(dataset)
            >>> model.fit(method="teams", ball_method="max", orient="home_away")
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

def load_dfl_spoho(data_path: str) -> None:
    """Load and process DFL-SPOHO data for pressing intensity calculations.
    
    This function processes DFL-SPOHO tracking data using the kloppy library,
    calculates pressing intensity for each match, and saves the results as
    pickle files. It handles the complete pipeline from raw data to processed
    pressing intensity metrics.
    
    Args:
        data_path: Path to the processed data directory where results will be saved.
        
    Example:
        >>> load_dfl_spoho("/path/to/dfl-spoho/processed")
    """
    
    coordinates = "secondspectrum"
    match_id_lst = os.listdir("/data/MHL/dfl-spoho/raw")
    total_dict = {match_id : {} for match_id in match_id_lst}
    
    for match_id in match_id_lst:
        os.makedirs(f"{data_path}/{match_id}", exist_ok=True)
        kloppy_dataset = sportec.load_open_tracking_data(
                match_id=match_id.split("-")[-1], coordinates=coordinates
            )
        orient = kloppy_dataset.metadata.orientation.value
        orient = orient.replace("-", "_")
        dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset, orient_ball_owning=False)
        model = CustomPressingIntensity(dataset=dataset)
        print(f"Calculate Pressing Intensity {match_id}")
        model.fit(
            method="teams",
            ball_method="max",
            orient=orient,
            speed_threshold=2.0,
        )
        
        with open(f"{data_path}/{match_id}/{match_id}_pressing_intensity.pkl", "wb") as f:
            pickle.dump(model.output, f)

def load_dfl_confidential(data_path: str) -> None:
    """Load and process DFL confidential data for pressing intensity calculations.
    
    This function processes pre-processed DFL confidential tracking data,
    calculates pressing intensity for each match, and saves the results as
    pickle files. It handles orientation detection and team identification
    automatically from the tracking data.
    
    Args:
        data_path: Path to the processed data directory containing match data.
        
    Example:
        >>> load_dfl_confidential("/path/to/dfl-confidential/processed")
    """
    match_id_lst = os.listdir(data_path)
    total_dict = {match_id : {} for match_id in match_id_lst}

    for match_id in match_id_lst:
        if os.path.exists(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl")):
            with open(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl"), "rb") as f:
                total_dict[match_id] = pickle.load(f)
            if total_dict[match_id]['tracking_df'] is None: #'DFL-MAT-J03YIY'
                print(f"Processed tracking data {match_id} not exists.")
                continue
        else:
            print(f"Processed data {match_id} not exists.")
        
        if total_dict[match_id]['tracking_df'] is None:
            continue
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
        print(f"Calculate Pressing Intensity {match_id}")
        model.fit(
            method="teams",
            ball_method="max",
            orient=orient,
            speed_threshold=2.0,
        )
        with open(f"{data_path}/{match_id}/{match_id}_pressing_intensity.pkl", "wb") as f:
            pickle.dump(model.output, f)



def load_bepro(data_path: str) -> None:
    """Load and process BePro data for pressing intensity calculations.
    
    This function processes pre-processed BePro tracking data, calculates
    pressing intensity for each match, and saves the results as pickle files.
    It handles orientation detection and team identification automatically
    from the tracking data.
    
    Args:
        data_path: Path to the processed data directory containing match data.
        
    Example:
        >>> load_bepro("/path/to/bepro/processed")
    """
    match_id_lst = os.listdir(data_path)
    total_dict = {match_id : {} for match_id in match_id_lst}

    for match_id in match_id_lst:
        # if match_id not in ["126424", "126433", "126444", "126458", "126466", "126473", "153373", "153385", "153387"]: continue
        if os.path.exists(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl")):
            with open(os.path.join(data_path, match_id, f"{match_id}_processed_dict.pkl"), "rb") as f:
                total_dict[match_id] = pickle.load(f)
            if total_dict[match_id]['tracking_df'] is None:
                print(f"Processed tracking data {match_id} not exists.")
                continue
        else:
            print(f"Processed data {match_id} not exists.")
        
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
        print(f"Calculate Pressing Intensity {match_id}")
        model.fit(
            method="teams",
            ball_method="max",
            orient=orient,
            speed_threshold=2.0,
        )
        with open(f"{data_path}/{match_id}/{match_id}_pressing_intensity.pkl", "wb") as f:
            pickle.dump(model.output, f)



def main() -> None:
    """Main function to calculate pressing intensity for different data sources.
    
    This function provides a command-line interface for processing different
    data sources (BePro, DFL-SPOHO, DFL-Confidential) and calculating
    pressing intensity metrics for each match in the dataset.
    
    Example:
        >>> python pressing_intensity.py --source bepro --data_path /path/to/bepro/processed
        >>> python pressing_intensity.py --source dfl-spoho --data_path /path/to/dfl-spoho/processed
        >>> python pressing_intensity.py --source dfl-confidential --data_path /path/to/dfl-confidential/processed
    """
    parser = argparse.ArgumentParser(description="Calculate pressing intensity for different data sources.")
    parser.add_argument("--source", type=str, default="bepro", 
                       choices=["bepro", "dfl-spoho", "dfl-spoho-local", "dfl-confidential"],
                       help="Data source to process (default: bepro)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the data directory")
    
    args = parser.parse_args()
    
    if args.source == 'dfl-spoho':
        load_dfl_spoho(args.data_path)
    elif args.source in ['dfl-spoho-local', 'dfl-confidential']:
        load_dfl_confidential(args.data_path)
    elif args.source == 'bepro':
        load_bepro(args.data_path)
    else:
        print(f"Unknown source: {args.source}")
        print("Available sources: bepro, dfl-spoho, dfl-spoho-local, dfl-confidential")
    
    print("Done")


if __name__=="__main__":
    main()
