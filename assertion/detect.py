import os
import sys
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from env import PROJECT_DIR
error_path = Path(PROJECT_DIR / "assertion" / "error_data" / "bepro")
error_path = Path(PROJECT_DIR / "assertion_statsbomb" / "error_data" / "statsbomb")

from datatools.data.bepro import BeproLoader
import datatools.lsdp as lsdp
from assertion.validator import Validator

os.makedirs(error_path, exist_ok=True)

provider = "bepro"
Bepro = BeproLoader()
competitions = Bepro.competitions()
games = pd.concat([
    Bepro.games(row.competition_id, row.season_id)
    for row in competitions.itertuples()
]).reset_index(drop=True)

total_data = 0
total_error = 0

games = games.iloc[len(games)//4:]
games_verbose = tqdm(list(games.itertuples()), desc="Loading game data")
for game in games_verbose:  
    events = Bepro.events(game.game_id)
    sequences = Bepro.sequences(game.game_id)

    actions= lsdp.bepro.convert_to_actions(
        events, 
        home_team_id=game.home_team_id, 
        sequences=sequences
    )

    # Validate sequence
    yaml_file = Path(PROJECT_DIR / "assertion" / "transitions.yaml")
    validator = Validator(actions, yaml_file=yaml_file)
    
    validator.validate_sequence()

    print(f"{game.game_id}: {len(validator.df_errors)}-------------------")
    total_error += len(validator.df_errors)
    total_data += len(validator.df_events)

    # Save df_errors to file
    validator.df_errors.to_csv(Path(error_path / f"{game.game_id}.csv"), index=False)  
    
print(f"Total Error: {total_error}")
print(f"Total Data: {total_data}")
