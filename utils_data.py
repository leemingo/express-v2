import numpy as np
import pandas as pd

import config as C
from config import Constant, Column, Group


def infer_ball_carrier(tracking_df, source='bepro'):
    # --- Helper Functions with corrected indentation and English comments ---
    def _determine_ball_owning_team_for_frame(frame_df, threshold, col_team_id, col_botid):
        # Check for existing BOTID in the frame
        current_botids = frame_df[col_botid].dropna()
        if not current_botids.empty and current_botids.iloc[0] != 'neutral':
            return current_botids.iloc[0]  # Use the first valid value
        
        # If existing BOTID is 'neutral' or absent, infer
        # Check if 'ball_dist' column has any valid (non-NaN) values
        if 'ball_dist' in frame_df.columns and frame_df['ball_dist'].notna().any():
            # Exclude NaN values and find the minimum value and its index
            valid_ball_dist_series = frame_df['ball_dist'].dropna()
            # Since .notna().any() was checked above, valid_ball_dist_series should not be empty
            # (if all values were NaN, .any() would be False)
            # However, it might be safer to double-check for emptiness after dropna()
            if not valid_ball_dist_series.empty:
                min_dist_idx = valid_ball_dist_series.idxmin()  # Index from the original frame_df
                min_dist = valid_ball_dist_series.min()

                if min_dist < threshold:
                    return frame_df.loc[min_dist_idx, col_team_id]
        return pd.NA  # or np.nan

    def _determine_ball_owning_player_for_frame(frame_df, threshold, col_object_id, col_team_id, col_bopid):
        # This function assumes that the frame's BALL_OWNING_TEAM_ID has already been determined
        # and is present in the 'final_botid_for_frame' column.
        frame_botid_value = frame_df['final_botid_for_frame'].iloc[0]  # Value applied uniformly to all rows within the frame.

        # Check for existing BOPID in the frame (among players of the determined owning team).
        if pd.notna(frame_botid_value):
            # If 'neutral', check for existing BOPID based on all players.
            if frame_botid_value == 'neutral':
                players_considered_for_existing_bopid = frame_df.copy()
            else:
                players_considered_for_existing_bopid = frame_df[frame_df[col_team_id] == frame_botid_value]
            
            if not players_considered_for_existing_bopid.empty: # Check if there are any players to consider.
                current_bopids_on_team = players_considered_for_existing_bopid[col_bopid].dropna()
                # Check if the existing BOPID is a valid value (not 'neutral').
                if not current_bopids_on_team.empty and current_bopids_on_team.iloc[0] != 'neutral': 
                    return current_bopids_on_team.iloc[0]
        
        # If no valid existing BOPID, or if the owning team is unknown (NA), try to infer (owning team must be known for inference).
        if pd.isna(frame_botid_value): # If owning team is NA, player inference is not possible.
            return pd.NA

        # Infer player based on the owning team.
        if frame_botid_value == 'neutral': # If 'neutral', calculate ball distance based on all players.
            players_to_search_in = frame_df.copy()
        else:
            players_to_search_in = frame_df[frame_df[col_team_id] == frame_botid_value]
        
        if not players_to_search_in.empty and \
           'ball_dist' in players_to_search_in.columns and \
           players_to_search_in['ball_dist'].notna().any():
            
            valid_ball_dist_series_on_team = players_to_search_in['ball_dist'].dropna()
            if not valid_ball_dist_series_on_team.empty:
                min_dist_on_team_idx = valid_ball_dist_series_on_team.idxmin()
                min_dist_on_team = valid_ball_dist_series_on_team.min()
                if min_dist_on_team < threshold:
                    return players_to_search_in.loc[min_dist_on_team_idx, col_object_id]
        
        return pd.NA

    # --- Main function logic with corrected indentation and English comments ---
    # 0. Initialize BALL_OWNING_PLAYER_ID and BALL_OWNING_TEAM_ID if they don't exist
    obj_id_dtype = tracking_df[Column.OBJECT_ID].dtype if Column.OBJECT_ID in tracking_df.columns else 'object'
    team_id_dtype = tracking_df[Column.TEAM_ID].dtype if Column.TEAM_ID in tracking_df.columns else 'object'

    if Column.BALL_OWNING_PLAYER_ID not in tracking_df.columns:
        tracking_df[Column.BALL_OWNING_PLAYER_ID] = pd.Series(dtype=obj_id_dtype, index=tracking_df.index)
    if Column.BALL_OWNING_TEAM_ID not in tracking_df.columns:
        tracking_df[Column.BALL_OWNING_TEAM_ID] = pd.Series(dtype=team_id_dtype, index=tracking_df.index)

    # 1. Separate ball and player data
    # Since the 'id' column is used, either Column.OBJECT_ID should point to 'id',
    # or 'id' should be used directly instead of Column.OBJECT_ID.
    # Here, 'id' is used as per the provided code.
    ball_df = tracking_df[tracking_df['id'] == 'ball'].copy() # Added .copy()
    players_df = tracking_df[tracking_df['id'] != 'ball'].copy() # Added .copy()

    # Defensive code for cases where data is missing
    if ball_df.empty or players_df.empty:
        tracking_df[Column.IS_BALL_CARRIER] = False
        # If BALL_OWNING_TEAM_ID doesn't exist, dropna might remove all rows, so check if the column exists.
        if Column.BALL_OWNING_TEAM_ID in tracking_df.columns:
            tracking_df = tracking_df.dropna(subset=[Column.BALL_OWNING_TEAM_ID])
        return tracking_df.reset_index(drop=True)


    # 2. Prepare ball positions: one row per frame, with ball's x, y, z
    # For bepro, just using x, y
    ball_pos_per_frame = ball_df.groupby(Group.BY_FRAME, as_index=False).first()  # Assumes one ball entry per frame
    ball_pos_per_frame = ball_pos_per_frame[Group.BY_FRAME + [Column.X, Column.Y]].rename(
        columns={Column.X: "ball_x", Column.Y: "ball_y"}
    )

    # Merge ball positions to player data
    players_df_with_ball_pos = pd.merge(players_df, ball_pos_per_frame, on=Group.BY_FRAME, how="left")

    # 3. Calculate distance to ball for each player
    # Ensure coordinates are numeric and handle potential NaNs (especially for Z)
    if source == 'bepro':
        coord_cols_to_numeric = ["ball_x", "ball_y", Column.X, Column.Y]

        for col in coord_cols_to_numeric:
            if col in players_df_with_ball_pos.columns: # Check if column exists
                players_df_with_ball_pos[col] = pd.to_numeric(players_df_with_ball_pos[col], errors='coerce')
            else: # To handle cases where columns might not be created due to merge, etc.
                players_df_with_ball_pos[col] = pd.NA 

        # If ball_x, ball_y are NA (ball position for the frame was not available in ball_pos_per_frame), distance calculation is not possible.
        # In this case, dist_sq will be NA, and np.sqrt(NA) will also be NA.
        dist_sq = (
            (players_df_with_ball_pos[Column.X] - players_df_with_ball_pos["ball_x"]) ** 2 +
            (players_df_with_ball_pos[Column.Y] - players_df_with_ball_pos["ball_y"]) ** 2
        )
    elif source == 'sportec':  # Using Z
        coord_cols_to_numeric = ["ball_x", "ball_y", Column.X, Column.Y]
        if Column.Z in players_df_with_ball_pos.columns: coord_cols_to_numeric.append(Column.Z)
        if "ball_z" in players_df_with_ball_pos.columns: coord_cols_to_numeric.append("ball_z")

        for col in coord_cols_to_numeric:
            if col in players_df_with_ball_pos.columns:
                players_df_with_ball_pos[col] = pd.to_numeric(players_df_with_ball_pos[col], errors='coerce')
            else:
                players_df_with_ball_pos[col] = pd.NA


        # Fill Z with 0.0 if missing or NaN (common for 2D data or if ball Z isn't always present)
        for col_z in [Column.Z, "ball_z"]:
            if col_z not in players_df_with_ball_pos.columns: # If Z column itself doesn't exist, create it with 0
                players_df_with_ball_pos[col_z] = 0.0
            players_df_with_ball_pos[col_z] = players_df_with_ball_pos[col_z].fillna(0.0)

        dist_sq = (
            (players_df_with_ball_pos[Column.X] - players_df_with_ball_pos["ball_x"]) ** 2 +
            (players_df_with_ball_pos[Column.Y] - players_df_with_ball_pos["ball_y"]) ** 2 +
            (players_df_with_ball_pos[Column.Z] - players_df_with_ball_pos["ball_z"]) ** 2
        )
    else: # Unknown source
        raise ValueError(f"Unknown source: {source}. Must be 'bepro' or 'sportec'.")

    players_df_with_ball_pos["ball_dist"] = np.sqrt(dist_sq)

    # 4. Determine BALL_OWNING_TEAM_ID per frame (Use original column)
    # BALL_CARRIER_THRESHOLD must be defined within the scope of this function.
    # If it's a class member, it would be self.BALL_CARRIER_THRESHOLD or self._ball_carrier_threshold.
    # Here, it's assumed to be a local or global variable.
    botid_series = players_df_with_ball_pos.groupby(Group.BY_FRAME, group_keys=True).apply(
        _determine_ball_owning_team_for_frame,
        threshold=C.BALL_CARRIER_THRESHOLD, 
        col_team_id=Column.TEAM_ID,
        col_botid=Column.BALL_OWNING_TEAM_ID # Use original BOTID column
    )

    # Ensure botid_series has a name for merging if it's not empty
    if not botid_series.empty:
        botid_series = botid_series.rename("final_botid_for_frame")
        # Merge determined BOTID back to player data for next step
        players_df_with_ball_pos = pd.merge(
            players_df_with_ball_pos,
            botid_series,
            on=Group.BY_FRAME,
            how="left"
        )
    else: # No player data or groups, create column with NaNs
        players_df_with_ball_pos["final_botid_for_frame"] = pd.NA
        
    # 5. Determine BALL_OWNING_PLAYER_ID per frame
    bopid_series = players_df_with_ball_pos.groupby(Group.BY_FRAME, group_keys=True).apply(
        _determine_ball_owning_player_for_frame,
        threshold=C.BALL_CARRIER_THRESHOLD,
        col_object_id=Column.OBJECT_ID,
        col_team_id=Column.TEAM_ID,
        col_bopid=Column.BALL_OWNING_PLAYER_ID # Use original BOPID column
    )

    if not bopid_series.empty:
        bopid_series = bopid_series.rename("final_bopid_for_frame")
    
    # 6. Consolidate frame-level inferences (BOTID, BOPID)
    frame_summary_components = []
    # Add to frame_summary_components only if botid_series and bopid_series are not None and are non-empty Series.
    if isinstance(botid_series, pd.Series) and not botid_series.empty:
        frame_summary_components.append(botid_series)
    if isinstance(bopid_series, pd.Series) and not bopid_series.empty:
        frame_summary_components.append(bopid_series)

    if not frame_summary_components: # If players_df was empty or grouping resulted in no groups
        # Create an empty frame_summary including Group.BY_FRAME columns
        frame_summary = pd.DataFrame(columns=Group.BY_FRAME + [Column.BALL_OWNING_TEAM_ID, Column.BALL_OWNING_PLAYER_ID])
    else:
        frame_summary = pd.concat(frame_summary_components, axis=1).reset_index()
        rename_map = {}
        if "final_botid_for_frame" in frame_summary.columns:
            rename_map["final_botid_for_frame"] = Column.BALL_OWNING_TEAM_ID
        if "final_bopid_for_frame" in frame_summary.columns:
            rename_map["final_bopid_for_frame"] = Column.BALL_OWNING_PLAYER_ID
        if rename_map:
            frame_summary = frame_summary.rename(columns=rename_map)

    # 7. Merge frame summary back to the original full DataFrame (tracking_df)
    # Original tracking_df's BALL_OWNING_PLAYER_ID is dropped, and BALL_OWNING_TEAM_ID is backed up.
    output_df = tracking_df.drop(columns=[Column.BALL_OWNING_PLAYER_ID], errors='ignore')
    # Check if the column exists before prefixing with 'ori_'
    if Column.BALL_OWNING_TEAM_ID in output_df.columns:
        output_df = output_df.rename(columns={Column.BALL_OWNING_TEAM_ID: "ori_" + Column.BALL_OWNING_TEAM_ID})
    
    # Check if Group.BY_FRAME key columns exist in frame_summary and merge
    # (If frame_summary is empty but has columns, merge will fill with NA)
    # (Defensive code for when frame_summary is completely empty or key columns are missing)
    all_keys_present_in_summary = all(key in frame_summary.columns for key in Group.BY_FRAME)
    
    if not frame_summary.empty and all_keys_present_in_summary:
        output_df = pd.merge(output_df, frame_summary, on=Group.BY_FRAME, how="left")
    else: # If frame_summary is unsuitable for merge, fill target columns with NA
        if Column.BALL_OWNING_TEAM_ID not in output_df.columns:
            output_df[Column.BALL_OWNING_TEAM_ID] = pd.NA
        if Column.BALL_OWNING_PLAYER_ID not in output_df.columns:
            output_df[Column.BALL_OWNING_PLAYER_ID] = pd.NA

    # 8. Set IS_BALL_CARRIER column
    # True if OBJECT_ID matches BALL_OWNING_PLAYER_ID and BOPID is not NA
    # Exclude 'neutral' state
    output_df[Column.IS_BALL_CARRIER] = \
        (output_df[Column.OBJECT_ID] == output_df[Column.BALL_OWNING_PLAYER_ID]) & \
        (output_df[Column.BALL_OWNING_PLAYER_ID].notna()) & \
        (output_df[Column.BALL_OWNING_PLAYER_ID] != 'neutral') 


    # 9. Drop rows where the (newly determined) BALL_OWNING_TEAM_ID is NA
    # If you want to keep 'neutral' values, fill with 'neutral' before dropna,
    # and then either don't dropna or modify the condition.
    # Currently, only NA is removed. If 'neutral' also needs to be removed, an additional condition is needed.
    if Column.BALL_OWNING_TEAM_ID in output_df.columns:
        output_df = output_df.dropna(subset=[Column.BALL_OWNING_TEAM_ID])
    
    # Part that finally deletes the BALL_OWNING_PLAYER_ID column (was in previous code).
    # If you want to keep this column, remove/comment out this line.
    # Actual column name needs verification. Using Column.BALL_OWNING_PLAYER_ID is recommended.
    # Assuming Column.BALL_OWNING_PLAYER_ID is the string "ball_owning_player_id" for this specific line from original code
    if 'ball_owning_player_id' in output_df.columns: 
         output_df = output_df.drop(columns=['ball_owning_player_id']) 

    return output_df.reset_index(drop=True) # Reset index
