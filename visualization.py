from kloppy import sportec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
import pandas as pd


def plot_single_frame_positions(total_df, period_id, frame_idx, pitch, home_team_info, away_team_info):
    """
    Plots all x/y positions for a single frame and annotates each player with their jersey number.
    The legend shows team names.
    
    Parameters:
        df (DataFrame): Positional data (Pandas or Polars converted to Pandas if needed)
        frame_idx (int): The target frame index.
        pitch: An mplsoccer Pitch object.
        home_team_info: Team object for the home team.
        away_team_info: Team object for the away team.
    """
    # 기본 팀 색상
    team_colors = {"Home": "blue", "Away": "red", "Ball": "black"}

    df = total_df[(total_df['period_id']==period_id) & (total_df['frame_id']==frame_idx)]
    # x/y 컬럼 검출 (컬럼 이름은 "pid_x" 또는 "pid_y" 형태)
    player_ids = df['id'].unique()
    
    # 팀 정보 딕셔너리 생성: key=player_id, value=[player.name, player.jersey_no]
    home_team_dict = {player.player_id: [player.name, player.jersey_no] for player in home_team_info.players}
    away_team_dict = {player.player_id: [player.name, player.jersey_no] for player in away_team_info.players}
    
    # 피치 그리기
    fig, ax = pitch.draw()
    ax.set_title(f"Frame {frame_idx} Player Positions", fontsize=14)
    
    # 각 선수 위치 플로팅 및 등번호 표시
    for pid in player_ids:
        x = df[df['id']==pid]["x"].iloc[0]
        y = df[df['id']==pid]["y"].iloc[0]
        if pd.isna(x) or pd.isna(y):
            continue  # 결측치 스킵
        
        if pid in home_team_dict:
            color = team_colors["Home"]
            jersey_no = home_team_dict[pid][1]
        elif pid in away_team_dict:
            color = team_colors["Away"]
            jersey_no = away_team_dict[pid][1]
        elif "ball" in pid.lower():
            color = team_colors["Ball"]
            jersey_no = ""  # 공은 등번호가 없으므로 빈 문자열
        else:
            color = "gray"
            jersey_no = ""
        
        # 선수 위치 scatter plot
        ax.scatter(x, y, color=color, s=100, alpha=0.8)
        # 선수 등번호 텍스트 추가 (중앙 정렬, 흰색 텍스트)
        if jersey_no != "":
            ax.text(x, y, str(jersey_no), fontsize=7, fontweight="bold", color="white",
                    ha="center", va="center")
    
    # 범례 생성: 홈팀, 어웨이팀, 볼 (필요한 경우)
    home_patch = mpatches.Patch(color=team_colors["Home"], label=home_team_info.name)
    away_patch = mpatches.Patch(color=team_colors["Away"], label=away_team_info.name)
    ball_patch = mpatches.Patch(color=team_colors["Ball"], label="Ball")
    ax.legend(handles=[home_patch, away_patch, ball_patch], loc='upper right')
    
    plt.show()

def plot_window_frame_positions(total_df, period_id, start_frame_idx, end_frame_idx, pitch, home_team_info, away_team_info):
    """
    Plots all x/y positions for a single frame and annotates each player with their jersey number.
    The legend shows team names.
    
    Parameters:
        df (DataFrame): Positional data (Pandas or Polars converted to Pandas if needed)
        frame_idx (int): The target frame index.
        pitch: An mplsoccer Pitch object.
        home_team_info: Team object for the home team.
        away_team_info: Team object for the away team.
    """
    # 기본 팀 색상
    team_colors = {"Home": "blue", "Away": "red", "Ball": "black"}

    window_df = total_df[(total_df['period_id'] == period_id) &
                  (total_df['frame_id'] >= start_frame_idx) &
                  (total_df['frame_id'] <= end_frame_idx)
                ]
    window_df.sort_values("frame_id", inplace=True)
    if window_df.empty:
        print("No frames found for the specified time window.")
        return
    
    # x/y 컬럼 검출 (컬럼 이름은 "pid_x" 또는 "pid_y" 형태)
    player_ids = window_df['id'].unique()
    
    # 팀 정보 딕셔너리 생성: key=player_id, value=[player.name, player.jersey_no]
    home_team_dict = {player.player_id: [player.name, player.jersey_no] for player in home_team_info.players}
    away_team_dict = {player.player_id: [player.name, player.jersey_no] for player in away_team_info.players}
    
    # 피치 그리기
    fig, ax = pitch.draw()
    ax.set_title(f"Frame {start_frame_idx} to Frame {end_frame_idx} Player Positions", fontsize=14)
        
    # 각 선수 위치 플로팅 및 등번호 표시
    for pid in player_ids:
        traj_x = window_df[window_df['id']==pid]["x"].values
        traj_y = window_df[window_df['id']==pid]["y"].values
        # if pd.isna(traj_x) or pd.isna(traj_y):
        #     continue  # 결측치 스킵
        
        if pid in home_team_dict:
            color = team_colors["Home"]
            jersey_no = home_team_dict[pid][1]
        elif pid in away_team_dict:
            color = team_colors["Away"]
            jersey_no = away_team_dict[pid][1]
        elif "ball" in pid.lower():
            color = team_colors["Ball"]
            jersey_no = ""  # 공은 등번호가 없으므로 빈 문자열
        else:
            color = "gray"
            jersey_no = ""
        
        # Plot trajectory
        ax.plot(traj_x, traj_y, color=color, linestyle='-', linewidth=2, alpha=0.5)
        
        # 마지막 프레임(window_df의 마지막 행)에서 현재 위치 표시
        current_x = traj_x[-1]
        current_y = traj_y[-1]
        # 선수 위치 scatter plot
        ax.scatter(current_x, current_y, color=color, s=100, alpha=0.8)
        # 선수 등번호 텍스트 추가 (중앙 정렬, 흰색 텍스트)
        if jersey_no != "":
            ax.text(current_x, current_y, str(jersey_no), fontsize=7, fontweight="bold", color="white",
                    ha="center", va="center")
    
    # 범례 생성: 홈팀, 어웨이팀, 볼 (필요한 경우)
    home_patch = mpatches.Patch(color=team_colors["Home"], label=home_team_info.name)
    away_patch = mpatches.Patch(color=team_colors["Away"], label=away_team_info.name)
    ball_patch = mpatches.Patch(color=team_colors["Ball"], label="Ball")
    ax.legend(handles=[home_patch, away_patch, ball_patch], loc='upper right')
    
    plt.show()
