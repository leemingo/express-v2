import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from mplsoccer import Pitch

import config as C

def plot_single_frame_positions(total_df, period_id, frame_idx, home_team_info, away_team_info):
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
    # Default team colors
    # team_colors = {"Home": "blue", "Away": "red", "Ball": "black"}
    # team_colors = {"Home": "red", "Away": "black", "Ball": "orange"}
    styles = {
        "Home": {"facecolor": "red",    "edgecolor": "black"},
        "Away": {"facecolor": "orange", "edgecolor": "red"},
        "Ball": {"facecolor": "blue", "edgecolor": "black"}
    }

    df = total_df[(total_df['period_id']==period_id) & (total_df['frame_id']==frame_idx)].copy()
    # Detect player IDs in the frame
    player_ids = df['id'].unique()
    
    home_map = {r.pID: r.jID for _, r in home_team_info.iterrows()}
    away_map = {r.pID: r.jID for _, r in away_team_info.iterrows()}
    
    # Create team info dictionaries: key=player_id, value=[player.name, player.jersey_no]
    home_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in home_team_info.iterrows()}
    away_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in away_team_info.iterrows()}
    
    # Draw the pitch
    # Set up pitch using kloppy_dataset metadata
    pitch_length = C.PITCH_X_MAX - C.PITCH_X_MIN
    pitch_width = C.PITCH_Y_MAX - C.PITCH_Y_MIN
    
    pitch = Pitch(
        pitch_type='secondspectrum', 
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color='white', 
        line_color='gray'
    )

    fig, ax = pitch.draw()
    ax.set_title(f"Frame {frame_idx} Player Positions", fontsize=14)
    nodes = []
    # Plot each player's position and jersey number
    for pid in player_ids:
        x = df[df['id']==pid]["x"].iloc[0]
        y = df[df['id']==pid]["y"].iloc[0]
        if pd.isna(x) or pd.isna(y):
            continue  
        
        if pid in home_team_dict:
            # color = team_colors["Home"]
            # jersey_no = home_team_dict[pid][1]
            cat = "Home"; jersey = home_map[pid]
        elif pid in away_team_dict:
            # color = team_colors["Away"]
            # jersey_no = away_team_dict[pid][1]
             cat = "Away"; jersey = away_map[pid]
        elif "ball" in pid.lower():
            # color = team_colors["Ball"]
            # jersey_no = "" 
             cat = "Ball"; jersey = ""
        else:
            # color = "gray"
            # jersey_no = ""
            cat = None; jersey = ""
         
        style = styles.get(cat, {"facecolor":"gray","edgecolor":"black"})
        #
        #  Scatter plot for player position
        # ax.scatter(x, y, color=color, s=100, alpha=0.8)
        ax.scatter(x, y, facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            s=100, alpha=0.8)
        # if jersey_no != "":
            # ax.text(x, y, str(jersey_no), fontsize=7, fontweight="bold", color="white",
            #         ha="center", va="center")
        if jersey:
            ax.text(x, y, str(jersey), fontsize=7, fontweight="bold", color="white",
                    ha="center", va="center")
        nodes.append((x, y))
    for (x1, y1), (x2, y2) in itertools.combinations(nodes, 2):
        ax.plot(
            [x1, x2], [y1, y2],
            color='gray', linewidth=0.5, alpha=0.6, zorder=1
        )    

    # Create legend: Home team, Away team, Ball (if necessary)
    # home_patch = mpatches.Patch(color=team_colors["Home"], label='Home')
    # away_patch = mpatches.Patch(color=team_colors["Away"], label='Away')
    # ball_patch = mpatches.Patch(color=team_colors["Ball"], label="Ball")
    # ax.legend(handles=[home_patch, away_patch, ball_patch], loc='upper right')
    
    plt.show()

def plot_window_frame_positions(total_df, period_id, start_frame_idx, end_frame_idx, home_team_info, away_team_info):
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
    # Default team colors
    team_colors = {"Home": "blue", "Away": "red", "Ball": "black"}

    window_df = total_df[(total_df['period_id'] == period_id) &
                  (total_df['frame_id'] >= start_frame_idx) &
                  (total_df['frame_id'] <= end_frame_idx)
                ].copy()
    if window_df.empty:
        print("No frames found for the specified time window.")
        return
    window_df.sort_values("frame_id", inplace=True)
    
    # Detect player IDs in the window
    player_ids = window_df['id'].unique()
    
    # Create team info dictionaries: key=player_id, value=[player.name, player.jersey_no]
    home_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in home_team_info.iterrows()}
    away_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in away_team_info.iterrows()}
    
    # Draw the pitch
    pitch_length = C.PITCH_X_MAX - C.PITCH_X_MIN
    pitch_width = C.PITCH_Y_MAX - C.PITCH_Y_MIN
    
    pitch = Pitch(
        pitch_type='secondspectrum', 
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color='white', 
        line_color='gray'
    )

    fig, ax = pitch.draw()
    ax.set_title(f"Frame {start_frame_idx} to Frame {end_frame_idx} Player Positions", fontsize=14)
        
    # Plot each player's position and jersey number
    for pid in player_ids:
        traj_x = window_df[window_df['id']==pid]["x"].values
        traj_y = window_df[window_df['id']==pid]["y"].values
        # if pd.isna(traj_x) or pd.isna(traj_y):
        #     continue  
        
        if pid in home_team_dict:
            color = team_colors["Home"]
            jersey_no = home_team_dict[pid][1]
        elif pid in away_team_dict:
            color = team_colors["Away"]
            jersey_no = away_team_dict[pid][1]
        elif "ball" in pid.lower():
            color = team_colors["Ball"]
            jersey_no = ""  
        else:
            color = "gray"
            jersey_no = ""
        
        # Plot trajectory
        ax.plot(traj_x, traj_y, color=color, linestyle='-', linewidth=2, alpha=0.5)

        # Plot direction
        last_frame = window_df[(window_df['id'] == pid) & (window_df['frame_id'] ==end_frame_idx)]
        current_x = last_frame['x'].iloc[0]
        current_y = last_frame['y'].iloc[0]
        current_vx = last_frame['vx'].iloc[0]
        current_vy = last_frame['vy'].iloc[0]
        current_v = last_frame['v'].iloc[0]
        
        
         # Scatter plot for the final position
        ax.scatter(current_x, current_y, color=color, s=100, alpha=0.8)
        if jersey_no != "":
            ax.text(current_x, current_y, str(jersey_no), fontsize=7, fontweight="bold", color="white",
                    ha="center", va="center")
    
         # Add velocity arrow
        if current_v > 0.1: # Ignore very small speeds
            # 화살표 길이 스케일링 (속도에 비례하도록 조절 가능)
            arrow_scale = 0.3
            ax.arrow(current_x, current_y,
                        current_vx * arrow_scale, current_vy * arrow_scale,
                        head_width=1, head_length=1.5, fc=color, ec=color, width=0.3)
    
    # Create legend: Home team, Away team, Ball (if necessary)           
    home_patch = mpatches.Patch(color=team_colors["Home"], label='Home')
    away_patch = mpatches.Patch(color=team_colors["Away"], label='Away')
    ball_patch = mpatches.Patch(color=team_colors["Ball"], label="Ball")
    ax.legend(handles=[home_patch, away_patch, ball_patch], loc='upper right')
    
    plt.show()

def create_match_animation(total_df, period_id, start_frame_idx, end_frame_idx, pitch, home_team_info, away_team_info, sampling_rate=25, filename='match_animation.mp4'):
    """
    Creates and saves an animation of player movements and velocities for a time window.

    Parameters:
        total_df (DataFrame): Positional data including 'id', 'x', 'y', 'frame_id', 'period_id'.
                               Must contain pre-calculated 'vx', 'vy', 'v' for velocity.
        period_id (int): The target period ID.
        start_frame_idx (int): The start frame index of the animation window.
        end_frame_idx (int): The end frame index of the animation window.
        pitch: An mplsoccer Pitch object.
        home_team_info: DataFrame/Info for the home team (needs 'pID', 'player', 'jID').
        away_team_info: DataFrame/Info for the away team (needs 'pID', 'player', 'jID').
        sampling_rate (int): Sampling rate in Hz (default: 25).
        filename (str): Name of the file to save the animation (e.g., 'animation.mp4', 'animation.gif').
                        Requires appropriate writers like ffmpeg or imagemagick.
    """
    # --- 0. Basic Settings and Data Preparation ---
    team_colors = {"Home": "blue", "Away": "red", "Ball": "black", "Unknown": "gray"}
    dt = 1.0 / sampling_rate
    arrow_scale = 0.3 # 화살표 길이 조절 인자

    # Filter data for the animation window
    anim_df = total_df[(total_df['period_id'] == period_id) &
                       (total_df['frame_id'] >= start_frame_idx) &
                       (total_df['frame_id'] <= end_frame_idx)
                      ].copy()

    if anim_df.empty:
        print(f"No frames found for Period {period_id}, Frames {start_frame_idx}-{end_frame_idx} to animate.")
        return

    anim_df.sort_values(['id', 'frame_id'], inplace=True)

     # Calculate velocity data if not present
    if 'vx' not in anim_df.columns:
        print("Calculating velocities for animation...")
        anim_df['dx'] = anim_df.groupby('id')['x'].diff()
        anim_df['dy'] = anim_df.groupby('id')['y'].diff()
        anim_df.loc[anim_df.groupby('id').head(1).index, ['dx', 'dy']] = 0 # 첫 프레임 diff는 0으로
        anim_df['vx'] = anim_df['dx'] / dt
        anim_df['vy'] = anim_df['dy'] / dt
        anim_df['v'] = np.sqrt(anim_df['vx']**2 + anim_df['vy']**2)
        print("Velocities calculated.")

    # Team info dictionaries
    home_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in home_team_info.iterrows()}
    away_team_dict = {player['pID']: [player['player'], player['jID']] for _, player in away_team_info.iterrows()}

    unique_frames = sorted(anim_df['frame_id'].unique())
    num_animation_frames = len(unique_frames)
    player_ids = anim_df['id'].unique()

    # --- 1. Animation Setup ---
    fig, ax = pitch.draw(figsize=(12, 8))

     # Initialize dictionaries to store plot objects for each player/ball
    player_plots = {}
    for pid in player_ids:
        player_plots[pid] = {
            'scatter': None,
            'text': None,
            'arrow': None,
            'trajectory': None
        }

    # 시간 표시 텍스트 객체 초기화
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

    # 범례 추가 (한 번만)
    handles = []
    labels = []
    if any(pid in home_team_dict for pid in player_ids):
        handles.append(mpatches.Patch(color=team_colors["Home"], label='Home'))
        labels.append('Home')
    if any(pid in away_team_dict for pid in player_ids):
        handles.append(mpatches.Patch(color=team_colors["Away"], label='Away'))
        labels.append('Away')
    if any("ball" in str(pid).lower() for pid in player_ids):
         handles.append(mpatches.Patch(color=team_colors["Ball"], label='Ball'))
         labels.append('Ball')
    ax.legend(handles=handles, labels=labels, loc='upper right')

    # --- 2. 애니메이션 업데이트 함수 정의 ---
    def update(frame_num):
        current_frame_id = unique_frames[frame_num]
        frame_data = anim_df[anim_df['frame_id'] == current_frame_id]

        updated_elements = [] # 업데이트된 요소들을 반환하기 위한 리스트

        for pid in player_ids:
            player_frame_data = frame_data[frame_data['id'] == pid]

            if not player_frame_data.empty:
                data = player_frame_data.iloc[0]
                x, y = data['x'], data['y']
                vx, vy, v = data['vx'], data['vy'], data['v']

                # 선수 정보 및 색상 결정
                if pid in home_team_dict:
                    color = team_colors["Home"]
                    jersey_no = home_team_dict[pid][1]
                elif pid in away_team_dict:
                    color = team_colors["Away"]
                    jersey_no = away_team_dict[pid][1]
                elif "ball" in str(pid).lower():
                    color = team_colors["Ball"]
                    jersey_no = ""
                else:
                    color = team_colors["Unknown"]
                    jersey_no = "?"

                # 객체 초기화 또는 업데이트
                if player_plots[pid]['scatter'] is None: # 첫 프레임에서 객체 생성
                    # Scatter plot 생성
                    player_plots[pid]['scatter'] = ax.scatter(x, y, color=color, s=100, alpha=0.8, zorder=3)
                    # 등번호 텍스트 생성
                    if jersey_no != "":
                        player_plots[pid]['text'] = ax.text(x, y, str(jersey_no), fontsize=7, fontweight="bold", color="white",
                                                            ha="center", va="center", zorder=4)
                    # 화살표 객체 초기화 (나중에 위치/방향 업데이트)
                    if v > 0.1:
                         # ax.arrow 대신 ax.quiver 사용 고려 가능 (더 효율적)
                         # 여기서는 간단하게 arrow를 매번 제거하고 다시 그리는 방식을 쓰거나,
                         # arrow 객체를 직접 다루기 어려우므로 quiver를 사용
                         # 우선 quiver 없이, 기존 arrow를 업데이트 하는 방식 시도 (주의: Arrow는 직접 수정 어려움)
                         # -> 매 프레임마다 이전 화살표 제거하고 새로 그리는 방식 선택
                         pass # 아래에서 처리
                    # 궤적 Plot (필요 시)
                    # player_plots[pid]['trajectory'], = ax.plot([], [], color=color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=1)

                else: # 기존 객체 업데이트
                    # Scatter 위치 업데이트
                    player_plots[pid]['scatter'].set_offsets([x, y])
                    # 텍스트 위치 업데이트
                    if player_plots[pid]['text'] is not None:
                        player_plots[pid]['text'].set_position((x, y))
                    # 궤적 데이터 업데이트 (필요 시)
                    # traj_data = anim_df[(anim_df['id'] == pid) & (anim_df['frame_id'] <= current_frame_id)]
                    # player_plots[pid]['trajectory'].set_data(traj_data['x'], traj_data['y'])

                # 화살표 처리: 이전 화살표 제거 후 새로 그리기
                if player_plots[pid]['arrow'] is not None:
                    player_plots[pid]['arrow'].remove()
                    player_plots[pid]['arrow'] = None

                if v > 0.1:
                    player_plots[pid]['arrow'] = ax.arrow(x, y,
                                                          vx * arrow_scale, vy * arrow_scale,
                                                          head_width=1, head_length=1.5, fc=color, ec=color, width=0.3, zorder=2)
                    updated_elements.append(player_plots[pid]['arrow']) # 제거/생성되므로 항상 추가

                # 업데이트된 요소 목록에 추가
                if player_plots[pid]['scatter'] is not None: updated_elements.append(player_plots[pid]['scatter'])
                if player_plots[pid]['text'] is not None: updated_elements.append(player_plots[pid]['text'])
                # if player_plots[pid]['trajectory'] is not None: updated_elements.append(player_plots[pid]['trajectory'])

            else: # 해당 프레임에 선수가 없으면 숨김 (또는 다른 처리)
                 if player_plots[pid]['scatter'] is not None:
                     player_plots[pid]['scatter'].set_visible(False)
                     updated_elements.append(player_plots[pid]['scatter'])
                 if player_plots[pid]['text'] is not None:
                     player_plots[pid]['text'].set_visible(False)
                     updated_elements.append(player_plots[pid]['text'])
                 if player_plots[pid]['arrow'] is not None:
                     player_plots[pid]['arrow'].remove()
                     player_plots[pid]['arrow'] = None
                 # if player_plots[pid]['trajectory'] is not None:
                 #    player_plots[pid]['trajectory'].set_visible(False)
                 #    updated_elements.append(player_plots[pid]['trajectory'])


        # 시간 텍스트 업데이트
        time_sec = (current_frame_id - start_frame_idx) / sampling_rate
        time_text.set_text(f'Time: {time_sec:.2f} s (Frame: {current_frame_id})')
        updated_elements.append(time_text)

        # 그림 제목 업데이트 (선택 사항)
        # ax.set_title(f"Period {period_id}, Frame {current_frame_id} Positions & Velocity", fontsize=14)

        return updated_elements # 업데이트된 matplotlib 객체들을 반환

    # --- 3. 애니메이션 생성 및 저장 ---
    print(f"Creating animation for {num_animation_frames} frames...")
    # interval: 프레임 간 간격 (밀리초)
    interval = 1000 / sampling_rate
    ani = animation.FuncAnimation(fig, update, frames=num_animation_frames,
                                  interval=interval, blit=True, repeat=False) # blit=True로 성능 향상 시도

    # 애니메이션 저장 (mp4 또는 gif 등)
    # 적절한 writer가 설치되어 있어야 함 (예: ffmpeg, imagemagick)
    try:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='ffmpeg', fps=sampling_rate, dpi=150) # dpi로 해상도 조절 가능
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure ffmpeg (for mp4) or imagemagick (for gif) is installed and accessible in your system PATH.")

    # (선택 사항) 인터랙티브하게 보기 (저장과 동시 사용 어려움)
    # plt.show()
