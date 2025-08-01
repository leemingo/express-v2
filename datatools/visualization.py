# visualization.ipynb

import matplotlib.pyplot as plt
import seaborn as sns

# Constants
COL_FACE = "lightgrey"
plt.style.use("ggplot")

# Count Plot for Event IDs
def plot_event_count(all_events):
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    sns.countplot(y=all_events["eID"], order=all_events["eID"].value_counts().index, ax=ax,color="indianred")
    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Event ID", size=14)
    ax.set_ylabel("Count", size=14)
    plt.show()

# KDE Plot
def plot_kde(xy_objects, pitch):
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(13, 9))
    for a in ax.flat:
        pitch.plot(ax=a)
        a.set_facecolor(COL_FACE)
        a.set_xlim(-55, 55)
        a.set_ylim(-37, 37)
        a.annotate("Attacking Direction", xy=(-51, -25), xytext=(-51, -28), xycoords="data", fontsize=12)
        a.annotate("", xy=(-22, -25), xytext=(-51, -25), xycoords="data",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), ha="center", va="top")

    ax[0, 0].set_xlabel("TW", size=14)
    ax[0, 1].set_xlabel("STZ", size=14)
    ax[1, 0].set_xlabel("OLM", size=14)
    ax[1, 1].set_xlabel("IVL", size=14)

    sns.kdeplot(x=xy_objects["firstHalf"]["Home"].xy[:, 22], y=xy_objects["firstHalf"]["Home"].xy[:, 23], fill=True, color="red", alpha=0.5, ax=ax[0, 0])
    sns.kdeplot(x=xy_objects["firstHalf"]["Home"].xy[:, 4], y=xy_objects["firstHalf"]["Home"].xy[:, 5], fill=True, color="green", alpha=0.5, ax=ax[0, 1])
    sns.kdeplot(x=xy_objects["firstHalf"]["Home"].xy[:, 30], y=xy_objects["firstHalf"]["Home"].xy[:, 31], fill=True, color="blue", alpha=0.5, ax=ax[1, 0])
    sns.kdeplot(x=xy_objects["firstHalf"]["Home"].xy[:, 8], y=xy_objects["firstHalf"]["Home"].xy[:, 9], fill=True, color="purple", alpha=0.5, ax=ax[1, 1])

    plt.show()

# Goal Positions Plot
def plot_goal_positions(xy_objects, events, pitch):
    framerate = xy_objects["secondHalf"]["Home"].framerate
    events["secondHalf"]["Home"].add_frameclock(framerate)
    goals = events["secondHalf"]["Home"].events.loc[events["secondHalf"]["Home"].events["eID"] == "ShotAtGoal_SuccessfulShot"]
    first_goal = goals.iloc[0]
    frame_first_goal = int(first_goal["gameclock"] * framerate + 1.6 * framerate)  # offset event clock, pos data
    second_before_goal = frame_first_goal - 5 * framerate

    fig, ax = plt.subplots(tight_layout=True, figsize=(16, 10))
    pitch.plot(ax=ax)
    ax.set_facecolor(COL_FACE)

    xy_objects["secondHalf"]["Home"].plot(t=(second_before_goal, frame_first_goal), plot_type="trajectories", color="blue", ax=ax)
    xy_objects["secondHalf"]["Away"].plot(t=(second_before_goal, frame_first_goal), plot_type="trajectories", color="red", ax=ax)
    xy_objects["secondHalf"]["Ball"].plot(t=(second_before_goal, frame_first_goal), plot_type="trajectories", color="black", ax=ax)

    xy_objects["secondHalf"]["Home"].plot(t=frame_first_goal, color="blue", ax=ax)
    xy_objects["secondHalf"]["Away"].plot(t=frame_first_goal, color="red", ax=ax)
    xy_objects["secondHalf"]["Ball"].plot(t=frame_first_goal, color="black", ax=ax)

    plt.show()

""" 
Pitch plotting functions, using code from Laurie on Tracking Github - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.git
"""

from mplsoccer import Pitch
from matplotlib import animation
from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np


def plot_pitch():
    pitch = Pitch(
        pitch_type="uefa", pitch_length=105, pitch_width=68, axis=True, label=True
    )
    fig, ax = pitch.draw()
    return fig, ax


"""
Gets the positions of all the players for a particular frame in the game
"""


def animate(i, tracking_df, home_df, away_df, ball, away, home):
    ball.set_data(float(tracking_df["ball_x"][i]), float(tracking_df["ball_y"][i]))
    x_coords = tracking_df.loc[i].filter(like="_x")
    y_coords = tracking_df.loc[i].filter(like="_y")

    home_ids_x = ["player_" + h + "_x" for h in home_df["player_id"].values]
    away_ids_x = ["player_" + a + "_x" for a in away_df["player_id"].values]
    home_ids_y = ["player_" + h + "_y" for h in home_df["player_id"].values]
    away_ids_y = ["player_" + a + "_y" for a in away_df["player_id"].values]
    home_x_coords = pd.to_numeric(x_coords.filter(items=home_ids_x))
    home_y_coords = pd.to_numeric(y_coords.filter(items=home_ids_y))
    away_x_coords = pd.to_numeric(x_coords.filter(items=away_ids_x))
    away_y_coords = pd.to_numeric(y_coords.filter(items=away_ids_y))
    home.set_data(home_x_coords, home_y_coords)
    away.set_data(away_x_coords, away_y_coords)
    return ball, home, away


"""
Animates the football game using animate function by continously plotting the player locations for each row of tracking data at 25fps
"""


def generate_video(tracking_df, home_df, away_df):

    fig, ax = plot_pitch()
    marker_kwargs = {"marker": "o", "markeredgecolor": "black", "linestyle": "None"}
    (ball,) = ax.plot([], [], ms=6, markerfacecolor="w", zorder=3, **marker_kwargs)
    (away,) = ax.plot([], [], ms=10, markerfacecolor="r", **marker_kwargs)
    (home,) = ax.plot([], [], ms=10, markerfacecolor="b", **marker_kwargs)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(tracking_df),
        interval=40,
        blit=True,
        fargs=(tracking_df, home_df, away_df, ball, away, home),
    )
    anim.save("tracking.mp4", fps=25)


"""
Plot a frame of the game using a row of tracking data
"""

#Plots a frame of data with the 360 data included
def plot_frame(event, locations, figax=None, team_colors=('r','b'), PlayerMarkerSize=15, PlayerAlpha=1, annotate=True ):
    
    if figax is None: # create new pitch 
        fig,ax = plot_event(event)
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    
    #Get locations of players for teammates and opponents
    teammate_locs = locations[locations['team_on_ball'] == True]
    opp_locs = locations[locations['team_on_ball'] == False]
    
    #Loop through teammates and plot teammates, and then loop through opponents and plot opponents.
    for i,row in teammate_locs.iterrows():
        if (row['position'] == 'GK'):
            ax.plot(row['x'],row['y'],'g'+'o',markersize=PlayerMarkerSize, alpha=PlayerAlpha)
        else:
            ax.plot(row['x'],row['y'],team_colors[0]+'o',markersize=PlayerMarkerSize, alpha=PlayerAlpha)
                
    for i,row in opp_locs.iterrows():
        if (row['position'] == 'GK'):
            ax.plot(row['x'],row['y'],'g'+'o',markersize=PlayerMarkerSize, alpha=PlayerAlpha)
        else:
            ax.plot(row['x'],row['y'],team_colors[1]+'o',markersize=PlayerMarkerSize, alpha=PlayerAlpha)
    return fig,ax


# Plot a sequence of events by plotting where the event took place, and an arrow of where the ball is going if that data is available through relative_event
def plot_events(
    events,
    figax=None,
    indicators=["Marker", "Arrow"],
    color="r",
    marker_style="o",
    alpha=0.8,
    annotate=False,
):

    if figax is None:  # create new pitch
        fig, ax = plot_pitch()
    else:  # overlay on a previously generated pitch
        fig, ax = figax

    # print(events)
    for i, row in events.iterrows():
        if "Marker" in indicators:
            ax.plot(row["x"], row["y"], color + marker_style, alpha=alpha)
        if not math.isnan(row["relative_event_x"]) & ("Arrow" in indicators):
            ax.annotate(
                "",
                xy=row[["relative_event_x", "relative_event_y"]],
                xytext=row[["x", "y"]],
                alpha=alpha,
                arrowprops=dict(
                    alpha=alpha, width=0.5, headlength=4.0, headwidth=4.0, color=color
                ),
                annotation_clip=False,
            )
        if annotate:
            textstring = row["event_types_0_eventType"] + ": " + str(row["player_id"])
            ax.text(row["x"], row["y"], textstring, fontsize=10, color=color)
    return fig, ax


#Plot a single event rather than a sequence. Make it so colour varies based on home/away
def plot_event(event, figax=None, indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 1, annotate=False):

    if figax is None: # create new pitch 
        fig,ax = plot_pitch()
    else: # overlay on a previously generated pitch
        fig,ax = figax 
        
    color = 'k'
    
    if 'Marker' in indicators:
        ax.plot( event['ballx'], event['bally'], color+marker_style,markersize=15, alpha=alpha )
    return fig,ax



# Plots the pitch control at a given event
def plot_pitchcontrol_for_event(
    event_id,
    events,
    tracking_df,
    home_df,
    away_df,
    PPCF,
    alpha=0.7,
    include_player_velocities=True,
    annotate=False,
    field_dimen=(105.0, 68),
):

    # Gets the event time and team on the ball, and the index in tracking dataframe to plot the frame based on event time
    pass_frame = events.loc[event_id]["event_time"]
    pass_team = events.loc[event_id]["team_id"]
    pass_half = events.loc[event_id]["event_period"]
    tracking_index = ut.get_tracking_index_from_event_time(
        tracking_df, pass_frame, pass_half
    )
    # print(tracking_index)

    # plot frame and event
    fig, ax = plot_pitch()

    plot_frame(
        tracking_df.loc[tracking_index],
        home_df,
        away_df,
        figax=(fig, ax),
        PlayerAlpha=alpha,
        include_player_velocities=include_player_velocities,
        annotate=annotate,
    )
    plot_events(
        events.loc[event_id:event_id],
        figax=(fig, ax),
        indicators=["Marker", "Arrow"],
        annotate=False,
        color="k",
        alpha=1,
    )

    # plot pitch control surface, chooses color to plot based on team on the ball
    if pass_team == ut.get_teams(events, home_df)[0]:
        cmap = "bwr"
    else:
        cmap = "bwr_r"

    # Plots the surface, extent represents the dimensions of the field, uses the color map specified
    # Flips the pitch control vector to match how pitch is plotted (first values on the bottom working up)
    ax.imshow(
        np.flipud(PPCF),
        extent=(0, field_dimen[0], 0.0, field_dimen[1]),
        interpolation="spline36",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        alpha=0.5,
    )

    return fig, ax