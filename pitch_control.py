"""
Code adapted from Friends of Tracking Github - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.git
"""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(base_path)
sys.path.append(base_path)

import numpy as np
from datatools.visualization import plot_frame

#Pass in players for attacking/defending team, their team name, and params
def initialise_players(team_df,teamname,params, att, event):
    team_players = []
    for i,p in team_df.iterrows():
        if att == True:
            team_player = player(p,teamname,params,p['position'], False, p['player_on_ball'])
        else:
            team_player = player(p,teamname,params,p['position'], False, False)
        if team_player.inframe:
            team_players.append(team_player)
    return team_players

#Plots pitch control for an event
def plot_pitchcontrol_for_event(PPCF, action, locs, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (105.0,68)):
    fig,ax = plot_frame(action, locs)
    cmap = 'bwr'

    pc = ax.imshow(np.flipud(PPCF), extent=(0, field_dimen[0], 0, field_dimen[1]),vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
    cbar = fig.colorbar(pc,orientation="horizontal")
    tick_font_size = 25
    cbar.ax.tick_params(labelsize=tick_font_size)
    return fig,ax
    
#Creates a player class
class player(object):
    
    # Pass in the line for that player (from 360 data), their team name(e.g. home/away), default params and if they are goalkeeper or not
    def __init__(self,team,teamname,params,GK, off, actor):
        if GK == 'TW':#'GK':
            self.is_gk = True
        else:
            self.is_gk = False
        self.is_actor = actor
        self.teamname = teamname
        self.accel = params['max_player_accel']
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.offside = off
        self.PPCF = 0. # initialise this for later
        
    def get_position(self,team):
        self.position = np.array( [ team['x'], team['y'] ] )
        self.inframe = not np.any( np.isnan(self.position) )
    
    # 속도가 0인 버전
    # def get_velocity(self,team):
    #     self.velocity = np.array( [ 0, 0 ] )
    #     if np.any( np.isnan(self.velocity) ):
    #         self.velocity = np.array([0.,0.])
    # def simple_time_to_intercept(self, r_final):
    #     self.PPCF = 0. # initialise this for later
    #     # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
    #     # and then runs at full speed to the target position.
    #     r_reaction = self.position + self.velocity*self.reaction_time
    #     self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
    #     return self.time_to_intercept

    # 속도가 1인 버전
    def get_velocity(self,team):
        self.velocity = np.array( [ team['vx'], team['vy'] ] )

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. 
        dist_to_loc = np.linalg.norm(r_final-self.position)
        time_to_reach_terminal_velocity = self.vmax / self.accel
        dist_reached_before_TV = 0.5 * self.accel * (time_to_reach_terminal_velocity ** 2)
        
        if dist_to_loc < dist_reached_before_TV:
            time = np.sqrt((2*dist_to_loc)/self.accel)
        else:
            time = time_to_reach_terminal_velocity + ((dist_to_loc - dist_reached_before_TV) / self.vmax)
        
        self.time_to_intercept = time
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f
    
#Sets default Pitch Control parameters
def default_model_params(time_to_control_veto=3):
    
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1 # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att']) 
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params

#Calculates pitch control probability for attacking and defending team at a given target position
def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ 
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
     
    tti = [p.simple_time_to_intercept(target_position) for p in attacking_players]
    ttca = params['time_to_control_att']
    if attacking_players[tti.index(np.nanmin(tti))].is_actor == True:
        ttca = params['time_to_control_att']
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= ttca:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                if player.is_actor:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_att)
                else:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # calculate ball control probablity for 'player' in time interval T+dt
                #dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                #if (player.teamname == 'Home'):
                #    behind_loc = player.position[0] < target_position[0] - 1
                #else:
                #    behind_loc = player.position[0] > target_position[0] - 1
            
                # calculate ball control probablity for 'player' in time interval T+dt
                #Gives defender the advantage if they are behind the ball, if not, no advantage
                #if behind_loc:
                if player.is_gk:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_def * 3)
                else:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_def)
                #else:
                #    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]
    
def generate_pitch_control_for_event(event, locations, params, field_dimen = (105.,68.,), n_grid_cells_x = 16, n_grid_cells_y=16, offsides=False):
    # get the details of the event (frame, team in possession, ball_start_position)
    ball_start_pos = np.array([event['ballx'],event['bally']])
    # break the pitch down into a grid
    #n_grid_cells_y = n_grid_cells_x * 0.75
    
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x)*dx + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy + dy/2.
   
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    att_players = initialise_players(locations[locations['team_on_ball'] == True], "Home", params, True, event)
    def_players = initialise_players(locations[locations['team_on_ball'] == False], "Away", params, False, event)
    
        
    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        att_players = [a for a in att_players if a.offside == False]
        #for att_p in att_players:
        #    if att_p.offside == True:
        #        att_players.remove(att_p)
        
    count = 0
    # calculate pitch pitch control model at each location on the pitch
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, att_players, def_players, ball_start_pos, params)
                
            
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(count)#float(n_grid_cells_y*n_grid_cells_x ) 
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid