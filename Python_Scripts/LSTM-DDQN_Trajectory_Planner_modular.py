# This script has Simple DQN as model
# The state space is changed as well
# It contains the ray on the front of the car 
# The length of the ray is always 30 except when ego_car gets close to the car in the front

# SPECIAL to LSTM based HDQN
# 1} Uses LSTM layer of 512 hidden nodes instead of the fully connected layer
# 2} Uses the 3 time steps sequence for the training and buffer
# 3} Noise added to the state space using the gaussian noise

from __future__ import print_function
# NORMAL FUNCTION IMPORTS

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tqdm import tqdm
import pylab
import csv

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# These lines of code are important if you want to import other scripts
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from PID_Controller import VehiclePIDController
from Network_LSTM import Network_Model
from Car_EnvRL import CarEnv

# TENSORFLOW Imports 
import keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Lambda, TimeDistributed, LSTM, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise


# The command to set frequency is: "carlaUE4 -benchmark -fps=30

# GLOBAL VARIABLES
# =========================================
#SHOW_PREVIEW = False
#IM_WIDTH = 640
#IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 10_000
MIN_REWARD = -200
EPISODES = 100
AGGREGATE_STATS_EVERY = 10
FPS = 60
name = "HDQN"
# ========================================


# This import and code prevents "prediction function callback" error
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# =====================================================================
# CLASS ENVIRONEMENT "CarEnv" for lane follow and lane change manuever  
# =====================================================================



class CarEnv:
    #SHOW_CAM = SHOW_PREVIEW
    #STEER_AMT = 0.3
    #im_width = IM_WIDTH
    #im_height = IM_HEIGHT

    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.world.set_weather(getattr(carla.WeatherParameters, 'ClearNoon'))
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        self.model_4 = self.blueprint_library.filter("model3")[0]
        self.model_4.set_attribute('color', "255, 0 , 0")        

        self.model_5 = self.blueprint_library.filter("model3")[0]
        self.model_5.set_attribute('color', "0, 255, 0")


        #self.model_6 = self.blueprint_library.filter("model3")[0]
        #self.model_6.set_attribute('color', "0, 255, 0")

    def reset(self):
        self.epi_steps = 0
        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []
        self.lane_counter = 0
        self.is_alive = True
        self.collision_boolean = False
        self.lane_boolean = False
        self.goals = [0,1]
        self.option2_failed = False
        self.transform = carla.Transform(carla.Location(x=46.0, y=3.7, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.controller = VehiclePIDController(self.vehicle)
        self.actor_list.append(self.vehicle)
        self.collision_counter = 0
        self.fail_counter = 0 
        self.initial_distance = 0      

        rand_num = random.randrange(0,4)
        if rand_num == 0:
            self.transform2 = carla.Transform(carla.Location(x=105.0, y=4.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle = self.world.spawn_actor(self.model_4, self.transform2)
            self.actor_list.append(self.target_vehicle)
        elif rand_num == 1:
            self.transform2 = carla.Transform(carla.Location(x=95.0, y=4.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle = self.world.spawn_actor(self.model_4, self.transform2)
            self.actor_list.append(self.target_vehicle)
        elif rand_num == 2:
            self.transform2 = carla.Transform(carla.Location(x=85.0, y=3.8, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle = self.world.spawn_actor(self.model_4, self.transform2)
            self.actor_list.append(self.target_vehicle)
        elif rand_num == 3:
            self.transform2 = carla.Transform(carla.Location(x=75.0, y=3.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle = self.world.spawn_actor(self.model_4, self.transform2)
            self.actor_list.append(self.target_vehicle) 

        self.initial_distance = carla.Location.distance(self.vehicle.get_location(), self.target_vehicle.get_location()) - 20.0
        self.safe_distance = carla.Transform(carla.Location(x=70.0, y=4.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000)) 
                        
        self.transform3 = carla.Transform(carla.Location(x =46.0, y = 7.0, z = 1.84312), carla.Rotation(pitch = 0.00000, yaw = 0.855823, roll = 0.00000))
        self.moving_vehicle = self.world.spawn_actor(self.model_5, self.transform3)
        self.actor_list.append(self.moving_vehicle)

        self.transform4 = carla.Transform(carla.Location(x =55.0, y = 7.4, z = 1.84312), carla.Rotation(pitch = 0.00000, yaw = 0.855823, roll = 0.00000))
        self.moving_vehicle2 = self.world.spawn_actor(self.model_5, self.transform4)
        self.actor_list.append(self.moving_vehicle2)        

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        #self.target_vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake=0.0))
        time.sleep(4)

        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, carla.Transform(), attach_to = self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event_lane: self.invasion_data(event_lane))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        obs = self.observe()
        return obs

    def get_speed(self, vehicle):
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


    def observe(self):


        # State vector size = 15

        ego_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road = True)
        target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location(), project_to_road = True)
        movingA_waypoint = self.map.get_waypoint(self.moving_vehicle.get_location(), project_to_road = True)
        movingB_waypoint = self.map.get_waypoint(self.moving_vehicle2.get_location(), project_to_road = True)

        self.observe_vector = []

        # ego car velocity
        v = self.vehicle.get_velocity()
        ego_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.observe_vector.append(ego_kmh)

        # Ego Lane ID 
        ego_lane_id = abs(ego_waypoint.lane_id)
        self.observe_vector.append(ego_lane_id)

        # Target car velocity
        v = self.target_vehicle.get_velocity()
        target_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.observe_vector.append(target_kmh)    

        # Target Lane ID
        target_lane_id = abs(target_waypoint.lane_id)
        self.observe_vector.append(target_lane_id)        

        if ego_lane_id == target_lane_id:
            if self.ego_target() - 10 <= 30:
                safe_ray = self.ego_target() - 10
            else:
                safe_ray = 30.0
        else:
            safe_ray = 30.0

        safe_ratio = safe_ray/10.0

        self.observe_vector.append(safe_ray)
        self.observe_vector.append(safe_ratio)

        # Moving vehicle # 1

        v = self.moving_vehicle.get_velocity()
        movingA_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.observe_vector.append(movingA_kmh)

        movingA_lane_id = abs(movingA_waypoint.lane_id)
        self.observe_vector.append(movingA_lane_id)

        if ego_lane_id == target_lane_id:
            if self.ego_target() - 10 <= 30:
                mov_safe_ray = self.ego_moving() - 10
            else:
                mov_safe_ray = 10
            mov_safe_ratio = mov_safe_ray/10.0
        else:
            if self.ego_moving() - 6.0 >= 10.0:
                mov_safe_ray = 10
            else:
                mov_safe_ray = self.ego_moving() - 6.0
            mov_safe_ratio = mov_safe_ray/6.0

        self.observe_vector.append(mov_safe_ray)
        self.observe_vector.append(mov_safe_ratio)            

        # Moving vehicle 2

        v1 = self.moving_vehicle2.get_velocity()
        movingB_kmh = int(3.6 * math.sqrt(v1.x**2 + v1.y**2 + v1.z**2))
        self.observe_vector.append(movingB_kmh)

        movingB_lane_id = abs(movingB_waypoint.lane_id)
        self.observe_vector.append(movingB_lane_id)

        if ego_lane_id == target_lane_id:
            if self.ego_target() - 10 <= 30:
                movB_safe_ray = self.ego_moving2() - 10
            else:
                movB_safe_ray = 10
            movB_safe_ratio = movB_safe_ray/10.0
        else:
            if self.ego_moving2() - 6.0 >= 10.0:
                movB_safe_ray = 10
            else:
                movB_safe_ray = self.ego_moving2() - 6.0
            movB_safe_ratio = movB_safe_ray/6.0

        self.observe_vector.append(movB_safe_ray)
        self.observe_vector.append(movB_safe_ratio)


        #print(tuple(self.observe_vector))
        return tuple(self.observe_vector)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def collision_check(self):
        if len(self.collision_hist) != 0:
            self.collision_boolean = True
            self.is_alive = False
            return True
        elif self.ego_target() <= 3:
            self.collision_boolean = True
            self.is_alive = False
            return True           
        else:
            return False

    def invasion_data(self,event_lane):
        self.invasion_hist.append(event_lane)

    def ego_target(self):
        # Distance between target and ego car
        self.dist_ET = carla.Location.distance(self.vehicle.get_location(),self.target_vehicle.get_location())
        return self.dist_ET

    def ego_goal(self):
        # Distance between ego and goal destination
        goal_position = carla.Transform(carla.Location(x=135.9, y=8.7, z=0), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
        dist_EG = carla.Location.distance(self.vehicle.get_location(), goal_position.location)
        return dist_EG

    def ego_sd(self):
        # This is the distance from the ego-car to the safety threshold
        chase_distance = carla.distance(self.vehicle.get_location(), self.safe_distance.location)  
        return chase_distance 

    def ego_moving(self):
        self.dist_EM = carla.Location.distance(self.vehicle.get_location(), self.moving_vehicle.get_location())
        return self.dist_EM

    def ego_moving2(self):
        self.dist_EM2 = carla.Location.distance(self.vehicle.get_location(), self.moving_vehicle2.get_location())
        return self.dist_EM2

    def is_done(self):
        # Gives the information if the termination state is reached
        if self.is_alive == False or abs(self.ego_goal()) <= 12:
            return True
        else:
            return False

    def goal (self, goal, action):
        if goal == 0:
            status = self.option1_action(action)
        elif goal == 1:
            status = self.option2_action(action) 
        return status


    def option1_action(self, action):
        # Follow Lane/ Wait Option
        ego_location = self.vehicle.get_location()
        ego_waypoint = self.map.get_waypoint(ego_location, project_to_road = True)
        target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location(), project_to_road = True)
        if action == 0:
            #print('5m')
            waypoint_final = ego_waypoint.next(15.0)[0]
            if (abs(ego_waypoint.lane_id) == abs(target_waypoint.lane_id)): 
                target_speed = 0.0
                # 0.1
                to_draw_waypoint_final = ego_waypoint.next(3.0)[0]
                self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.0)
                #self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)
                self.world.debug.draw_point(to_draw_waypoint_final.transform.location, size = 0.3, life_time = 2.0)          
                self.world.debug.draw_line(ego_waypoint.transform.location, to_draw_waypoint_final.transform.location, thickness = 0.3, life_time = 2.0)                
                atime = 0.33
                start_time = time.time()
                while time.time() < start_time + atime:
                    control = self.controller.run_step(target_speed, waypoint_final)
                    self.vehicle.apply_control(control)
            else:
                target_speed = 7.0
                to_draw_waypoint_final = ego_waypoint.next(7.0)[0]
                self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.0)
                #self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)
                self.world.debug.draw_point(to_draw_waypoint_final.transform.location, size = 0.3, life_time = 2.0)          
                self.world.debug.draw_line(ego_waypoint.transform.location, to_draw_waypoint_final.transform.location, thickness = 0.3, life_time = 2.0)                 
                atime = 0.65
                start_time = time.time()
                while time.time() < start_time + atime:
                    control = self.controller.run_step(target_speed, waypoint_final)
                    self.vehicle.apply_control(control)                
            '''
            print("WAIT")
            atime = 1.0
            start_time = time.time()
            while time.time() < start_time + atime:
                control = carla.VehicleControl(throttle = 0.0 ,steer = 0, brake = 1.0)
                self.vehicle.apply_control(control)
            '''  
        # Slow Moving Foward      
        elif action == 1:
            #print("14m")
            waypoint_final = ego_waypoint.next(18.0)[0]
            self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.0)
            self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)
            self.world.debug.draw_point(waypoint_final.transform.location, size = 0.3, life_time = 2.0)
            #self.world.debug.draw_string(waypoint_final.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)            
            self.world.debug.draw_line(ego_waypoint.transform.location, waypoint_final.transform.location, thickness = 0.3, life_time = 2.0)            
            target_speed = 14
            # 0.2, 
            a2time = 0.60
            start_time_2 = time.time()
            while time.time() < start_time_2 + a2time:
                control = self.controller.run_step(target_speed, waypoint_final)
                self.vehicle.apply_control(control)
            final_location = self.vehicle.get_location() 
        
        # Fast Moving Forward    
        elif action == 2:
            #print("20m")
            waypoint_final = ego_waypoint.next(23.0)[0]
            self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.5)
            self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.5)
            self.world.debug.draw_point(waypoint_final.transform.location, size = 0.3, life_time = 2.5)
            #self.world.debug.draw_string(waypoint_final.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)            
            self.world.debug.draw_line(ego_waypoint.transform.location, waypoint_final.transform.location, thickness = 0.3, life_time = 2.5)            
            target_speed = 17
            # 0.40
            a3time = 0.78
            start_time_3 = time.time()
            while time.time() < start_time_3 + a3time:
                control = self.controller.run_step(target_speed, waypoint_final)
                self.vehicle.apply_control(control)
            final_location = self.vehicle.get_location()        
        
        # Check if collision occured then returned false
        status  = self.collision_check()
        return status

    def option2_action(self, action):
        # Lane Change Option
        status = False
        ego_location = self.vehicle.get_location()
        ego_waypoint = self.map.get_waypoint(ego_location, project_to_road = True)
        lane_change = str(ego_waypoint.lane_change)
        if lane_change == 'Right':
            self.option2_failed = False        
            waypoint_rightlane = ego_waypoint.get_right_lane()
            if action == 0:
                print("waypoint at 10")
                final_waypoint = waypoint_rightlane.next(10.0)[0]
                target_speed = 13
            elif action == 1:
                print("waypoint at 12")
                final_waypoint = waypoint_rightlane.next(12.0)[0]
                target_speed = 15
            elif action == 2:
                print("waypoint at 14")
                final_waypoint = waypoint_rightlane.next(14.0)[0]
                target_speed = 17 

            self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, color = carla.Color(255, 255, 0), life_time = 2.0)
            self.world.debug.draw_point(final_waypoint.transform.location, size = 0.1, color = carla.Color(255, 255, 0), life_time = 2.0)
            self.world.debug.draw_line(ego_waypoint.transform.location, final_waypoint.transform.location, thickness = 0.4, color = carla.Color(255, 255, 0), life_time = 2.0)
            lchange_time = 0.50
            start_time = time.time()
            while time.time() < start_time + lchange_time:
                control = self.controller.run_step(target_speed, final_waypoint)    
                self.vehicle.apply_control(control)        
            final_location = self.vehicle.get_location()
            #self.world.debug.draw_point(final_location, size = 0.3, life_time = 3.5)            
            self.world.debug.draw_line(ego_waypoint.transform.location, final_location, thickness = 0.3, color = carla.Color(0, 255, 0), life_time = 2.5)            
            #ego_waypoint2 = self.map.get_waypoint(self.vehicle.get_location(), project_to_road = True)
            waypoint_final2 = final_waypoint.next(16.0)[0]
            self.world.debug.draw_point(waypoint_final2.transform.location, size = 0.3, color = carla.Color(255,255, 0), life_time = 2.0)            
            self.world.debug.draw_line(final_waypoint.transform.location, waypoint_final2.transform.location, thickness = 0.4, color = carla.Color(255, 255, 0), life_time = 2.0)             
            correct_time = 0.75
            start_time2 = time.time()
            while time.time() < start_time2 + correct_time:
                control = self.controller.run_step(target_speed, waypoint_final2)
                self.vehicle.apply_control(control)
            last_location = self.vehicle.get_location()
            self.world.debug.draw_line(final_location, waypoint_final2.transform.location, thickness = 0.3, color = carla.Color(0, 255, 0), life_time =2.0)

            status = self.collision_check()
            # If status returns true this means that collision occured and then the option will be penalise 
            return status


        else:
            self.option2_failed = True
            status  = self.collision_check()
            return status
            # Give a penalty for choosing wrong option


    def reward(self, goal, action, status):
        
        ego_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road = True)
        target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location(), project_to_road = True)

        reward_option = 0
        reward_action = 0
        # Time penalty
        time_penalty = 0        
        time_out = 15
        if self.episode_start + time_out > 15:
            time_penalty += -15

        if status:
            collision_value = 1
        else:
            collision_value = 0

        # The focus should be on the options
        if self.ego_goal() < 12:
            success_reward = 1000
        else:
            success_reward = 0

        # OPTIONS
        if self.epi_steps < 15:
            if not(status):
                if not(self.option2_failed):
                    reward_action += 70
                    reward_option += 70

        # if ego car and target car are on the same lane
        if (abs(ego_waypoint.lane_id) == abs(target_waypoint.lane_id)):   
            if goal == 0:
                if self.ego_target() - 10 <= 30:
                    # IF the ego-car is less than 40 meters from safe distance then it can select between options
                    # But if ego-car is less than 15 m and it waits then it gets positive reward
                    if (self.ego_moving() -10 <= 10) or (self.ego_moving2() - 10 <= 10):
                        reward_option += 40                        
                        reward_action += -40 * math.exp((-(self.ego_target()) - 10)/10) + (collision_value * -100)
                        if reward_action == 0:
                            reward_action += 40
                    else:
                        reward_option += -80
                        reward_action += -40 * math.exp((-(self.ego_target()) - 10)/10) + (collision_value * -100)
                else:
                    reward_option += 40
                    if action == 0:
                        reward_action += -40
                    # Here the lane change option does not get any positive reward
            elif goal == 1:
                reward_option += -60 * math.exp((-(self.ego_moving()) - 6.0)/6.0) -60 * math.exp((-(self.ego_moving2()) - 6.0)/6.0) - 40 * math.exp((-(self.ego_target()) - 10)/10) + (collision_value * -100)
            #    if self.ego_target() - 10 >= 30:
            #        reward_option += -40 
        else:
            if goal == 0:
                reward_option += 40
                reward_action += -40 * math.exp((-(self.ego_moving()) - 6.0)/6.0) - 40 * math.exp((-(self.ego_moving2()) - 6.0)/6.0) + (collision_value * -100)
                if (self.ego_moving() - 6.0 <= 12) or (self.ego_moving2() - 6.0 <= 12):
                    if action == 0:
                        reward_action += 40
                #if action == 0 and self.ego_moving() - 6.0 >= 12
                #if action == 0 and (self.ego_moving() - 6.0 >= 12 or self.ego_moving2() - 6.0 >= 12):
                #    reward_action += -40
                #else:
                #    reward_action += 40
            elif goal == 1:
                # *****
                # One more thing can be added here is the distance to the front or back vehicle and the option get penlised as 
                # the car gets really close to the one of the green car in the target lane

                
                #reward_action += -40 * math.exp((-(self.ego_target()) - 10)/10) + (collision_value * -100)
                if self.option2_failed:
                    reward_option += -40
                else:
                    reward_option += -70 * math.exp((-(self.ego_moving()) - 6.0)/6.0) -70 * math.exp((-(self.ego_moving2()) - 6.0)/6.0) + (collision_value * -100)                    
                    reward_action += -60 * math.exp((-(self.ego_moving()) - 6.0)/6.0) -60 * math.exp((-(self.ego_moving2()) - 6.0)/6.0) + (collision_value * -100)


        reward_action += (success_reward)
        reward_option += (success_reward)

        return reward_option, reward_action


    def step(self, goal, action):
        #print("distance between ego and target: ", self.ego_target())
        # One thing is missing here. The car should only do the lane change when there is a 
        # car in the same line and 
        #ego_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        #target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location())
        #if (ego_waypoint.lane_id == target_waypoint.lane_id):
        status = self.goal(goal,action)
        reward_option, reward_action = self.reward(goal, action, status)
        obs = self.observe()


        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            self.is_alive = False

        done = self.is_done()

        return obs, reward_option, reward_action, done, None

    def rule_based(self):
        # Lane 4: ego_laneID == obstacle_laneID
        # Lane 5: ego_laneID == obstacle_laneUD
        # Lets suppose there are 2 cars is in the lane 4
        # IF rule based is called then full episode is run and  
        ego_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road = True)
        target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location(), project_to_road = True)

        if (abs(ego_waypoint.lane_id) == abs(target_waypoint.lane_id)):
            if self.ego_target() - 10 > 25:
                goal = 0
                action = random.randrange(1,3)
            else:
                if (self.ego_moving() -10 <= 10) or (self.ego_moving2() -10 <= 10):
                    goal = 0
                    action = 0
                else:
                    goal = 1
                    action = random.randrange(0,3)
        else:
            goal = 0
            if (self.ego_moving() - 6.0 < 12) or (self.ego_moving2() -10 <= 10):
                action = 0
            else:
                action = random.randrange(1,3)
        return goal, action


# ======================================================================================================================================
# DQN Agent
# ======================================================================================================================================


class DQNAgent:
    def __init__(self):
        self.state_size = 14
        self.goal_size = 2
        self.action_size = 3
        self.vector_size = 15
        self.EPISODES = 9999
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.writing_bool = False
        self.reading_bool = False
        self.network = Network_Model()

        
        self.gamma = 0.8    # discount rate
        # The exploration has to be decreased after each batch of training
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 68
        self.TAU = 0.1
        self.scores, self.scores_r, self.scores_g, self.scores_a, self.episodes, self.episodes_r, self.average,self.average_r, self.average_g, self.average_a = [], [], [], [], [], [], [], [], [], [] 
        self.resume_training = False 


        # Create model for meta goals
        self.meta_model =  self.network.meta_model(input_shape = (3, 14), goal_space = self.goal_size)

        # create model for lower level actions
        self.model = self.network.controller_model(goal_shape = (1,), input_shape = (3,14), action_space = self.action_size)


        if (self.resume_training == True):
            print("Training resumed...")
            self.load()

        if (self.reading_bool == True):
            print("Building replay memory from csv file...")
            print("Might take few seconds")
            self.reading_and_replay()


    # There has to be change in the way we store state information and read it.
    # Writing lookes fine   


    def remember(self, episode, state, history, goal, action, goal_reward, action_reward, next_state, next_history, done):
        self.memory.append((state, history, goal, action, goal_reward, action_reward, next_state, next_history, done))
        if episode >= 100:
            if len(self.memory) > self.train_start:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        if self.writing_bool:
            with open('state_file1.csv', 'a+', newline = '') as csv_log_file:
                self.log_writer = csv.writer(csv_log_file)
                self.log_writer.writerow([float(state[0][0]), float(state[0][1]), float(state[0][2]) , float(state[0][3]), float(state[0][4]),float(state[0][5]),float(state[0][6]),float(state[0][7]),float(state[0][8]),float(state[0][9]), float(state[0][10]), float(state[0][11]), float(state[0][12]),float(state[0][13])])

            with open('next_state_file1.csv', 'a+', newline = '') as csv_log_file:
                self.log_writer = csv.writer(csv_log_file)
                self.log_writer.writerow([float(next_state[0][0]), float(next_state[0][1]), float(next_state[0][2]), float(next_state[0][3]), float(next_state[0][4]), float(next_state[0][5]), float(next_state[0][6]), float(next_state[0][7]), float(next_state[0][8]), float(next_state[0][9]), float(next_state[0][10]), float(next_state[0][11]), float(next_state[0][12]), float(next_state[0][13])])


            with open('other_files1.csv', 'a+', newline = '') as csv_other_file:
                self.other_writer = csv.writer(csv_other_file)
                self.other_writer.writerow([int(goal), int(action), int(goal_reward), int(action_reward), bool(done)])        
        #if len(self.memory) > self.train_start:
            #if self.epsilon > self.epsilon_min:
                #self.epsilon *= self.epsilon_decay

        # Save the value of the 

    def reading_and_replay(self):
        if self.reading_bool:
            state_list = []
            single_state_list = []
            with open('state_file.csv') as log_file:
                log_reader = csv.reader(log_file)
                for row in log_reader:
                    single_state_list.append(float(row[0]))
                    single_state_list.append(float(row[1]))                     
                    single_state_list.append(float(row[2]))
                    single_state_list.append(float(row[3]))
                    single_state_list.append(float(row[4]))
                    single_state_list.append(float(row[5]))
                    single_state_list.append(float(row[6]))
                    single_state_list.append(float(row[7]))
                    single_state_list.append(float(row[8]))
                    single_state_list.append(float(row[9]))
                    single_state_list.append(float(row[10]))
                    single_state_list.append(float(row[11]))                                                                               
                    single_state_list.append(float(row[12]))
                    single_state_list.append(float(row[13]))
                    single_state_list = tuple(single_state_list)

                    single_state_list = np.reshape(single_state_list, [1, 14])
                    #print(len(single_state_list))
                    #print(single_state_list)
                    state_list.append(single_state_list)
                    #print(state_list)
                    #print(state_list[0])
                    single_state_list = [] 
                    
            next_state_list = []
            single_state_list = []
            with open('next_state_file.csv') as log_file:
                log_reader = csv.reader(log_file)
                for row in log_reader:
                    single_state_list.append(float(row[0]))
                    single_state_list.append(float(row[1]))                     
                    single_state_list.append(float(row[2]))
                    single_state_list.append(float(row[3]))
                    single_state_list.append(float(row[4]))
                    single_state_list.append(float(row[5]))
                    single_state_list.append(float(row[6]))                                                                                                    
                    single_state_list.append(float(row[7]))
                    single_state_list.append(float(row[8]))
                    single_state_list.append(float(row[9]))
                    single_state_list.append(float(row[10]))
                    single_state_list.append(float(row[11]))
                    single_state_list.append(float(row[12]))
                    single_state_list.append(float(row[13]))
                    single_state_list = tuple(single_state_list)

                    single_state_list = np.reshape(single_state_list, [1, 14])
                    #print(len(single_state_list))
                    #print(single_state_list)
                    next_state_list.append(single_state_list)
                    #print(state_list)
                    #print(state_list[0])
                    single_state_list = []  

            action_list = []
            goal_list = []
            action_reward_list = []
            goal_reward_list = []
            done_list = []

            with open('other_files.csv') as log_other:
                log_reader_2 = csv.reader(log_other)
                for row in log_reader_2:
                    goal_list.append(int(row[0])) 
                    action_list.append(int(row[1]))                   
                    goal_reward_list.append(int(row[2]))
                    action_reward_list.append(int(row[3]))
                    done_list.append(eval(row[4]))


            for i in range(len(state_list)):
                #print(state_list[i])
                experience = (state_list[i], goal_list[i], action_list[i], goal_reward_list[i], action_reward_list[i], next_state_list[i], done_list[i])
                #print(experience)
                self.memory.append(experience)
            print(len(self.memory))


    def select_goal(self, history):
        if np.random.random() <= self.epsilon:
            #print("random--goal")
            return random.randrange(self.goal_size)
            time.sleep(1/FPS)
        else:
            #print("policy--GOAL")
            goal = np.argmax(self.meta_model.predict(history))
            return goal

    
    def act(self, goal, history):
        _goal = []
        _goal.append(goal)
        _goal = tuple(_goal)
        _goal = np.reshape(_goal,[1,1])   
        if np.random.random() <= self.epsilon:
            time.sleep(0.035)
            return random.randrange(self.action_size)
        else:
            #print("In here")
            action = np.argmax(self.model.predict([_goal, history]))
            #print("FINE UNTIL HERE CHECK 0")
            return action


    def replay(self):
        if len(self.memory) < self.train_start:
            print(len(self.memory))
            print("Returning back")
            return

        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        history = np.zeros((self.batch_size, 3, 14))
        next_state = np.zeros((self.batch_size, self.state_size))
        next_history = np.zeros((self.batch_size, 3, 14))
        n_goal = np.zeros((self.batch_size, 1))        
        goal, action, goal_reward, action_reward, done = [], [], [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            history[i] = minibatch[i][1]
            goal.append(minibatch[i][2])
            np_goal = []
            np_goal.append(minibatch[i][2])
            np_goal = tuple(np_goal)
            np_goal = np.reshape(np_goal,[1,1])
            n_goal[i] = np_goal
            action.append(minibatch[i][3])
            goal_reward.append(minibatch[i][4])
            action_reward.append(minibatch[i][5])
            next_state[i] = minibatch[i][6]
            next_history[i] = minibatch[i][7]
            done.append(minibatch[i][8])


        # do batch predeiction to save speed for the goal network
        target = self.meta_model.predict(history)
        target_next = self.meta_model.predict(next_history)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][goal[i]] = goal_reward[i]
            else:
                target[i][goal[i]] = goal_reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.meta_model.fit(history, target, batch_size=self.batch_size, verbose=0)


        target2 = self.model.predict([n_goal, history])
        #print("FINE UNTIL HERE CHECK 1")
        target_next2 = self.model.predict([n_goal, next_history])
        #print("FINE UNTIL HERE CHECK 2")
        for i in range(self.batch_size):
            if done[i]:
                target2[i][action[i]] = action_reward[i]
            else:
                target2[i][action[i]] = action_reward[i] + self.gamma * (np.amax(target_next2[i]))

        #print("FINE UNTIL HERE CHECK 3")

        self.model.fit([n_goal, history], target2, batch_size=self.batch_size, verbose = 0) #, callbacks = [self.tensorboard] if log_this_step else None)



    def load(self):
        # Saving meta model
        episode = 1000
        self.meta_model = load_model("options_model"+str(episode)+".h5")
        self.model = load_model("planner_model"+str(episode)+".h5")

    def save(self, episode):
        # Saving the meta model
        print("Saving model and values...")
        self.meta_model.save("options_model"+str(episode)+".h5")
        # SAaving the planner model
        self.model.save("planner_model"+str(episode)+".h5")

        # Saving the values of the epsilon 
        f = open("HDQN_values.txt", "a")
        episode_str = str(episode)
        epsilon_str = str(self.epsilon)
        to_write = episode_str+" episode_epsilon_value"+" : "+ epsilon_str
        f.write("\n")
        f.write(to_write)
        f.close()

    pylab.figure(figsize=(18,9))
    def PlotReward(self, score, episode):
        self.scores_r.append(score)
        self.episodes_r.append(episode)
        self.average_r.append(sum(self.scores_r[-50:]) / len(self.scores_r[-50:]))
        pylab.plot(self.episodes_r, self.scores_r, 'C1')
        pylab.plot(self.episodes_r, self.average_r, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)
        dqn = 'Reward_Test2_LSTMHDQN_Double_01-1000'        

        try:
            pylab.savefig(dqn+".png")
        except OSError:
            pass

        return str(self.average_r[-1])[:5]            

'''
# Test the main
if __name__ == '__main__':
    agent = DQNAgent()
    env = CarEnv()
    episode_number = 1

    for episode in tqdm(range(episode_number + 10), ascii = True, unit = 'episodes'):

        env.collision_hist = []
        env.invasion_hist = []
        episode_reward = 0
        egoal_reward = 0
        eaction_reward = 0        
        step = 1
        state = env.reset()
        time.sleep(0.5)
        state = np.reshape(state, [1, agent.state_size])

        done = False
        episode_start = time.time()
        num_rand = random.randrange(0, 2)

        while True:
            if num_rand == 0:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.4, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
            else:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))                
            #env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = (random.randrange(8))/10, brake = 0.0, steer = 0.0))                                
            goal = np.argmax(agent.meta_model.predict(state))

            # Action selection based on weights
            _goal = []
            _goal.append(goal)             
            vector = tuple(np.concatenate([_goal, state[0]]))
            vector = np.reshape(vector, [1, 9])
            action = np.argmax(agent.model.predict(vector))

            #print("Into step")
            next_state, goal_reward, action_reward, done, _ = env.step(goal,action)
            #print("Out of step")
            egoal_reward += goal_reward
            eaction_reward += action_reward
            episode_reward += (goal_reward + action_reward)

            next_state = np.reshape(next_state, [1, agent.state_size])

            # Every step we update replay memory
            state = next_state
            step += 1

            if done:
                average = agent.PlotModel(episode_reward, episode)
                #agent.tensorboard.update_stats(reward_avg=average, episode_reward=episode_reward, options_reward=egoal_reward) #Episode_Reward=episode_reward, Options_reward=egoal_reward, Planner_Reward=eaction_reward)                    
                print('averag reward: ', average)
                print("Episode %d finished with total reward = %f." % (episode, episode_reward))
                for actor in env.actor_list:
                    actor.destroy()
                break
'''

if __name__ == '__main__':

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()
    episode_number = 1

    # Iterate over episodes
    for episode in tqdm(range(episode_number, episode_number + 1000), ascii=True, unit='episodes'):
        env.collision_hist = []
        env.invasion_hist = []
        episode_reward = 0
        egoal_reward = 0
        eaction_reward = 0
        state = env.reset()
        time.sleep(0.5)
        state = np.reshape(state, [1, agent.state_size])
        done = False
        episode_start = time.time()
        num_rand = random.randrange(0, 4)
        step = 0
        # History from states
        history = np.stack((state, state, state), axis=0)
        history = np.reshape([history], (1,3,14))

        while True:
            if num_rand == 0:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
            elif num_rand == 1:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.4, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
            elif num_rand == 2:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0))
            elif num_rand == 3:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0)) 

            # Rule Based Episodes                               
            if episode < 100:
                goal, action = env.rule_based()
            else:
                goal = agent.select_goal(history)                   
                action = agent.act(goal, history)                  


            next_state, goal_reward, action_reward, done, _ = env.step(goal,action)
            egoal_reward += goal_reward
            eaction_reward += action_reward
            episode_reward += (goal_reward + action_reward)
            next_state = np.reshape(next_state, [1, agent.state_size])
           

            # Appending new state to the history containing three time step experiences
            next_history = next_state.reshape(1,1,14)
            next_history = np.append(next_history, history[:,:2,:], axis = 1)
            next_history = np.reshape([next_history], (1,3,14))         

            # Every step we update replay memory
            agent.remember(episode, state, history, goal, action, goal_reward, action_reward, next_state, next_history, done)

            state = next_state
            history = next_history
            step += 1
            #print("state vector", state)
            env.epi_steps += 1

            if done:
                for actor in env.actor_list:
                    actor.destroy()
                #average = agent.PlotModel(episode_reward, egoal_reward, eaction_reward, episode)
                average = agent.PlotReward(episode_reward, episode)
                #agent.tensorboard.update_stats(reward_avg=average, episode_reward=episode_reward, options_reward=egoal_reward) #Episode_Reward=episode_reward, Options_reward=egoal_reward, Planner_Reward=eaction_reward)
                print('averag reward: ', average)            
                print("Episode %d finished in %d steps with total reward = %f." % (episode, env.epi_steps, episode_reward))
                if (episode % 50) == 0:
                    agent.save(episode)
                #agent.remember(episode_buffer)
                agent.replay()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                break
            #print("going into replay")
            # End of episode - destroy agents


