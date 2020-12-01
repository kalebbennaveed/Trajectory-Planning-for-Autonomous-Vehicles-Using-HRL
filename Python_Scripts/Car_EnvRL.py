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

# Might not be used in this particular script
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
