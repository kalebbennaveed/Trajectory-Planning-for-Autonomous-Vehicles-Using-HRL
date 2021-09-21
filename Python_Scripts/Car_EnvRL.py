from __future__ import print_function
# NORMAL FUNCTION IMPORTS

import glob
import os
import sys
import random
import time
import numpy as np
#import cv2
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
SECONDS_PER_EPISODE = 25
REPLAY_MEMORY_SIZE = 10_000
MIN_REWARD = -200
EPISODES = 100
AGGREGATE_STATS_EVERY = 10
FPS = 30
name = "HDQN"
# ========================================


# This import and code prevents "prediction function callback" error
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
# =====================================================================
# CLASS ENVIRONEMENT "CarEnv" for lane follow and lane change manuever
*** 
The environment now consists of the 10 - 12 vehicles which follow lane 
perform random lane change when required and 5 vehicels are at the front
and few at the back 

# =====================================================================
'''


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
        self.collision_counter = 0
        self.fail_counter = 0 
        self.initial_distance = 0 

        # Set up traffiic manager
        self.traffic_manager  = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        # This vehicle list include the information about the list of the vehicles spawned
        # This is only made to know which three vehicles are near the ego-car
        self.vehicles = []

        # Vehicles classified as obstacles
        self.obstacle = [] 

        # Vehicles classified as lane-follower cars
        self.control_cars = []

        # Vehicles classified as danger cars
        self.danger_cars = []

        # Vehicles classifies as slow cars
        self.slow_cars = []
        
        # ....     
        # For all scenerios, the behaviours dynamic and while the starting positions of the cars are random
        # In scenerio 1, traffic is dense and car has to navigate its way out either through lane follow, wait or lane change
        # In scenerio 2, traffic flow is medium
        # In scenerio 5, one car is not given any autpilot option, it acts as an obstace and blocks the left side of the road.

        # Commands on traffic manager
        '''
        tm = client.get_trafficmanager(port)
        tm_port = tm.get_port()
        for v in my_vehicles:
            v.set_autopilot(True,tm_port)
        danger_car = my_vehicles[0]
        tm.global_distance_to_leading_vehicle(5)
        tm.global_percentage_speed_difference(80)
        for v in my_vehicles: 
            tm.auto_lane_change(v,False)
        '''
        # 

        rand_num = random.randrange(0,4)

        # Dense traffic Simulation
        if rand_num == 0:

            # Spawning ego_vehicle
            self.transform_ego = carla.Transform(carla.Location(x=104.8, y=4.6, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
            self.vehicle_ego = self.world.spawn_actor(self.model_3, self.transform_ego)
            self.actor_list.append(self.vehicle_ego)
            self.controller = VehiclePIDController(self.vehicle_ego)

            # ======= Front Vehicles ======== 
            # Target vehicle 1
            self.transform1 = carla.Transform(carla.Location(x=215.0, y=6.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_1 = self.world.spawn_actor(self.model_4, self.transform1)
            #self.target_vehicle_1.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_1)
            self.actor_list.append(self.target_vehicle_1)
            self.obstacle.append(self.target_vehicle_1)
            # Target vehicle 2
            self.transform2 = carla.Transform(carla.Location(x=198.7, y=6.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_2= self.world.spawn_actor(self.model_4, self.transform2)
            #self.target_vehicle_2.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_2)                        
            self.actor_list.append(self.target_vehicle_2)
            self.control_cars.append(self.target_vehicle_2)
            # Target vehicle 3
            self.transform3 = carla.Transform(carla.Location(x=177.1, y=5.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_3 = self.world.spawn_actor(self.model_4, self.transform3)
            #self.target_vehicle_3.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_3)                        
            self.actor_list.append(self.target_vehicle_3)
            self.control_cars.append(self.target_vehicle_3)
            # Target vehicle 4
            self.transform4 = carla.Transform(carla.Location(x=158.8, y=5.4, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_4 = self.world.spawn_actor(self.model_4, self.transform4)
            #self.target_vehicle_4.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_4)                        
            self.actor_list.append(self.target_vehicle_4)
            self.control_cars.append(self.target_vehicle_4)
            # Target vehicle 5
            self.transform5 = carla.Transform(carla.Location(x=136.5, y=5.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_5 = self.world.spawn_actor(self.model_4, self.transform5)
            #self.target_vehicle_5.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_5)                        
            self.actor_list.append(self.target_vehicle_5)
            self.control_cars.append(self.target_vehicle_5)

            # ======= Back or side vehicles
            # Target vehicle 6
            self.transform6 = carla.Transform(carla.Location(x=61.1, y=3.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_6 = self.world.spawn_actor(self.model_4, self.transform6)
            #self.target_vehicle_6.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_6)                        
            self.actor_list.append(self.target_vehicle_6)
            self.control_cars.append(self.target_vehicle_6)
            # Target vehicle 7
            self.transform7 = carla.Transform(carla.Location(x=37.3, y=3.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_7= self.world.spawn_actor(self.model_4, self.transform7)
            #self.target_vehicle_7.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_7)                        
            self.actor_list.append(self.target_vehicle_7)
            self.control_cars.append(self.target_vehicle_7)            
            # Target vehicle 8
            self.transform8 = carla.Transform(carla.Location(x=200.6, y=9.7, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_8 = self.world.spawn_actor(self.model_4, self.transform8)
            #self.target_vehicle_8.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_8)                        
            self.actor_list.append(self.target_vehicle_8)
            self.danger_cars.append(self.target_vehicle_8)            
            # Target vehicle 9
            self.transform9 = carla.Transform(carla.Location(x=156.4, y=8.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_9 = self.world.spawn_actor(self.model_4, self.transform9)
            #self.target_vehicle_9.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_9)                        
            self.actor_list.append(self.target_vehicle_9)
            self.danger_cars.append(self.target_vehicle_9)            
            # Target vehicle 10
            self.transform10 = carla.Transform(carla.Location(x=37.2, y=7.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_10 = self.world.spawn_actor(self.model_4, self.transform10)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_10)                        
            self.actor_list.append(self.target_vehicle_10)
            self.danger_cars.append(self.target_vehicle_10)
            # Target vehicle 11
            self.transform11 = carla.Transform(carla.Location(x=54.2, y=7.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_11 = self.world.spawn_actor(self.model_4, self.transform11)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_11)                        
            self.actor_list.append(self.target_vehicle_11)
            self.danger_cars.append(self.target_vehicle_11) 
            # Target vehicle 12
            self.transform12 = carla.Transform(carla.Location(x=72.2, y=7.5, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_12 = self.world.spawn_actor(self.model_4, self.transform12)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_12)                        
            self.actor_list.append(self.target_vehicle_12)
            self.danger_cars.append(self.target_vehicle_12) 
            # Target vehicle 13
            self.transform13 = carla.Transform(carla.Location(x=89.2, y=7.7, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_13 = self.world.spawn_actor(self.model_4, self.transform13)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_13)                        
            self.actor_list.append(self.target_vehicle_13)
            self.danger_cars.append(self.target_vehicle_13) 
            # Target vehicle 14
            self.transform14 = carla.Transform(carla.Location(x=106.2, y=8.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_14 = self.world.spawn_actor(self.model_4, self.transform14)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_14)                        
            self.actor_list.append(self.target_vehicle_14)
            self.danger_cars.append(self.target_vehicle_14) 
            # Target vehicle 15
            self.transform15 = carla.Transform(carla.Location(x=123.2, y=8.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_15 = self.world.spawn_actor(self.model_4, self.transform15)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_15)                        
            self.actor_list.append(self.target_vehicle_15)
            self.danger_cars.append(self.target_vehicle_15) 
            # Target vehicle 16
            self.transform16 = carla.Transform(carla.Location(x=173.2, y=8.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_16 = self.world.spawn_actor(self.model_4, self.transform16)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_16)                        
            self.actor_list.append(self.target_vehicle_16)
            self.danger_cars.append(self.target_vehicle_16) 
                                                   
            
        # Medium traffic simulation
        elif rand_num == 1:
            # ======= Front Vehicles ======== 
            # Spawning ego_vehicle
            # Spawing car around x = 80.0
            self.transform_ego = carla.Transform(carla.Location(x=104.8, y=4.6, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
            self.vehicle_ego = self.world.spawn_actor(self.model_3, self.transform_ego)
            self.actor_list.append(self.vehicle_ego)
            self.controller = VehiclePIDController(self.vehicle_ego)

            # Target vehicle 3
            self.transform3 = carla.Transform(carla.Location(x=177.1, y=5.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_3 = self.world.spawn_actor(self.model_4, self.transform3)
            #self.target_vehicle_3.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_3)                        
            self.actor_list.append(self.target_vehicle_3)
            self.control_cars.append(self.target_vehicle_3)

            # Target vehicle 4
            self.transform4 = carla.Transform(carla.Location(x=158.8, y=5.4, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_4 = self.world.spawn_actor(self.model_4, self.transform4)
            #self.target_vehicle_4.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_4)                        
            self.actor_list.append(self.target_vehicle_4)
            self.control_cars.append(self.target_vehicle_4)

            # ======= Back or side vehicles
            # Target vehicle 6
            self.transform6 = carla.Transform(carla.Location(x=50.1, y=3.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_6 = self.world.spawn_actor(self.model_4, self.transform6)
            #self.target_vehicle_6.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_6)                        
            self.actor_list.append(self.target_vehicle_6)
            self.control_cars.append(self.target_vehicle_6)
            # Target vehicle 7
            self.transform7 = carla.Transform(carla.Location(x=37.3, y=3.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_7= self.world.spawn_actor(self.model_4, self.transform7)
            #self.target_vehicle_7.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_7)                        
            self.actor_list.append(self.target_vehicle_7)
            self.control_cars.append(self.target_vehicle_7)
            # Target vehicle 8
            self.transform8 = carla.Transform(carla.Location(x=185.6, y=9.5, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_8 = self.world.spawn_actor(self.model_4, self.transform8)
            #self.target_vehicle_8.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_8)                        
            self.actor_list.append(self.target_vehicle_8)
            self.danger_cars.append(self.target_vehicle_8)
            
            
            # Target vehicle 9
            self.transform9 = carla.Transform(carla.Location(x=99.0, y=8.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_9 = self.world.spawn_actor(self.model_4, self.transform9)
            #self.target_vehicle_9.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_9)                        
            self.actor_list.append(self.target_vehicle_9)
            self.danger_cars.append(self.target_vehicle_9)

            # Target vehicle 10
            self.transform10 = carla.Transform(carla.Location(x=37.2, y=7.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_10 = self.world.spawn_actor(self.model_4, self.transform10)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_10)                        
            self.actor_list.append(self.target_vehicle_10)
            self.danger_cars.append(self.target_vehicle_10)

        # Light Traffic
        elif rand_num == 2:

            # Spawning ego_vehicle
            self.transform_ego = carla.Transform(carla.Location(x=104.8, y=4.6, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
            self.vehicle_ego = self.world.spawn_actor(self.model_3, self.transform_ego)
            self.actor_list.append(self.vehicle_ego)
            self.controller = VehiclePIDController(self.vehicle_ego)

            # ======= Front Vehicles ======== 
            # Target vehicle 1
            self.transform1 = carla.Transform(carla.Location(x=215.0, y=6.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_1 = self.world.spawn_actor(self.model_4, self.transform1)
            #self.target_vehicle_1.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_1)
            self.actor_list.append(self.target_vehicle_1)
            self.slow_cars.append(self.target_vehicle_1)
            # Target vehicle 2
            self.transform2 = carla.Transform(carla.Location(x=198.7, y=6.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_2= self.world.spawn_actor(self.model_4, self.transform2)
            #self.target_vehicle_2.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_2)                        
            self.actor_list.append(self.target_vehicle_2)
            self.slow_cars.append(self.target_vehicle_2)
            # Target vehicle 3
            self.transform3 = carla.Transform(carla.Location(x=177.1, y=5.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_3 = self.world.spawn_actor(self.model_4, self.transform3)
            #self.target_vehicle_3.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_3)                        
            self.actor_list.append(self.target_vehicle_3)
            self.slow_cars.append(self.target_vehicle_3)
            # Target vehicle 4
            self.transform4 = carla.Transform(carla.Location(x=158.8, y=5.4, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_4 = self.world.spawn_actor(self.model_4, self.transform4)
            #self.target_vehicle_4.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_4)                        
            self.actor_list.append(self.target_vehicle_4)
            self.slow_cars.append(self.target_vehicle_4)
            # Target vehicle 5
            self.transform5 = carla.Transform(carla.Location(x=136.5, y=5.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_5 = self.world.spawn_actor(self.model_4, self.transform5)
            #self.target_vehicle_5.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_5)                        
            self.actor_list.append(self.target_vehicle_5)
            self.slow_cars.append(self.target_vehicle_5)
            # Target vehicle 17
            self.transform17 = carla.Transform(carla.Location(x=119.5, y=4.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_17 = self.world.spawn_actor(self.model_4, self.transform17)
            #self.target_vehicle_5.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_17)                        
            self.actor_list.append(self.target_vehicle_17)
            self.slow_cars.append(self.target_vehicle_17)

            # ======= Back or side vehicles
            # Target vehicle 6
            self.transform6 = carla.Transform(carla.Location(x=61.1, y=3.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_6 = self.world.spawn_actor(self.model_4, self.transform6)
            #self.target_vehicle_6.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_6)                        
            self.actor_list.append(self.target_vehicle_6)
            self.control_cars.append(self.target_vehicle_6)
            # Target vehicle 7
            self.transform7 = carla.Transform(carla.Location(x=37.3, y=3.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_7= self.world.spawn_actor(self.model_4, self.transform7)
            #self.target_vehicle_7.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_7)                        
            self.actor_list.append(self.target_vehicle_7)
            self.control_cars.append(self.target_vehicle_7)            
            # Target vehicle 8
            self.transform8 = carla.Transform(carla.Location(x=200.6, y=9.7, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_8 = self.world.spawn_actor(self.model_4, self.transform8)
            #self.target_vehicle_8.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_8)                        
            self.actor_list.append(self.target_vehicle_8)
            self.danger_cars.append(self.target_vehicle_8)            
            # Target vehicle 9
            self.transform9 = carla.Transform(carla.Location(x=156.4, y=8.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_9 = self.world.spawn_actor(self.model_4, self.transform9)
            #self.target_vehicle_9.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_9)                        
            self.actor_list.append(self.target_vehicle_9)
            self.danger_cars.append(self.target_vehicle_9)            
            # Target vehicle 10
            self.transform10 = carla.Transform(carla.Location(x=37.2, y=7.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_10 = self.world.spawn_actor(self.model_4, self.transform10)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_10)                        
            self.actor_list.append(self.target_vehicle_10)
            self.danger_cars.append(self.target_vehicle_10)
            # Target vehicle 11
            self.transform11 = carla.Transform(carla.Location(x=54.2, y=7.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_11 = self.world.spawn_actor(self.model_4, self.transform11)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_11)                        
            self.actor_list.append(self.target_vehicle_11)
            self.danger_cars.append(self.target_vehicle_11) 
            # Target vehicle 12
            self.transform12 = carla.Transform(carla.Location(x=72.2, y=7.5, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_12 = self.world.spawn_actor(self.model_4, self.transform12)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_12)                        
            self.actor_list.append(self.target_vehicle_12)
            self.danger_cars.append(self.target_vehicle_12) 
            # Target vehicle 13
            self.transform13 = carla.Transform(carla.Location(x=89.2, y=7.7, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_13 = self.world.spawn_actor(self.model_4, self.transform13)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_13)                        
            self.actor_list.append(self.target_vehicle_13)
            self.danger_cars.append(self.target_vehicle_13) 
            # Target vehicle 14
            self.transform14 = carla.Transform(carla.Location(x=106.2, y=8.0, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_14 = self.world.spawn_actor(self.model_4, self.transform14)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_14)                        
            self.actor_list.append(self.target_vehicle_14)
            self.danger_cars.append(self.target_vehicle_14) 
            # Target vehicle 15
            self.transform15 = carla.Transform(carla.Location(x=123.2, y=8.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_15 = self.world.spawn_actor(self.model_4, self.transform15)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_15)                        
            self.actor_list.append(self.target_vehicle_15)
            self.danger_cars.append(self.target_vehicle_15) 
            # Target vehicle 16
            self.transform16 = carla.Transform(carla.Location(x=173.2, y=8.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_16 = self.world.spawn_actor(self.model_4, self.transform16)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_16)                        
            self.actor_list.append(self.target_vehicle_16)
            self.danger_cars.append(self.target_vehicle_16) 
                                                   

        # Light Blocked traffic
        elif rand_num == 3:

            # Spawning ego vehicle (Using different model)
            # Spawning ego_vehicle
            self.transform_ego = carla.Transform(carla.Location(x=87.5, y=4.3, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
            self.vehicle_ego = self.world.spawn_actor(self.model_3, self.transform_ego)
            self.actor_list.append(self.vehicle_ego)
            self.controller = VehiclePIDController(self.vehicle_ego)            

            # ======= Front Vehicles ======== 
            # Target vehicle 1 (Fixed Obstacle)
            self.transform1 = carla.Transform(carla.Location(x=215.0, y=6.3, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_1 = self.world.spawn_actor(self.model_4, self.transform1)
            self.actor_list.append(self.target_vehicle_1)
            self.vehicles.append(self.target_vehicle_1)
            self.obstacle.append(self.target_vehicle_1)                       

            # Target vehicle 3
            self.transform3 = carla.Transform(carla.Location(x=177.1, y=5.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_3 = self.world.spawn_actor(self.model_4, self.transform3)
            #self.target_vehicle_3.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_3)                        
            self.actor_list.append(self.target_vehicle_3)
            self.control_cars.append(self.target_vehicle_3)

            # ======= Back or side vehicles
            # Target vehicle 6
            self.transform6 = carla.Transform(carla.Location(x=61.1, y=3.9, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_6 = self.world.spawn_actor(self.model_4, self.transform6)
            #self.target_vehicle_6.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_6)                        
            self.actor_list.append(self.target_vehicle_6)
            self.control_cars.append(self.target_vehicle_6)
            # Target vehicle 7
            self.transform7 = carla.Transform(carla.Location(x=37.3, y=3.2, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_7= self.world.spawn_actor(self.model_4, self.transform7)
            #self.target_vehicle_7.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_7)                        
            self.actor_list.append(self.target_vehicle_7)
            self.control_cars.append(self.target_vehicle_7)
            # Target vehicle 8
            self.transform8 = carla.Transform(carla.Location(x=200.6, y=9.7, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_8 = self.world.spawn_actor(self.model_4, self.transform8)
            #self.target_vehicle_8.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_8)            
            self.actor_list.append(self.target_vehicle_8)
            self.control_cars.append(self.target_vehicle_8)
            # Target vehicle 9
            self.transform9 = carla.Transform(carla.Location(x=156.4, y=8.6, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_9 = self.world.spawn_actor(self.model_4, self.transform9)
            #self.target_vehicle_9.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_9)            
            self.actor_list.append(self.target_vehicle_9)
            self.control_cars.append(self.target_vehicle_9)
            # Target vehicle 10
            self.transform10 = carla.Transform(carla.Location(x=90.4, y=8.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))
            self.target_vehicle_10 = self.world.spawn_actor(self.model_4, self.transform10)
            #self.target_vehicle_10.set_autopilot(True)
            self.vehicles.append(self.target_vehicle_10)            
            self.actor_list.append(self.target_vehicle_10)
            self.control_cars.append(self.target_vehicle_10) 

        

        # Do we still need this?
        #self.initial_distance = carla.Location.distance(self.vehicle.get_location(), self.target_vehicle.get_location()) - 20.0
        #self.safe_distance = carla.Transform(carla.Location(x=70.0, y=4.1, z=1.843102), carla.Rotation(pitch = 0.0000, yaw = 0.855823, roll = 0.00000))  

        #self.vehicle_ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        #self.target_vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake=0.0))
        time.sleep(4)

        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, carla.Transform(), attach_to = self.vehicle_ego)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event_lane: self.invasion_data(event_lane))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle_ego)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        time.sleep(0.01)

        self.episode_start = time.time()
        #self.vehicle_ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        obs = self.observe()
        return obs

    def get_speed(self, vehicle):
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


    def observe(self):

        self.feature_cars = self.select_vehicles()

        # State vector size = 15

        # Selected Target cars
        self.feature_car1 = self.feature_cars[0]
        self.feature_car2 = self.feature_cars[1]
        self.feature_car3 = self.feature_cars[2]

        # Waypoints for all the cars
        ego_waypoint = self.map.get_waypoint(self.vehicle_ego.get_location(), project_to_road = True)
        feature1_waypoint = self.map.get_waypoint(self.feature_car1.get_location(), project_to_road = True)
        feature2_waypoint = self.map.get_waypoint(self.feature_car2.get_location(), project_to_road = True)
        feature3_waypoint = self.map.get_waypoint(self.feature_car3.get_location(), project_to_road = True)

        self.observe_vector = []

        # ego car velocity
        v = self.vehicle_ego.get_velocity()
        ego_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.observe_vector.append(ego_kmh)

        # Ego Lane ID 
        ego_lane_id = abs(ego_waypoint.lane_id)
        self.observe_vector.append(ego_lane_id)

        # Feature car 1 specifications
        v_1 = self.feature_car1.get_velocity()
        feature1_kmh = int(3.6 * math.sqrt(v_1.x**2 + v_1.y**2 + v_1.z**2))
        self.observe_vector.append(feature1_kmh)    

        # Feature 1 lane ID
        feature1_lane_id = abs(feature1_waypoint.lane_id)
        self.observe_vector.append(feature1_lane_id)        

        if ego_lane_id == feature1_lane_id:
            if self.ego_target(self.feature_car1) - 10 <= 30:
                safe_ray = self.ego_target(self.feature_car1) - 10
            else:
                safe_ray = 10.0
            safe_ratio = safe_ray/10.0
        else:
            if self.ego_target(self.feature_car1) - 6.0 >= 10.0:
                safe_ray = 10
            else:
                safe_ray = self.ego_target(self.feature_car1) - 6.0
            safe_ratio = safe_ray/6.0


        self.observe_vector.append(safe_ray)
        self.observe_vector.append(safe_ratio)

        # Feature veicle # 2

        # Feature car 2 specifications
        v_2 = self.feature_car2.get_velocity()
        feature2_kmh = int(3.6 * math.sqrt(v_2.x**2 + v_2.y**2 + v_2.z**2))
        self.observe_vector.append(feature2_kmh)    

        # Feature 1 lane ID
        feature2_lane_id = abs(feature2_waypoint.lane_id)
        self.observe_vector.append(feature2_lane_id)        

        if ego_lane_id == feature2_lane_id:
            if self.ego_target(self.feature_car2) - 10 <= 30:
                safe_ray_2 = self.ego_target(self.feature_car2) - 10
            else:
                safe_ray_2 = 10.0
            safe_ratio_2 = safe_ray_2/10.0
        else:
            if self.ego_target(self.feature_car2) - 6.0 >= 10.0:
                safe_ray_2 = 10
            else:
                safe_ray_2 = self.ego_target(self.feature_car2) - 6.0
            safe_ratio_2 = safe_ray_2/6.0


        self.observe_vector.append(safe_ray_2)
        self.observe_vector.append(safe_ratio_2)

        # Feature veicle # 3
        
        # Feature car 3 specifications
        v_3 = self.feature_car3.get_velocity()
        feature3_kmh = int(3.6 * math.sqrt(v_3.x**2 + v_3.y**2 + v_3.z**2))
        self.observe_vector.append(feature3_kmh)    

        # Feature 1 lane ID
        feature3_lane_id = abs(feature3_waypoint.lane_id)
        self.observe_vector.append(feature3_lane_id)        

        if ego_lane_id == feature3_lane_id:
            if self.ego_target(self.feature_car3) - 10 <= 30:
                safe_ray_3 = self.ego_target(self.feature_car3) - 10
            else:
                safe_ray_3 = 10.0
            safe_ratio_3 = safe_ray_3/10.0
        else:
            if self.ego_target(self.feature_car3) - 6.0 >= 10.0:
                safe_ray_3 = 10
            else:
                safe_ray_3 = self.ego_target(self.feature_car3) - 6.0
            safe_ratio_3 = safe_ray_3/6.0


        self.observe_vector.append(safe_ray_3)
        self.observe_vector.append(safe_ratio_3)
        return tuple(self.observe_vector)

    def collision_data(self, event):
        self.collision_hist.append(event)

    # Change this aswell
    # Go through all the cars which are selected and see if any of there cars distance is below that level
    def collision_check(self):
        self.col_vehicles = self.select_vehicles()
        if len(self.collision_hist) != 0:
            self.collision_boolean = True
            self.is_alive = False
            return True
        else:
            for v in self.col_vehicles:
                if self.ego_target(v) <= 3:
                    self.collision_boolean = True
                    self.is_alive = False
                    return True
                else:
                    return False
        #elif self.ego_target() <= 3:
        #    self.collision_boolean = True
        #    self.is_alive = False
        #    return True           
        #else:
        #    return False

    def invasion_data(self,event_lane):
        self.invasion_hist.append(event_lane)

    def select_vehicles(self):
        '''
        This function is meant to return three vehicles which are closest to the ego_car and return
        there respective distances from the ego_car. The function simply goes over the list of the 
        vehicles in the list of the vehicles.
        '''

        first_min = 10000
        second_min = 10000
        third_min = 10000
        first_index = 0
        second_index = 0
        third_index = 0
        vehicles_index = []

        self.distances = []
        for v in self.vehicles:
            self.distances.append(carla.Location.distance(self.vehicle_ego.get_location(), v.get_location()))
        self.length = len(self.distances)
        for i in range(0,self.length):
            if self.distances[i] < first_min:
                third_min = second_min
                second_min = first_min
                first_min = self.distances[i]
                first_index = i
            elif self.distances[i] < second_min:
                third_min = second_min
                second_min = self.distances[i]
                second_index = i
            elif self.distances[i] < third_min:
                third_min = self.distances[i]
                third_index = i

        vehicles_index = [first_index, second_index, third_index]
        self.selected_car_1 = self.vehicles[first_index]
        self.selected_car_2 = self.vehicles[second_index]
        self.selected_car_3 = self.vehicles[third_index]

        return self.selected_car_1, self.selected_car_2, self.selected_car_3


    

    def ego_target(self, target_car):
        # Distance between target and ego car
        self.dist_ET = carla.Location.distance(self.vehicle_ego.get_location(),target_car.get_location())
        return self.dist_ET

    def ego_goal(self):
        # Distance between ego and goal destination
        goal_position = carla.Transform(carla.Location(x=215.0, y=6.3, z=1.843102), carla.Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000))
        dist_EG = carla.Location.distance(self.vehicle_ego.get_location(), goal_position.location)
        return dist_EG

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
        ego_location = self.vehicle_ego.get_location()
        ego_waypoint = self.map.get_waypoint(ego_location, project_to_road = True)
        #target_waypoint = self.map.get_waypoint(self.target_vehicle_ego.get_location(), project_to_road = True)
        if action == 0:
            '''
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
                self.vehicle_ego.apply_control(control)
            '''
            waypoint_final = ego_waypoint.next(15.0)[0]
            target_speed = 0.0
            to_draw_waypoint_final = ego_waypoint.next(7.0)[0]
            self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.0)
            #self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)
            self.world.debug.draw_point(to_draw_waypoint_final.transform.location, size = 0.3, life_time = 2.0)          
            self.world.debug.draw_line(ego_waypoint.transform.location, to_draw_waypoint_final.transform.location, thickness = 0.3, life_time = 2.0)                 
            atime = 0.65
            start_time = time.time()
            while time.time() < start_time + atime:
                control = self.controller.run_step(target_speed, waypoint_final)
                self.vehicle_ego.apply_control(control)                
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
           # self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle_ego.get_velocity()).x**2 + (self.vehicle_ego.get_velocity()).y**2 + (self.vehicle_ego.get_velocity()).z**2))), life_time = 2.0)
            self.world.debug.draw_point(waypoint_final.transform.location, size = 0.3, life_time = 2.0)
            #self.world.debug.draw_string(waypoint_final.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)            
            self.world.debug.draw_line(ego_waypoint.transform.location, waypoint_final.transform.location, thickness = 0.3, life_time = 2.0)            
            target_speed = 14
            # 0.2, 
            a2time = 0.60
            start_time_2 = time.time()
            while time.time() < start_time_2 + a2time:
                control = self.controller.run_step(target_speed, waypoint_final)
                self.vehicle_ego.apply_control(control)
            final_location = self.vehicle_ego.get_location() 
        
        
        # Fast Moving Forward    
        elif action == 2:
            #print("20m")
            waypoint_final = ego_waypoint.next(23.0)[0]
            self.world.debug.draw_point(ego_waypoint.transform.location, size = 0.3, life_time = 2.5)
            #self.world.debug.draw_string(ego_waypoint.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle_ego.get_velocity()).x**2 + (self.vehicle_ego.get_velocity()).y**2 + (self.vehicle_ego.get_velocity()).z**2))), life_time = 2.5)
            self.world.debug.draw_point(waypoint_final.transform.location, size = 0.3, life_time = 2.5)
            #self.world.debug.draw_string(waypoint_final.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt((self.vehicle.get_velocity()).x**2 + (self.vehicle.get_velocity()).y**2 + (self.vehicle.get_velocity()).z**2))), life_time = 2.0)            
            self.world.debug.draw_line(ego_waypoint.transform.location, waypoint_final.transform.location, thickness = 0.3, life_time = 2.5)            
            target_speed = 17
            # 0.40
            a3time = 0.78
            start_time_3 = time.time()
            while time.time() < start_time_3 + a3time:
                control = self.controller.run_step(target_speed, waypoint_final)
                self.vehicle_ego.apply_control(control)
            final_location = self.vehicle_ego.get_location()        
        
        # Check if collision occured then returned false
        status  = self.collision_check()
        return status

    def option2_action(self, action):
        # Lane Change Option
        status = False
        ego_location = self.vehicle_ego.get_location()
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
                self.vehicle_ego.apply_control(control)        
            final_location = self.vehicle_ego.get_location()
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
                self.vehicle_ego.apply_control(control)
            last_location = self.vehicle_ego.get_location()
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

        # Same reward for both the lanes
        # Each step penalty 
        
        ego_waypoint = self.map.get_waypoint(self.vehicle_ego.get_location(), project_to_road = True)
        #target_waypoint = self.map.get_waypoint(self.target_vehicle.get_location(), project_to_road = True)

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
        
        self.same_lane = []
        self.next_lane = []


        # New reward structure
        # Simpler reward structure
        self.chosen_vehicles = self.select_vehicles()

        # The car's status (Praise God for this idea!!!)
        for v in self.chosen_vehicles:
            # Vehicle waypoint
            v_waypoint = self.map.get_waypoint(v.get_location(), project_to_road = True)
            if (abs(ego_waypoint.lane_id) == abs(v_waypoint.lane_id)):
                self.same_lane.append(v)
            else:
                self.next_lane.append(v)

        for s in self.same_lane:
            if (self.ego_target(s) - 10 >= 30):
                if goal == 0:
                    reward_option += 40
                    if action == 0:
                        reward_action -= 40
            else:
                for n in self.next_lane:
                    if (self.ego_target(n) - 10 <= 10):
                        if goal == 0:
                            reward_option += 40
                            reward_action += -40 * math.exp((-(self.ego_target(s)) - 10)/10) + (collision_value * -100)
                        if goal == 1:
                            if self.option2_failed:
                                reward_option += -40
                            else:
                                reward_option -= -40 * math.exp((-(self.ego_target(s)) - 10)/10)+ (collision_value * -100)
                                reward_action -= -40 * math.exp((-(self.ego_target(s)) - 10)/10) + (collision_value * -100)

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
