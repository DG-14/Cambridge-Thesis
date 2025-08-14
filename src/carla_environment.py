import carla
from gymnasium import spaces
import random
import time
import datetime
import math
import os
import numpy as np
import tensorflow as tf
from collections import deque
import transforms3d
from src.waypoints import *
from src.utilies import classify_actor_type, compute_actor_risks_in_cone, log_episode_metrics,get_at_risk_actors,is_actor_in_group, log_episode_summary_stats_csv
from src.risk_models import ped_cyclists_injury_probability, car_passenger_injury_probability
# from src.CARLA_process_manager import restart_CARLA
from src.normalise_scores import normalise_step, normalise_terminal


class CarlaEnv:
    """
        Carla environment used for training RL agents for ethical decision-making or collision avoidance tasks.
    """
    def __init__(self,log_dir, **kwargs):
        """
        Initialize the Carla environment.

        Parameters:
        kwargs (dict): Dictionary of configuration parameters.
        """
        # Setting up Carla simulator
        self.client = carla.Client('localhost', 2000)
        self.CARLA_timeout_counter = 0
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()

        self.episode_max_time = kwargs['episode_timeout']
        self.episode_start_time = None
        self.actor_list = []

        self.simulation_step_count = 0
        self.last_harm_metrics = []
        self.episode_log = []

        self.log_dir = log_dir

        self.track_train_util = False

        if kwargs['track_training']:
            self.track_train_util = True
            self.train_util_dir = os.path.join("training_util", "baseline", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
            os.makedirs(self.train_util_dir, exist_ok=True)
            self.train_util_tracker = {}

        # curriculum mode
        self.curriculum_mode = kwargs['curriculum_mode']

        # Ego vehicle parameters
        self.a_x_max_acc = kwargs['a_x_max_acceleration']
        self.a_x_max_braking = kwargs['a_x_max_braking']
        self.a_y_max = kwargs['a_y_max']
        self.mass = kwargs['mass']
        self.mu = kwargs['mu']
        self.t_gear_change = kwargs['t_gear_change']

        # Chrono physics parameters
        self.chrono_enabled = kwargs['use_chrono']
        self.chrono_path = kwargs['chrono_path']
        self.vehicle_json = kwargs['vehicle_json']
        self.powertrain_json = kwargs['powertrain_json']
        self.tire_json = kwargs['tire_json']

        # Inputted frames parameters
        self.image_width = kwargs['image_width']
        self.image_height = kwargs['image_height']
        self.history_length = kwargs['history_length']
        self.knob = kwargs['knob_value']
        # Could set to several contious axis rather than discrete
        self.action_space = spaces.Discrete(9)  # Actions from 1 to 9 as defined
        
        # greyscale
        # self.observation_space = spaces.Box(
            # low=0, high=255, shape=(self.image_width, self.image_height, self.history_length), dtype=np.float32)
        
        # semantic
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.image_width, self.image_height, self.history_length), dtype=np.float32)

        
        self.state_buffer = None

        # Obstacle blueprints which are being strategically placed to restrict the potential driving space of the agent
        self.obstacle_bp = self.bp.filter("chainbarrierend")[0]

        # Road friction parameters
        friction_bp = self.bp.find('static.trigger.friction')
        extent = carla.Location(700.0, 700.0, 700.0)
        friction_bp.set_attribute('friction', str(self.mu))
        friction_bp.set_attribute('extent_x', str(extent.x))
        friction_bp.set_attribute('extent_y', str(extent.y))
        friction_bp.set_attribute('extent_z', str(extent.z))

        # Blueprints of ego vehicle etc.
        self.bp_ego = self.bp.filter(blueprints_dict['ego_car'][0])[0]
        # self.bp_ego.set_attribute('color', '0, 0, 255') # set blue color for ego car
        self.collision_bp = self.bp.find('sensor.other.collision')

        # self.camera_bp = self.bp.find('sensor.camera.rgb')
        self.camera_bp = self.bp.find('sensor.camera.semantic_segmentation')


        self.camera_bp.set_attribute("image_size_x", f"{self.image_width}")
        self.camera_bp.set_attribute("image_size_y", f"{self.image_height}")
        self.sp_sensors = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.ego = None
        self.collision_sensor = None
        self.camera = None
        self.spawned = None
        self.crossed = False
        self.cross_points = [0, 0, 0, 0]
        self.collisions = []
        self.collision_history = []
        self.collision_speed = 0
        self.cars = []
        self.motorcycles = []
        self.bicycles = []
        self.pedestrians1 = []
        self.pedestrians2 = []
        self.evaluate = False
        self.seed = None
        self.break_limit = 1

        # Difficulty default
        self.difficulty = None

        # Location
        self.location = None

        self.DEDUPLICATION_INTERVAL = kwargs['deduplication_interval']  # currently 1 second by default

        # Spawn obstacles in the environment
        for intersection, transforms in obstacles.items():
            for i, transform in enumerate(transforms):
                obstacle_spawn = transform
                self.world.spawn_actor(self.obstacle_bp, obstacle_spawn)
        self.world.spawn_actor(friction_bp, carla.Transform(
            carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0)))
        
    def set_difficulty(self,difficulty_level):

        self.difficulty = difficulty_level
        self._create_difficulty_levels()
        # print("Blergh: " + str(self.difficulty))


        return
    
    def get_summary_writer_from_agent(self,summary_writer_ref):
        self.summary_writer = summary_writer_ref

    def reset(self, evaluate,episode):
        """
        Reset the environment to start a new episode.

        Parameters:
        evaluate (bool): Flag indicating whether to evaluate the environment.

        Returns:
        np.array: Initial state of the environment.
        """

        self.episode = episode
        self.evaluate = evaluate

        self.train_util_tracker = {
                                    "episode":self.episode,
                                    "evaluate":self.evaluate,
                                    "difficulty":self.difficulty
                                   }
        

        # self.seed = random.randint(100, 400)
        self.seed = 42

        self.state_buffer = deque([np.zeros((self.image_width, self.image_height), dtype=np.float32)
                                   for _ in range(self.history_length)], maxlen=self.history_length)
        self.cars = []
        self.motorcycles = []
        self.bicycles = []
        self.pedestrians1 = []
        self.pedestrians2 = []
        self.crossed = False
        self.cross_points = [0, 0, 0, 0]
        self.collisions = []
        self.collision_history = []
        self.collision_log = []
        self.collided_actor_ids = set()
        self.steps_after_collision = 0
        
        self.last_harm_metrics = []  # clear for next episode

        if bool(self.actor_list):
            self.destroy_actors()

        # tell it what scenarios/traffic logic to use for each curriculum mode
        if self.curriculum_mode == 'baseline':
            self._generate_traffic_baseline(evaluate)
        elif self.curriculum_mode == 'fixed':
            self._generate_traffic_curriculum(evaluate)
        elif self.curriculum_mode == 'adaptive':
            self._generate_traffic_curriculum(evaluate)
        elif self.curriculum_mode == 'tscl':
            self._generate_traffic_curriculum(evaluate)

        # print("groups")
        # print(self.pedestrians1)
        # print(self.pedestrians2)
        # print(self.motorcycles)
        # print(self.bicycles)
        # print(self.cars)
        
        # print("refs")
        # print(self.start_actor_refs)

        # --- Collect all relevant actors for new scene ---
        # self.start_actor_refs = list(self.world.get_actors().filter('*walker*')) + self.motorcycles + self.bicycles + self.cars
        self.start_actor_refs = list([d['object'] for d in self.pedestrians1] + [d['object'] for d in self.pedestrians2]+[d['object'] for d in self.cars]+[d['object'] for d in self.motorcycles]+[d['object'] for d in self.bicycles])

        self.start_actor_data = [{'id': a.id, 'pos': (a.get_location().x, a.get_location().y), 'ref': a} for a in self.start_actor_refs]

        self.start_ego_transform = self.ego.get_transform()
        self.start_ego_pos = (self.start_ego_transform.location.x, self.start_ego_transform.location.y)
        self.start_ego_heading = np.radians(self.start_ego_transform.rotation.yaw)


        self.episode_start_time = datetime.datetime.now()
        return self._get_state()

    def step(self, action):
        """
        Perform the action and return the next state, reward, done flag, and additional info.

        Parameters:
        action (int): Action to be performed.

        Returns:
        tuple: (observation, reward, done, info)
        """
        self.simulation_step_count += 1

        self._convert_action_index_to_control(action)  # Perform selected action to ego car

        observation = self._get_state()
        done,termination_type= self._check_done()
        reward = self._compute_step_reward(self.seed) if done == False else 20 * self._compute_terminal_reward(self.seed)
        return observation, reward, done, {'termination_type':termination_type}
    
    def _create_difficulty_levels(self):

        if self.difficulty == 0:
            # Empty scene, no obstacles
            self.min_pedestrians_1 = 0
            self.min_pedestrians_2 = 0
            self.min_bicycles = 0
            self.min_vehicles = 0

            self.max_pedestrians_1 = 0
            self.max_pedestrians_2 = 0
            self.max_bicycles = 0
            self.max_vehicles = 0

        elif self.difficulty == 1:
            # Single vehicle
            self.min_pedestrians_1 = 0
            self.min_pedestrians_2 = 0
            self.min_bicycles = 0
            self.min_vehicles = 1

            self.max_pedestrians_1 = 0
            self.max_pedestrians_2 = 0
            self.max_bicycles = 0
            self.max_vehicles = 1

        elif self.difficulty == 2:
            # One pedestrian and one vehicle
            self.min_pedestrians_1 = 1
            self.min_pedestrians_2 = 0
            self.min_bicycles = 0
            self.min_vehicles = 1

            self.max_pedestrians_1 = 1
            self.max_pedestrians_2 = 0
            self.max_bicycles = 0
            self.max_vehicles = 1

        elif self.difficulty == 3:
            # Multiple types, low count
            self.min_pedestrians_1 = 1
            self.min_pedestrians_2 = 1
            self.min_bicycles = 0
            self.min_vehicles = 1

            self.max_pedestrians_1 = 2
            self.max_pedestrians_2 = 2
            self.max_bicycles = 1
            self.max_vehicles = 2

        elif self.difficulty == 4:
            # Moderately dense and diverse scene
            self.min_pedestrians_1 = 1
            self.min_pedestrians_2 = 1
            self.min_bicycles = 1
            self.min_vehicles = 2

            self.max_pedestrians_1 = 3
            self.max_pedestrians_2 = 2
            self.max_bicycles = 2
            self.max_vehicles = 3

        elif self.difficulty == 5:
            # High density, complex ethical scenario
            self.min_pedestrians_1 = 2
            self.min_pedestrians_2 = 2
            self.min_bicycles = 1
            self.min_vehicles = 3

            self.max_pedestrians_1 = 4
            self.max_pedestrians_2 = 3
            self.max_bicycles = 2
            self.max_vehicles = 4

    def _generate_traffic_baseline(self, evaluate):
        """
        Spawn the ego vehicle, other actors, and set up sensors.

        Parameters:
        evaluate (bool): Flag indicating whether to evaluate the environment.
        """
        self.evaluate = False if evaluate == 0 else True
        spawn_points_new = spawn_points()
        self.location = random.choice(list(spawn_points_new)[1:]) if not self.evaluate else list(spawn_points_new)[0]
        # self.location = "intersection4"
        print("Location: " + str(self.location))
        sp_ego = spawn_points_new[self.location]['ego_car']
        self.cross_points[0],  self.cross_points[1] = sp_ego.location.x, sp_ego.location.y

        control = carla.WalkerControl()
        control.direction.x = 0
        control.direction.y = 0
        control.direction.z = 0
        control.speed = 1.55  # 5.6 km/h

        if spawn_points_new[self.location]['pedestrian_direction'][0] == 0:
            control.direction.y = spawn_points_new[self.location]['pedestrian_direction'][1]
            self.cross_points[2] = spawn_points_new[self.location]['pedestrian_direction'][2]
        else:
            control.direction.x = spawn_points_new[self.location]['pedestrian_direction'][0]
            self.cross_points[3] = spawn_points_new[self.location]['pedestrian_direction'][2]

        self.spawned = False
        while not self.spawned:
            try:
                self.ego = self.world.spawn_actor(self.bp_ego, sp_ego)
                self.ego.set_simulate_physics(enabled=True)
                physics_control = self.ego.get_physics_control()
                physics_control.gear_switch_time = self.t_gear_change
                physics_control.mass = self.mass
                physics_control.center_of_mass = carla.Vector3D(x=-0.100000, y=0.000000, z=-0.350000)
                self.ego.apply_physics_control(physics_control)
                self.actor_list.append(self.ego)

                # self.spectator.set_transform(carla.Transform(sp_ego.location + carla.Location(z=12),carla.Rotation(pitch=-53, yaw=sp_ego.rotation.yaw)))

                
                self.spectator.set_transform(carla.Transform(sp_ego.location + carla.Location(z=5),carla.Rotation(pitch=-23, yaw=sp_ego.rotation.yaw)))

                self.camera = self.world.spawn_actor(self.camera_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.camera)
                self.camera.listen(lambda data: self._get_camera_image(data))

                self.collision_sensor = self.world.spawn_actor(self.collision_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.collision_sensor)
                self.collision_sensor.listen(lambda event: self._collision_data(event))

                if evaluate == 0:
                    
                    self.pedestrian_actors = []  # Add this before the loops

                    for i in range(random.randint(0, len(spawn_points_new[self.location]['pedestrians1']))):
                        # wildcard = random.choice(blueprints_dict['pedestrians1'])

                        wildcard = (blueprints_dict['pedestrians1'])[0]

                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians1'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians1.append(wildcard)
                        self.pedestrians1.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})


                    for i in range(random.randint(0, len(spawn_points_new[self.location]['pedestrians2']))):
                        # wildcard = random.choice(blueprints_dict['pedestrians2'])
                        wildcard = (blueprints_dict['pedestrians1'])[0]


                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians2'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians2.append(wildcard)
                        self.pedestrians2.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})




                    non_autopilot_actors_num = len(self.actor_list)

                    for i in range(random.randint(0, len(spawn_points_new[self.location]['bicycles']))):
                        wildcard = random.choice(blueprints_dict['bicycles'])
                        bp_bicycle = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['bicycles'][i]
                        bicycle = self.world.spawn_actor(bp_bicycle, spawn_point)
                        self.actor_list.append(bicycle)
                        # self.bicycles.append(wildcard)
                        self.bicycles.append({'object':bicycle,'id':bicycle.id,'bp':wildcard})


                    autopilot_actors_bicycle = len(self.actor_list)

                    for i in range(random.randint(0, len(spawn_points_new[self.location]['cars_motorcycles']))):
                        
                        spawn_point = spawn_points_new[self.location]['cars_motorcycles'][i]
                        
                        if random.random() < 0.6:  # 60% chance to spawn a car
                            wildcard = random.choice(blueprints_dict['cars'])
                            bp_car = self.bp.filter(wildcard)[0]
                            car = self.world.spawn_actor(bp_car, spawn_point)
                            self.actor_list.append(car)
                            # self.cars.append(wildcard)
                            self.cars.append({'object':car,'id':car.id,'bp':wildcard})

                        else:  # 40% chance to spawn a motorcycle
                            wildcard = random.choice(blueprints_dict['motorcycles'])
                            bp_motorcycle = self.bp.filter(wildcard)[0]
                            motorcycle = self.world.spawn_actor(bp_motorcycle, spawn_point)
                            self.actor_list.append(motorcycle)
                            # self.motorcycles.append(wildcard)
                            self.motorcycles.append({'object':motorcycle,'id':motorcycle.id,'bp':wildcard})


                else:
                    bp_red_car = self.bp.filter("tesla")[0]
                    # bp_red_car.set_attribute('color', '255, 0, 0') set Red color
                    bp_motorcycle = self.bp.filter("yamaha")[0]
                    bp_bicycle = self.bp.filter("gazelle")[0]
                    bp_ped_adult_man = self.bp.filter("0028")[0]
                    bp_ped_adult_mother = self.bp.filter("0008")[0]
                    bp_ped_child = self.bp.filter("0011")[0]
                    bp_ped_old_man = self.bp.filter("0017")[0]
                    sp_red_car = spawn_points_new[self.location]['cars_motorcycles'][0]
                    sp_motorcycle = spawn_points_new[self.location]['cars_motorcycles'][1]
                    sp_bicycle = spawn_points_new[self.location]['bicycles'][0]
                    sp_ped_adult_man = spawn_points_new[self.location]['pedestrians1'][0]

                    if evaluate == 1:
                        ped_adult_man = self.world.spawn_actor(bp_ped_adult_man, sp_ped_adult_man)
                        # self.pedestrians1.append("0028")
                        self.pedestrians1.append({'object':ped_adult_man,'id':ped_adult_man.id,'bp':"0028"})

                        self.actor_list.append(ped_adult_man)
                    else:
                        sp_ped_adult_man = spawn_points_new[self.location]['pedestrians2'][0]
                        # self.pedestrians2.append("0028")


                        sp_ped_adult_mother = spawn_points_new[self.location]['pedestrians2'][1]
                        # self.pedestrians2.append("0008")


                        sp_ped_child = spawn_points_new[self.location]['pedestrians2'][2]
                        # self.pedestrians2.append("0011")


                        sp_ped_old_man = spawn_points_new[self.location]['pedestrians2'][3]
                        # self.pedestrians2.append("0017")


                        ped_adult_man = self.world.spawn_actor(bp_ped_adult_man, sp_ped_adult_man)
                        self.pedestrians2.append({'object':ped_adult_man,'id':ped_adult_man.id,'bp':"0028"})

                        ped_adult_mother = self.world.spawn_actor(bp_ped_adult_mother, sp_ped_adult_mother)
                        self.pedestrians2.append({'object':ped_adult_mother,'id':ped_adult_mother.id,'bp':"0008"})

                        ped_child = self.world.spawn_actor(bp_ped_child, sp_ped_child)
                        self.pedestrians2.append({'object':ped_child,'id':ped_child.id,'bp':"0011"})

                        ped_old_man = self.world.spawn_actor(bp_ped_old_man, sp_ped_old_man)
                        self.pedestrians2.append({'object':ped_old_man,'id':ped_old_man.id,'bp':"0017"})


                        self.actor_list.append(ped_adult_man)
                        self.actor_list.append(ped_adult_mother)
                        self.actor_list.append(ped_child)
                        self.actor_list.append(ped_old_man)

                    non_autopilot_actors_num = len(self.actor_list)

                    bicycle = self.world.spawn_actor(bp_bicycle, sp_bicycle)
                    self.bicycles.append({'object':bicycle,'id':bicycle.id,'bp':"gazelle"})


                    autopilot_actors_bicycle = len(self.actor_list)

                    motorcycle = self.world.spawn_actor(bp_motorcycle, sp_motorcycle)
                    # self.motorcycles.append("yamaha")
                    self.motorcycles.append({'object':motorcycle,'id':motorcycle.id,'bp':"yamaha"})


                    red_car = self.world.spawn_actor(bp_red_car, sp_red_car)
                    # self.cars.append("tesla")
                    self.cars.append({'object':red_car,'id':red_car.id,'bp':"tesla"})


                    self.actor_list.append(red_car)
                    self.actor_list.append(motorcycle)
                    self.actor_list.append(bicycle)

                time.sleep(0.5)
                self.ego.apply_control(carla.VehicleControl(throttle=0, brake=0, manual_gear_shift=True, gear=3))
                self.ego.enable_constant_velocity(carla.Vector3D(60 / 3.6, 0, 0))
                time.sleep(0.5)
                self.ego.apply_control(carla.VehicleControl(manual_gear_shift=False))
                self.ego.disable_constant_velocity()

                if self.chrono_enabled:
                    self.ego.enable_chrono_physics(5000, 0.002, self.vehicle_json, self.powertrain_json, self.tire_json,self.chrono_path)

                for actor in self.actor_list[3:non_autopilot_actors_num]:
                    actor.apply_control(control)

                for actor in self.actor_list[non_autopilot_actors_num:autopilot_actors_bicycle]:
                    actor.enable_constant_velocity(carla.Vector3D(3.05, 0, 0))
                
                for actor in self.actor_list[autopilot_actors_bicycle:]:
                    random.seed(self.seed)
                    speed = random.uniform(5, 12) if not self.evaluate else 6.94  # 25 km/h for red car and motorcycle
                    actor.enable_constant_velocity(carla.Vector3D(speed, 0, 0))

                self.spawned = True

                # self.train_util_tracker["actor_list"] = self.actor_list
            
            except RuntimeError:
                print("Spawn failed, Retrying in 5s. Please check server connection and whether simulator is running.")
                self.CARLA_timeout_counter += 1
                time.sleep(5)
                self.destroy_actors()
                self.spawned = False
                if self.CARLA_timeout_counter > 4:
                    self.CARLA_timeout_counter = 0
                    # restart_CARLA()
            pass

    def _generate_traffic_curriculum(self, evaluate):
        """
        Spawn the ego vehicle, other actors, and set up sensors.

        Parameters:
        evaluate (bool): Flag indicating whether to evaluate the environment.
        """
        self.evaluate = False if evaluate == 0 else True
        spawn_points_new = spawn_points()
        self.location = random.choice(list(spawn_points_new)[1:]) if not self.evaluate else list(spawn_points_new)[0]
        # self.location = "intersection4"
        print("Location: " + str(self.location))
        sp_ego = spawn_points_new[self.location]['ego_car']
        self.cross_points[0],  self.cross_points[1] = sp_ego.location.x, sp_ego.location.y

        control = carla.WalkerControl()
        control.direction.x = 0
        control.direction.y = 0
        control.direction.z = 0
        control.speed = 1.55  # 5.6 km/h

        if spawn_points_new[self.location]['pedestrian_direction'][0] == 0:
            control.direction.y = spawn_points_new[self.location]['pedestrian_direction'][1]
            self.cross_points[2] = spawn_points_new[self.location]['pedestrian_direction'][2]
        else:
            control.direction.x = spawn_points_new[self.location]['pedestrian_direction'][0]
            self.cross_points[3] = spawn_points_new[self.location]['pedestrian_direction'][2]

        self.spawned = False
        while not self.spawned:
            try:
                self.ego = self.world.spawn_actor(self.bp_ego, sp_ego)
                self.ego.set_simulate_physics(enabled=True)
                physics_control = self.ego.get_physics_control()
                physics_control.gear_switch_time = self.t_gear_change
                physics_control.mass = self.mass
                physics_control.center_of_mass = carla.Vector3D(x=-0.100000, y=0.000000, z=-0.350000)
                self.ego.apply_physics_control(physics_control)
                self.actor_list.append(self.ego)

                # self.spectator.set_transform(carla.Transform(sp_ego.location + carla.Location(z=12),carla.Rotation(pitch=-53, yaw=sp_ego.rotation.yaw)))

                
                self.spectator.set_transform(carla.Transform(sp_ego.location + carla.Location(z=5),carla.Rotation(pitch=-23, yaw=sp_ego.rotation.yaw)))

                self.camera = self.world.spawn_actor(self.camera_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.camera)
                self.camera.listen(lambda data: self._get_camera_image(data))

                self.collision_sensor = self.world.spawn_actor(self.collision_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.collision_sensor)
                self.collision_sensor.listen(lambda event: self._collision_data(event))

                if evaluate == 0:
                    
                    self.pedestrian_actors = []  # Add this before the loops

                    for i in range(random.randint(min(self.min_pedestrians_1,len(spawn_points_new[self.location]['pedestrians1'])), min(self.max_pedestrians_1, len(spawn_points_new[self.location]['pedestrians1'])))):
                        # wildcard = random.choice(blueprints_dict['pedestrians1'])

                        wildcard = (blueprints_dict['pedestrians1'])[0]

                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians1'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians1.append(wildcard)
                        self.pedestrians1.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})


                    for i in range(random.randint(min(self.min_pedestrians_2,len(spawn_points_new[self.location]['pedestrians2'])), min(self.max_pedestrians_2, len(spawn_points_new[self.location]['pedestrians2'])))):
                        # wildcard = random.choice(blueprints_dict['pedestrians2'])
                        wildcard = (blueprints_dict['pedestrians1'])[0]


                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians2'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians2.append(wildcard)
                        self.pedestrians2.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})

                    non_autopilot_actors_num = len(self.actor_list)

                    for i in range(random.randint(min(self.min_bicycles,len(spawn_points_new[self.location]['bicycles'])), min(self.max_bicycles, len(spawn_points_new[self.location]['bicycles'])))):
                        wildcard = random.choice(blueprints_dict['bicycles'])
                        bp_bicycle = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['bicycles'][i]
                        bicycle = self.world.spawn_actor(bp_bicycle, spawn_point)
                        self.actor_list.append(bicycle)
                        # self.bicycles.append(wildcard)
                        self.bicycles.append({'object':bicycle,'id':bicycle.id,'bp':wildcard})


                    autopilot_actors_bicycle = len(self.actor_list)

                    for i in range(random.randint(min(self.min_vehicles,len(spawn_points_new[self.location]['cars_motorcycles'])), min(self.max_vehicles, len(spawn_points_new[self.location]['cars_motorcycles'])))):
                        
                        spawn_point = spawn_points_new[self.location]['cars_motorcycles'][i]
                        
                        if random.random() < 0.6:  # 60% chance to spawn a car
                            wildcard = random.choice(blueprints_dict['cars'])
                            bp_car = self.bp.filter(wildcard)[0]
                            car = self.world.spawn_actor(bp_car, spawn_point)
                            self.actor_list.append(car)
                            # self.cars.append(wildcard)
                            self.cars.append({'object':car,'id':car.id,'bp':wildcard})

                        else:  # 40% chance to spawn a motorcycle
                            wildcard = random.choice(blueprints_dict['motorcycles'])
                            bp_motorcycle = self.bp.filter(wildcard)[0]
                            motorcycle = self.world.spawn_actor(bp_motorcycle, spawn_point)
                            self.actor_list.append(motorcycle)
                            # self.motorcycles.append(wildcard)
                            self.motorcycles.append({'object':motorcycle,'id':motorcycle.id,'bp':wildcard})


                else:
                    
                    self.pedestrian_actors = []  # Add this before the loops

                    for i in range(random.randint(min(self.min_pedestrians_1,len(spawn_points_new[self.location]['pedestrians1'])), min(self.max_pedestrians_1, len(spawn_points_new[self.location]['pedestrians1'])))):
                        # wildcard = random.choice(blueprints_dict['pedestrians1'])

                        wildcard = (blueprints_dict['pedestrians1'])[0]

                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians1'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians1.append(wildcard)
                        self.pedestrians1.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})


                    for i in range(random.randint(min(self.min_pedestrians_2,len(spawn_points_new[self.location]['pedestrians2'])), min(self.max_pedestrians_2, len(spawn_points_new[self.location]['pedestrians2'])))):
                        # wildcard = random.choice(blueprints_dict['pedestrians2'])
                        wildcard = (blueprints_dict['pedestrians1'])[0]


                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['pedestrians2'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrian_actors.append(pedestrian)  # Track for deletion
                        # self.pedestrians2.append(wildcard)
                        self.pedestrians2.append({'object':pedestrian,'id':pedestrian.id,'bp':wildcard})

                    non_autopilot_actors_num = len(self.actor_list)

                    for i in range(random.randint(min(self.min_bicycles,len(spawn_points_new[self.location]['bicycles'])), min(self.max_bicycles, len(spawn_points_new[self.location]['bicycles'])))):
                        wildcard = random.choice(blueprints_dict['bicycles'])
                        bp_bicycle = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[self.location]['bicycles'][i]
                        bicycle = self.world.spawn_actor(bp_bicycle, spawn_point)
                        self.actor_list.append(bicycle)
                        # self.bicycles.append(wildcard)
                        self.bicycles.append({'object':bicycle,'id':bicycle.id,'bp':wildcard})


                    autopilot_actors_bicycle = len(self.actor_list)

                    for i in range(random.randint(min(self.min_vehicles,len(spawn_points_new[self.location]['cars_motorcycles'])), min(self.max_vehicles, len(spawn_points_new[self.location]['cars_motorcycles'])))):
                        
                        spawn_point = spawn_points_new[self.location]['cars_motorcycles'][i]
                        
                        if random.random() < 0.6:  # 60% chance to spawn a car
                            wildcard = random.choice(blueprints_dict['cars'])
                            bp_car = self.bp.filter(wildcard)[0]
                            car = self.world.spawn_actor(bp_car, spawn_point)
                            self.actor_list.append(car)
                            # self.cars.append(wildcard)
                            self.cars.append({'object':car,'id':car.id,'bp':wildcard})

                        else:  # 40% chance to spawn a motorcycle
                            wildcard = random.choice(blueprints_dict['motorcycles'])
                            bp_motorcycle = self.bp.filter(wildcard)[0]
                            motorcycle = self.world.spawn_actor(bp_motorcycle, spawn_point)
                            self.actor_list.append(motorcycle)
                            # self.motorcycles.append(wildcard)
                            self.motorcycles.append({'object':motorcycle,'id':motorcycle.id,'bp':wildcard})

                time.sleep(0.5)
                self.ego.apply_control(carla.VehicleControl(throttle=0, brake=0, manual_gear_shift=True, gear=3))
                self.ego.enable_constant_velocity(carla.Vector3D(60 / 3.6, 0, 0))
                time.sleep(0.5)
                self.ego.apply_control(carla.VehicleControl(manual_gear_shift=False))
                self.ego.disable_constant_velocity()

                if self.chrono_enabled:
                    self.ego.enable_chrono_physics(5000, 0.002, self.vehicle_json, self.powertrain_json, self.tire_json,self.chrono_path)

                for actor in self.actor_list[3:non_autopilot_actors_num]:
                    actor.apply_control(control)

                for actor in self.actor_list[non_autopilot_actors_num:autopilot_actors_bicycle]:
                    actor.enable_constant_velocity(carla.Vector3D(3.05, 0, 0))
                
                for actor in self.actor_list[autopilot_actors_bicycle:]:
                    random.seed(self.seed)
                    speed = random.uniform(5, 12) if not self.evaluate else 6.94  # 25 km/h for red car and motorcycle
                    actor.enable_constant_velocity(carla.Vector3D(speed, 0, 0))

                self.spawned = True

                # self.train_util_tracker["actor_list"] = self.actor_list
            
            except RuntimeError:
                print("Spawn failed, Retrying in 5s. Please check server connection and whether simulator is running.")
                self.CARLA_timeout_counter += 1
                time.sleep(5)
                self.destroy_actors()
                self.spawned = False
                if self.CARLA_timeout_counter > 4:
                    self.CARLA_timeout_counter = 0
                    # restart_CARLA()
            pass


    def _convert_action_index_to_control(self, action):
        """
        Map the action index to vehicle control commands.

        Parameters:
        action (int): Action index.
        """

        # Special test case for collision debugging
        # action = 0
        # if action == 0:  # No Action
        #     self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        if action == 1:  # No Action
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        elif action == 2:  # Braking
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=self.break_limit))
        elif action == 3:  # Accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        elif action == 4:  # Turning left
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.8, brake=0.0))
        elif action == 5:  # Turning left and accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.8, brake=0.0))
        elif action == 6:  # Turning left and braking
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.8, brake=self.break_limit))
        elif action == 7:  # Turning right
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.8, brake=0.0))
        elif action == 8:  # Turning right and accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.8, brake=0.0))
        elif action == 9:  # Turning right and braking
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.8, brake=self.break_limit))

    def _get_state(self):
        """
        Get the current state of the environment.

        Returns:
        np.array: State buffer as an array.
        """
        # Stack frames along the last dimension to form (84, 84, 4)
        state = np.stack(list(self.state_buffer), axis=-1)
        # state = np.concatenate(list(self.state_buffer), axis=-1) 
        # print("Final stacked state shape:", state.shape)

        return state

    # semantic version
    def _get_camera_image(self, image):
        # Convert CARLA image to numpy array (semantic class IDs)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        semantic_map = array[:, :, 2]  # Red channel = class ID
        semantic_map = semantic_map.astype(np.float32) / 255.0  # Normalize to [0, 1]

        self.state_buffer.append(semantic_map)


    # greyscale version
    # def _get_camera_image(self, data):
    #     """
    #     Convert the image from the camera sensor to the state observation and add it to state buffer.
    #     """
    #     raw_data = np.frombuffer(data.raw_data, dtype=np.uint8)
    #     raw_data = np.reshape(raw_data, (data.height, data.width, 4))
    #     raw_data = raw_data[:, :, :3]  # Drop alpha channel (RGBA → RGB)

    #     # Convert to Tensor before any TensorFlow operations
    #     raw_tensor = tf.convert_to_tensor(raw_data, dtype=tf.uint8)

    #     image = tf.image.rgb_to_grayscale(raw_tensor)
    #     image = tf.image.resize(image, [self.image_width, self.image_height])
    #     image = tf.cast(image, tf.float32)

    #     # Safe squeeze
    #     image = tf.squeeze(image, axis=-1)

    #     # Convert back to NumPy if needed for buffer
    #     image = image.numpy()

    #     # ✅ Add this:
    #     # image = np.expand_dims(image, axis=-1)  # Shape now (84, 84, 1)

    #     # ✅ Add logging here
    #     # print("Processed image shape:", image.shape)

    #     self.state_buffer.append(image)


    def add_noise_to_image(self, image, variance=1.4):
        noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=(4, 84, 84))
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255)

    # for greyscale
    # def get_noisy_image(self):
    #     # method to get noisy image
    #     images = self.state_buffer
    #     noisy_image = self.add_noise_to_image(images, variance=1.4)
    #     return np.stack(list(noisy_image), axis=-1)

    # for semantic
    def get_noisy_image(self):
        return np.copy(self.state_buffer)


    def change_lighting_conditions(self, weather_params):
        weather = carla.WeatherParameters(
            cloudiness=weather_params['cloudiness'],
            sun_altitude_angle=weather_params['sun_altitude_angle']
        )
        self.world.set_weather(weather)
        time.sleep(1)
        pass

    def _collision_data(self, event):
        """
        Handle collision data from the collision sensor and log summary info.

        Parameters:
        event (carla.CollisionEvent): Collision event data.
        """
        speed = self._get_speed() * 3.6  # Convert m/s to km/h
        self.collision_speed = speed
        wildcard = event.other_actor.type_id
        found = False
        categories = ['cars', 'motorcycles', 'bicycles', 'pedestrians1', 'pedestrians2']

        # Initialize logging structures
        if not hasattr(self, 'collision_log'):
            self.collision_log = []
        if not hasattr(self, 'last_collision_time'):
            self.last_collision_time = {}
        if not hasattr(self, 'collided_actor_ids'):
            self.collided_actor_ids = set()


        actor_id = event.other_actor.id
        actor_type = classify_actor_type(event.other_actor.type_id)

        now = datetime.datetime.now()

        for category in categories:
            for actor in blueprints_dict[category]:
                if wildcard.find(actor) != -1:
                    if not self.evaluate:
                        if category == 'pedestrians1':
                            for p in [d['bp'] for d in self.pedestrians1]:
                                self._collision_history(p)
                        elif category == 'pedestrians2':
                            for p in [d['bp'] for d in self.pedestrians2]:
                                self._collision_history(p)
                        else:
                            self._collision_history(actor)
                    elif wildcard == "0011" or wildcard == "0008":
                        self._collision_history("0011")
                        self._collision_history("0008")
                    else:
                        self._collision_history(actor)

                    # Deduplication check
                    last_time = self.last_collision_time.get(actor_id)
                    if last_time is None or (now - last_time).total_seconds() > self.DEDUPLICATION_INTERVAL:
                        self.collision_log.append({
                            "time": now,
                            "speed": speed,
                            "actor": actor,
                            "actor_id": actor_id,
                            "actor_type": classify_actor_type(actor)
                        })
                        self.last_collision_time[actor_id] = now
                        self.collided_actor_ids.add(actor_id)

                    found = True
                    break
            if found:
                break

        if not found:
            self._collision_history("static")
            last_time = self.last_collision_time.get("static")
            if last_time is None or (now - last_time).total_seconds() > self.DEDUPLICATION_INTERVAL:
                self.collision_log.append({
                    "time": now,
                    "speed": speed,
                    "actor": "static",
                    "actor_type": "static"
                })
                self.last_collision_time["static"] = now


        # Optional: only attempt to destroy if in pedestrian category
        if actor_type in ['child', 'old_adult', 'regular_adult']:
            for ped in getattr(self, 'pedestrian_actors', []):
                if ped.id == actor_id:
                    # print(f"Destroying pedestrian actor {ped.id} after collision.")
                    ped.destroy()
                    self.pedestrian_actors.remove(ped)
                    break

    def _collision_history(self, item):
        if item not in self.collisions:
            self.collisions.append(item)
        pass

    def get_collision_summary(self):
        """
        Returns a summary of all collisions that occurred during the episode.

        Returns:
            dict: {
                'total_collisions': int,
                'average_speed': float,
                'actor_types': dict (actor type -> count)
            }
        """
        if not hasattr(self, 'collision_log') or not self.collision_log:
            return {
                'total_collisions': 0,
                'average_speed': 0.0,
                'actor_IDs': {},
                "actor_type" : {}
            }

        total_collisions = len(self.collision_log)
        total_speed = sum(event['speed'] for event in self.collision_log)
        average_speed = total_speed / total_collisions if total_collisions > 0 else 0.0

        actor_IDs = {}
        for event in self.collision_log:
            actor = event['actor']
            if actor in actor_IDs:
                actor_IDs[actor] += 1
            else:
                actor_IDs[actor] = 1

        actor_types = {}
        for event in self.collision_log:
            actor_type = event['actor_type']
            if actor_type in actor_types:
                actor_types[actor_type] += 1
            else:
                actor_types[actor_type] = 1

        # self.train_util_tracker["collisions"] = {
        #     'total_collisions': total_collisions,
        #     'average_speed': average_speed,
        #     'actor_IDs' : actor_IDs,
        #     'actor_types' : actor_types
        #     }


        return {
            'total_collisions': total_collisions,
            'average_speed': average_speed,
            'actor_IDs' : actor_IDs,
            'actor_types' : actor_types
        }

    def get_collision_history(self):
        # For evaluation scenario only.
        if not bool(self.collisions):
            return 1  # no collision
        elif self.collisions[0] == "static":
            return 2  # self-sacrifice
        elif self.collisions[0] == "tesla":
            return 3  # the red car
        elif self.collisions[0] == "yamaha":
            return 4  # motorcycle
        elif self.collisions[0] == "gazelle":
            return 5  # bicycle
        elif self.collisions[0] == "0028":
            return 6  # adult man
        elif self.collisions[0] == "0008" or self.collisions[0] == "0011":
            return 7  # mother + child
        elif self.collisions[0] == "0017":
            return 8  # old man
        else:
            print("something weird!")
            return None
        
    def _compute_step_reward(self, seed):
        """
        Compute the reward for non-terminal steps using risk-cone-filtered injury metrics.

        Parameters:
        seed (int): Random seed for reproducibility of age values.

        Returns:
        float: Step reward.
        """
        step_reward = 0
        speed = self._get_speed() * 3.6
        random.seed(seed)

        # Random ages for probabilistic harm modeling
        r1 = random.randint(20, 51)
        r2 = random.randint(20, 51)
        r3 = random.randint(5, 10)
        r4 = random.randint(60, 70)
        r5 = random.randint(20, 51)
        r6 = random.randint(5, 10)
        r7 = random.randint(60, 70)
        r8 = random.randint(20, 51)

        x = 1 - self.knob  # ethical blending knob

        ego_prob = []
        other_prob = []
        harm_values = []

        a_x, a_y = self.get_acceleration()

        # Ego position and heading
        ego_transform = self.ego.get_transform()
        ego_pos = (ego_transform.location.x, ego_transform.location.y)
        ego_heading = np.radians(ego_transform.rotation.yaw)

        # --- Collect all relevant actors ---
        # Check this works for cars

        # Optimise this whole section

        actor_refs = list(self.world.get_actors().filter('*walker*')) +[d['object'] for d in self.cars]+[d['object'] for d in self.motorcycles]+[d['object'] for d in self.bicycles]
        
        # print("Actor Refs: ")
        # print(actor_refs)
        actor_data = [{'id': a.id, 'pos': (a.get_location().x, a.get_location().y), 'ref': a} for a in actor_refs]

        # --- Risk cone filter ---
        risk_dict = compute_actor_risks_in_cone(ego_pos, ego_heading, actor_data)
        # print("Risk Dict: ")
        # print(risk_dict)

        for actor_id, cone_risk in risk_dict.items():
            actor = self.world.get_actor(actor_id)
            # print("Actor: ")
            # print(actor)
            # print(actor.id)
            # print(actor.type_id.split('.')[-1])

            # optimise this so its not converting each time

            # Motorcycle
            if is_actor_in_group(actor, self.motorcycles):
                # print("MOTORBIKE")

                age = r1 if not self.evaluate else 40
                prob = ped_cyclists_injury_probability(a=-4.555, b=0.040, c=0.011, vel=speed, age=age)
                harm = cone_risk * prob
                harm_values.append(harm)
                other_prob.append(harm)

            # Bicycle
            elif is_actor_in_group(actor, self.bicycles):
                # print("BIKE")

                age = r2 if not self.evaluate else 25
                prob = ped_cyclists_injury_probability(a=-7.467, b=0.079, c=0.047, vel=speed, age=age)
                harm = cone_risk * prob
                harm_values.append(harm)
                other_prob.append(harm)

            # Pedestrian
            elif is_actor_in_group(actor, self.pedestrians1) or is_actor_in_group(actor, self.pedestrians2):
                # print("PED")
                pedestrian_type = actor.type_id.split('.')[-1]
                if pedestrian_type in pedestrians_age_gp['child']:
                    age = r3 if pedestrian_type in self.pedestrians1 else r6
                elif pedestrian_type in pedestrians_age_gp['old']:
                    age = r4 if pedestrian_type in self.pedestrians1 else r7
                else:
                    age = r5 if pedestrian_type in self.pedestrians1 else r8 if not self.evaluate else 35
                prob = ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, vel=speed, age=age)
                # print("Cone Risk: " + str(cone_risk) + " prob: " + str(prob) + " speed: " + str(speed))
                harm = cone_risk * prob
                harm_values.append(harm)
                other_prob.append(harm)

            # Car collision (affects both ego and actor)
            elif is_actor_in_group(actor, self.cars):
                # print("CAR")
                prob = car_passenger_injury_probability(a=-6.6986, b=0.0333, vel=speed)
                harm = cone_risk * prob
                harm_values.append(harm)
                other_prob.append(harm)
                ego_prob.append(harm)

        # Harm statistics
        harm_total = sum(harm_values) if harm_values else 0.0
        # print("Harm Total: " + str(harm_total))
        harm_max = max(harm_values) if harm_values else 0.0
        harm_var = np.var(harm_values) if harm_values else 0.0
        harm_avg = np.average(harm_values) if harm_values else 0.0

        # Inverted injury scores for reward
        # inversed_prob = [1 / p for p in other_prob if p > 0]
        # inversed_prob_ego = [1 / p for p in ego_prob if p > 0]

        # Acceleration constraints
        a = 0 if self.a_x_max_braking < a_x < self.a_x_max_acc else 1
        b = 0 if a_y < abs(self.a_y_max) else 1

        # Final reward
        # current: acceleration laterally * other actor prob divided by knob) + (ego prob divided by knob)
        # step_reward = ((1 - 0.2 * a) * (1 - 0.8 * b) / 10000) * ((sum(inversed_prob) / x) + (sum(inversed_prob_ego) / self.knob))

        # New Final Reward: inverse acceleration - total risk - variance risk - max risk
        step_reward = (1 * ((1 - 0.2 * a) * (1 - 0.8 * b) / 10000)) -1*(harm_avg) -1*(harm_var) -1*(harm_max)
        scaled_step_reward = normalise_step(step_reward,self.difficulty)

        if self.evaluate == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('step_reward', step_reward, step=self.simulation_step_count)

        # Harm log metrics
        self.last_harm_metrics.append({
            'simulation_step': self.simulation_step_count,
            'episode': self.episode,
            'evaluation':self.evaluate,
            'harm_total': harm_total,
            'harm_max': harm_max,
            'harm_var': harm_var,
            'harm_avg': harm_avg,
            'risk_dict': risk_dict,  # Include full per-actor risk data
            'step_reward': step_reward,
            'step_reward_scaled': scaled_step_reward
        })


        # want at each step to score based on:
        # want to minise aggresive/excess movement
        # total risk (want to minimse) - estimated harm * risk coefficient, 
        # variance in risk (want to minimise) - estimated harm * risk coefficient
        # Risk to the ego vehicle
        # need step score to be quite small compared to terminal score

        # In future maybe add some static obstacles

        # print("Step Reward: " + str(step_reward))
        # print("Scaled Step Reward: " + str(scaled_step_reward))
        
        return scaled_step_reward



    def _compute_terminal_reward(self, seed):
        """
        Compute the reward at the end of the episode.

        Parameters:
        seed (int): Random seed for generating the reward.

        Returns:
        float: Terminal reward.
        """

        ego_prob = []
        other_prob = []

        if self._check_crossed() and not bool(self.collisions): #  eq. 11
            
            # revise this
            terminal_reward = 5
        elif not self._check_crossed() and not bool(self.collisions):
            
            # timeout case
            terminal_reward = 2
        
        else:

            terminal_reward = 0
            speed = self.collision_speed
            random.seed(seed)
            r1 = random.randint(20, 51)
            r2 = random.randint(20, 51)
            r3 = random.randint(5, 10)
            r4 = random.randint(60, 70)
            r5 = random.randint(20, 51)
            r6 = random.randint(5, 10)
            r7 = random.randint(60, 70)
            r8 = random.randint(20, 51)

            x = 1 - self.knob  # eq. 12


            # print("------------------------------------")
            # print("Collisions: " + str(self.collisions))
            # print("cars: " + str(self.cars))
            # print("motorcycles: " + str(self.motorcycles))
            # print("bicycles: " + str(self.bicycles))
            # print("pedestrians1: " + str(self.pedestrians1))
            # print("pedestrians2: " + str(self.pedestrians2))
            # print("------------------------------------")



            for i in self.collisions:
                if any(car['bp'] == i for car in self.cars):
                    prob = car_passenger_injury_probability(a=-6.6986, b=0.0333, vel=speed)
                    other_prob.append(prob)
                    ego_prob.append(prob)
                    # print("Car Collision: " + str(i) + " with prob: " + str(prob))
                elif any(m['bp'] == i for m in self.motorcycles):
                    age = r1 if not self.evaluate else 40
                    prob = ped_cyclists_injury_probability(a=-4.555, b=0.040, c=0.011, vel=speed, age=age)
                    other_prob.append(prob)
                elif any(b['bp'] == i for b in self.bicycles):
                    age = r2 if not self.evaluate else 25
                    prob = ped_cyclists_injury_probability(a=-7.467, b=0.079, c=0.047, vel=speed, age=age)
                    other_prob.append(prob)
                elif any(p['bp'] == i for p in self.pedestrians1):
                    if i in pedestrians_age_gp.get('child', []):
                        age = r3
                    elif i in pedestrians_age_gp.get('old', []):
                        age = r4
                    else:
                        age = r5 if not self.evaluate else 38
                    prob = ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, vel=speed, age=age)
                    other_prob.append(prob)
                    # print("Pedestrian Collision: " + str(i) + " with age: " + str(age) + " and prob: " + str(prob))
                elif any(p['bp'] == i for p in self.pedestrians2):
                    if i in pedestrians_age_gp.get('child', []):
                        age = r6 if not self.evaluate else 7
                    elif i in pedestrians_age_gp.get('old', []):
                        age = r7 if not self.evaluate else 65
                    else:
                        age = r8 if not self.evaluate else 35
                    prob = ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, vel=speed, age=age)
                    other_prob.append(prob)
                    # print("Pedestrian Collision: " + str(i) + " with age: " + str(age) + " and prob: " + str(prob))
                elif i == "static":
                    prob = car_passenger_injury_probability(a=-5.7641, b=0.0239, vel=speed)  # Single car crash
                    ego_prob.append(1.5*prob)
                    # print("Static Collision: " + str(i) + " with prob: " + str(prob))


            # print("Ego Prob: " + str(ego_prob))
            # print("Other Prob: " + str(other_prob))
            
            # Old scoring: weight * weighted ego prob + weighted ego prob
            # terminal_reward = -400 * ((sum(other_prob) * x) + (sum(ego_prob)*self.knob))  # eq. 11
                    

        # In this case, we want:
        # the final harm to ego vehicle (minimise)
        # final harm to actors (minimise)

        at_risk_ids = get_at_risk_actors(self.start_ego_pos, self.start_ego_heading, self.start_actor_data)
        collided_ids = set(self.collided_actor_ids)

        # Fair collision rate
        fair_collision_rate = len(collided_ids & at_risk_ids) / max(len(at_risk_ids), 1)

        # print("Fair Collision Rate: " + str(fair_collision_rate*100) + "%")

        # self.summary = log_episode_metrics(self.last_harm_metrics, self.episode,self.evaluate,fair_collision_rate,terminal_reward,ego_prob,other_prob)
        self.summary = log_episode_metrics(self.last_harm_metrics, self.episode,self.evaluate,fair_collision_rate,terminal_reward,ego_prob,other_prob,self.difficulty)
        # print("Episode Summary: ")
        # print(self.summary)
        # self.episode_log.append(self.summary)


        if terminal_reward != 5 and terminal_reward != 2:
            # terminal_reward = -1 * (sum(other_prob)) -1 * (sum(ego_prob)) - 1*(self.summary['mean_harm_avg'])-1*(self.summary['mean_harm_var'])-1*(self.summary['max_harm_max']) -1*(fair_collision_rate)
            terminal_reward = (
                -1.5 * sum(other_prob)
                -1 * sum(ego_prob)
                # -1 * self.summary.get('mean_harm_avg', 0)
                # -1 * self.summary.get('mean_harm_var', 0)
                # -1 * self.summary.get('max_harm_max', 0)
                # -1 * fair_collision_rate
            )

        scaled_terminal_reward = normalise_terminal(terminal_reward,self.difficulty)

        self.summary = log_episode_metrics(self.last_harm_metrics, self.episode,self.evaluate,fair_collision_rate,terminal_reward,ego_prob,other_prob,self.difficulty)
        self.episode_log.append(self.summary)
        log_episode_summary_stats_csv(self.summary,filepath=self.log_dir)


        if self.evaluate == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('terminal_reward', scaled_terminal_reward, step=self.episode)
                tf.summary.scalar('Fair Collision Rate', (fair_collision_rate*100), step=self.episode)
                tf.summary.scalar('Utilitarian', 1 - self.summary['mean_harm_avg'], step=self.episode)
                tf.summary.scalar('Egalitarian', 1 - self.summary['mean_harm_var'], step=self.episode)
                # Check Rawlian
                tf.summary.scalar('Rawlian', 1 - self.summary['max_harm_max'], step=self.episode)
            
        # print("Original Terminal Reward: " + str(terminal_reward))
        # print("Scaled Terminal Reward: " + str(scaled_terminal_reward))

        return scaled_terminal_reward

    def get_location(self):  # for test scenario only
        x = 50.9  # ego start position at intersection_1 (evaluation)
        y = 28.35
        location = self.ego.get_transform().location
        return location.x-x, y-location.y

    def get_velocity(self):
        velocity = self.ego.get_velocity()
        return velocity.x, velocity.y

    def _get_speed(self):  # m/s
        velocity = self.ego.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def get_acceleration(self):  # m/s^2
        acc = self.ego.get_acceleration()
        rot = self.ego.get_transform().rotation
        pitch = np.radians(rot.pitch)  # deg to radian
        roll = np.radians(rot.roll)
        yaw = np.radians(rot.yaw)
        global_acc = np.array([acc.x, acc.y, acc.z])
        r = transforms3d.euler.euler2mat(roll, pitch, yaw).T
        acc_relative = np.dot(r, global_acc)
        return acc_relative[0], acc_relative[1]

    def _check_done(self):
        """
        Check if the episode is done.

        Returns:
        tuple: (bool indicating done, int indicating reason)
            reason codes:
            1 - post-collision timeout or stop
            2 - scenario crossed
            3 - max episode time
        """
        # If collision occurred, start post-collision tracking
        if bool(self.collisions):
            if not hasattr(self, 'steps_after_collision'):
                self.steps_after_collision = 0  # Init on first detection

            self.steps_after_collision += 1
            speed = self._get_speed() * 3.6  # Convert to km/h

            if speed < 2.0 or self.steps_after_collision >= 25:
                print("Ending 1")
                return True, 1

        # Still respect scenario crossing
        if self._check_crossed():
            print("Ending 2")
            return True, 2

        # Or hard time limit
        if (datetime.datetime.now() - self.episode_start_time).total_seconds() > self.episode_max_time:
            print("Ending 3")
            return True, 3

        return False, None


    def _check_crossed(self):
        ego_location = self.ego.get_transform().location
        if self.cross_points[2] != 0:
            if np.sign(self.cross_points[0]-self.cross_points[2]) != np.sign(ego_location.x-self.cross_points[2]):
                self.crossed = True
        elif np.sign(self.cross_points[1]-self.cross_points[3]) != np.sign(ego_location.y-self.cross_points[3]):
            self.crossed = True
        return self.crossed

    # careful of memory leak
    def destroy_actors(self):
        """
        Destroys all the actors in the actor_list and stops any active sensors.
        """
        
        # Stop any active sensors
        
        if self.collision_sensor:
            self.collision_sensor.stop()
        
        if self.camera:
            self.camera.stop()

        # Iterate over the actor list to destroy them
        for actor in self.actor_list:
            try:
                if actor.is_alive:
                    # print(f"Destroying actor: {actor.id}")
                    actor.destroy()
            except Exception as e:
                print(f"Failed to destroy actor {actor.id}: {e}")
        
        time.sleep(0.5)
        self.actor_list = []
        self.cars = []
        self.motorcycles = []
        self.bicycles = []
        self.pedestrians1 = []
        self.pedestrians2 = []

        # print("Actor list here 1")
        # print(self.actor_list)

    def get_train_util(self):
        
        if self.track_train_util:
            print("Train Util Tracker: ")
            print(self.train_util_tracker)
            return self.train_util_tracker, self.train_util_dir+"/train_util_tracker.json"
        else:
            return {},None