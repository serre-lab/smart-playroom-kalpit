# -*- coding: utf-8 -*-

"""Script for creating all the global variables used across scripts
This module is responsible for initializing all the global variables
that are used across scripts and are important for the analysis.
"""

import pandas

class Flags:
    """Class to initialize some flags required for computations in the code

    To add a new flag, just add that attribute to the class and set it's
    value to either True or False.
    """
    def __init__(self):
        self.stop_at_grasp_times = True
        self.plot_3d_animation = False
        # self. some_other_flag = False

subs = [3773, 3920, 3868, 3823, 3822, 3805, 3941, 3944, 3829, 3947, 3809,
        3930, 3927, 3798, 3934, 3898, 3899, 3871, 3872, 3894, 3859, 3834,
        3835, 3939, 3765, 3870, 3867, 3818, 3936, 3839, 3918, 3864, 3884,
        3881, 3883, 3952, 3848, 3954, 3950, 3991, 3998, 3999]
subs.sort()

dataRoot = '/media/CLPS_Amso_Lab/Playroom/NEPIN_SmartPlayroom/SubjectData/'

trialNames = ['pink_cylinder_placard',
              'blue_cube_placard',
              'red_ball_placard',
              'yellow_pyramid_placard',
              'orange_box_placard',
              'green_cone_placard',
              'lion_placard',
              'boat_placard',
              'cow_placard',
              'airplane_placard',
              'elephant_placard',
              'book_placard',
              'car_placard',
              'hippo_placard',
              'frog_placard',
              'pooh_placard',
              'raptor_placard',
              'duck_placard']

trial_name_mapper = {
    'pink': 'pink_cylinder_placard',
    'orange': 'orange_box_placard',
    'blue': 'blue_cube_placard',
    'yellow': 'yellow_pyramid_placard',
    'green': 'green_cone_placard',
    'red': 'red_ball_placard',
    'frog': 'frog_placard',
    'boat': 'boat_placard',
    'lion': 'lion_placard',
    'cow': 'cow_placard',
    'elephant': 'elephant_placard',
    'hippo': 'hippo_placard',
    'car':'car_placard',
    'dinosaur': 'raptor_placard',
    'raptor': 'raptor_placard',
    'duck': 'duck_placard',
    'winnie': 'pooh_placard',
    'pooh': 'pooh_placard',
    'plane': 'airplane_placard',
    'airplane': 'airplane_placard',
    'book': 'book_placard'
}

condition_types = {1: 'Geometric', 2: 'Near', 3: 'Far'}

colorDict = {}

grasp_timing_dir = '../../grasp_return_timing_mov/'
grasp_timing_files = []
grasp_timing_subids = []
grasp_timings = {}
grasp_timing_problems = {}

sets = [{}, {}]
objconds = [{}, {}]

triplet_names = {}
NUM_COLORS = 3

flagged_trials = {}
dfTurnPoints = pandas.DataFrame([], index=subs, columns=trialNames)

pathlengths = {}
num_turns = {}

startPoints = {}
endPoints = {}
trial_types = {}

skelDataFrames = {}
trialStartEnd = {}

flags = Flags()

ref_stats = {}
near_stats = {}
far_stats = {}

attraction_metric = {}
attraction_stats = {}

straightLineDists = {}
curviness = {}
