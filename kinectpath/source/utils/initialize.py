# -*- coding: utf-8 -*-

"""Script for initializing several variables for automated analysis of
smartplayroom data.

This module contains the initialization functions for several of the
important variables required for consequent analysis of important
attributes that should be extracted from the smartplayroom data in order to
make conclusions about the visual search strategies and performance of kids
participating in the study.

Notes
-----
    (Everything noted here was before myself and Lakshmi went over all the data
    and fixed erroneous annotations / added missing annotations, wherever possible)
    Notes on each subject's data:
		3765
		Trial S/E annotations only for 15 / 18 trials
		S/E missing: Boat, Hippo, Duck
		Grasp time missing: Raptor
		3D trajectory data for 14 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3773
		Trial S/E annotations present for all 18 trials
		S/E missing: None
		Grasp time missing: Green Cone
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3798
		Trial S/E annotations only for 16 / 18 trials
		S/E missing: Hippo, Lion
		Grasp time missing: None
		3D trajectory data for 16 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3809
		Trial S/E annotations only for 17 / 18 trials
		S/E missing: Frog
		Grasp time missing: None
		3D trajectory data for 9 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp times out of bounds for
		Pink Cylinder, Cow, Airplane, Blue Cube, Car, Yellow Pyramid, Book
		and Elephant

		3818
		Trial S/E annotations only for 16 / 18 trials
		S/E missing: Blue Cube, Car
		Grasp time missing: None
		3D trajectory data for 9 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: (1) Pink Cylinder, Red Ball,
		Yellow Pyramid and Orange Box have missing data in Kinect,
		(2) Grasp times out of bounds for Book, Raptor and Elephant

		3823
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: Boat
		3D trajectory data for 16 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp times out of bounds for
		Book

		3829
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3834
		Trial S/E annotations for 17 / 18 trials
		S/E missing: Airplane
		Grasp time missing: None
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3835
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: Green Cone
		3D trajectory data for 0 trials extracted
		Reason for missing trajectory data: Except Green Cone, all grasp
		times out of bounds

		3839
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3848
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3859
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 15 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Pink Cylinder, Red Ball and
		Green Cone have missing data in Kinect

		3864
		Trial S/E annotations for 17 / 18 trials
		S/E missing: Airplane
		Grasp time missing: None
		3D trajectory data for 14 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Red Ball, Book and Elephant
		have missing data in Kinect

		3867
		Trial S/E annotations for 17 / 18 trials
		S/E missing: Cow
		Grasp time missing: None
		3D trajectory data for 15 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		Orange Box

		3868
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3871
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: Airplane
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3872
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 0 trials extracted
		Reason for missing trajectory data: All trials have missing data
		in Kinect

		3881
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 0 trials extracted
		Reason for missing trajectory data: All trials have missing data
		in Kinect

		3883
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3884
		Trial S/E annotations for all 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 0 trials extracted
		Reason for missing trajectory data: All trials have missing data
		in Kinect

		3894
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: Frog
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3898
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3899
		Trial S/E annotations for 17 / 18 trials
		S/E missing: Duck
		Grasp time missing: Hippo
		3D trajectory data for 15 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3918
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3920
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3930
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3934
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: Yellow Pyramid
		3D trajectory data for 7 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		Cow, Red Ball, Pink Cylinder, Boat, Orange Box, Car, Pooh, Green
		Cone, Raptor, Elephant

		3936
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 6 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		Cow, Airplane, Boat, Car, Pooh, Lion, Yellow Pyramid, Hippo, Book,
		Orange Box, Frog, Raptor

		3941
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 1 trial extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		all trials except Hippo

		3944
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 1 trial extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		all trials except Pink Cylinder

		3947
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: Hippo, Lion, Yellow Pyramid
		3D trajectory data for 3 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Grasp time out of bounds for
		Pink Cylinder, Frog, Blue Cube, Airplane, Cow, Book, Duck, Boat,
		Orange Box, Pooh, Green Cone, Raptor

		3950
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3952
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 0 trials extracted
		Reason for missing trajectory data: All trials have missing data
		in Kinect

		3954
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)
		Reason for missing trajectory data: Missing data in Kinect for
		Duck

		3991
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3998
		Trial S/E annotations for 18 trials
		S/E missing: None
		Grasp time missing: None
		3D trajectory data for 18 trials extracted
		Remarks on trajectory data: (Good / Bad)

		3999
		Trial S/E annotations for 17 / 18 trials
		S/E missing: Car
		Grasp time missing: None
		3D trajectory data for 17 trials extracted
		Remarks on trajectory data: (Good / Bad)

"""

from variables import trialNames, trial_name_mapper, condition_types, grasp_timing_dir, \
        grasp_timing_files, sets, objconds, triplet_names, NUM_COLORS, dfTurnPoints, \
        flagged_trials, subs, grasp_timing_subids, dataRoot, skelDataFrames, trialStartEnd, flags, \
        ref_stats, near_stats, far_stats, trial_types, startPoints, endPoints, pathlengths, colorDict, \
        grasp_timings, grasp_timing_problems, num_turns

import os
import matplotlib.pyplot as plt
import glob
import datetime
import pandas
import numpy as np
import pickle
import h5py

from RDPpath import getTurningPoints
from misc import calc_distance, calc_line_point_distance, reject_outliers

import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

def init_triplet_sets(triplet_sets_filepath='./data/triplet_sets.csv'):
    """Function to initialize the triplets for (ref, near, far) according
    to the two unique configurations present in trials.

    The function takes the predefined sets of triplets for the two
    configurations and initializes an object that stores this information.

    Parameters
    ----------
    triplet_sets_filepath : str
        Path to the file containing the triplet sets for two configs

    """
    with open(triplet_sets_filepath, 'r') as f:
        tlines = f.readlines()[7:]

    """This loop initializes the sets indexed by Set ID (0 or 1), each
    index containing a dictionary mapping each object to it's corresponding
    reference object.
    To ascertain whether the object is reference, near or far, a separate
    dictionary stores their type (1: ref, 2: near, 3: far).
    """
    for line in tlines:
        vals = line.split(',')
        vals = [x.strip() for x in vals]
        sets[int(vals[0])-1][trial_name_mapper[vals[1]]] = trial_name_mapper[vals[-1]]
        objconds[int(vals[0])-1][trial_name_mapper[vals[1]]] = 2 if int(vals[3]) else 3

    """Using these sets and object type dictionaries, wwe create a dict
    that stores the triplets in a dictionary.
    For e.g.:
    Consider the dictionary is called `trip`. It should look
    something like this:
        trip = {
            0: {'ref_obj_placard': {
                1: 'ref_obj_placard',
                2: 'near_obj_placard_0',
                3: 'far_obj_placard_0',
            }, ...},
            1: {'ref_obj_placard': {
                1: 'ref_obj_placard',
                2: 'near_obj_placard_1',
                3: 'far_obj_placard_1',
            }, ...},
        }
    """
    for i in range(2):
        triplet_names[i] = {}
        for k in sets[i].keys():
            if sets[i][k] not in triplet_names[i].keys():
                triplet_names[i][sets[i][k]] = {}
            triplet_names[i][sets[i][k]][1] = sets[i][k]
            triplet_names[i][sets[i][k]][objconds[i][k]] = k

def init_colormaps(maps=['gist_rainbow', 'gist_earth', 'ocean', 'hsv', 'cubehelix', 'brg']):
    """Function for initializing the colors to use for plotting (ref, near, far) paths

    Parameters
    ----------
    maps : list
        A predefined list of colormaps you want to use for plotting. You
        can choose from them while plotting

    """
    cms = [plt.get_cmap(colmap) for colmap in maps]
    colorlist = [[cms[0](1. * i / NUM_COLORS) for i in range(NUM_COLORS)] for x in range(len(cms))]
    colorDict.update(dict(zip(trialNames[:6], colorlist)))

def init_grasp_time_files():
    """Function to initialize an object that stores the grasp times for
    the trials where it is annotated.

    There are no parameters passed to this function as we store all the
    necessary variables as global variables imported from the `variables`
    module.

    """
    files_grasp = glob.glob(os.path.join(grasp_timing_dir, '*.p'))
    # Initialize a map for checking which subjects have these annotations
    for fn in files_grasp:
        if '.p' in fn:
            grasp_timing_subids.append(fn.split('/')[-1].split('.')[0])

    for subject_id in subs:
        noFile = True
        for file_grasp in files_grasp:
            if str(subject_id) in file_grasp:
                noFile = False
                grasp_timing_files.append(file_grasp)
        if noFile:
            grasp_timing_files.append('')

    for ix, sub in enumerate(subs):
        if str(sub) in grasp_timing_subids:
            with open(grasp_timing_files[ix], 'rb') as f:
                grasp_times = pickle.load(f)
            grasp_times = {trial_name_mapper[k]: grasp_times[k] for k in grasp_times.keys() if k in trial_name_mapper.keys()}
        grasp_timings[sub] = grasp_times

def init_flagged_trials_map(flag_trials_filepath='./data/flag_trials.csv'):
    """Function to initialize a map for marking the trials with a specific
    strategy: where the kid fixates at the object being searched before
    making a move to get it.

    Note: The trials were visually inspected to mark the ones which
    demonstrate this strategy being used. Hence, there could be some degree
    of error that can be assumed in terms of understanding whether fixation
    was done or not (+/- 5 pixels, I'd say).

    """
    with open(flag_trials_filepath, 'r') as f:
        lines = f.readlines()[3:]
        lines = [x.strip() for x in lines]

    for line in lines:
        sub = line.split(',')[0]
        if sub not in flagged_trials.keys():
            flagged_trials[sub] = []
        trial = line.split(',')[1].split()[0].lower()
        trial_name = trial_name_mapper[trial]
        flagged_trials[sub].append(trial_name)

def prepare_trial_set_array():
    """Function to create a map that sets the arrangement set ID for each
    subject - according to the triplet sets we initialized in one of the
    previous functions.
    """
    for sub in subs:
        trial_conditions_filepath = glob.glob(os.path.join(dataRoot, str(sub), 'NaturalisticVisSearch',
                                '*_playroom.csv'))[0]
        trial_set = 0
        with open(trial_conditions_filepath, 'r') as f:
            lines = f.readlines()
            conds, trial_names = [x.split(',')[-3].strip() for x in lines[1:]], [x.split(',')[-1].strip() for x in lines[1:]]
            conditions = {}
            dict_trials = {trial_name_mapper[k]: int(x) for (k,x) in zip(trial_names, conds)}
            shared_items_0 = {k: objconds[0][k] for k in objconds[0] if k in dict_trials and objconds[0][k] == dict_trials[k]}
            shared_items_1 = {k: objconds[1][k] for k in objconds[1] if k in dict_trials and objconds[1][k] == dict_trials[k]}
            if len(shared_items_0) == 12:
                trial_set = 1
            if len(shared_items_1) == 12:
                trial_set = 2

        trial_types[sub] = trial_set
    print trial_types

def prepare_skel_dataframe():
    """Function to prepare the Pandas dataframe that stores the 3D
    skeleton coordinates for each of the trials using the S/E annotations
    that are initialized from the annotation files.
    """
    for sub in subs:
        # Get kinect file for visual search task
        kinFiles = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + '*.xef')
        if not kinFiles:
            print('No kinect files found for subject: ' + str(sub) + '\n\n\n\n')
            continue
        # Sort so the the first one is the one we want
        kinFiles.sort(key=os.path.getmtime)

        # get kinect1 stop time - using the file creation / modified time
        k1MTime = os.path.getmtime(kinFiles[0])
        k1MDateTime = datetime.datetime.fromtimestamp(k1MTime)  # .strftime('%Y%m%d%H%M%S%f')
        k1CDateTime = datetime.datetime.fromtimestamp(os.path.getctime(kinFiles[0]))  # .strftime('%Y%m%d%H%M%S%f')

        # Read the coordinates stored in a csv for that kinect1 file into
        # a Pandas dataframe
        df = pandas.read_csv(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + 'output_body.csv',
                                low_memory=False)

        # get rows for joint type 13(body center)
        temp = df.loc[df[' jointType'] == 13]
        if flags.plot_3d_animation:
            temp = df.dropna()
        # copy dataframe so that original dataframe stays unaltered
        skelDataFrame = temp.copy()
        # subtract the first timestamp from all the timestamps to get zero based stamps
        skelDataFrame['# timestamp'] = skelDataFrame['# timestamp'] - skelDataFrame['# timestamp'].iloc[0]

        if k1CDateTime > k1MDateTime:
            # recording duration of kinect in millisecs(the last element of column)
            k1Dur = skelDataFrame['# timestamp'].iloc[-1]
            # get start of kinect recording by subtracting duration from file modification time
            k1CDateTime = k1MDateTime - datetime.timedelta(milliseconds=k1Dur)

        # add absolute start time to relative timestamps and format it
        skelDataFrame['# timestamp'] = [(datetime.timedelta(milliseconds=ms) + k1CDateTime).strftime('%Y%m%d%H%M%S%f') for
                                        ms in skelDataFrame['# timestamp']]
        skelDataFrame['# timestamp'] = skelDataFrame['# timestamp'].astype(pandas.datetime)

        skelDataFrames[sub] = skelDataFrame

def prepare_trial_start_stops():
    for sub in subs:
        # Store the trials for each subject which have faults in grasp time
        grasp_timing_problems[sub] = {}
        trialStartEnd[sub] = {}
        # File containing the Eyetracker log with timestamped S/E
        feyetrackPSData = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1_*.txt')
        if not feyetrackPSData:
            print('No eyetracking PSData text file. Moving to next subject \n\n')
            continue

        # Initially, time on eyetracker PC was a ahead of flycapture video
        # recording and biopac PC's. We set it off by following number of
        # seconds to match.
        eyetrack_offset = 0  # seconds
        eyePSDatafread = open(feyetrackPSData[0])
        eyePSDataLines = eyePSDatafread.read().split('\n')
        eyePSDatafread.close()

        i_start = [i for i, s in enumerate(eyePSDataLines) if 'recording scene' in s][0]
        i_stop = [i for i, s in enumerate(eyePSDataLines) if 'Stopped recording' in s][0]
        eyetrackStart = datetime.datetime.strptime(eyePSDataLines[i_start][0:23], '%Y-%m-%d %I:%M:%S.%f')
        if eyetrackStart.hour < 8:
            eyetrackStart = eyetrackStart + datetime.timedelta(hours=12)

        eyetrack_stop = datetime.datetime.strptime(eyePSDataLines[i_stop][0:23], '%Y-%m-%d %I:%M:%S.%f')
        if eyetrack_stop.hour < 8:
            eyetrack_stop = eyetrack_stop + datetime.timedelta(hours=12)

        # Read eyetracked output file - frame numbers timestamped
        feyetrackInfo = glob.glob(
            dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1a' + os.sep + str(sub) + '*.txt')
        if not feyetrackInfo:
            print('No eyetracking text file. Moving to next subject \n\n')
            continue
        eyetrOutput = pandas.read_csv(feyetrackInfo[0], skiprows=5, delimiter=' ', index_col=False)

        # Read the file for this subject containing trial S/E annotations
        vsStartEnd = pandas.read_csv('../vs_trials' + os.sep + str(sub) + '.csv', header=None, index_col=2)
        trStartTimes = {}
        trEndTimes = {}
        for index, row in vsStartEnd.iterrows():
            trName = index
            startFr = row[0]
            endFr = row[1]

            # The scene start and end times (according *.mov) using the
            # start and end frame annotations. TODO: Check the difference
            # between recordFrameCount / sceneFrameCount
            stSceneQTtime = eyetrOutput[eyetrOutput['recordFrameCount'] >= startFr].iloc[0][
                'sceneQTtime(d:h:m:s.tv/ts)'].split('.')
            enSceneQTtime = eyetrOutput[eyetrOutput['recordFrameCount'] >= endFr].iloc[0][
                'sceneQTtime(d:h:m:s.tv/ts)'].split('.')

            stSceneTimeStr = stSceneQTtime[0][2:] + '.' + str(int(eval(stSceneQTtime[1] + '.0') * 1000))
            enSceneTimeStr = enSceneQTtime[0][2:] + '.' + str(int(eval(enSceneQTtime[1] + '.0') * 1000))
            t = datetime.datetime.strptime(stSceneTimeStr, "%H:%M:%S.%f")
            tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

            trStartTime = (eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f')

            t = datetime.datetime.strptime(enSceneTimeStr, "%H:%M:%S.%f")
            tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

            trEndTime = (eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f')

            if flags.stop_at_grasp_times:           # Until grasp times?
                if str(sub) in grasp_timing_subids:
                    if trName in grasp_timings[sub].keys():
                        lastFr = grasp_timings[sub][trName]
                        # If annotation not in between S and E, reject
                        if not (lastFr > startFr and lastFr < endFr):
                            #print str(sub) + "_" + trName + " : Grasp " + \
                            #    "frame not within S/E annotation " + \
                            #    "bounds. S:" + str(startFr) + " E:" + \
                            #    str(endFr) + " G:" + str(lastFr)

                            # OOBS: Out Of BoundS (S/E)
                            grasp_timing_problems[sub][trName] = "OOBS"

                        enSceneQTtimeRFC = eyetrOutput[eyetrOutput['recordFrameCount'] >= lastFr].iloc[0][
                            'sceneQTtime(d:h:m:s.tv/ts)'].split('.')
                        enSceneTimeRFCStr = enSceneQTtimeRFC[0][2:] + '.' + str(int(eval(enSceneQTtimeRFC[1] + '.0') * 1000))
                        t = datetime.datetime.strptime(enSceneTimeRFCStr, "%H:%M:%S.%f")
                        tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

                        trEndTimeRFC = (eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f')

                        if trEndTimeRFC > trStartTime and trEndTimeRFC < trEndTime:
                            trEndTime = trEndTimeRFC
                        else:
                            grasp_timing_problems[sub][trName] += " - No Fix"
                            continue

                        endFr = lastFr
                    else:
                        #print str(sub) + "_" + trName + " : Missing " + \
                        #    "grasp time"

                        # MISS: MISSing annotation
                        grasp_timing_problems[sub][trName] = "MISS - No fix / annotate?"
                        continue

            trStartTimes[trName] = trStartTime
            trEndTimes[trName] = trEndTime

        trialStartEnd[sub]['starts'] = trStartTimes
        trialStartEnd[sub]['ends'] = trEndTimes

def plot_3d_animation(sub, trialname, data):
    """Function to create the animations showing the Kinect data for trials

    The functions takes the 3D trajectory data for a trial, generates an
    animation from the data and saves it as *.mp4 using ffmpeg as the
    backend to generate the video.

    Parameters
    ----------
    sub : int
        The subject ID for which the trials are being processed
    trialname : str
        The name of the trial for which the animation is being generated
    data : ndarray
        Numpy array containing the data for the trial.
        Shape is of the format (3, N, 25), where N is the number of frames
        covering the entire trial.
    """
    path_to_mp4 = os.path.join('path_vids', sub+'_'+trialname+'.mp4')
    if os.path.exists(path_to_mp4):
        print sub+'_'+trialname+'.mp4'+" already exists. [SKIPPING]"
        return

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    def update(num, data, line):
        line.set_data(data[:2, num, :])
        line.set_3d_properties(data[2, num, :])

    scatter, = ax.plot(data[0, 0, :], data[1, 0, :], data[2, 0, :], marker='o')

    ax.set_xlim3d([2.0, -2.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-2.0, 2.0])
    ax.set_zlabel('Z')

    ax.view_init(elev=17, azim=-43)
    ax.set_title(sub + '-' + trialname)

    ani = animation.FuncAnimation(
        fig, update, data.shape[1], fargs=(data, scatter),
        interval=10000/data.shape[1], blit=False
    )
    ani.save(path_to_mp4, writer='ffmpeg', fps=30)

    plt.close(fig)
    print sub+'_'+trialname+'.mp4'+" [DONE]"

def prepare_start_end_points_and_pathlens():
    """Function to initialize the trial data from S/E annotations and
    calculating path lengths for each object search (trial) trajectory

    Simply put, we store the dataframes related to each trial mapped to
    that trial name with the subject. We calculate the length of the search
    path by dividing it into small line segments and adding up the lengths.
    """
    subs_data = {}
    for sub in subs:
        print "Writing coords for " + str(sub) + " - ",
        subs_data[str(sub)] = {}
        #hf = h5py.File(str(sub)+'_coords.h5', 'w')
        #g = hf.create_group('coords')

        startPoints[sub] = []
        trial_set = trial_types[sub]-1
        endPoints[sub] = {}
        trStartTimes, trEndTimes = trialStartEnd[sub]['starts'], trialStartEnd[sub]['ends']
        skelDataFrame = skelDataFrames[sub]
        pathlengths[sub] = {}
        num_turns[sub] = {}

        ref = []; near = []; far = []
        for i, k in enumerate(trialNames[:6]):
            triplet_name = [triplet_names[trial_set][k][x] for x in range(1, 4)]
            for j, name in enumerate(triplet_name):
                print sub, name,
                if name in trStartTimes.keys():
                    trStartTime = trStartTimes[name]
                    trEndTime = trEndTimes[name]
                else:
                    # There could be two reasons for no S/E annotations:
                    # (1) Missing grasp time annotation for that trial
                    # (2) Missing S/E annotation for that trial
                    if sub in grasp_timing_problems.keys():
                        if name in grasp_timing_problems[sub].keys():
                            if "No" in grasp_timing_problems[sub][name]:
                                #print "Breakpoint 1: No start-end annotation ",
                                # MISS or OOBS?
                                #print "Grasp time problem: " + \
                                #    grasp_timing_problems[sub][name],
                                continue
                            else:
                                pass
                                #print "Found a fix for faulty annotation - ",
                                #print grasp_timing_problems[sub][name]
                        else:
                            #print "Breakpoint 1: No start-end annotation"
                            continue
                    else:
                        #print "Breakpoint 1: No start-end annotation"
                        continue

                # To get the dataframes we compare S/E timestamps
                trialDf = skelDataFrame[
                    (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

                if trialDf.empty:
                    # missing trial probably - print the S/E timestamps
                    #print "Breakpoint 2: No trial data - " +trStartTime+ \
                    #    " " + trEndTime
                    continue

                # Remove all the samples that are too close to sensor. They are likely wrong tracking
                trialDf = trialDf.drop(trialDf[trialDf[' position.Z'] < 0.5].index)

                # Find the points where tracking was lost
                sampleGaps = np.diff(np.array(trialDf['# timestamp'].astype(float)))

                # Get length of trial. Exclude the time when the skeleton was lost
                trTrackTime = sum(sampleGaps[sampleGaps < 35000])

                # If samples are >40 millisec apart, there was a jump in skeleton
                # skelJumpIdx = np.add(np.where(sampleGaps > 35000)[0],1)

                if flags.plot_3d_animation:
                    x = trialDf[' position.X'].astype(float)
                    y = trialDf[' position.Y'].astype(float)
                    z = trialDf[' position.Z'].astype(float)

                    X = []; Y = []; Z = []
                    for idx in range(len(x.index)):
                        c1 = x.iloc[idx*25 : (idx+1)*25]
                        c2 = y.iloc[idx*25 : (idx+1)*25]
                        c3 = z.iloc[idx*25 : (idx+1)*25]
                        if (not c1.empty) and (not c3.empty) and (not c2.empty):
                            # Sometimes the last frame has incomplete entries in
                            # the file 'output_body.csv'
                            if len(c1.index) < 25:
                                continue
                            X.append(c1.as_matrix())
                            Y.append(c3.as_matrix())
                            Z.append(c2.as_matrix())

                    # (N, 25)
                    X = np.stack(X)
                    Y = np.stack(Y)
                    Z = np.stack(Z)
                    data = [X, Y, Z]
                    # (3, N, 25)
                    data = np.stack(data)
                    subs_data[str(sub)][name] = data

                    # plot_3d_animation(str(sub), name, data)

                x = trialDf[' position.X'].astype(float)
                y = trialDf[' position.Z'].astype(float)
                z = trialDf[' position.Y'].astype(float)

                #x_np = x.as_matrix()
                #y_np = z.as_matrix()
                #z_np = y.as_matrix()
                #coords = np.stack([x_np, y_np, z_np])
                #g.create_dataset(name, data=coords)

                # Get the points where we consider there was a turn
                # according to some threshold in the angle between two
                # segments - ref. RDPpath.py
                simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

                sx, sy, sz = np.vstack((x, z, y))

                # Initialize the start and end point 2D coordinates for
                # each trial - used to calculate the straight line paths
                # for a trial used in consequent metric calculations.
                startPoints[sub].append((x.values[0], z.values[0], y.values[0]))
                #import ipdb; ipdb.set_trace()
                if flags.stop_at_grasp_times:
                    if str(sub) in grasp_timing_subids:
                        if name in grasp_timings[sub].keys():
                            endPoints[sub][name] = (x.values[-1], z.values[-1], y.values[-1])

                # Considering each segment between two turning points as
                # straight, calculate the length - adding all these segment
                # lengths should give the path length.
                sx, sy = simplePath.T
                plen = 0
                for idx, _idx in zip(turnIdx[:-1], turnIdx[1:]):
                    x1, y1 = sx[idx], sy[idx]
                    x2, y2 = sx[_idx], sy[_idx]

                    dist = calc_distance([x1, y1], [x2, y2])
                    plen += dist
                #for x1,y1,x2,y2 in zip(sx[:-1],sy[:-1],sx[1:],sy[1:]):
                #    dist = calc_distance([x1, y1], [x2, y2])

                pathlengths[sub][name] = plen
                num_turns[sub][name] = len(turnIdx)
                if j == 0:
                    ref.append(plen)
                if j == 1:
                    near.append(plen)
                if j == 2:
                    far.append(plen)

                print "[COMPLETE]"

        #with open('subs_data.pkl', 'wb') as f:
        #    pickle.dump(subs_data, f)
        #hf.close()
        #print "[DONE]"
        # Use the calculated path lengths for trial trajectories to store
        # some statistics used in further calculations.
        ref = np.asarray(ref); near = np.asarray(near); far = np.asarray(far)
        ref_stats[sub] = (np.mean(ref), np.std(ref), len(ref))
        near_stats[sub] = (np.mean(near), np.std(near), len(near))
        far_stats[sub] = (np.mean(far), np.std(far), len(far))

def get_framewise_correspondence_eye_kin(sub=None):
    """Function to find the corresponding frames in Kinect for Eyetracker

    The function is aimed at getting a framewise correspondence as the way
    we extract frames from Kinect data is using timestamp bounds. This way
    each frame in Eyetracker video (scene video) does not have a
    corresponding frame in Kinect video. As the frame rates of the two
    videos are different, we sample from nearby frames if there is not a
    perfect match. This can result in multiple frames being matched to a
    single frame in the correspondence, which works for our case.

    Parameters
    ----------
    sub : int
        The subject ID for which you want framewise correspondece
    """
    if sub is None:
        print "Please pass the subject ID you need the correspondence for, as a parameter"
        return None

    # eyetracking
    feyetrackPSData = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1_*.txt')
    if not feyetrackPSData:
        print('No eyetracking PSData text file. Moving to next subject \n\n')
        return None

    # initially, time on eyetracker PC was a ahead of flycapture video recording and biopac PC's
    # We set it off by following number of seconds to match
    eyetrack_offset = 0  # seconds
    eyePSDatafread = open(feyetrackPSData[0])
    eyePSDataLines = eyePSDatafread.read().split('\n')
    eyePSDatafread.close()

    i_start = [i for i, s in enumerate(eyePSDataLines) if 'recording scene' in s][0]
    i_stop = [i for i, s in enumerate(eyePSDataLines) if 'Stopped recording' in s][0]
    eyetrackStart = datetime.datetime.strptime(eyePSDataLines[i_start][0:23], '%Y-%m-%d %I:%M:%S.%f')
    if eyetrackStart.hour < 8:
        eyetrackStart = eyetrackStart + datetime.timedelta(hours=12)

    eyetrack_stop = datetime.datetime.strptime(eyePSDataLines[i_stop][0:23], '%Y-%m-%d %I:%M:%S.%f')
    if eyetrack_stop.hour < 8:
        eyetrack_stop = eyetrack_stop + datetime.timedelta(hours=12)

    # read eyetracked output file
    feyetrackInfo = glob.glob(
        dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1a' + os.sep + str(sub) + '*.txt')
    if not feyetrackInfo:
        print('No eyetracking text file. Moving to next subject \n\n')
        return None
    eyetrOutput = pandas.read_csv(feyetrackInfo[0], skiprows=5, delimiter=' ', index_col=False)

    vsStartEnd = pandas.read_csv('../vs_trials' + os.sep + str(sub) + '.csv', header=None, index_col=2)
    trTimes = {}
    perTrialDf = {}
    tempDf = skelDataFrames[sub]
    colnames = list(tempDf.columns.values)
    for index, row in vsStartEnd.iterrows():
        trName = index
        trTimes[trName] = []
        corrDataFrame = pandas.DataFrame(columns=colnames)
        startFr = row[0]
        endFr = row[1]

        if flags.stop_at_grasp_times:           # Until grasp times?
                if str(sub) in grasp_timing_subids:
                    if trName in grasp_timings[sub].keys():
                        lastFr = grasp_timings[sub][trName]
                        # If annotation not in between S and E, reject
                        if not (lastFr > startFr and lastFr < endFr):
                            #print str(sub) + "_" + trName + " : Grasp " + \
                            #    "frame not within S/E annotation " + \
                            #    "bounds. S:" + str(startFr) + " E:" + \
                            #    str(endFr) + " G:" + str(lastFr)
                            continue
                        endFr = lastFr
                    else:
                        #print str(sub) + "_" + trName + " : Missing " + \
                        #    "grasp time"
                        continue

        sceneQTtime = eyetrOutput[eyetrOutput['recordFrameCount'] >= startFr][
            'sceneQTtime(d:h:m:s.tv/ts)'].str.split('.')
        QTtimes = sceneQTtime[sceneQTtime.index <= endFr]
        for idx in range(len(QTtimes)):
            tstamp = QTtimes.iloc[idx]
            tstampStr = tstamp[0][2:] + '.' + str(int(eval(tstamp[1] + '.0') * 1000))
            t = datetime.datetime.strptime(tstampStr, "%H:%M:%S.%f")
            tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

            trTimes[trName].append((eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f'))

        # TODO: Understand and explain this logic for future reference
        for j, tstamp in enumerate(trTimes[trName]):
            tdf = tempDf[tempDf['# timestamp'] >= tstamp].iloc[0,:]
            revtdf = tempDf[tempDf['# timestamp'] <= tstamp].iloc[-1,:]
            length = len(trTimes[trName]) - (j + 1)
            revlength = j + 1
            ix = 1
            goBack = False
            while tdf[4] < 0.5:
                if ix > length-1:
                    goBack = True
                    break
                tdf = tempDf[tempDf['# timestamp'] >= tstamp].iloc[ix,:]
                ix += 1
            if goBack:
                ix = -2
                while revtdf[4] < 0.5:
                    if ix < -revlength:
                        break
                    revtdf = tempDf[tempDf['# timestamp'] <= tstamp].iloc[ix,:]
                ix -= 1
            if goBack:
                corrDataFrame.loc[j] = revtdf
            else:
                corrDataFrame.loc[j] = tdf
        perTrialDf[trName] = corrDataFrame

    return perTrialDf

def get_kinectpath_features(trialDf):
    """Function to extract the 2D coordinates and the difference of
    these coordinates across two consecutive frames in time: features for
    time series evaluation.

    Parameters
    ----------
    trialDf : pandas.DataFrame
        The dataframe containing the data for a trial
    """
    x = trialDf[' position.X'].astype(float)
    y = trialDf[' position.Z'].astype(float)

    vel_x = x - trialDf[' position.X'].iloc[0].astype(float)
    vel_y = y - trialDf[' position.Z'].iloc[0].astype(float)

    # Return the features: all of them are Pandas dataframes
    return x, y, vel_x, vel_y
