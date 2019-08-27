import pandas
import os, glob
import time
import platform
import datetime
import matplotlib.pyplot as plt
import scipy
import scipy.signal as scsig
from RDPpath import getTurningPoints
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('TkAgg')

subs = [3773, 3920, 3868, 3823, 3822, 3805, 3941, 3944, 3829, 3947, 3809,
        3930, 3927, 3798, 3934, 3898, 3899, 3871, 3872, 3894, 3859, 3834,
        3835, 3939, 3765, 3870, 3867, 3818, 3936, 3839, 3918, 3864, 3884,
        3881, 3883, 3952, 3848, 3954, 3950, 3991, 3998, 3999]

__STOP_AT_GRASP_TIMES = True

subs.sort()
dataRoot = '/media/CLPS_Amso_Lab/Playroom/NEPIN_SmartPlayroom/SubjectData/'
if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_withgrasp_new.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

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

sets = [{}, {}]
objconds = [{}, {}]

with open('triplet_sets.csv', 'r') as f:
    tlines = f.readlines()[7:]

for line in tlines:
    vals = line.split(',')
    vals = [x.strip() for x in vals]
    sets[int(vals[0])-1][trial_name_mapper[vals[1]]] = trial_name_mapper[vals[-1]]
    objconds[int(vals[0])-1][trial_name_mapper[vals[1]]] = 2 if int(vals[3]) else 3

triplet_names = {}
for i in range(2):
    triplet_names[i] = {}
    for k in sets[i].keys():
        if sets[i][k] not in triplet_names[i].keys():
            triplet_names[i][sets[i][k]] = {}
        triplet_names[i][sets[i][k]][1] = sets[i][k]
        triplet_names[i][sets[i][k]][objconds[i][k]] = k

NUM_COLORS = 3
cms = [plt.get_cmap('gist_rainbow'), plt.get_cmap('gist_earth'), plt.get_cmap('ocean'),
       plt.get_cmap('hsv'), plt.get_cmap('cubehelix'), plt.get_cmap('brg')]
colorlist = [[cms[0](1. * i / NUM_COLORS) for i in range(NUM_COLORS)] for x in range(len(cms))]
colorDict = dict(zip(trialNames[:6], colorlist))

dfTurnPoints = pandas.DataFrame([], index=subs, columns=trialNames)

grasp_timing_dir = '/home/kalpit/ImportantStuff/smart_playroom/grasp_return_timing_mov/'
files_grasp = os.listdir(grasp_timing_dir)
grasp_timing_subids = [x.split('.')[0] for x in files_grasp]
grasp_timing_files = []
#subs = grasp_timing_subids
#subs.sort()

#subs = [x for x in subs if x not in ['3835', '3941', '3944']]

for subject_id in subs:
    noFile = True
    for file_grasp in files_grasp:
        if str(subject_id) in file_grasp:
            noFile = False
            grasp_timing_files.append(os.path.join(grasp_timing_dir, file_grasp))
    if noFile:
        grasp_timing_files.append('')

flagged_trials = {}
with open('flag_trials.csv', 'r') as f:
    lines = f.readlines()[3:]
    lines = [x.strip() for x in lines]

for line in lines:
    sub = line.split(',')[0]
    if sub not in flagged_trials.keys():
        flagged_trials[sub] = []
    trial = line.split(',')[1].split()[0].lower()
    trial_name = trial_name_mapper[trial]
    flagged_trials[sub].append(trial_name)

import pickle
pathlengths = {}

def calc_line_point_distance(line, point):
    slope = ((line[1] - line[3]) / (line[0] - line[2]))
    y_intercept = line[1] - slope * line[0]
    b = -1.0
    a = slope

    return ((a*point[0] + b*point[1] + y_intercept) / (np.sqrt(a**2 + b**2)))

def calc_distance(x1, y1, x2, y2):
    return (np.sqrt((x1-x2)**2 + (y1-y2)**2))

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0)]

startPoints = []
endPoints = {}
endPoints[0] = {}
endPoints[1] = {}
trial_types = {}

with open('pathlengths_trials.csv', 'w') as pf:
    pf.write("Subject ID, Ref, Near, Far\n")
    for ix, sub in enumerate(subs):
        pathlengths[sub] = {}
        if str(sub) not in grasp_timing_subids:
            dur_grasp_times = None
        else:
            with open(grasp_timing_files[ix], 'rb') as f:
                grasp_times = pickle.load(f)
            grasp_times = {trial_name_mapper[k]: grasp_times[k] for k in grasp_times.keys() if k in trial_name_mapper.keys()}
            #if sub in [3835, 3941, 3944]:
            #    print grasp_times

        # Get kinect file for visual search task
        kinFiles = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + '*.xef')
        if not kinFiles:
            print('No kinect files found for subject: ' + str(sub) + '\n\n\n\n')
            continue
        kinFiles.sort(key=os.path.getmtime)
        # get kinect1 stop time
        k1MTime = os.path.getmtime(kinFiles[0])
        k1MDateTime = datetime.datetime.fromtimestamp(k1MTime)  # .strftime('%Y%m%d%H%M%S%f')
        k1CDateTime = datetime.datetime.fromtimestamp(os.path.getctime(kinFiles[0]))  # .strftime('%Y%m%d%H%M%S%f')

        df = pandas.read_csv(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + 'output_body.csv',
                             low_memory=False)

        # get rows for joint type 13(body center)
        temp = df.loc[df[' jointType'] == 13]
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

        trial_conditions_file = glob.glob(os.path.join(dataRoot, str(sub), 'NaturalisticVisSearch', '*_playroom.csv'))
        trial_conditions_filepath = trial_conditions_file[0]
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

            if trial_set == 0:
                print "Fault: ", trial_conditions_filepath
                break
        trial_types[sub] = trial_set

        #############################################################
        # eyetracking
        feyetrackPSData = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1_*.txt')
        if not feyetrackPSData:
            print('No eyetracking PSData text file. Moving to next subject \n\n')
            continue

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
            continue
        eyetrOutput = pandas.read_csv(feyetrackInfo[0], skiprows=5, delimiter=' ', index_col=False)

        vsStartEnd = pandas.read_csv('vs_trials' + os.sep + str(sub) + '.csv', header=None, index_col=2)
        trStartTimes = {}
        trEndTimes = {}
        for index, row in vsStartEnd.iterrows():
            print(sub, index, row[0], row[1])
            trName = index
            startFr = row[0]
            endFr = row[1]

            if __STOP_AT_GRASP_TIMES:
                if str(sub) in grasp_timing_subids:
                    if trName in grasp_times:
                        lastFr = grasp_times[trName]
                        endFr = lastFr

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

            trStartTimes[trName] = trStartTime
            trEndTimes[trName] = trEndTime

        ref = 0; near = 0; far = 0
        nref = 0; nnear = 0; nfar = 0
        for i, k in enumerate(trialNames[:6]):
            triplet_name = [triplet_names[trial_set-1][k][x] for x in range(1, 4)]
            for j, name in enumerate(triplet_name):
                objcond = j+1
                if name in trStartTimes.keys():
                    trStartTime = trStartTimes[name]
                    trEndTime = trEndTimes[name]
                else:
                    continue
                trialDf = skelDataFrame[
                    (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

                if trialDf.empty:
                    # missing trial probably
                    continue

                # Remove all the samples that are too close to sensor. They are likely wrong tracking
                trialDf = trialDf.drop(trialDf[trialDf[' position.Z'] < 0.5].index)

                # Find the points where tracking was lost
                sampleGaps = np.diff(np.array(trialDf['# timestamp'].astype(float)))

                # Get length of trial. Exclude the time when the skeleton was lost
                trTrackTime = sum(sampleGaps[sampleGaps < 35000])

                # If samples are >40 millisec apart, there was a jump in skeleton
                # skelJumpIdx = np.add(np.where(sampleGaps > 35000)[0],1)

                x = trialDf[' position.X'].astype(float)
                y = trialDf[' position.Z'].astype(float)

                #xsmooth = scsig.savgol_filter(x, 5, 2)
                #ysmooth = scsig.savgol_filter(y, 5, 2)

                simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)
                sx, sy = simplePath.T

                startPoints.append((x.values[0], y.values[0]))
                if __STOP_AT_GRASP_TIMES:
                    if str(sub) in grasp_timing_subids:
                        if name in grasp_times.keys():
                            if name not in endPoints[trial_set-1].keys():
                                endPoints[trial_set-1][name] = []
                            endPoints[trial_set-1][name].append((x.values[-1], y.values[-1]))

                plen = 0
                for idx, _idx in zip(turnIdx[:-1], turnIdx[1:]):
                    x1, y1 = sx[idx], sy[idx]
                    x2, y2 = sx[_idx], sy[_idx]

                    dist = calc_distance(x1, y1, x2, y2)
                    plen += dist
                pathlengths[sub][name] = plen
                if j == 0:
                    ref += plen
                    nref += 1
                if j == 1:
                    near += plen
                    nnear += 1
                if j == 2:
                    far += plen
                    nfar += 1
        ag_ref = ref/nref if not nref == 0 else 0
        ag_near = near/nnear if not nnear == 0 else 0
        ag_far = far/nfar if not nfar == 0 else 0
        pf.write(str(sub)+','+str(ag_ref)+','+str(ag_near)+','+str(ag_far)+'\n')


'''
with open('trial_proportions.csv', 'w') as f:
    f.write("Subject ID, #(near with fixate before locomotion), #(far with fixate before locomotion), #(completed near), #(completed far)\n")
    for sub in subs:
        comp_near = 0; comp_far = 0; fix_near = 0; fix_far = 0
        ttype = trial_types[sub] - 1
        for i, k in enumerate(trialNames[:6]):
            triplet_name = [triplet_names[ttype][k][x] for x in range(1, 4)]
            for j, name in enumerate(triplet_name[1:]):
                if name in trStartTimes.keys():
                    if j == 0:
                        if str(sub) in flagged_trials.keys():
                            if name in flagged_trials[str(sub)]:
                                fix_near += 1
                        comp_near += 1
                    if j == 1:
                        if str(sub) in flagged_trials.keys():
                            if name in flagged_trials[str(sub)]:
                                fix_far += 1
                        comp_far += 1
        f.write(str(sub)+','+str(fix_near)+','+str(fix_far)+','+str(comp_near)+','+str(comp_far)+'\n')
'''

attraction_metric = {}
#--------------------------------------------------------------------------------------------------------
with open('attraction_metric.csv', 'w') as pf:
    pf.write("Subject ID, Ref, Near, Near - Ref (Near path: expect +ve), Near - Ref (Ref path: expect +ve)\n")
    for ix, sub in enumerate(subs):
        attraction_metric[sub] = {}
        if str(sub) not in grasp_timing_subids:
            dur_grasp_times = None
        else:
            with open(grasp_timing_files[ix], 'rb') as f:
                grasp_times = pickle.load(f)
            grasp_times = {trial_name_mapper[k]: grasp_times[k] for k in grasp_times.keys() if k in trial_name_mapper.keys()}

        # Get kinect file for visual search task
        kinFiles = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + '*.xef')
        if not kinFiles:
            print('No kinect files found for subject: ' + str(sub) + '\n\n\n\n')
            continue
        kinFiles.sort(key=os.path.getmtime)
        # get kinect1 stop time
        k1MTime = os.path.getmtime(kinFiles[0])
        k1MDateTime = datetime.datetime.fromtimestamp(k1MTime)  # .strftime('%Y%m%d%H%M%S%f')
        k1CDateTime = datetime.datetime.fromtimestamp(os.path.getctime(kinFiles[0]))  # .strftime('%Y%m%d%H%M%S%f')

        df = pandas.read_csv(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + 'output_body.csv',
                             low_memory=False)

        # get rows for joint type 13(body center)
        temp = df.loc[df[' jointType'] == 13]
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

        # Figure out the order for the near, far and geometric objects in current trial
        trial_conditions_file = glob.glob(os.path.join(dataRoot, str(sub), 'NaturalisticVisSearch', '*_playroom.csv'))
        trial_conditions_filepath = trial_conditions_file[0]
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

            if trial_set == 0:
                print "Fault: ", trial_conditions_filepath
                break
        trial_types[sub] = trial_set

        #############################################################
        # eyetracking
        feyetrackPSData = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1_*.txt')
        if not feyetrackPSData:
            print('No eyetracking PSData text file. Moving to next subject \n\n')
            continue

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
            continue
        eyetrOutput = pandas.read_csv(feyetrackInfo[0], skiprows=5, delimiter=' ', index_col=False)

        # read trials
        vsStartEnd = pandas.read_csv('vs_trials' + os.sep + str(sub) + '.csv', header=None, index_col=2)
        fig = plt.figure(figsize=(24, 12))
        if __STOP_AT_GRASP_TIMES:
            fig.suptitle(str(sub)+("(no grasp times)" if (str(sub) not in grasp_timing_subids) else "(with grasp)"), fontsize=12)
        else:
            fig.suptitle(str(sub), fontsize=12)

        trStartTimes = {}
        trEndTimes = {}
        for index, row in vsStartEnd.iterrows():
            print(sub, index, row[0], row[1])
            trName = index
            startFr = row[0]
            endFr = row[1]

            if __STOP_AT_GRASP_TIMES:
                if str(sub) in grasp_timing_subids:
                    if trName in grasp_times:
                        lastFr = grasp_times[trName]
                        endFr = lastFr

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

            trStartTimes[trName] = trStartTime
            trEndTimes[trName] = trEndTime

        mux, muy = np.mean(np.asarray(startPoints), axis=0)
        for i, k in enumerate(trialNames[:6]):
            ax = fig.add_subplot(2,3,i+1,title=k)
            #ln, = ax.plot(mux, muy, 'o', c='black', markersize=10, label='start')
            triplet_name = [triplet_names[trial_set-1][k][x] for x in range(1, 4)]
            scatter_st_en = []
            for j, name in enumerate(triplet_name):
                #if sub in flagged_trials.keys():
                #    if name in flagged_trials[sub]:
                if triplet_name[0] not in attraction_metric[sub].keys():
                    attraction_metric[sub][triplet_name[0]] = (0, 0)
                emux, emuy = np.mean(np.asarray(endPoints[trial_set-1][name]), axis=0)
                #ln, = ax.plot(emux, emuy, 'o', c=colorDict[k][j], markersize=10, label="Pos "+name)
                objcond = j+1
                if name in trStartTimes.keys():
                    trStartTime = trStartTimes[name]
                    trEndTime = trEndTimes[name]
                else:
                    if triplet_name[0] not in attraction_metric[sub].keys():
                        attraction_metric[sub][triplet_name[0]] = (0, 0)
                    if not j == 2:
                        pf.write(str(sub)+','+triplet_name[0]+','+triplet_name[1]+','+"0"+','+"0"+'\n')
                    continue
                trialDf = skelDataFrame[
                    (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

                if trialDf.empty:
                    # missing trial probably
                    continue

                # Remove all the samples that are too close to sensor. They are likely wrong tracking
                trialDf = trialDf.drop(trialDf[trialDf[' position.Z'] < 0.5].index)

                # Find the points where tracking was lost
                sampleGaps = np.diff(np.array(trialDf['# timestamp'].astype(float)))

                # Get length of trial. Exclude the time when the skeleton was lost
                trTrackTime = sum(sampleGaps[sampleGaps < 35000])

                # If samples are >40 millisec apart, there was a jump in skeleton
                # skelJumpIdx = np.add(np.where(sampleGaps > 35000)[0],1)

                x = trialDf[' position.X'].astype(float)
                y = trialDf[' position.Z'].astype(float)

                #xsmooth = scsig.savgol_filter(x, 5, 2)
                #ysmooth = scsig.savgol_filter(y, 5, 2)

                simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

                sx, sy = simplePath.T

                xGapIdx = np.where(abs(np.diff(np.array(x))) > 0.5)[0]
                yGapIdx = np.where(abs(np.diff(np.array(y))) > 0.5)[0]
                skelJumpIdx = np.add(np.union1d(xGapIdx, yGapIdx), 1)

                # Add index 0 at the beginning to create the split range
                skelJumpIdx = np.insert(skelJumpIdx,0,0)
                skelJumpIdx = np.append(skelJumpIdx,len(trialDf)-1)

                start = skelJumpIdx[0]
                end = skelJumpIdx[-1]
                for st, en in zip(skelJumpIdx, skelJumpIdx[1:]):
                    ln, = ax.plot(x[st:en], y[st:en], '-', color=colorDict[k][j], label='')
                    ax.plot(x[en-1:en+1], y[en-1:en+1], ':', color=colorDict[k][j], label='')

                ln.set_label(name+': '+condition_types[objcond])
                ax.legend(loc='best', prop={'size': 8})
                scatter_st_en.append((x[:start+1], y[:start+1], x[end:], y[end:]))

                # X limit is width of the room on either side of the kinect sensor
                ax.set_xlim([-2, 2])
                # Y limit is the maximum depth in meters of the room (Y is taken as the Z of Kinect here)
                ax.set_ylim([0, 5])
                ax.invert_xaxis()
                #ax.plot(sx[turnIdx], sy[turnIdx], 'o', color=colorDict[k][j], markersize=4)

                dfTurnPoints[name][sub] = len(turnIdx)/trTrackTime

                ttype = trial_types[sub] - 1
                dist_1 = 0
                dist_2 = 0
                ref_dist_1 = 0
                ref_dist_2 = 0
                if j == 0:
                    ref_emux, ref_emuy = np.mean(np.asarray(endPoints[ttype][name]), axis=0)
                    ref_turnIdx = turnIdx
                    ref_sx = sx
                    ref_sy = sy
                    continue
                if j == 1:
                    emux, emuy = np.mean(np.asarray(endPoints[ttype][name]), axis=0)
                    for idx in turnIdx:
                        x1, y1 = sx[idx], sy[idx]
                        dist1 = calc_line_point_distance([mux, muy, ref_emux, ref_emuy], [x1, y1])
                        dist2 = calc_line_point_distance([mux, muy, emux, emuy], [x1, y1])
                        dist_1 += dist1
                        dist_2 += dist2
                    for idx in ref_turnIdx:
                        x1, y1 = ref_sx[idx], ref_sy[idx]
                        dist1 = calc_line_point_distance([mux, muy, ref_emux, ref_emuy], [x1, y1])
                        dist2 = calc_line_point_distance([mux, muy, emux, emuy], [x1, y1])
                        ref_dist_1 += dist1
                        ref_dist_2 += dist2
                    pf.write(str(sub)+','+triplet_name[0]+','+triplet_name[1]+','+str(dist_2-dist_1)+','+str(ref_dist_2-ref_dist_1)+'\n')
                if triplet_name[0] not in attraction_metric[sub].keys():
                    attraction_metric[sub][triplet_name[0]] = ((dist_2 - dist_1), (ref_dist_2 - ref_dist_1))
                #if j == 2:
                #    if triplet_name[0] in attraction_metric[sub].keys():
                #        attraction_metric[sub][triplet_name[0]] -= dist
                #    else:
                #        attraction_metric[sub][triplet_name[0]] = 0

            for j, pts in enumerate(scatter_st_en):
                ax.plot(pts[0], pts[1], '*', c=colorDict[k][j], label='')
                ax.plot(pts[2], pts[3], '^', c=colorDict[k][j], label='')

        #save maximised figures in pdf
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        pp.savefig(fig)
        plt.close()

pp.close()
'''
if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_withgrasp_randomness.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

straightLineDists = {}
abs_ratios = {}

fig = plt.figure(figsize=(24,12))
fig.suptitle("Location of objects")

if __STOP_AT_GRASP_TIMES:
    mux, muy = np.mean(np.asarray(startPoints), axis=0)
    for ttype in range(2):
        ax = fig.add_subplot(1,2,ttype+1,title="Trial type "+str(ttype+1))
        # X limit is width of the room on either side of the kinect sensor
        ax.set_xlim([-2, 2])
        # Y limit is the maximum depth in meters of the room (Y is taken as the Z of Kinect here)
        ax.set_ylim([0, 5])
        ax.invert_xaxis()
        ln, = ax.plot(mux, muy, '*', c='black', label='start')
        straightLineDists[ttype] = {}
        abs_ratios[ttype] = {}
        endPts = endPoints[ttype]
        for idx, k in enumerate(endPts.keys()):
            emux, emuy = np.mean(np.asarray(endPts[k]), axis=0)
            ln, = ax.plot(emux, emuy, 'o', c=cms[3](1. * (idx+1) / 18.), label=k)
            straightLineDists[ttype][k] = calc_distance(mux, muy, emux, emuy)
            abs_ratios[ttype][k] = {}
            for sub in subs:
                if trial_types[sub]-1 == ttype:
                    if sub in flagged_trials.keys():
                        if k in flagged_trials[sub]:
                            abs_ratios[ttype][k][sub] = 0
                            continue
                    if k in pathlengths[sub].keys():
                        abs_ratios[ttype][k][sub] = (pathlengths[sub][k] / straightLineDists[ttype][k])
                    else:
                        abs_ratios[ttype][k][sub] = 0
        ax.legend(loc='best', prop={'size': 8})

pp.savefig(fig)
plt.close()

for ttype in range(2):
    for i, tname in enumerate(triplet_names[ttype].keys()):
        names = triplet_names[ttype][tname]
        subratios0 = abs_ratios[ttype][names[1]]
        subratios1 = abs_ratios[ttype][names[2]]
        subratios2 = abs_ratios[ttype][names[3]]
        ratios = []; labels = []; colors = []
        for idx, (r1, r2, r3) in enumerate(zip(subratios0.values(), subratios1.values(), subratios2.values())):
            ratios.append(r1); ratios.append(r2); ratios.append(r3)
            labels.append('*'); labels.append(subratios0.keys()[idx]); labels.append('*')
            cols = [cms[3](1. * (x+1) / 3.0001) for x in range(3)]
            colors.append(cols[0]); colors.append(cols[1]); colors.append(cols[2])

        #ratios = np.asarray(ratios)
        #labels = np.asarray(labels)
        fig = plt.figure(figsize=(24,12))
        fig.suptitle("Curviness measure for trajectories")
        ax = fig.add_subplot(1,1,1,title="Type "+str(ttype+1)+", Ref "+tname+", "+
                             "Near "+names[2]+", Far "+names[3])
        xs = np.linspace(0.0, 2.0, 3)
        last = 2.0
        space = 2.0
        for i in range((len(ratios)/3)-1):
            xs = np.concatenate([xs, np.linspace(last+space, last+space+2.0, 3)])
            last = last+space+2.0

        pos = ['Reference', 'Near', 'Far']
        ax.bar(x=xs, height=ratios, width=0.5, tick_label=labels, color=colors)
        pp.savefig(fig)

pp.close()
plt.close()

if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_withgrasp_randomness_alltrials.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

diff_per_subj = {}
#diff_across_trials = {}
for sub in subs:
    ttype = trial_types[sub]-1
    diff_per_subj[sub] = {}
    for k in triplet_names[ttype].keys():
        if (sub in abs_ratios[ttype][k].keys()):
            if(sub in abs_ratios[ttype][triplet_names[ttype][k][2]].keys()):
                if abs_ratios[ttype][triplet_names[ttype][k][2]][sub] == 0:
                    diff_per_subj[sub][k] = (0, 0)
                else:
                    curv_ref_near = abs_ratios[ttype][triplet_names[ttype][k][1]][sub]
                    curv_ref = abs_ratios[ttype][triplet_names[ttype][k][2]][sub]
            else:
                diff_per_subj[sub][k] = (0, 0)
            if(sub in abs_ratios[ttype][triplet_names[ttype][k][3]].keys()):
                if abs_ratios[ttype][triplet_names[ttype][k][3]][sub] == 0:
                    diff_per_subj[sub][k] = (0, 0)
                else:
                    curv_ref_far = (abs_ratios[ttype][triplet_names[ttype][k][1]][sub]
                                 / abs_ratios[ttype][triplet_names[ttype][k][3]][sub])
            else:
                diff_per_subj[sub][k] = (0, 0)
        else:
            diff_per_subj[sub][k] = (0, 0)
        if k not in diff_per_subj[sub].keys():
            diff_per_subj[sub][k] = (curv_ref_near, curv_ref)

_eps = 1e-4
colors1 = [cms[2](1. / 3.) for _ in range(6)]
colors2 = [cms[0](1. / 3.) for _ in range(6)]
space = 1.0
for i, sub in enumerate(subs):
    fig = plt.figure(figsize=(24,12))
    fig.suptitle("Distribution of curviness difference Near - Ref for subject "+str(sub)+" across all trials")
    diffs = diff_per_subj[sub].values()
    xs = np.linspace(0.0, 6.0, 6)
    labels = np.asarray(diff_per_subj[sub].keys())
    near = np.asarray([x[0] for x in diffs])
    ref = np.asarray([x[1] for x in diffs])
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_tick_params(labelrotation=45)
    ratio = near - ref
    ax.bar(x=xs, height=ratio, width=0.5, tick_label=labels, color=colors2)
    #ax.bar(x=xs, height=ref_far, width=0.5, bottom=ref_near, tick_label=labels, color=colors2)
    #ax.plot([0, 10], [1, 1], c='black')
    #3np.insert(xs, 0, [0])
    #np.insert(ratio, 0, [0])
    #ax.plot(xs, ratio, c='khaki')
    pp.savefig(fig)

pp.close()
plt.close()

if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_withgrasp_attraction.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

for i, sub in enumerate(subs):
    fig = plt.figure(figsize=(24,12))
    fig.suptitle("Attraction measure for subject "+str(sub)+" across all trials")
    ax = fig.add_subplot(1,1,1)
    attractions = attraction_metric[sub].values()
    xs = np.linspace(0.0, 6.0, len(attractions))
    labels = np.asarray([x for x in attraction_metric[sub].keys()])
    #ratios = np.asarray([attractions[x] for x in attractions.keys()])
    ax.xaxis.set_tick_params(labelrotation=45)
    ax.bar(x=xs, height=attractions, width=0.75, tick_label=labels, color=colors1)
    pp.savefig(fig)

plt.close()
pp.close()

if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_mean_curviness_subject.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

_eps = 1e-4
avg_ref = []
avg_near = []
avg_far = []
for i, sub in enumerate(subs):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("Average curviness score in trials for Ref, Near and Far for "+str(sub))
    ax = fig.add_subplot(1,1,1)
    ttype = trial_types[sub]-1
    sums = [0] * 3
    lens = [0] * 3
    std = [[]] * 3
    for k in trialNames[:6]:
        triplet_name = [triplet_names[ttype][k][x] for x in range(1, 4)]
        for j, name in enumerate(triplet_name):
            if name in abs_ratios[ttype].keys():
                std[j].append(abs_ratios[ttype][name][sub])
                sums[j] += abs_ratios[ttype][name][sub]
                lens[j] += 1
    std = np.asarray(std)
    std = np.std(std, axis=1)
    sums = np.asarray(sums)
    lens = np.asarray(lens)
    for i in range(len(sums)):
        if lens[i] > 0:
            sums[i] = sums[i] / lens[i]
        else:
            sums[i] = 0
    avg_ref.append(sums[0])
    avg_near.append(sums[1])
    avg_far.append(sums[2])
    xs = np.linspace(0.0, 6.0, 3)
    labels = ['Ref', 'Near', 'Far']
    ax.bar(x=xs, height=sums, yerr=std, width=0.75, tick_label=labels, color=colors2, ecolor='black', capsize=10)
    pp.savefig(fig)

plt.close()
pp.close()

if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_mean_attraction.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

fig = plt.figure(figsize=(24, 12))
fig.suptitle("Average attraction score per subject")
avgs = [0] * len(subs)
std = [0] * len(subs)
for i, sub in enumerate(subs):
    attractions = attraction_metric[sub].values()
    avgs[i] = np.mean(np.asarray(attractions))
xs = np.linspace(0.0, 20.0, len(subs))
labels = subs
ax = fig.add_subplot(1,1,1)
ax.bar(x=xs, height=avgs, width=0.5, tick_label=labels, color=colors1)
pp.savefig(fig)

plt.close()
pp.close()

avg_ref = np.asarray(avg_ref)
avg_near = np.asarray(avg_near)
avg_far = np.asarray(avg_far)
avg_att = np.asarray(avgs)

ref_mu, ref_std = np.mean(avg_ref), np.std(avg_ref)
near_mu, near_std = np.mean(avg_near), np.std(avg_near)
far_mu, far_std = np.mean(avg_far), np.std(avg_far)
att_mu, att_std = np.mean(avg_att), np.std(avg_att)

print ref_mu, near_mu, far_mu, att_mu

if __STOP_AT_GRASP_TIMES:
    pp = PdfPages('kinectPathResults_mean_std_attributes.pdf')
else:
    pp = PdfPages('kinectPathResults_withoutgrasp.pdf')

print "Final plot"
fig = plt.figure(figsize=(24, 12))
fig.suptitle("Average and Standard deviation for attribute scores across subjects")
xs = np.linspace(0.0, 6.0, 4)
labels = ['Curviness_Ref', 'Curviness_Near', 'Curviness_Far', 'Attraction_Near_Ref']
ax = fig.add_subplot(1,1,1)
avg_final = np.asarray([ref_mu, near_mu, far_mu, att_mu])
std_final = np.asarray([ref_std, near_std, far_std, att_std])
ax.bar(x=xs, height=avg_final, yerr=std_final, width=0.75, tick_label=labels, color=colors2, ecolor='black', capsize=10)
pp.savefig(fig)

plt.close()
pp.close()
'''
