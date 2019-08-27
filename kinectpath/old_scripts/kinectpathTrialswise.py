import pandas
import os, glob
import time
import platform
import datetime
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.signal
from RDPpath import getTurningPoints
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

mpl.style.use('seaborn')

if not platform.system() == 'Windows':
    print('This code should only be run on Windows platform. Obtaining creation time only works in Windows')
    exit()
subs = [3773, 3920, 3868, 3823, 3822, 3805, 3941, 3944, 3829, 3947, 3809,
        3930, 3927, 3798, 3934, 3898, 3899, 3871, 3872, 3894, 3859, 3834,
        3835, 3939, 3765, 3870, 3867, 3818, 3936, 3839, 3918, 3864, 3884,
        3881, 3883, 3952, 3848, 3954, 3950, 3986, 3991, 3998, 3999]

subs.sort()
# dataRoot = '/media/CLPS_Amso_Lab/Playroom/NEPIN_SmartPlayroom/SubjectData/'
dataRoot = '\\\\files.brown.edu\Research\CLPS_Amso_Lab\Playroom\NEPIN_SmartPlayroom\SubjectData'

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

orders = [1,2]

for order in orders:
    pertrialFigures = {}
    for tr in trialNames:
        fig, ax = plt.subplots(1)
        pertrialFigures[tr] = fig, ax

    NUM_COLORS = len(subs)
    cm = plt.get_cmap('gist_rainbow')
    colorlist = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    colorDict = dict(zip(subs, colorlist))

    dfTurnPoints = pandas.DataFrame([], index=subs, columns=trialNames)
    pp = PdfPages('kinectPathResultsOrder' + str(order) + '.pdf')
    for sub in subs:

        nOrder = pandas.read_csv(dataRoot + os.sep + str(sub) + os.sep + 'NaturalisticVisSearch' + os.sep + str(sub) + '_order_playroom.csv', delimiter=',', index_col=False)["order"][0]

        #if nOrder != order:
        #    continue

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

        #############################################################
        # eyetracking
        feyetrackPSData = glob.glob(dataRoot + os.sep + str(sub) + os.sep + 'EyeTracking' + os.sep + str(sub) + '_1_*.txt')
        if not feyetrackPSData:
            print('No eyetracking PSData text file. Moving to next subject \n\n')
            continue

        #       feyetrackVid = '/eyetracker/3620_1/3620_1.mov'
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
        writer = pandas.ExcelWriter(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + str(sub) + 'VS.xlsx',
                                    engine='xlsxwriter')


        #ax = fig.add_subplot(111, projection='3d')
        for index, row in vsStartEnd.iterrows():
            print(sub, index, row[0], row[1])
            trName = index
            startFr = row[0]
            endFr = row[1]

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

            trialDf = skelDataFrame[
                (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

            if trialDf.empty:
                #missing trial probably
                continue

            # remove all the samples that are too close to sensor. They are likely wrong tracking
            trialDf = trialDf.drop(trialDf[trialDf[' position.Z'] < 0.5].index)

            #find the points where tracking was lost
            sampleGaps = np.diff(np.array(trialDf['# timestamp'].astype(float)))
            #get length of trial. Exclude the time when the skeleton was lost
            trTrackTime = sum(sampleGaps[sampleGaps < 35000])
            #if samples are >40 millisec apart, there was a jump in skeleton
            #skelJumpIdx = np.add(np.where(sampleGaps > 35000)[0],1)

            #smooth the path and display
            # plt.scatter(trialDf[' position.X'], trialDf[' position.Z'],label=trName)
            # xhat = scipy.signal.savgol_filter(trialDf[' position.X'], 11, 2)  # window size 51, polynomial order 3
            # zhat = scipy.signal.savgol_filter(trialDf[' position.Z'], 11, 2)  # window size 51, polynomial order 3
            # #plt.plot(trialDf[' position.X'], trialDf[' position.Z'])
            # plt.plot(xhat, zhat,label=trName)
            # plt.legend(prop={'size': 6})
            # plt.show()

            x = trialDf[' position.X'].astype(float)
            y = trialDf[' position.Z'].astype(float)


            simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

            sx, sy = simplePath.T

            xGapIdx = np.where(abs(np.diff(np.array(x))) > 0.5)[0]
            yGapIdx = np.where(abs(np.diff(np.array(y))) > 0.5)[0]
            skelJumpIdx = np.add(np.union1d(xGapIdx, yGapIdx), 1)
            #add index 0 at the beginning to create the split range
            skelJumpIdx = np.insert(skelJumpIdx,0,0)
            skelJumpIdx = np.append(skelJumpIdx,len(trialDf)-1)

            for st, en in zip(skelJumpIdx, skelJumpIdx[1:]):
                ln, = pertrialFigures[trName][1].plot(x[st:en], y[st:en], '-', color=colorDict[sub], label='')
                pertrialFigures[trName][1].plot(x[en-1:en+1], y[en-1:en+1], ':', color=colorDict[sub], label='')

            ln.set_label(sub)
            #ax.plot(x, y, '-', color=colorDict[trName], label=trName)
            #ax.plot(sx, sy, '--', label='simplified path')
            pertrialFigures[trName][1].plot(sx[turnIdx], sy[turnIdx], 'o', color=colorDict[sub], markersize=4)

            #plt.show()

            trialDf.to_excel(writer, sheet_name=trName, index=False)
            #normalize the number of turns
            dfTurnPoints[trName][sub] = len(turnIdx)/trTrackTime

        #save all the trial data for subject
        writer.save()
        #x-limit is width of the room on either side of the kinect sensor
        pertrialFigures[trName][1].set_xlim(-2, 2)
        pertrialFigures[trName][1].set_ylim(0, 5)
        #skeleton data is in meters
        pertrialFigures[trName][1].set_xlabel('meters')
        pertrialFigures[trName][1].set_ylabel('meters')
        pertrialFigures[trName][1].legend(loc='best', prop={'size': 6})
        #the data we get is axis inverted. So invert in back to display
        pertrialFigures[trName][1].invert_xaxis()
        #save maximised figures in pdf
        #manager = pertrialFigures[trName].get_current_fig_manager()
        #manager.window.showMaximized()


    for tr in trialNames:
        pertrialFigures[tr][1].set_title(tr + 'order:' + str(order))
        pertrialFigures[tr][1].legend()
        pp.savefig(pertrialFigures[tr][0])
        plt.close(pertrialFigures[tr][0])

    #dfTurnPoints.to_csv('vsTurnPoints.csv')
    pp.close()