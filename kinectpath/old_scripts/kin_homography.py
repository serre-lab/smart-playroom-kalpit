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
import pickle

#mpl.style.use('seaborn')

# if not platform.system() == 'Windows':
#     print('This code should only be run on Windows platform. Obtaining creation time only works in Windows')
#     exit()
# subs = [3773, 3920, 3868, 3823, 3822, 3805, 3941, 3944, 3829, 3947, 3809,
#         3930, 3927, 3798, 3934, 3898, 3899, 3871, 3872, 3894, 3859, 3834,
#         3835, 3939, 3765, 3870, 3867, 3818, 3936, 3839, 3918, 3864, 3884,
#         3881, 3883, 3952, 3848, 3954, 3950, 3986, 3991, 3998, 3999]
subs = [3898]
subs.sort()
dataRoot = '/media/CLPS_Amso_Lab/Playroom/NEPIN_SmartPlayroom/SubjectData/'
# dataRoot = '\\\\files.brown.edu\Research\CLPS_Amso_Lab\Playroom\NEPIN_SmartPlayroom\SubjectData'
pp = PdfPages('kinectWithHomography.pdf')

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

NUM_COLORS = len(trialNames)
cm = plt.get_cmap('gist_rainbow')
colorlist = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
colorDict = dict(zip(trialNames, colorlist))

dfTurnPoints = pandas.DataFrame([], index=subs, columns=trialNames)

for sub in subs:
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

    time_ms = skelDataFrame['# timestamp']
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

    fig = plt.figure()
    plt.title(str(sub))


    homFile = open(dataRoot + os.sep + str(sub) + os.sep + 'kinect1' + os.sep + '/homography_dict.p', 'rb')
    [scale, rot] = pickle.load(homFile).values()
    scale = np.array(scale)
    rot = np.array(rot)

    startFr = 8559
    endFr = 18832
    stSceneQTtime = eyetrOutput[eyetrOutput['recordFrameCount'] >= startFr].iloc[0][
        'sceneQTtime(d:h:m:s.tv/ts)'].split('.')
    enSceneQTtime = eyetrOutput[eyetrOutput['recordFrameCount'] >= endFr].iloc[0][
        'sceneQTtime(d:h:m:s.tv/ts)'].split('.')

    stSceneTimeStr = stSceneQTtime[0][2:] + '.' + str(int(eval(stSceneQTtime[1] + '.0') * 1000))
    enSceneTimeStr = enSceneQTtime[0][2:] + '.' + str(int(eval(enSceneQTtime[1] + '.0') * 1000))
    t = datetime.datetime.strptime(stSceneTimeStr, "%H:%M:%S.%f")
    tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

    sessStartTime = (eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f')

    t = datetime.datetime.strptime(enSceneTimeStr, "%H:%M:%S.%f")
    tDelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

    sessEndTime = (eyetrackStart + tDelta).strftime('%Y%m%d%H%M%S%f')

    sessDf = skelDataFrame[
        (skelDataFrame['# timestamp'] >= sessStartTime) & (skelDataFrame['# timestamp'] <= sessEndTime)]

    sessEyeDf = eyetrOutput[
        (eyetrOutput['recordFrameCount'] > startFr) & (eyetrOutput['recordFrameCount'] <= endFr)]

    aa = sessEyeDf['sceneQTtime(d:h:m:s.tv/ts)']
    a = [x.split('.') for x in aa]
    b = [x[0][2:] + '.' + str(int(eval(x[1] + '.0') * 1000)) for x in a]
    c = [datetime.datetime.strptime(x, "%H:%M:%S.%f") for x in b]
    d = [datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds=x.microsecond) for x in c]
    e = np.array([float((eyetrackStart + x).strftime('%Y%m%d%H%M%S%f')) for x in d])

    if sessDf.empty:
        # missing trial probably
        continue

    # remove all the samples that are too close to sensor. They are likely wrong tracking
    sessDf = sessDf.drop(sessDf[sessDf[' position.Z'] < 0.5].index)

    # find the points where tracking was lost
    sampleGaps = np.diff(np.array(sessDf['# timestamp'].astype(float)))
    # get length of trial. Exclude the time when the skeleton was lost
    trTrackTime = sum(sampleGaps[sampleGaps < 35000])
    # if samples are >40 millisec apart, there was a jump in skeleton
    # skelJumpIdx = np.add(np.where(sampleGaps > 35000)[0],1)

    # smooth the path and display
    # plt.scatter(trialDf[' position.X'], trialDf[' position.Z'],label=trName)
    # xhat = scipy.signal.savgol_filter(trialDf[' position.X'], 11, 2)  # window size 51, polynomial order 3
    # zhat = scipy.signal.savgol_filter(trialDf[' position.Z'], 11, 2)  # window size 51, polynomial order 3
    # #plt.plot(trialDf[' position.X'], trialDf[' position.Z'])
    # plt.plot(xhat, zhat,label=trName)
    # plt.legend(prop={'size': 6})
    # plt.show()

    sessX = sessDf[' position.X'].astype(float)
    sessY = sessDf[' position.Z'].astype(float)

    # simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T, use_rdp=True)
    simplePath, turnIdx = getTurningPoints(np.vstack((sessX, sessY)).T)

    sx, sy = simplePath.T

    for i, n in enumerate(scale):
        if n > 2:
            scale[i] = 2
        if n == -1:
            scale[i] = 0

    ax1 = plt.subplot(311)
    ax1.plot(sessDf['# timestamp'].astype(float), sessX)
    ax1.plot(sessDf['# timestamp'].astype(float), sessY)
    ax1.set_ylabel('kinect x,y loc.')
    ax2 = plt.subplot(312)
    ax2.set_ylabel('scale')
    ax2.plot(e, scale)
    ax3 = plt.subplot(313)
    ax3.plot(e, rot)
    ax3.set_xlabel('time')
    ax3.set_ylabel('rotation')


    #ax = fig.add_subplot(111, projection='3d')
    for index, row in vsStartEnd.iterrows():
        #print(sub, index, row[0], row[1])
        trName = index
        startFr = row[0]
        endFr = row[1]
        print "Trial Name: ", trName

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

        ax1.axvline(x=float(trStartTime), color='red',linewidth=3)
        ax1.axvline(x=float(trEndTime), color='red',linewidth=3)
        ax2.axvline(x=float(trStartTime), color='red',linewidth=3)
        ax2.axvline(x=float(trEndTime), color='red',linewidth=3)
        ax3.axvline(x=float(trStartTime), color='red',linewidth=3)
        ax3.axvline(x=float(trEndTime), color='red',linewidth=3)
        trialDf = skelDataFrame[
            (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]
        startMS = time_ms[skelDataFrame['# timestamp'] >= trStartTime].iloc[0]
        endMS = time_ms[skelDataFrame['# timestamp'] <= trEndTime].iloc[-1]
        print "Start: ", (datetime.datetime.fromtimestamp(startMS/1000.0)).strftime("%M:%S.%f")
        print "End: ", (datetime.datetime.fromtimestamp(endMS/1000.0)).strftime("%M:%S.%f")

        if trialDf.empty:
            #missing trial probably
            continue

        # remove all the samples that are too close to sensor. They are likely wrong tracking
        trialDf = trialDf.drop(trialDf[trialDf[' position.Z'] < 0.5].index)

        bb = sessDf[(sessDf['# timestamp'].astype(float) > float(trStartTime)) & (sessDf['# timestamp'].astype(float) < float(trEndTime))]
        d1 = np.where((e >= float(trStartTime)) & (e <= float(trEndTime)))

        fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, sharex=True)
        fig2.suptitle(trName)
        ax11.plot(bb['# timestamp'].astype(float), bb[' position.Z'].astype(float))
        ax11.plot(bb['# timestamp'].astype(float), bb[' position.X'].astype(float))
        # ax11.autoscale(enable=True, axis='y', tight=True)
        ax22.plot(e[d1], scale[d1])
        ax33.plot(e[d1], rot[d1])
        # plt.autoscale(enable=True, axis='y', tight=True)
        pp.savefig(fig2)
        plt.close(fig2)


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

        # simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T, use_rdp=True)
        simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

        sx, sy = simplePath.T


        for i, n in enumerate(scale):
            if n > 2:
                scale[i] = 2
            if n == -1:
                scale[i] = 0

        xGapIdx = np.where(abs(np.diff(np.array(x))) > 0.5)[0]
        yGapIdx = np.where(abs(np.diff(np.array(y))) > 0.5)[0]
        skelJumpIdx = np.add(np.union1d(xGapIdx, yGapIdx), 1)
        #add index 0 at the beginning to create the split range
        skelJumpIdx = np.insert(skelJumpIdx,0,0)
        skelJumpIdx = np.append(skelJumpIdx,len(trialDf)-1)

        # for st, en in zip(skelJumpIdx, skelJumpIdx[1:]):
        #     ln, = plt.plot(x[st:en], y[st:en], '-', color=colorDict[trName], label='')
        #     plt.plot(x[en-1:en+1], y[en-1:en+1], ':', color=colorDict[trName], label='')
        #
        # ln.set_label(trName)
        # #ax.plot(x, y, '-', color=colorDict[trName], label=trName)
        # #ax.plot(sx, sy, '--', label='simplified path')
        # plt.plot(sx[turnIdx], sy[turnIdx], 'o', color=colorDict[trName], markersize=4)

        #plt.show()

        # trialDf.to_excel(writer, sheet_name=trName, index=False)
        #normalize the number of turns
        dfTurnPoints[trName][sub] = len(turnIdx)/trTrackTime

    pp.savefig(fig)
    plt.legend(loc='best', prop={'size': 8})

    #save maximised figures in pdf
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    pp.savefig(fig)
    plt.close()

# dfTurnPoints.to_csv('vsTurnPoints.csv')
pp.close()
