import pandas
import os, glob
import time
import platform
import datetime
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from RDPpath import getTurningPoints
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from scipy import stats
import seaborn as sn
#mpl.style.use('seaborn')
import pylab as pl

if not platform.system() == 'Windows':
    print('This code should only be run on Windows platform. Obtaining creation time only works in Windows')
    exit()
subs = [3773, 3920, 3868, 3823, 3822, 3805, 3941, 3944, 3829, 3947, 3809,
        3930, 3927, 3798, 3934, 3898, 3899, 3871, 3872, 3894, 3859, 3834,
        3835, 3939, 3765, 3870, 3867, 3818, 3936, 3839, 3918, 3864, 3884,
        3881, 3883, 3952, 3848, 3954, 3950, 3991, 3998, 3999]

subs.sort()
# dataRoot = '/media/CLPS_Amso_Lab/Playroom/NEPIN_SmartPlayroom/SubjectData/'
dataRoot = '\\\\files.brown.edu\Research\CLPS_Amso_Lab\Playroom\NEPIN_SmartPlayroom\SubjectData'
pp = PdfPages('vs_path_spatial_scaling.pdf')

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
max_pathlen = 128
keys = []
for tr in trialNames:
    keys.append(tr+'_spatial')
    keys.append(tr+'_temporal')

dfTurnPoints = pandas.DataFrame([], index=subs, columns=keys )

slope_arr = []

def spatial_scaling_factor(x, y):
    global i, k, ln, slope, intercept
    #steps increase in geometric progression
    n_steps = int(np.log2(len(x))) + 1
    #generate steps
    steps_arr = [2 ** i for i in range(0, n_steps)]
    #spatial scaling array
    spsc_arr = []
    for step in steps_arr:
        #resample coordinates in step size
        resample_x = x[0::step]
        resample_y = y[0::step]
        #get lengths of above chunks and sum up
        resample_diff_x = np.diff(resample_x)
        resample_diff_y = np.diff(resample_y)
        total_dist = sum(np.sqrt(np.square(resample_diff_x) + np.square(resample_diff_y)))
        spsc_arr.append(total_dist / step)
    k = np.log2(steps_arr)
    d = np.log2(spsc_arr)
    ln, = ax1.plot(k, d, '.', color=colorDict[trName], label='')
    # determine best fit line
    # slope value is the spatial scaling
    slope, intercept, r_value, p_value, std_err = stats.linregress(k, d)

    line = slope * k + intercept
    ax1.plot(k, line)
    ax1.set_xlabel(r'$\log_2$(k)')
    ax1.set_ylabel(r'$\log_2$(d)')



min_stepsize = 1
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
    skelDataFrame['# reltimestamp'] = skelDataFrame['# timestamp']

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
        slope_arr = []
        pathlen_arr = []
        dist_arr = []

        fig = plt.figure()

        ax1 = fig.add_subplot(221)


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

        trialDf['# reltimestamp'] = trialDf['# reltimestamp'] - trialDf['# reltimestamp'].iloc[0]
        
        trialDf[' relposition.X'] = trialDf[' position.X'].astype(float) - trialDf[' position.X'].iloc[0].astype(float)
        trialDf[' relposition.Z'] = trialDf[' position.Z'].astype(float) - float(trialDf[' position.Z'].iloc[0])

        # spatial_scaling_factor(trialDf[' position.X'].astype(float), trialDf[' position.Z'].astype(float))




        # coefficients = np.polyfit(k, d, 1)
        # polynomial = np.poly1d(coefficients)
        # ys = polynomial(k)
        # ax1.plot(k, d)
        # ax1.plot(k, ys)

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

        ax2 = fig.add_subplot(222)
        for st, en in zip(skelJumpIdx, skelJumpIdx[1:]):

            if sum(pathlen_arr, en - st) > max_pathlen:
                en = max_pathlen - sum(pathlen_arr) + 1
            # too short a sequence
            if not en - st > 2:
                continue

            pathlen_arr.append(en - st)
            ln, = ax2.plot(x[st:en], y[st:en], '-', color=colorDict[trName], label='')
            spatial_scaling_factor(x[st:en], y[st:en])
            slope_arr.append(slope)
            ax2.plot(x[en-1:en+1], y[en-1:en+1], ':', color=colorDict[trName], label='')

            # temporal scaling
            diff_x = np.diff(x[st:en])
            diff_y = np.diff(y[st:en])
            dist_arr.extend(np.sqrt(np.square(diff_x) + np.square(diff_y)))

        slope_avg = np.average(slope_arr, weights=pathlen_arr)
        ln.set_label(trName + ' ' + "Slope: {0:.2f}".format(slope_avg))
        #ax.plot(x, y, '-', color=colorDict[trName], label=trName)
        #ax.plot(sx, sy, '--', label='simplified path')
        ax1.set_title("{0} : {1} : weighted avg $d$ {2:.2f}".format(str(sub), trName, abs(slope_avg)))

        # mark start point
        ax2.plot(x.iloc[0], y.iloc[0], 'D',label=('Start'))
        # mark stop point
        ax2.plot(x.iloc[-1], y.iloc[-1], 's',label=('End'))
        ax2.plot(sx[turnIdx], sy[turnIdx], 'o', color=colorDict[trName], markersize=4)

        # x-limit is width of the room on either side of the kinect sensor
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(0, 5)
        # skeleton data is in meters
        ax2.set_xlabel('meters')
        ax2.set_ylabel('meters')
        ax2.legend(loc='best', prop={'size': 8})
        # the data we get is axis inverted. So invert in back to display
        ax2.invert_xaxis()
        ax2.legend()

        #plt.show()

        #temporal scaling
        dist_arr_norm = np.divide(dist_arr, 1) # 1.004e-6 was found to be the minimum step size in all dataset
        # h = sorted(dist_arr_norm)
        # fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
        ax3 = fig.add_subplot(223)
        # ax3.plot(h, fit, 'b-o')
        # ax3.set_xscale('log')
        counts, bins, bars = ax3.hist(dist_arr_norm, normed=True, bins=2 ** np.linspace(np.log2(min(dist_arr_norm)), np.log2(max(dist_arr_norm)), 10))
        # ax3.plot(counts)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('step size')
        ax3.set_ylabel('count')
        counts_nzidx = np.nonzero(counts)
        ax3.plot(bins[counts_nzidx], counts[counts_nzidx])

        m, c = np.polyfit(np.log2(counts[counts_nzidx]), np.log2(bins[counts_nzidx]), 1)
        # y_fit = np.exp(m * np.log2(counts[2:]) + c)
        # ax3.plot(counts[2:], y_fit)
        ax3.set_title("{0} : {1} : $alpha$ {2:.2f}".format(str(sub), trName, abs(m)))
        plt.tight_layout()

        ax4 = fig.add_subplot(224)
        ax4.plot(dist_arr_norm, label='motion/step')
        ax4.set_xlabel('samples')
        ax4.set_ylabel('displacement')
        ax4.legend()

        min_stepsize = min(min_stepsize, min(dist_arr))
        print('min step size: ' + str(min_stepsize))

        trialDf.to_excel(writer, sheet_name=trName, index=False)
        #normalize the number of turns
        # dfTurnPoints[trName][sub] = len(turnIdx)/trTrackTime
        dfTurnPoints[trName+'_spatial'][sub] = slope_avg
        dfTurnPoints[trName+'_temporal'][sub] = m


        # save maximised figures in pdf
        manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # manager.frame.Maximize(True)

        plt.tight_layout()
        pp.savefig(fig)
        plt.close()


    #save all the trial data for subject
    writer.save()

dfTurnPoints.to_csv('vs_path_spatial_scaling.csv')

dd = dfTurnPoints.replace(r'\s+', np.nan, regex=True)
plt.figure(figsize=(10, 7))
mask = dd.isnull()
sn.heatmap(dd, mask=mask, annot=True)

pp.close()