import pandas
import os, glob
import time
import platform
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.signal as scsig
import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages

import mpl_toolkits.mplot3d.axes3d as p3

from utils import *
from utils.misc import reject_outliers

def write_num_turns_ref_near_far(num_turns_filepath='./results/per_trial_num_turns_ref_near_far.csv'):
    with open(num_turns_filepath, 'w') as f:
        f.write("Subject ID, trial_name, #turns, ref/near/far\n")
        for sub in subs:
            ttype= trial_types[sub] - 1
            nturns_sub = num_turns[sub]
            ref = []; near = []; far = []
            for i, k in enumerate(trialNames[:6]):
                triplet_name = [triplet_names[ttype][k][x] for x in range(1, 4)]
                if triplet_name[0] in nturns_sub.keys():
                    ref_val = nturns_sub[triplet_name[0]]
                    f.write(",".join([str(sub), str(triplet_name[0]), str(ref_val), "ref"]) + "\n")
                    ref.append(nturns_sub[triplet_name[0]])
                if triplet_name[1] in nturns_sub.keys():
                    near_val = nturns_sub[triplet_name[1]]
                    f.write(",".join([str(sub), str(triplet_name[1]), str(near_val), "near"]) + "\n")
                    near.append(nturns_sub[triplet_name[1]])
                if triplet_name[2] in nturns_sub.keys():
                    far_val = nturns_sub[triplet_name[2]]
                    f.write(",".join([str(sub), str(triplet_name[2]), str(far_val), "far"]) + "\n")
                    far.append(nturns_sub[triplet_name[2]])
            # Average stats if you need to use them
            avg_ref = sum(ref) * 1. / len(ref)
            avg_near = sum(near) * 1. / len(near)
            avg_far = sum(far) * 1. / len(far)
            avg_turns = (sum(ref) + sum(near) + sum(far)) * 1. / (len(ref) + len(near) + len(far))
            avgs = np.around([avg_ref, avg_near, avg_far, avg_turns], 3)
    return

def write_pathlengths_ref_near_far(num_turns_filepath='./results/per_trial_pathlengths_ref_near_far.csv'):
    with open(num_turns_filepath, 'w') as f:
        f.write("Subject ID, trial_name, pathlength, ref/near/far\n")
        for sub in subs:
            ttype = trial_types[sub] - 1
            plens_sub = pathlengths[sub]
            ref = []; near = []; far = []
            for i, k in enumerate(trialNames[:6]):
                triplet_name = [triplet_names[ttype][k][x] for x in range(1, 4)]
                if triplet_name[0] in plens_sub.keys():
                    ref_val = plens_sub[triplet_name[0]]
                    f.write(",".join([str(sub), str(triplet_name[0]), str(ref_val), "ref"]) + "\n")
                    ref.append(plens_sub[triplet_name[0]])
                if triplet_name[1] in plens_sub.keys():
                    near_val = plens_sub[triplet_name[1]]
                    f.write(",".join([str(sub), str(triplet_name[1]), str(near_val), "near"]) + "\n")
                    near.append(plens_sub[triplet_name[1]])
                if triplet_name[2] in plens_sub.keys():
                    far_val = plens_sub[triplet_name[2]]
                    f.write(",".join([str(sub), str(triplet_name[2]), str(far_val), "far"]) + "\n")
                    far.append(plens_sub[triplet_name[2]])
            # Average stats if you need to use them
            avg_ref = sum(ref) * 1. / len(ref)
            avg_near = sum(near) * 1. / len(near)
            avg_far = sum(far) * 1. / len(far)
            avg_turns = (sum(ref) + sum(near) + sum(far)) * 1. / (len(ref) + len(near) + len(far))
            avgs = np.around([avg_ref, avg_near, avg_far, avg_turns], 3)
    return

def write_trial_proportions_file(trial_proportions_filepath='./results/trial_proportions.csv'):
    with open(trial_proportions_filepath, 'w') as f:
        f.write("Subject ID, #(near with fixate before locomotion), \
                #(far with fixate before locomotion), #(completed near), #(completed far)\n")
        for sub in subs:
            trStartTimes = trialStartEnd[sub]['starts']
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

def write_triplet_pathlength_file(triplet_pathlength_filepath='./results/triplet_pathlengths.csv'):
    with open(triplet_pathlength_filepath, 'w') as pf:
        pf.write("Subject ID, Ref, Near, Far\n")
        for sub in subs:
            ag_ref = ref_stats[sub]
            ag_near = near_stats[sub]
            ag_far = far_stats[sub]
            ref_str_to_write = str(ag_ref[0]) + ' | ' + str(ag_ref[1]) + ' | ' + str(ag_ref[2])
            near_str_to_write = str(ag_near[0]) + ' | ' + str(ag_near[1]) + ' | ' + str(ag_near[2])
            far_str_to_write = str(ag_far[0]) + ' | ' + str(ag_far[1]) + ' | ' + str(ag_far[2])
            pf.write(str(sub)+','+ref_str_to_write+','+near_str_to_write+','+far_str_to_write+'\n')

def write_attraction_measures_file(attraction_measures_filepath='./results/attraction_metric_dist.csv'):
    with open(attraction_measures_filepath, 'w') as f:
        f.write("Subject ID, Ref, Near, Distance of Near from Near - Distance of Near from Ref, Distance of Ref from Near - Distance of Ref from Ref\n")
        for sub in subs:
            for k in trialNames[:6]:
                if k in attraction_metric[sub].keys():
                    x, y = attraction_metric[sub][k]
                    ref = triplet_names[trial_types[sub]-1][k][1]
                    near = triplet_names[trial_types[sub]-1][k][2]
                    f.write(str(sub)+','+ref+','+near+','+str(x)+','+str(y)+'\n')

def plot_path_trajectories_trial(path_trajs_plotpath='./results/per_trial_path_trajectories_1.pdf',
                            trial_set=1,
                            write_attraction=False,
                            attraction_filepath=None,
                            write_turnpoints=False):
    pp = PdfPages(path_trajs_plotpath)
    set_subs = []
    hw_sets = {0: (5, 5), 1: (4, 5)}
    for sub in subs:
        if str(sub) in grasp_timing_subids:
            sub = int(sub)
            if trial_types[sub]-1 == trial_set:
                set_subs.append(sub)
    mux, muy, muz = np.mean(reject_outliers(np.asarray(startPoints[sub])), axis=0)
    for i, k in enumerate(trialNames[:6]):
        fig = plt.figure(figsize=(24,12))
        fig.suptitle(k)
        h = hw_sets[trial_set][0]
        w = hw_sets[trial_set][1]
        for n, sub in enumerate(set_subs):
            ax = fig.add_subplot(h, w, n+1, title=str(sub))
            trStartTimes, trEndTimes = trialStartEnd[sub]['starts'], trialStartEnd[sub]['ends']
            skelDataFrame = skelDataFrames[sub]
            triplet_name = [triplet_names[trial_set][k][x] for x in range(1, 4)]
            scatter_st_en = []
            skips = []
            for j, name in enumerate(triplet_name):
                print k, j, colorDict[k][j]
                if name in trStartTimes.keys():
                    trStartTime = trStartTimes[name]
                    trEndTime = trEndTimes[name]
                else:
                    scatter_st_en.append(())
                    skips.append(j)
                    continue

                trialDf = skelDataFrame[
                    (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

                if trialDf.empty:
                    # missing trial probably
                    scatter_st_en.append(())
                    skips.append(j)
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

                simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

                sx, sy = simplePath.T

                xGapIdx = np.where(abs(np.diff(np.array(x))) > 0.5)[0]
                yGapIdx = np.where(abs(np.diff(np.array(y))) > 0.5)[0]
                skelJumpIdx = np.add(np.union1d(xGapIdx, yGapIdx), 1)

                # Add index 0 at the beginning to create the split range
                skelJumpIdx = np.insert(skelJumpIdx,0,0)
                skelJumpIdx = np.append(skelJumpIdx,len(trialDf)-1)

                #ax.scatter(sx, sy)
                start = skelJumpIdx[0]
                end = skelJumpIdx[-1]
                for st, en in zip(skelJumpIdx, skelJumpIdx[1:]):
                    ln, = ax.plot(x[st:en], y[st:en], '-', color=colorDict[k][j], label='')
                    ax.plot(x[en-1:en+1], y[en-1:en+1], ':', color=colorDict[k][j], label='')

                ln.set_label(name+': '+condition_types[j+1])
                #ax.legend(loc='best', prop={'size': 8})
                scatter_st_en.append((x[:start+1], y[:start+1], x[end:], y[end:]))

                # X limit is width of the room on either side of the kinect sensor
                ax.set_xlim([-2, 2])
                # Y limit is the maximum depth in meters of the room (Y is taken as the Z of Kinect here)
                ax.set_ylim([0, 5])
                ax.invert_xaxis()

                dfTurnPoints[name][sub] = len(turnIdx)/trTrackTime

            # '*' is the start and '^' is the end of the paths
            for j, pts in enumerate(scatter_st_en):
                if j not in skips:
                    ax.plot(pts[0], pts[1], '*', c=colorDict[k][j], label='')
                    ax.plot(pts[2], pts[3], '^', c=colorDict[k][j], label='')
            
            # 'o' plots the average start point calculated from the population of all start points for this subject
            # Write out the attraction values for this trial for the subject
            if k in attraction_metric[sub].keys():
                ax.plot(mux, muz, 'o', c='black')
                ax.text(1.90, 0.05, s=str(np.float32(attraction_metric[sub][k][0])), fontsize='small', color='green')
                ax.text(1.90, 0.40, s=str(np.float32(attraction_metric[sub][k][1])), fontsize='small', color='red')

        pp.savefig(fig)
    plt.close()
    pp.close()

    if write_attraction:
        if attraction_filepath is not None:
            write_attraction_measures_file(attraction_filepath)
        else:
            write_attraction_measures_file()
    if write_turnpoints:
        # Write the dfTurnPoints panda dataframe to file
        pass

def plot_path_trajectories(path_trajs_plotpath='./results/per_subject_path_trajectories.pdf',
                            write_attraction=True,
                            attraction_filepath=None,
                            write_turnpoints=False):
    pp = PdfPages(path_trajs_plotpath)
    for sub in subs:
        attraction_metric[sub] = {}
        fig = plt.figure(figsize=(24,12))
        trial_set = trial_types[sub]-1
        trStartTimes, trEndTimes = trialStartEnd[sub]['starts'], trialStartEnd[sub]['ends']
        skelDataFrame = skelDataFrames[sub]
        mux, muy, muz = np.mean(reject_outliers(np.asarray(startPoints[sub])), axis=0)
        for i, k in enumerate(trialNames[:6]):
            ax = fig.add_subplot(2,3,i+1,title=k)
            triplet_name = [triplet_names[trial_set][k][x] for x in range(1, 4)]
            scatter_st_en = []
            for j, name in enumerate(triplet_name):
                if name in trStartTimes.keys():
                    trStartTime = trStartTimes[name]
                    trEndTime = trEndTimes[name]
                else:
                    if not j == 2:
                        if triplet_name[0] not in attraction_metric[sub].keys():
                            attraction_metric[sub][triplet_name[0]] = (0, 0)
                    continue

                trialDf = skelDataFrame[
                    (skelDataFrame['# timestamp'] >= trStartTime) & (skelDataFrame['# timestamp'] <= trEndTime)]

                if trialDf.empty:
                    # missing trial probably
                    if not j == 2:
                        if triplet_name[0] not in attraction_metric[sub].keys():
                            attraction_metric[sub][triplet_name[0]] = (0, 0)
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
                z = trialDf[' position.Y'].astype(float)

                simplePath, turnIdx = getTurningPoints(np.vstack((x, y)).T)

                sx, sy, sz = np.vstack((x, z, y))

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

                ln.set_label(name+': '+condition_types[j+1])
                ax.legend(loc='best', prop={'size': 8})
                scatter_st_en.append((x[:start+1], y[:start+1], x[end:], y[end:]))

                # X limit is width of the room on either side of the kinect sensor
                # ax.set_xlim([-2, 2])
                # Y limit is the maximum depth in meters of the room (Y is taken as the Z of Kinect here)
                # ax.set_ylim([0, 5])
                ax.invert_xaxis()

                dfTurnPoints[name][sub] = len(turnIdx)/trTrackTime

                dist_1 = 0; dist_2 = 0; ref_dist_1 = 0; ref_dist_2 = 0
                if j == 0:
                    ref_emux, ref_emuy, ref_emuz = sx[-1], sy[-1], sz[-1]
                    ref_turnIdx = turnIdx
                    ref_sx = sx
                    ref_sy = sy
                    ref_sz = sz
                    continue
                if j == 1:
                    emux, emuy, emuz = sx[-1], sy[-1], sz[-1]
                    for x1, y1, z1 in zip(sx, sy, sz):
                        dist1 = calc_line_point_distance(
                            [mux, muy, muz, ref_emux, ref_emuy, ref_emuz],
                            [x1, y1, z1]
                        )
                        dist2 = calc_line_point_distance(
                            [mux, muy, muz, emux, emuy, emuz],
                            [x1, y1, z1]
                        )
                        dist_1 += dist1
                        dist_2 += dist2
                    for x1, y1, z1 in zip(ref_sx, ref_sy, ref_sz):
                        dist1 = calc_line_point_distance(
                            [mux, muy, muz, ref_emux, ref_emuy, ref_emuz],
                            [x1, y1, z1]
                        )
                        dist2 = calc_line_point_distance(
                            [mux, muy, muz, emux, emuy, emuz],
                            [x1, y1, z1]
                        )
                        ref_dist_1 += dist1
                        ref_dist_2 += dist2
                    '''
                    if sub == 3765:
                        fig = plt.figure()
                        ax = p3.Axes3D(fig)
                        ax.set_title(triplet_name[0]+','+triplet_name[1]+','+triplet_name[2])
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([0, 5])
                        ax.plot([mux, ref_emux], [muy, ref_emuy], [muz, ref_emuz], c='red'); ax.plot([mux, emux], [muy, emuy], [muz, emuz], c='green'); ax.plot(sx, sy, sz, c='green'); ax.plot(ref_sx, ref_sy, ref_sz, c='red'); plt.show()
                        import ipdb; ipdb.set_trace()
                        plt.close(fig)
                    '''
                # Our attraction metric is the value of normalized signed difference of a quantity calulated wrt straight-line paths to near and far objects.
                # The quantity is the sum of distance of each point on the chosen search trajectory from the chosen straight-line path. Hence, for near
                # search trajectory, the two quantities are d(near_from_near) and d(near_from_ref) and for ref search trajectory the quantities are 
                # d(ref_from_near) and d(ref_from_ref). Now the attraction metric is:
                # A(near) = (d(near_from_near) - d(near_from_ref)) / num_points_in_near and A(ref) = (d(ref_from_near) - d(ref_from_ref)) / num_points_in_ref
                # As the value is proportional to distance, the attraction is "towards straight-line near path" if the value is negative. Otherwise, the attraction
                # is "towards straight-line ref path".
                if triplet_name[0] not in attraction_metric[sub].keys():
                    # Near - Ref
                    val_near = (dist_2 - dist_1) * 1.0 / len(sx)
                    val_ref = (ref_dist_2 - ref_dist_1) * 1.0 / len(ref_sx) 
                    # Ref - Near
                    #val_near = (dist_1 - dist_2) * 1.0 / len(sx)
                    #val_ref = (ref_dist_1 - ref_dist_2) * 1.0 / len(ref_sx)
                    # Old way
                    #attraction_metric[sub][triplet_name[0]] = (1.0 / val_near, 1.0 / val_ref)
                    attraction_metric[sub][triplet_name[0]] = (val_near, val_ref)
            for j, pts in enumerate(scatter_st_en):
                ax.plot(pts[0], pts[1], '*', c=colorDict[k][j], label='')
                ax.plot(pts[2], pts[3], '^', c=colorDict[k][j], label='')

            near_att = np.asarray([x[0] for x in attraction_metric[sub].values()])
            ref_att = np.asarray([x[1] for x in attraction_metric[sub].values()])
            attraction_stats[sub] = [(np.mean(near_att), np.mean(ref_att)),
                            (np.std(near_att), np.std(ref_att))]

        pp.savefig(fig)
    plt.close()
    pp.close()

    if write_attraction:
        if attraction_filepath is not None:
            write_attraction_measures_file(attraction_filepath)
        else:
            write_attraction_measures_file()
    if write_turnpoints:
        # Write the dfTurnPoints panda dataframe to file
        pass

def plot_per_subject_triplet_randomness(per_sub_triplet_randomness_plotpath='./results/curviness.pdf'):
    pp = PdfPages(per_sub_triplet_randomness_plotpath)
    def _calculate_curviness():
        mux, muy, _ = np.mean(np.asarray(startPoints[sub]), axis=0)
        for ttype in range(2):
            straightLineDists[ttype] = {}
            curviness[ttype] = {}
            #endPts = endPoints[ttype]
            for sub in subs:
            #for idx, k in enumerate(endPts.keys()):
                #emux, emuy = np.mean(np.asarray(endPts[k]), axis=0)
                for idx, k in enumerate(endPts[sub].keys()):
                    if k not in curviness[ttype].keys():
                        curviness[ttype][k] = {}
                #for sub in subs:
                    emux, emuy, _ = endPoints[sub][k]
                    straightLineDists[ttype][k] = calc_distance(
                        [mux, muy],
                        [emux, emuy]
                    )
                    if trial_types[sub]-1 == ttype:
                        #if sub in flagged_trials.keys():
                        #    if k in flagged_trials[sub]:
                        # curviness[ttype][k][sub] = 0
                        # continue
                        if k in pathlengths[sub].keys():
                            curviness[ttype][k][sub] = (pathlengths[sub][k] / straightLineDists[ttype][k])
                        else:
                            curviness[ttype][k][sub] = 0

    _calculate_curviness()
    for sub in subs:
        ttype = trial_types[sub]-1
        labels = []
        heights = []
        fig = plt.figure(figsize=(24,12))
        fig.suptitle("Distribution of curviness difference Near - Ref for subject "+str(sub)+" across all trials")
        for k in triplet_names[ttype].keys():
            if (sub in curviness[ttype][k].keys()):
                if(sub in curviness[ttype][triplet_names[ttype][k][2]].keys()):
                    if curviness[ttype][triplet_names[ttype][k][2]][sub] == 0:
                        curv_near = 0
                        curv_ref = 0
                    else:
                        curv_near = curviness[ttype][triplet_names[ttype][k][1]][sub]
                        curv_ref = curviness[ttype][triplet_names[ttype][k][2]][sub]
            heights.append(curv_near - curv_ref)
            labels.append(k)
        xs = np.linspace(0.0, 6.0, 6)
        ax = fig.add_subplot(1,1,1)
        ax.xaxis.set_tick_params(labelrotation=45)
        ax.bar(x=xs, height=heights, width=0.5, tick_label=labels, color='black')
        pp.savefig(fig)
    plt.close()
    pp.close()

def plot_per_subject_trial_attraction_values(per_sub_attraction_plotpath='./results/attraction.pdf'):
    pp = PdfPages(per_sub_attraction_plotpath)
    for sub in subs:
        fig = plt.figure(figsize=(24,12))
        fig.suptitle("Attraction measure for subject "+str(sub)+" across all trials")
        attractions = attraction_metric[sub].values()

        ax = fig.add_subplot(1,2,1)
        att_1 = [x[0] for x in attractions]
        xs = np.linspace(0.0, 6.0, len(att_1))
        labels = np.asarray([x for x in attraction_metric[sub].keys()])
        ax.xaxis.set_tick_params(labelrotation=45)
        ax.bar(x=xs, height=att_1, width=0.75, tick_label=labels, color='black')

        ax = fig.add_subplot(1,2,2)
        att_2 = [x[1] for x in attractions]
        xs = np.linspace(0.0, 6.0, len(att_2))
        labels = np.asarray([x for x in attraction_metric[sub].keys()])
        ax.xaxis.set_tick_params(labelrotation=45)
        ax.bar(x=xs, height=att_2, width=0.75, tick_label=labels, color='black')
        pp.savefig(fig)
    plt.close()
    pp.close()

def plot_average_ref_near_far_curviness(avg_curviness_plotpath='./results/average_curviness.pdf'):
    pp = PdfPages(avg_curviness_plotpath)
    xs = np.linspace(0.0, 6.0, 3)
    labels = ['Ref', 'Near', 'Far']
    for sub in subs:
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle("Average curviness score in trials for Ref, Near and Far for "+str(sub))
        ax = fig.add_subplot(1,1,1)
        avgs = [ref_stats[sub][0], near_stats[sub][0], far_stats[sub][0]]
        stds = [ref_stats[sub][1], near_stats[sub][1], far_stats[sub][1]]
        ax.bar(x=xs, height=avgs, yerr=stds, width=0.75, tick_label=labels, color='green', ecolor='black',
                capsize=10)
        pp.savefig(fig)
    plt.close()
    pp.close()

def plot_average_attraction_vals(avg_attraction_plotpath='./results/average_attraction.pdf'):
    pp = PdfPages(avg_attraction_plotpath)
    xs = np.linspace(0.0, 20.0, len(subs))
    labels = subs

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("Average attraction scores for all subjects (on near path)")
    avgs_near = [x[0][0] for x in attraction_stats.values()]
    stds_near = [x[1][0] for x in attraction_stats.values()]
    ax = fig.add_subplot(1,1,1)
    ax.bar(x=xs, height=avgs_near, yerr=stds_near, width=0.5, tick_label=labels, color='green', ecolor='black',
            capsize=4)
    pp.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("Average attraction scores for all subjects (on reference path)")
    avgs_ref = [x[0][1] for x in attraction_stats.values()]
    stds_ref = [x[1][1] for x in attraction_stats.values()]
    ax = fig.add_subplot(1,1,1)
    ax.bar(x=xs, height=avgs_ref, yerr=stds_ref, width=0.5, tick_label=labels, color='green', ecolor='black',
            capsize=4)
    pp.savefig(fig)
    plt.close()
    pp.close()

def plot_average_attributes_all(avg_attributes_plotpath='./results/average_attributes.pdf'):
    pp = PdfPages(avg_attributes_plotpath)
    xs = np.linspace(0.0, 10.0, 5)
    labels = ['curviness_ref', 'curviness_near', 'curviness_far', 'attraction_nearsearch', \
            'attraction_refsearch']
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("Average attribute values for all trials & subjects")
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_tick_params(labelrotation=45)

    print np.asarray([x[0][0] for x in attraction_stats.values() if not math.isnan(x[0][0])])

    avg_ref = np.mean(np.asarray([x[0] for x in ref_stats.values() if not math.isnan(x[0])]))
    std_ref = np.mean(np.asarray([x[1] for x in ref_stats.values() if not math.isnan(x[1])]))
    avg_near = np.mean(np.asarray([x[0] for x in near_stats.values() if not math.isnan(x[0])]))
    std_near = np.mean(np.asarray([x[1] for x in near_stats.values() if not math.isnan(x[1])]))
    avg_far = np.mean(np.asarray([x[0] for x in far_stats.values() if not math.isnan(x[0])]))
    std_far = np.mean(np.asarray([x[1] for x in far_stats.values() if not math.isnan(x[1])]))
    avg_natt = np.mean(np.asarray([x[0][0] for x in attraction_stats.values() if not math.isnan(x[0][0])]))
    std_natt = np.mean(np.asarray([x[1][0] for x in attraction_stats.values() if not math.isnan(x[1][0])]))
    avg_ratt = np.mean(np.asarray([x[0][1] for x in attraction_stats.values() if not math.isnan(x[0][1])]))
    std_ratt = np.mean(np.asarray([x[1][1] for x in attraction_stats.values() if not math.isnan(x[1][1])]))

    avg_final = [avg_ref, avg_near, avg_far, avg_natt, avg_ratt]
    std_final = [std_ref, std_near, std_far, std_natt, std_ratt]
    print avg_final
    print std_final
    ax.bar(x=xs, height=avg_final, yerr=std_final, width=0.75, tick_label=labels, color='green', \
            ecolor='black', capsize=8)
    pp.savefig(fig)
    plt.close()
    pp.close()

def get_start_end_kinect_times(subject_ids=[3898]):
    for sub in subject_ids:
        print "********************** Subject: "+str(sub)+" **************************"
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
        vsStartEnd = pandas.read_csv('../vs_trials' + os.sep + str(sub) + '.csv', header=None, index_col=2)

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

            startMS = time_ms[skelDataFrame['# timestamp'] >= trStartTime].iloc[0]
            endMS = time_ms[skelDataFrame['# timestamp'] <= trEndTime].iloc[-1]
            print "Start: ", (datetime.datetime.fromtimestamp(startMS/1000.0)).strftime("%M:%S.%f")
            print "End: ", (datetime.datetime.fromtimestamp(endMS/1000.0)).strftime("%M:%S.%f")
