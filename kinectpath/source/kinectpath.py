"""Main module for starting the processing of data and setting the
program input arguments.

TODO: Change the config file format to YAML so that it is very easy to set
the argument values by changing a single file.
"""

from utils import *
import argparse
from attributes import *

def setup_variables(cfg):
    """Function to setup the analysis environment. Initialize variables
    using the scripts in utils.

    Parameters
    ----------
    cfg : dict
        A dictionary containing the value of arguments captured by
        argument parser
    """
    if cfg['triplet_sets_filepath'] is not None:
        init_triplet_sets(cfg['triplet_sets_filepath'])
    else:
        init_triplet_sets()

    if cfg['maps'] is not None:
        init_colormaps(cfg['maps'])
    else:
        init_colormaps()

    init_grasp_time_files()

    if cfg['flag_trials_filepath'] is not None:
        init_flagged_trials_map(cfg['flag_trials_filepath'])
    else:
        init_flagged_trials_map()

    prepare_trial_set_array()
    print "Preparation of trial set ID for each subject - [DONE]"
    prepare_trial_start_stops()
    print "Preparation of trial start and end frame annotations through eye tracking data - [DONE]"
    prepare_skel_dataframe()
    print "Preparation of the trajectories for each subject trial using Kinect1 data - [DONE]"
    prepare_start_end_points_and_pathlens()
    print "Estimating the start location and end points for trials + pathlengths using turn points - [DONE]"

def calculate_attributes(cfg):
    """Function to do the computation for several metrics important for
    analysis of smartplayroom data

    Parameters
    ----------
    cfg : dict
        A dictionary containing the value of arguments captured by
        argument parser
    """
    setup_variables(cfg)

    if cfg['results_dir'] is not None:
        if not os.path.exists(cfg['results_dir']):
            os.makedirs(cfg['results_dir'])
    else:
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    #write_num_turns_ref_near_far()
    #write_pathlengths_ref_near_far()
    if cfg['write_props']:
        if cfg['props_filepath'] is not None:
            write_trial_proportions_file(cfg['props_filepath'])
        else:
            write_trial_proportions_file()

    if cfg['write_plen']:
        if cfg['plen_filepath'] is not None:
            write_triplet_pathlength_file(cfg['plen_filepath'])
        else:
            write_triplet_pathlength_file()

    if cfg['plot_traj']:
        if cfg['traj_plotpath'] is not None:
            if cfg['write_attraction']:
                plot_path_trajectories(cfg['traj_plotpath'],
                                        write_attraction=True,
                                        attraction_filepath=cfg['attraction_filepath'])
            else:
                plot_path_trajectories(cfg['traj_plotpath'],
                                        write_attraction=False)
        else:
            if cfg['write_attraction']:
                plot_path_trajectories(write_attraction=True,
                                        attraction_filepath=cfg['attraction_filepath'])
            else:
                plot_path_trajectories(write_attraction=False)

    if cfg['plot_traj_trial']:
        if cfg['traj_plotpath_trial'] is not None:
            if cfg['write_attraction']:
                plot_path_trajectories_trial(cfg['traj_plotpath_trial'],
                                        write_attraction=True,
                                        attraction_filepath=cfg['attraction_filepath'])
            else:
                plot_path_trajectories_trial(cfg['traj_plotpath_trial'],
                                        write_attraction=False)
        else:
            plot_path_trajectories_trial(write_attraction=False)

    if cfg['plot_persub_curve']:
        if cfg['persub_curve_plotpath'] is not None:
            plot_per_subject_triplet_randomness(cfg['persub_curve_plotpath'])
        else:
            plot_per_subject_triplet_randomness()

    if cfg['plot_persub_attract']:
        plot_path_trajectories(write_attraction=False)
        if cfg['persub_attract_plotpath'] is not None:
            plot_per_subject_trial_attraction_values(cfg['persub_attract_plotpath'])
        else:
            plot_per_subject_trial_attraction_values()

    if cfg['plot_avg_curve']:
        if cfg['avg_curve_plotpath'] is not None:
            plot_average_ref_near_far_curviness(cfg['avg_curve_plotpath'])
        else:
            plot_average_ref_near_far_curviness()

    if cfg['plot_avg_attract']:
        plot_path_trajectories(write_attraction=False)
        if cfg['avg_attract_plotpath'] is not None:
            plot_average_attraction_vals(cfg['avg_attract_plotpath'])
        else:
            plot_average_attraction_vals()

    if cfg['plot_avg_all']:
        if cfg['avg_all_plotpath'] is not None:
            plot_average_attributes_all(cfg['avg_all_plotpath'])
        else:
            plot_average_attributes_all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Smart Playroom: Attribute analysis pipeline')
    parser.add_argument('--triplet_sets_filepath', '-tsp', type=str, help='Path to the file having \
                            triplet sets configuration')
    parser.add_argument('--flag_trials_filepath', '-ftp', type=str, help='Path to file having \
                            the trials flagged for strategy "Fixate, Approach and Grab"')
    parser.add_argument('--maps', '-m', type=str, nargs='+', help='The different colormaps you want \
                            to use while plotting different trial triplets (one per reference)')
    parser.add_argument('--results-dir', '-rd', type=str, help='The path to directory where the \
                            results should be stored', default='./results')

    parser.add_argument('--write-props', '-wp', action='store_true', help='Boolean flag for \
                            specifying if trial proportions using flagged trials should be written to file')
    parser.add_argument('--write-plen', '-wl', action='store_true', help='Boolean flag for \
                            specifying if path lengths for ref, far, near should be written to file')
    parser.add_argument('--write-attraction', '-wa', action='store_true', help='Boolean flag for \
                            specifying if attraction measures should be written to file')
    parser.add_argument('--props-filepath', '-pfp', type=str, help='Path to the file where the trial \
                            proportions should be written')
    parser.add_argument('--plen-filepath', '-lfp', type=str, help='Path to the file where the path length \
                            for ref, near and far should be written')
    parser.add_argument('--attraction-filepath', '-afp', type=str, help='Path to the file where the attraction \
                            measures should be written')

    parser.add_argument('--plot-traj', '-pt', action='store_true', help='Boolean flag for \
                            specifying if path trajectories for trials are to be plotted')
    parser.add_argument('--plot-traj-trial', '-ptt', action='store_true', help='Boolean flag for \
                            specifying if path trajectories for trials, clustered per trial, are to be plotted')
    parser.add_argument('--plot-persub-curve', '-ppc', action='store_true', help='Boolean flag for \
                            specifying if per subject curviness measures are to be plotted')
    parser.add_argument('--plot-persub-attract', '-ppa', action='store_true', help='Boolean flag for \
                            specifying if per subject attraction measures are to be plotted')
    parser.add_argument('--plot-avg-curve', '-pac', action='store_true', help='Boolean flag for \
                            specifying if average ref, near and far curviness measures are to be plotted')
    parser.add_argument('--plot-avg-attract', '-paa', action='store_true', help='Boolean flag for \
                            specifying if average attraction measures are to be plotted')
    parser.add_argument('--plot-avg-all', '-pal', action='store_true', help='Boolean flag for \
                            specifying if average values for all attribute measures are to be plotted')
    parser.add_argument('--traj-plotpath', '-tpp', type=str, help='Path to the file where the trajectories \
                            plot PDF should be stored')
    parser.add_argument('--traj-plotpath-trial', '-tpt', type=str, help='Path to the file where the trajectories \
                            plot PDF, where plots are clustered by trial, should be stored')
    parser.add_argument('--persub-curve-plotpath', '-pcpp', type=str, help='Path to the file where the per \
                            subject curviness measure plot PDF should be stored')
    parser.add_argument('--persub-attract-plotpath', '-papp', type=str, help='Path to the file where the per \
                            subject attraction measure plot PDF should be stored')
    parser.add_argument('--avg-curve-plotpath', '-acpp', type=str, help='Path to the file where the average \
                            curviness measure plot PDF should be stored')
    parser.add_argument('--avg-attract-plotpath', '-aapp', type=str, help='Path to the file where the average \
                            attraction measure plot PDF should be stored')
    parser.add_argument('--avg-all-plotpath', '-alpp', type=str, help='Path to the file where the average \
                            of all attribute measures plot PDF should be stored')

    parser.add_argument('--use-grasp-times', '-gt', action='store_true', help='Boolean flag for \
                            specifying if the grasp timings for trials should be used for processing')
    parser.add_argument('--ignore-flagged-trials', '-ft', action='store_true', help='Boolean \
                            flag for specifying if the flagged trials should be excluded')

    args = parser.parse_args()
    cfg = vars(args)

    calculate_attributes(cfg)

    print "All attributes calculated - [DONE]"
