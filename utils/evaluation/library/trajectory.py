#!/usr/bin/env python

import os
import pickle
import sys
import numpy as np

# uzh trajectory evaluation toolbox
sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) \
                                + '/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation'))

import trajectory_utils as traj_utils
import associate_timestamps as associ
import results_writer as res_writer
import compute_trajectory_errors as traj_err
import align_utils as au
import transformations as tf
import plot_utils as pu

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

class Trajectory:
    """
    Class to perform trajectory evaluation given the input groundtruth and estimated pose.

    Source: https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/trajectory.py

    """
    rel_error_cached_nm = 'cached_rel_err.pickle'
    rel_error_prefix = 'relative_error_statistics_'
    saved_res_dir_nm = 'saved_results'

    def __init__(self, results_dir, run_name='', gt_traj_file=None, estimated_traj_file=None, align_type='sim3',
                 align_num_frames=-1, preset_boxplot_distances=[]):

        assert os.path.exists(results_dir), \
            "Specified directory {0} does not exist.".format(results_dir)
        assert align_type in ['posyaw', 'sim3', 'se3', 'none']

        # information of the results, useful as labels
        self.alg = run_name
        self.uid = self.alg
        self.data_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.data_loaded = False
        self.data_aligned = False
        self.saved_results_dir = os.path.join(self.data_dir, Trajectory.saved_res_dir_nm)

        if not os.path.exists(self.saved_results_dir):
            os.makedirs(self.saved_results_dir)
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        self.align_type = align_type
        self.align_num_frames = int(align_num_frames)

        self.align_str = self.align_type + '_' + str(self.align_num_frames)

        self.abs_errors = {}

        # we cache relative error since it is time-comsuming to compute
        self.rel_errors = {}
        self.cached_rel_err_fn = os.path.join(self.saved_results_dir,
                                              self.rel_error_cached_nm)

        self.load_data(gt_traj_file, estimated_traj_file)

        if len(preset_boxplot_distances) != 0:
            self.preset_boxplot_distances = preset_boxplot_distances
        else:
            self.compute_boxplot_distances()

        self.align_trajectory()

    def load_data(self, gt_traj_file=None, estimated_traj_file=None):
        """
        Loads the trajectory data. The resuls {p_es, q_es, p_gt, q_gt} is
        synchronized and has the same length.
        """
        print('Loading trajectory data...')

        # only timestamped pose series is supported
        self.t_es, self.p_es, self.q_es, self.t_gt, self.p_gt, self.q_gt = self.load_stamped_dataset(self.data_dir,
                                                                                                     fn_gt=gt_traj_file,
                                                                                                     fn_es=estimated_traj_file)
        self.accum_distances = traj_utils.get_distance_from_start(self.p_gt)
        self.traj_length = self.accum_distances[-1]

        if os.path.isfile(self.cached_rel_err_fn):
            print('Loading cached relative (odometry) errors from ' +
                  self.cached_rel_err_fn)
            with open(self.cached_rel_err_fn, 'rb') as f:
                self.rel_errors = pickle.load(f)
            print("Loaded odometry error calcualted at {0}".format(
                self.rel_errors.keys()))

        self.data_loaded = True
        print('...done.')

    def cache_current_error(self):
        if self.rel_errors:
            with open(self.cached_rel_err_fn, 'wb') as f:
                pickle.dump(self.rel_errors, f)

    def compute_boxplot_distances(self):
        pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
        print("Computing preset subtrajectory lengths for relative errors...")
        print("Use percentage {0} of trajectory length.".format(pcts))
        self.preset_boxplot_distances = [np.floor(pct * self.traj_length)
                                         for pct in pcts]

        print("...done. Computed preset subtrajecory lengths:"
              " {0}".format(self.preset_boxplot_distances))

    def align_trajectory(self):
        if self.data_aligned:
            print("Trajectory already aligned")
            return
        print("Aliging the trajectory estimate to the groundtruth...")

        print("Alignment type is {0}.".format(self.align_type))
        n = int(self.align_num_frames)
        if n < 0.0:
            print('To align all frames.')
            n = len(self.p_es)
        else:
            print('To align trajectory using ' + str(n) + ' frames.')

        self.trans = np.zeros((3,))
        self.rot = np.eye(3)
        self.scale = 1.0
        if self.align_type == 'none':
            pass
        else:
            self.scale, self.rot, self.trans = au.alignTrajectory(
                self.p_es, self.p_gt, self.q_es, self.q_gt,
                self.align_type, self.align_num_frames)

        self.p_es_aligned = np.zeros(np.shape(self.p_es))
        self.q_es_aligned = np.zeros(np.shape(self.q_es))
        for i in range(np.shape(self.p_es)[0]):
            self.p_es_aligned[i, :] = self.scale * \
                                      self.rot.dot(self.p_es[i, :]) + self.trans
            q_es_R = self.rot.dot(
                tf.quaternion_matrix(self.q_es[i, :])[0:3, 0:3])
            q_es_T = np.identity(4)
            q_es_T[0:3, 0:3] = q_es_R
            self.q_es_aligned[i, :] = tf.quaternion_from_matrix(q_es_T)

        self.data_aligned = True
        print("... trajectory alignment done.")

    def compute_absolute_error(self):
        if self.abs_errors:
            print("Absolute errors already calculated")
        else:
            print('Calculating RMSE...')
            # align trajectory if necessary
            self.align_trajectory()
            e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = \
                traj_err.compute_absolute_error(self.p_es_aligned,
                                                self.q_es_aligned,
                                                self.p_gt,
                                                self.q_gt)
            stats_trans = res_writer.compute_statistics(e_trans)
            stats_rot = res_writer.compute_statistics(e_rot)
            stats_scale = res_writer.compute_statistics(e_scale_perc)

            self.abs_errors['abs_e_trans'] = e_trans
            self.abs_errors['abs_e_trans_stats'] = stats_trans

            self.abs_errors['abs_e_trans_vec'] = e_trans_vec

            self.abs_errors['abs_e_rot'] = e_rot
            self.abs_errors['abs_e_rot_stats'] = stats_rot

            self.abs_errors['abs_e_ypr'] = e_ypr

            self.abs_errors['abs_e_scale_perc'] = e_scale_perc
            self.abs_errors['abs_e_scale_stats'] = stats_scale
            print('...RMSE calculated.')
        return

    def get_abs_errors_stats(self):
        errors_dict = dict()

        labels = ['abs_e_trans_stats', 'abs_e_rot_stats', 'abs_e_scale_stats']
        stats = ['rmse', 'mean', 'median']

        for label in labels:
            for stat in stats:
                errors_dict[label + '_' + stat] = self.abs_errors[label][stat]

        return errors_dict

    def get_rel_errors_stats(self):
        errors_dict = dict()

        labels = ['rel_trans_stats', 'rel_trans_perc_stats', 'rel_rot_stats']
        stats = ['rmse', 'mean', 'median']

        for dist in self.rel_errors:
            for label in labels:
                for stat in stats:
                    errors_dict[label + '_' + stat + '_trajlen_'+ str(dist)] = self.rel_errors[dist][label][stat]

        return errors_dict

    def write_errors_to_yaml(self):
        self.abs_err_stats_fn = os.path.join(
            self.saved_results_dir, 'absolute_err_statistics' + '_' +
                                    self.align_str + '.yaml')
        res_writer.update_and_save_stats(
            self.abs_errors['abs_e_trans_stats'], 'trans',
            self.abs_err_stats_fn)
        res_writer.update_and_save_stats(
            self.abs_errors['abs_e_rot_stats'], 'rot',
            self.abs_err_stats_fn)
        res_writer.update_and_save_stats(
            self.abs_errors['abs_e_scale_stats'], 'scale',
            self.abs_err_stats_fn)

        self.rel_error_stats_fns = []
        for dist in self.rel_errors:
            cur_err = self.rel_errors[dist]
            dist_str = "{:3.1f}".format(dist).replace('.', '_')
            dist_fn = os.path.join(
                self.saved_results_dir,
                Trajectory.rel_error_prefix + dist_str + '.yaml')
            res_writer.update_and_save_stats(
                cur_err['rel_trans_stats'], 'trans', dist_fn)
            res_writer.update_and_save_stats(
                cur_err['rel_rot_stats'], 'rot', dist_fn)
            res_writer.update_and_save_stats(
                cur_err['rel_trans_perc_stats'], 'trans_perc', dist_fn)
            res_writer.update_and_save_stats(
                cur_err['rel_yaw_stats'], 'yaw', dist_fn)
            res_writer.update_and_save_stats(
                cur_err['rel_gravity_stats'], 'gravity', dist_fn)
            self.rel_error_stats_fns.append(dist_fn)

    def compute_relative_error_at_subtraj_len(self, subtraj_len,
                                              max_dist_diff=-1):
        if max_dist_diff < 0:
            max_dist_diff = 0.2 * subtraj_len

        if self.rel_errors and (subtraj_len in self.rel_errors):
            print("Relative error at sub-trajectory length {0} is already "
                  "computed or loaded from cache.".format(subtraj_len))
        else:
            print("Computing relative error at sub-trajectory "
                  "length {0}".format(subtraj_len))
            Tcm = np.identity(4)
            _, e_trans, e_trans_perc, e_yaw, e_gravity, e_rot = \
                traj_err.compute_relative_error(
                    self.p_es, self.q_es, self.p_gt, self.q_gt, Tcm,
                    subtraj_len, max_dist_diff, self.accum_distances,
                    self.scale)
            dist_rel_err = {'rel_trans': e_trans,
                            'rel_trans_stats':
                                res_writer.compute_statistics(e_trans),
                            'rel_trans_perc': e_trans_perc,
                            'rel_trans_perc_stats':
                                res_writer.compute_statistics(e_trans_perc),
                            'rel_rot': e_rot,
                            'rel_rot_stats':
                                res_writer.compute_statistics(e_rot),
                            'rel_yaw': e_yaw,
                            'rel_yaw_stats':
                                res_writer.compute_statistics(e_yaw),
                            'rel_gravity': e_gravity,
                            'rel_gravity_stats':
                                res_writer.compute_statistics(e_gravity)}
            self.rel_errors[subtraj_len] = dist_rel_err

    def compute_relative_errors(self, subtraj_lengths=[]):
        if subtraj_lengths:
            for l in subtraj_lengths:
                self.compute_relative_error_at_subtraj_len(l)
        else:
            print("Computing the relative errors based on preset"
                  " subtrajectory lengths...")
            for l in self.preset_boxplot_distances:
                self.compute_relative_error_at_subtraj_len(l)

    def plot_trajectory(self, FORMAT='.png'):
        print(">>> Plotting trajectory...")
        # trajectory view from top
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='y [m]')
        pu.plot_trajectory_top(ax, self.p_es_aligned, 'b', 'Estimate')
        pu.plot_trajectory_top(ax, self.p_gt, 'm', 'Groundtruth')
        pu.plot_aligned_top(ax, self.p_es_aligned, self.p_gt, self.align_num_frames)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(self.plots_dir + '/trajectory_top' + '_' + self.align_str + FORMAT,
                    bbox_inches="tight")

        # trajectory view from side
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='y [m]')
        pu.plot_trajectory_side(ax, self.p_es_aligned, 'b', 'Estimate')
        pu.plot_trajectory_side(ax, self.p_gt, 'm', 'Groundtruth')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(self.plots_dir + '/trajectory_side' + '_' + self.align_str + FORMAT,
                    bbox_inches="tight")

    def plot_rel_traj_error(self, FORMAT='.png'):
        print(">>> Plotting relative (odometry) error...")

        distances = self.preset_boxplot_distances
        rel_trans_err = [[self.rel_errors[d]['rel_trans'] for d in distances]]
        rel_trans_err_perc = [[self.rel_errors[d]['rel_trans_perc']
                               for d in distances]]
        rel_yaw_err = [[self.rel_errors[d]['rel_yaw'] for d in distances]]
        labels = ['Estimate']
        colors = ['k', 'b']

        # plot relative translation error (in meters)
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(111, xlabel='Distance traveled [m]', ylabel='Translation error [m]')
        pu.boxplot_compare(ax, distances, rel_trans_err, labels, colors)
        fig.tight_layout()
        fig.savefig(self.plots_dir + '/rel_translation_error' + FORMAT, bbox_inches="tight")
        plt.close(fig)

        # plot relative translation error (in %)
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot( 111, xlabel='Distance traveled [m]',ylabel='Translation error [\%]')
        pu.boxplot_compare(ax, distances, rel_trans_err_perc, labels, colors)
        fig.tight_layout()
        fig.savefig(self.plots_dir + '/rel_translation_error_perc' + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

        # plot relative rotation error (in degrees)
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(111, xlabel='Distance traveled [m]', ylabel='Yaw error [deg]')
        pu.boxplot_compare(ax, distances, rel_yaw_err, labels, colors)
        fig.tight_layout()
        fig.savefig(self.plots_dir + '/rel_yaw_error' + FORMAT, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def remove_cached_error(data_dir):
        print("Remove cached error in {0}".format(data_dir))
        rel_error_fn = os.path.join(data_dir, Trajectory.saved_res_dir_nm,
                                    Trajectory.rel_error_cached_nm)
        if os.path.exists(rel_error_fn):
            os.remove(rel_error_fn)

    @staticmethod
    def load_stamped_dataset(results_dir, fn_es=None, fn_gt=None, max_diff=0.02):
        '''
        read synchronized estimation and groundtruth and associate the timestamps
        '''
        print('loading dataset in ' + results_dir)

        if fn_es == None:
            fn_es = os.path.join(results_dir, 'stamped_traj_estimate.txt')
        if fn_gt == None:
            fn_gt = os.path.join(results_dir, 'stamped_groundtruth.txt')
        fn_matches = os.path.join(results_dir, 'stamped_est_gt_matches.txt')

        matches = associ.read_files_and_associate(fn_es, fn_gt, 0.0, max_diff)
        dict_matches = dict(matches)
        np.savetxt(fn_matches, np.array(matches, dtype=int), fmt='%d')

        data_es = np.loadtxt(fn_es)
        data_gt = np.loadtxt(fn_gt)

        p_es = []
        p_gt = []
        q_es = []
        q_gt = []
        t_gt = []
        for es_id, es in enumerate(data_es):
            if es_id in dict_matches:
                gt = data_gt[dict_matches[es_id]]
                p_es.append(es[1:4])
                p_gt.append(gt[1:4])
                q_es.append(es[4:8])
                q_gt.append(gt[4:8])
                t_gt.append(gt[0])
        p_es = np.array(p_es)
        p_gt = np.array(p_gt)
        q_es = np.array(q_es)
        q_gt = np.array(q_gt)
        t_gt = np.array(t_gt)

        return t_gt, p_es, q_es, t_gt, p_gt, q_gt

if __name__ == '__main__':
    # example usage

    # define parameters
    results_dir = 'eval/2011_09_30_drive_0034/random_perturbations/run_0'
    run_name = 'run_0'
    gt_traj_file = 'KITTI_parameters/2011_09_30_drive_0034/stamped_groundtruth.txt'
    estimated_traj_file = results_dir + '/stamped_traj_estimate.txt'
    align_type = 'sim3'
    output_dir = os.path.join(results_dir, 'evaluation')

    # create folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # create instance of the class
    traj = Trajectory(output_dir, run_name, gt_traj_file, estimated_traj_file, align_type=align_type)

    # compute errors
    traj.compute_absolute_error()
    traj.compute_relative_errors()
    traj.cache_current_error()

    # write/plot the computed results
    traj.write_errors_to_yaml()
    traj.plot_trajectory('.png')

# EOF
