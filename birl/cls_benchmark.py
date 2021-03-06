"""
General benchmark template for all registration methods.
It also serves for evaluating the input registration pairs
(while no registration is performed, there is only the initial deformation)

EXAMPLE (usage):
>> mkdir ./results
>> python benchmarks/bm_registration.py \
    -c data_images/pairs-imgs-lnds_histol.csv -d ./data_images \
    -o ./results --unique

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import time
import logging
import shutil
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import (
    update_path, create_folder, image_size, load_landmarks, load_image, save_image)
from birl.utilities.evaluate import compute_points_dist_statistic, compute_affine_transf_diff
from birl.utilities.experiments import exec_commands, string_dict, wrap_execute_sequence
from birl.utilities.visualisation import (
    export_figure, draw_image_points, draw_images_warped_landmarks)
from birl.utilities.registration import estimate_affine_transform
from birl.utilities.cls_experiment import Experiment

#: number of available threads on this computer
NB_THREADS = int(mproc.cpu_count())
#: default number of threads used by benchmarks
NB_THREADS_USED = max(1, int(NB_THREADS * .8))
# some needed files
NAME_CSV_REGISTRATION_PAIRS = 'registration-results.csv'
#: default file for exporting results in table format
NAME_CSV_RESULTS = 'results-summary.csv'
#: default file for exporting results in formatted text format
NAME_TXT_RESULTS = 'results-summary.txt'
#: logging file for registration experiments
NAME_LOG_REGISTRATION = 'registration.log'
#: output image name in experiment folder for reg. results - image and landmarks are warped
NAME_IMAGE_MOVE_WARP_POINTS = 'image_warped_landmarks_warped.jpg'
#: output image name in experiment folder for reg. results - warped landmarks in reference image
NAME_IMAGE_REF_POINTS_WARP = 'image_ref_landmarks_warped.jpg'
#: output image name in experiment folder for showing improved alignment by used reguistration
NAME_IMAGE_WARPED_VISUAL = 'registration_visual_landmarks.jpg'
# columns names in cover and also registration table
#: reference (registration target) image
COL_IMAGE_REF = 'Target image'
#: moving (registration source) image
COL_IMAGE_MOVE = 'Source image'
#: reference image warped to the moving frame
COL_IMAGE_REF_WARP = 'Warped target image'
#: moving image warped to the reference frame
COL_IMAGE_MOVE_WARP = 'Warped source image'
#: reference (registration target) landmarks
COL_POINTS_REF = 'Target landmarks'
#: moving (registration source) landmarks
COL_POINTS_MOVE = 'Source landmarks'
#: reference landmarks warped to the moving frame
COL_POINTS_REF_WARP = 'Warped target landmarks'
#: moving landmarks warped to the reference frame
COL_POINTS_MOVE_WARP = 'Warped source landmarks'
#: registration folder for each particular experiment
COL_REG_DIR = 'Registration folder'
#: define robustness as improved image alignment from initial state
COL_ROBUSTNESS = 'Robustness'
#: measured time of image registration in minutes
COL_TIME = 'Execution time [minutes]'
#: tuple of image size
COL_IMAGE_SIZE = 'Image size [pixels]'
#: image diagonal in pixels
COL_IMAGE_DIAGONAL = 'Image diagonal [pixels]'

# list of columns in cover csv
COVER_COLUMNS = (COL_IMAGE_REF, COL_IMAGE_MOVE, COL_POINTS_REF, COL_POINTS_MOVE)
COVER_COLUMNS_EXT = tuple(list(COVER_COLUMNS) + [COL_IMAGE_SIZE, COL_IMAGE_DIAGONAL])
COVER_COLUMNS_WRAP = tuple(list(COVER_COLUMNS) + [COL_IMAGE_REF_WARP, COL_IMAGE_MOVE_WARP,
                                                  COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP])


# fixing ImportError: No module named 'copy_reg' for Python3
if sys.version_info.major == 2:
    import types
    import copy_reg

    def _reduce_method(m):
        # SOLVING issue: cPickle.PicklingError:
        #   Can't pickle <type 'instancemethod'>:
        #       attribute lookup __builtin__.instancemethod failed
        if m.im_self is None:
            tp = m.im_class
        else:
            tp = m.im_self
        return getattr, (tp, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _reduce_method)


class ImRegBenchmark(Experiment):
    """ General benchmark class for all registration methods.
    It also serves for evaluating the input registration pairs.

    :param {str: str|float} params: dictionary with experiment configuration,
        the required options are names in `REQUIRED_PARAMS`,
        note that the basic parameters are inherited

    The benchmark has following steps:
    1. check all necessary pathers and required parameters
    2. load cover file and set all paths as absolute
    3. run individual registration experiment in sequence or in parallel
       (nb_workers > 1); if the particular experiment folder exist (assume
       completed experiment) and skip it
        a) create experiment folder and init experiment
        b) generate execution command
        c) run the command (an option to lock it in single thread)
        d) evaluate experiment, set the expected outputs and visualisation
        e) clean all extra files if any
    4. visualise results abd evaluate registration results

    NOTE: The actual implementation simulates the "IDEAL" registration while
    it blindly copies the reference landmarks as results of the registration.
    In contrast to the right registration, it copies the moving images so there
    is alignment (consistent warping) between resulting landmarks and image.

    Running in single thread:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_workers': 1, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover': path_csv}
    >>> benchmark = ImRegBenchmark(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)

    Running in multiple parallel threads:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_workers': 2, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover': path_csv}
    >>> benchmark = ImRegBenchmark(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ['path_cover', 'path_out', 'nb_workers']

    def __init__(self, params):
        """ initialise benchmark

        :param dict params:  {str: value}
        """
        assert 'unique' in params, 'missing "unique" among %r' % params.keys()
        super(ImRegBenchmark, self).__init__(params, params['unique'])
        logging.info(self.__doc__)
        self._df_cover = None
        self._df_experiments = None
        self.nb_workers = params.get('nb_workers', NB_THREADS)
        self._path_csv_regist = os.path.join(self.params['path_exp'],
                                             NAME_CSV_REGISTRATION_PAIRS)

    def _check_required_params(self):
        """ check some extra required parameters for this benchmark """
        logging.debug('.. check if the BM have all required parameters')
        super(ImRegBenchmark, self)._check_required_params()
        for n in self.REQUIRED_PARAMS:
            assert n in self.params, 'missing "%s" among %r' % (n, self.params.keys())

    def _update_path(self, path, destination='data'):
        """ update te path to the dataset or output

        :param str path: original path
        :param str destination: type of update - data | output | general
        :return str: updated path
        """
        if destination == 'data' and 'path_dataset' in self.params:
            path = os.path.join(self.params['path_dataset'], path)
        elif destination == 'expt' and 'path_exp' in self.params:
            path = os.path.join(self.params['path_exp'], path)
        path = update_path(path, absolute=True)
        return path

    def _relativize_path(self, path, destination='path_exp'):
        """ extract relative path according given parameter

        :param str path: the original path to file/folder
        :param str destination: use path from parameters
        :return str: relative or the original path
        """
        if path is None or not os.path.exists(path):
            logging.debug('Source path does not exists: %s', path)
            return path
        assert destination in self.params, 'Missing path in params: %s' % destination
        base_path = self.params['path_exp']
        base_dir = os.path.basename(base_path)
        path_split = path.split(os.sep)
        if base_dir not in path_split:
            logging.debug('Missing requested folder "%s" in source path: %s',
                          base_dir, path_split)
            return path
        path_split = path_split[path_split.index(base_dir) + 1:]
        path_rltv = os.sep.join(path_split)
        if os.path.exists(os.path.join(self.params[destination], path_rltv)):
            return path_rltv
        else:
            logging.debug('Not existing relative path: %s', path)
            return path

    def _copy_config_to_expt(self, field_path):
        """ copy particular configuration to the experiment folder

        :param str field_path: field from parameters containing a path to file
        """
        path_source = self.params.get(field_path, '')
        path_config = os.path.join(self.params['path_exp'], os.path.basename(path_source))
        if os.path.isfile(path_source):
            shutil.copy(path_source, path_config)
            self.params[field_path] = path_config
        else:
            logging.warning('Missing config: %s', path_source)

    def _get_paths(self, row):
        """ expand the relative paths to absolute

        :param row: row from cover file with relative paths
        :return (str, str, str, str): path to reference and moving image
            and reference and moving landmarks
        """
        paths = [self._update_path(row[col], 'data') for col in COVER_COLUMNS]
        return paths

    def _get_path_reg_dir(self, record):
        return self._update_path(str(record[COL_REG_DIR]), 'expt')

    def _load_data(self):
        """ loading data, the cover file with all registration pairs """
        logging.info('-> loading data...')
        # loading the csv cover file
        assert os.path.isfile(self.params['path_cover']), \
            'path to csv cover is not defined - %s' % self.params['path_cover']
        self._df_cover = pd.read_csv(self.params['path_cover'], index_col=None)
        assert all(col in self._df_cover.columns for col in COVER_COLUMNS), \
            'Some required columns are missing in the cover file.'

    def _run(self):
        """ perform complete benchmark experiment """
        logging.info('-> perform set of experiments...')

        # load existing result of create new entity
        if os.path.isfile(self._path_csv_regist):
            logging.info('loading existing csv: "%s"', self._path_csv_regist)
            self._df_experiments = pd.read_csv(self._path_csv_regist,
                                               index_col=None)
            if 'ID' in self._df_experiments.columns:
                self._df_experiments.set_index('ID', inplace=True)
        else:
            self._df_experiments = pd.DataFrame()

        # run the experiment in parallel of single thread
        self.__execute_method(self._perform_registration, self._df_cover,
                              self._path_csv_regist, 'registration experiments',
                              aggr_experiments=True)

    def __execute_method(self, method, input_table, path_csv=None, desc='',
                         aggr_experiments=False, nb_workers=None):
        """ execute a method in sequence or parallel

        :param func method: used method
        :param DF input_table: iterate over table
        :param str path_csv: path to the output temporal csv
        :param str desc: name of the running process
        :param bool aggr_experiments: append output to experiment DF
        :param int|None nb_workers: number of jobs, by default using class setting
        :return:
        """
        # setting the temporal split
        self._main_thread = False
        # run the experiment in parallel of single thread
        nb_workers = self.nb_workers if nb_workers is None else nb_workers
        iter_table = ((idx, dict(row)) for idx, row, in input_table.iterrows())
        for res in wrap_execute_sequence(method, iter_table, ordered=True,
                                         nb_workers=nb_workers, desc=desc):
            if res is None or not aggr_experiments:
                continue
            self._df_experiments = self._df_experiments.append(res, ignore_index=True)
            self.__export_df_experiments(path_csv)
        self._main_thread = True

    def __export_df_experiments(self, path_csv=None):
        """ export the DataFrame with registration results

        :param str | None path_csv: path to output CSV file
        """
        if path_csv is not None:
            if 'ID' in self._df_experiments.columns:
                self._df_experiments.set_index('ID').to_csv(path_csv)
            else:
                self._df_experiments.to_csv(path_csv, index=None)

    def __check_exist_regist(self, idx, path_dir_reg):
        """ check whether the particular experiment already exists and have results

        if the folder with experiment already exist and it is also part
        of the loaded finished experiments, sometimes the oder may mean
        failed experiment

        :param int idx: index of particular
        :param str path_dir_reg:
        :return bool:
        """
        b_df_col = ('ID' in self._df_experiments.columns and idx in self._df_experiments['ID'])
        b_df_idx = idx in self._df_experiments.index
        check = os.path.exists(path_dir_reg) and (b_df_col or b_df_idx)
        if check:
            logging.warning('particular registration experiment already exists:'
                            ' "%r"', idx)
        return check

    def _perform_registration(self, df_row):
        """ run single registration experiment with all sub-stages

        :param (int, dict) df_row: tow from iterated table
        """
        idx, row = df_row
        logging.debug('-> perform single registration #%d...', idx)
        # create folder for this particular experiment
        row['ID'] = idx
        row[COL_REG_DIR] = str(idx)
        path_dir_reg = self._get_path_reg_dir(row)
        # check whether the particular experiment already exists and have result
        if self.__check_exist_regist(idx, path_dir_reg):
            return None
        create_folder(path_dir_reg)

        row = self._prepare_img_registration(row)

        # measure execution time
        time_start = time.time()
        row = self._execute_img_registration(row)
        # if the experiment failed, return back None
        if not row:
            return None
        # compute the registration time in minutes
        row[COL_TIME] = (time.time() - time_start) / 60.

        row = self._parse_regist_results(row)
        row = self._clear_after_registration(row)
        return row

    def _summarise(self):
        """ summarise complete benchmark experiment """
        logging.info('-> summarise experiment...')
        # load _df_experiments and compute stat
        _compute_landmarks_statistic = partial(
            compute_registration_statistic,
            df_experiments=self._df_experiments,
            path_dataset=self.params.get('path_dataset', None),
            path_experiment=self.params.get('path_exp', None))
        self.__execute_method(_compute_landmarks_statistic, self._df_experiments,
                              desc='compute TRE', nb_workers=1)
        # add visualisations
        _visualise_registration = partial(
            visualise_registration,
            path_dataset=self.params.get('path_dataset', None),
            path_experiment=self.params.get('path_exp', None))
        if self.params.get('visual', False):
            self.__execute_method(_visualise_registration, self._df_experiments,
                                  desc='visualise results')
        # export stat to csv
        if self._df_experiments.empty:
            logging.warning('no experimental results were collected')
            return
        self.__export_df_experiments(self._path_csv_regist)
        # export simple stat to txt
        export_summary_results(self._df_experiments, self.params['path_exp'], self.params)

    @classmethod
    def _prepare_img_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. no preparing before registration experiment')
        return record

    def _execute_img_registration(self, record):
        """ execute the image registration itself

        :param {} record:
        :return {}:
        """
        logging.debug('.. execute image registration as command line')
        path_dir_reg = self._get_path_reg_dir(record)

        commands = self._generate_regist_command(record)
        # in case it is just one command
        if not (isinstance(commands, list) or isinstance(commands, tuple)):
            commands = [commands]

        path_log = os.path.join(path_dir_reg, NAME_LOG_REGISTRATION)
        # TODO, add lock to single thread, create pool with possible thread ids
        # (USE taskset [native], numactl [need install])
        if not (isinstance(commands, list) or isinstance(commands, tuple)):
            commands = [commands]
        # measure execution time
        cmd_result = exec_commands(commands, path_log)
        # if the experiment failed, return back None
        if not cmd_result:
            return None
        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        logging.debug('.. simulate registration: '
                      'copy the target image and landmarks, simulate ideal case')
        path_im_ref, _, _, path_lnds_move = self._get_paths(record)
        path_reg_dir = self._get_path_reg_dir(record)
        name_img = os.path.basename(record[COL_IMAGE_MOVE])
        cmd_img = 'cp %s %s' % (path_im_ref, os.path.join(path_reg_dir, name_img))
        name_lnds = os.path.basename(record[COL_POINTS_MOVE])
        cmd_lnds = 'cp %s %s' % (path_lnds_move, os.path.join(path_reg_dir, name_lnds))
        commands = [cmd_img, cmd_lnds]
        return commands

    @classmethod
    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to img_ref_warp, img_move_warp,
                                                lnds_ref_warp, lnds_move_warp
        """
        # detect image
        path_img = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_POINTS_MOVE]))
        return None, path_img, path_lnd, None

    def _extract_execution_time(self, record):
        """ if needed update the execution time

        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        _ = self._get_path_reg_dir(record)
        return None

    def _parse_regist_results(self, record):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param record: {str: value}, dictionary with registration params
        :return: {str: value}
        """
        # Update the registration outputs / paths
        paths = self._extract_warped_image_landmarks(record)
        columns = (COL_IMAGE_REF_WARP, COL_IMAGE_MOVE_WARP,
                   COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP)

        for path, col in zip(paths, columns):
            # detect image and landmarks
            path = self._relativize_path(path, 'path_exp')
            if path is not None and os.path.isfile(self._update_path(path, 'expt')):
                record[col] = path

        # Update the registration time
        exec_time = self._extract_execution_time(record)
        if exec_time:
            # compute the registration time in minutes
            record[COL_TIME] = exec_time

        return record

    @classmethod
    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param {str: value} record: dictionary with regist. information
        :return {str: value}: the same or updated regist. info
        """
        logging.debug('.. no cleaning after registration experiment')
        return record


def update_path_(path, path_base=None):
    """ update the image path with possible base path

    :param str path: the last path of the path
    :param str|None path_base: optional base path
    :return str: update path
    """
    path = os.path.join(path_base, str(path)) if path_base else path
    return update_path(path, absolute=True)


def _image_diag(record, path_img_ref=None):
    """ get the image diagonal from several sources
        1. diagonal exists in the table
        2. image size exist in the table
        3. reference image exists

    :param {}|DF record: one row from the table
    :param str path_img_ref: optional path to the reference image
    :return float|None: image diagonal
    """
    img_diag = record[COL_IMAGE_DIAGONAL] if COL_IMAGE_DIAGONAL in record else None
    if not img_diag and path_img_ref and os.path.isfile(path_img_ref):
        _, img_diag = image_size(path_img_ref)
    return img_diag


def _load_landmarks(record, path_dataset):
    path_img_ref, _, path_lnds_ref, path_lnds_move = \
        [update_path_(record[col], path_dataset) for col in COVER_COLUMNS]
    points_ref = load_landmarks(path_lnds_ref)
    points_move = load_landmarks(path_lnds_move)
    return points_ref, points_move, path_img_ref


def compute_registration_statistic(idx_row, df_experiments,
                                   path_dataset=None, path_experiment=None):
    """ after successful registration load initial nad estimated landmarks
    afterwords compute various statistic for init, and finalNoBmTemplatene alignment

    :param (int, dict) idx_row: tow from iterated table
    :param DF df_experiments: DataFrame with experiments
    :param str|None path_dataset: path to the dataset folder
    :param str|None path_experiment: path to the experiment folder
    """
    idx, row = idx_row
    row = dict(row)  # convert even series to dictionary
    points_ref, points_move, path_img_ref = _load_landmarks(row, path_dataset)
    img_diag = _image_diag(row, path_img_ref)
    df_experiments.loc[idx, COL_IMAGE_DIAGONAL] = img_diag

    # compute landmarks statistic
    compute_registration_accuracy(df_experiments, idx, points_ref, points_move,
                                  'init', img_diag, wo_affine=False)

    # load transformed landmarks
    if (COL_POINTS_MOVE_WARP not in row) and (COL_POINTS_REF_WARP not in row):
        logging.error('Statistic: no output landmarks')
        return

    # define what is the target and init state according to the experiment results
    is_move_warp = COL_POINTS_MOVE_WARP in row and row[COL_POINTS_MOVE_WARP]
    points_init = points_move if is_move_warp else points_ref
    points_target = points_ref if is_move_warp else points_move
    col_lnds_warp = COL_POINTS_MOVE_WARP if is_move_warp else COL_POINTS_REF_WARP

    # load landmarks
    path_landmarks = update_path_(row[col_lnds_warp], path_experiment)
    if path_landmarks and os.path.isfile(path_landmarks):
        points_warp = load_landmarks(path_landmarks)
        points_warp = np.nan_to_num(points_warp)
    else:
        logging.warning('Invalid path to the landmarks: "%s" <- "%s"',
                        path_landmarks, row[col_lnds_warp])
        return

    # compute Affine statistic
    affine_diff = compute_affine_transf_diff(points_init, points_target, points_warp)
    for name in affine_diff:
        df_experiments.loc[idx, name] = affine_diff[name]

    # compute landmarks statistic
    compute_registration_accuracy(df_experiments, idx, points_target, points_warp,
                                  'elastic', img_diag, wo_affine=True)
    # compute landmarks statistic
    compute_registration_accuracy(df_experiments, idx, points_target, points_warp,
                                  'final', img_diag, wo_affine=False)
    row_ = dict(df_experiments.loc[idx])
    if 'TRE Mean (final)' in row_:
        robust = row_['TRE Mean (final)'] < row_['TRE Mean (init)']
        df_experiments.loc[idx, COL_ROBUSTNESS] = int(robust)


def compute_registration_accuracy(df_experiments, idx, points1, points2,
                                  state='', img_diag=None, wo_affine=False):
    """ compute statistic on two points sets

    :param DF df_experiments: DataFrame with experiments
    :param int idx: index of tha particular record
    :param points1: np.array<nb_points, dim>
    :param points2: np.array<nb_points, dim>
    :param str state: whether it was before of after registration
    :param float img_diag: target image diagonal
    :param bool wo_affine: without affine transform, assume only local/elastic deformation
    """
    if wo_affine and points1 is not None and points2 is not None:
        # removing the affine transform and assume only local/elastic deformation
        _, _, points1, _ = estimate_affine_transform(points1, points2)

    _, stat = compute_points_dist_statistic(points1, points2)
    if img_diag is not None:
        df_experiments.at[idx, COL_IMAGE_DIAGONAL] = img_diag
    # update particular idx
    for name in (n for n in stat if n not in ['overlap points']):
        if img_diag is not None:
            df_experiments.at[idx, 'rTRE %s (%s)' % (name, state)] = stat[name] / img_diag
        df_experiments.at[idx, 'TRE %s (%s)' % (name, state)] = stat[name]
    for name in ['overlap points']:
        df_experiments.at[idx, '%s (%s)' % (name, state)] = stat[name]


def _visual_image_move_warp_lnds_move_warp(record, path_dataset=None,
                                           path_experiment=None):
    """ visualise the case with warped moving image and landmarks
    to the reference frame so they are simple to overlap

    :param {} record: row with the experiment
    :param str|None path_dataset: path to the dataset folder
    :param str|None path_experiment: path to the experiment folder
    :return obj|None:
    """
    assert COL_POINTS_MOVE_WARP in record and isinstance(record[COL_POINTS_MOVE_WARP], str), \
        'Missing registered image "%s"' % COL_POINTS_MOVE_WARP
    path_points_warp = update_path_(record[COL_POINTS_MOVE_WARP], path_experiment)
    if not os.path.isfile(path_points_warp):
        logging.warning('missing warped landmarks for: %r', dict(record))
        return

    points_ref, points_move, path_img_ref = _load_landmarks(record, path_dataset)

    if COL_IMAGE_MOVE_WARP not in record or not isinstance(record[COL_IMAGE_MOVE_WARP], str):
        logging.warning('Missing registered image "%s"', COL_IMAGE_MOVE_WARP)
        image_warp = None
    else:
        path_image_warp = update_path_(record[COL_IMAGE_MOVE_WARP], path_experiment)
        image_warp = load_image(path_image_warp)

    points_warp = load_landmarks(path_points_warp)
    if not list(points_warp):
        return
    # draw image with landmarks
    image = draw_image_points(image_warp, points_warp)
    save_image(os.path.join(update_path_(record[COL_REG_DIR], path_experiment),
                            NAME_IMAGE_MOVE_WARP_POINTS), image)
    del image

    # visualise the landmarks move during registration
    image_ref = load_image(path_img_ref)
    fig = draw_images_warped_landmarks(image_ref, image_warp, points_move,
                                       points_ref, points_warp)
    del image_ref, image_warp
    return fig


def _visual_image_ref_warp_lnds_move_warp(record, path_dataset=None,
                                          path_experiment=None):
    """ visualise the case with warped reference landmarks to the move frame

    :param {} record: row with the experiment
    :param str|None path_dataset: path to the dataset folder
    :param str|None path_experiment: path to the experiment folder
    :return obj|None:
    """
    assert COL_POINTS_REF_WARP in record and isinstance(record[COL_POINTS_REF_WARP], str), \
        'Missing registered image "%s"' % COL_POINTS_REF_WARP
    path_points_warp = update_path_(record[COL_POINTS_REF_WARP], path_experiment)
    if not os.path.isfile(path_points_warp):
        logging.warning('missing warped landmarks for: %r', dict(record))
        return

    points_ref, points_move, path_img_ref = _load_landmarks(record, path_dataset)

    points_warp = load_landmarks(path_points_warp)
    if not list(points_warp):
        return
    # draw image with landmarks
    image_move = load_image(update_path_(record[COL_IMAGE_MOVE], path_dataset))
    # image_warp = tl_io.load_image(row['Moving image, Transf.'])
    image = draw_image_points(image_move, points_warp)
    save_image(os.path.join(update_path_(record[COL_REG_DIR], path_experiment),
                            NAME_IMAGE_REF_POINTS_WARP), image)
    del image

    # visualise the landmarks move during registration
    image_ref = load_image(path_img_ref)
    fig = draw_images_warped_landmarks(image_ref, image_move, points_ref,
                                       points_move, points_warp)
    del image_ref, image_move
    return fig


def visualise_registration(idx_row, path_dataset=None, path_experiment=None):
    """ visualise the registration results according what landmarks were
    estimated - in registration or moving frame

    :param (int, dict) idx_row: tow from iterated table
    :param str path_dataset: path to the dataset folder
    :param str path_experiment: path to the experiment folder
    """
    _, row = idx_row
    row = dict(row)  # convert even series to dictionary
    fig, path_fig = None, None
    # visualise particular experiment by idx
    if COL_POINTS_MOVE_WARP in row and isinstance(row[COL_IMAGE_MOVE_WARP], str):
        fig = _visual_image_move_warp_lnds_move_warp(row, path_dataset, path_experiment)
    elif COL_POINTS_REF_WARP in row and isinstance(row[COL_POINTS_REF_WARP], str):
        fig = _visual_image_ref_warp_lnds_move_warp(row, path_dataset, path_experiment)
    else:
        logging.error('Visualisation: no output image or landmarks')

    if fig is not None:
        path_fig = os.path.join(update_path_(row[COL_REG_DIR], path_experiment),
                                NAME_IMAGE_WARPED_VISUAL)
        export_figure(path_fig, fig)

    return path_fig


def export_summary_results(df_experiments, path_out, params=None,
                           name_txt=NAME_TXT_RESULTS, name_csv=NAME_CSV_RESULTS):
    """ export the summary as CSV and TXT

    :param DF df_experiments: DataFrame with experiments
    :param str path_out: path to the output folder
    :param {str: any} params: experiment parameters
    :param str name_csv: results file name
    :param str name_txt: results file name

    >>> export_summary_results(pd.DataFrame(), '')
    """
    costume_percentiles = np.arange(0., 1., 0.05)
    if df_experiments.empty:
        logging.error('No registration results found.')
        return
    if 'ID' in df_experiments.columns:
        df_experiments.set_index('ID', inplace=True)
    df_summary = df_experiments.describe(percentiles=costume_percentiles).T
    df_summary['median'] = df_experiments.median()
    nb_missing = np.sum(df_experiments['TRE Mean (init)'].isnull())
    df_summary['missing'] = nb_missing / float(len(df_experiments))
    df_summary.sort_index(inplace=True)
    path_csv = os.path.join(path_out, name_csv)
    logging.debug('exporting CSV summary: %s', path_csv)
    df_summary.to_csv(path_csv)

    path_txt = os.path.join(path_out, name_txt)
    logging.debug('exporting TXT summary: %s', path_txt)
    pd.set_option('display.float_format', '{:10,.3f}'.format)
    pd.set_option('expand_frame_repr', False)
    with open(path_txt, 'w') as fp:
        if params:
            fp.write(string_dict(params, 'CONFIGURATION:'))
        fp.write('\n' * 3 + 'RESULTS:\n')
        fp.write('completed registration experiments: %i' % len(df_experiments))
        fp.write('\n' * 2)
        fp.write(repr(df_summary[['mean', 'std', 'median', 'min', 'max', 'missing',
                                  '5%', '25%', '50%', '75%', '95%']]))
