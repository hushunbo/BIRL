"""
Benchmark for ANTs
see:
* http://stnava.github.io/ANTs
* https://sourceforge.net/projects/advants/
* https://github.com/ANTsX/ANTsPy

INSTALLATION:
See: https://brianavants.wordpress.com/2012/04/13/updated-ants-compile-instructions-april-12-2012/

* Do NOT download the binary code, there is an issue:
    - https://sourceforge.net/projects/advants/files/ANTS/ANTS_Latest
    - https://github.com/ANTsX/ANTs/issues/733
* Compile from source:
    > git clone git://github.com/stnava/ANTs.git
    > mkdir antsbin
    > cd antsbin
    > ccmake ../ANTs
    > make -j 4
* Install it as python package
    > pip install git+https://github.com/ANTsX/ANTsPy.git


Run the basic ANT registration with original parameters:
>> python bm_experiments/bm_ANTs.py \
    -c ./data_images/pairs-imgs-lnds_anhir.csv \
    -d ./data_images \
    -o ./results \
    --path_ANTs ./applications/ANTs/bin \
    --path_config ./configs/ANTs.txt


Disclaimer:
* needed to use own compiled version

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import glob
import shutil

import pandas as pd
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import (load_landmarks, save_landmarks, convert_image2nifti_gray,
                                    convert_nifti2image, get_image_size)
from birl.utilities.experiments import create_basic_parse, parse_arg_params, exec_commands
from birl.cls_benchmark import ImRegBenchmark, COL_IMAGE_REF, COL_IMAGE_MOVE
from birl.bm_template import main
from bm_experiments import bm_comp_perform


COMMAND_REGISTER = '%(path_ANTs)s/antsRegistration \
    --dimensionality 2 \
    %(config)s \
    --output [%(output)s/trans]'
COMMAND_WARP_IMAGE = '%(path_ANTs)s/antsApplyTransforms \
    --dimensionality 2 \
    --input %(img_source)s \
    --output %(output)s/%(img_name)s.nii \
    --reference-image %(img_target)s \
    --transform %(transfs)s \
    --interpolation Linear'
COMMAND_WARP_POINTS = '%(path_ANTs)s/antsApplyTransformsToPoints \
    --dimensionality 2 \
    --input %(path_points)s \
    --output %(output)s/%(pts_name)s.csv \
    --transform %(transfs)s'
COL_IMAGE_NII_REF = COL_IMAGE_REF + ' Nifty'
COL_IMAGE_NII_MOVE = COL_IMAGE_MOVE + ' Nifty'


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-ANTs', '--path_ANTs', type=str, required=True,
                          help='path to the ANTs executables')
    a_parser.add_argument('-config', '--path_config', required=True,
                          type=str, help='path to the ANTs regist. configuration')
    return a_parser


class BmANTs(ImRegBenchmark):
    """ Benchmark for ANTs
    no run test while this method requires manual compilation of ANTs

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> params = {'nb_workers': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': os.path.join(update_path('data_images'),
    ...                                      'pairs-imgs-lnds_mix.csv'),
    ...           'path_ANTs': '.', 'path_config': '.'}
    >>> benchmark = BmANTs(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_ANTs',
                                                        'path_config']
    REQUIRED_EXECUTABLES = ['antsRegistration',
                            'antsApplyTransforms',
                            'antsApplyTransformsToPoints']

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

        path_execs = [os.path.join(self.params['path_ANTs'], execute)
                      for execute in BmANTs.REQUIRED_EXECUTABLES]
        assert all(os.path.isfile(p) for p in path_execs), \
            'Some executables are missing: %r' % [p for p in path_execs if not os.path.isfile(p)]

    def _prepare_registration(self, record):
        """ prepare the experiment folder if it is required,

        * create registration macros

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. generate command before registration experiment')
        # set the paths for this experiment
        path_dir = self._get_path_reg_dir(record)
        path_im_ref, path_im_move, _, _ = self._get_paths(record)

        # Convert images to Nifty
        record[COL_IMAGE_NII_REF] = convert_image2nifti_gray(path_im_ref, path_dir)
        record[COL_IMAGE_NII_MOVE] = convert_image2nifti_gray(path_im_move, path_dir)

        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        path_dir = self._get_path_reg_dir(record)
        with open(self.params['path_config'], 'r') as fp:
            config = [l.rstrip().replace('\\', '') for l in fp.readlines()]

        config = ' '.join(config) % {
            'img_target': record[COL_IMAGE_NII_REF],
            'img_source': record[COL_IMAGE_NII_MOVE]
        }
        cmd = COMMAND_REGISTER % {
            'config': config,
            'path_ANTs': self.params['path_ANTs'],
            'output': path_dir
        }

        return cmd

    def _extract_warped_images_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to
        """
        path_dir = self._get_path_reg_dir(record)
        _, path_im_move, _, path_lnds_move = self._get_paths(record)
        name_im_move = os.path.splitext(os.path.basename(path_lnds_move))[0]
        name_lnds_move = os.path.splitext(os.path.basename(path_lnds_move))[0]

        # simplified version of landmarks
        lnds = load_landmarks(path_lnds_move)
        path_lnds = os.path.join(path_dir, name_lnds_move + '.csv')
        # https://github.com/ANTsX/ANTs/issues/733#issuecomment-472049427
        width, height = get_image_size(path_im_move)
        pd.DataFrame(np.vstack([width - lnds[:, 0], height - lnds[:, 1], ]).T * -1.,
                     columns=['x', 'y']).to_csv(path_lnds, index=None)

        # list output transformations
        tf_elast_inv = sorted(glob.glob(os.path.join(path_dir, 'trans*InverseWarp.nii*')))
        tf_elast = [os.path.join(os.path.dirname(p), os.path.basename(p).replace('Inverse', ''))
                    for p in tf_elast_inv]
        tf_affine = sorted(glob.glob(os.path.join(path_dir, 'trans*GenericAffine.mat')))
        # generate commands
        cmd_warp_img = COMMAND_WARP_IMAGE % {
            'path_ANTs': self.params['path_ANTs'],
            'output': path_dir,
            'img_target': record[COL_IMAGE_NII_REF],
            'img_source': record[COL_IMAGE_NII_MOVE],
            'transfs': ' -t '.join(sorted(tf_affine + tf_elast, reverse=True)),
            'img_name': name_im_move
        }
        cmd_warp_pts = COMMAND_WARP_POINTS % {
            'path_ANTs': self.params['path_ANTs'],
            'output': path_dir,
            'path_points': path_lnds,
            'transfs': ' -t '.join(['[ %s , 1]' % tf if 'Affine' in tf else tf
                                    for tf in sorted(tf_affine + tf_elast_inv)]),
            'pts_name': name_lnds_move
        }
        # execute commands
        exec_commands([cmd_warp_img, cmd_warp_pts],
                      path_logger=os.path.join(path_dir, 'warping.log'))
        path_regist = convert_nifti2image(os.path.join(path_dir, name_im_move + '.nii'), path_dir)

        lnds = pd.read_csv(path_lnds, index_col=None).values * -1.
        save_landmarks(path_lnds, np.vstack([width - lnds[:, 0], height - lnds[:, 1], ]).T)

        return None, path_regist, None, path_lnds

    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param {str: value} record: dictionary with regist. information
        :return {str: value}: the same or updated regist. info
        """
        [[os.remove(p) for p in glob.glob(os.path.join(self._get_path_reg_dir(record), ext))]
         for ext in ['*.nii', '*.nii.gz', '*.mat']]
        return record


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmANTs)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
