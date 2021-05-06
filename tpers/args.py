# from tpers.data import *
# from tpers.presets import *
from tpers.stats import MODELS
import numpy as np
import argparse


def get_parser(avail_data, avail_values, dir, dataset, testset, logfile, values, length, overlap, dim, presets=[], preset_dict=[]):
    parser = argparse.ArgumentParser(prog='tpers')
    parser.add_argument('--data', action='store_true', help='Print available data/test sets.')
    parser.add_argument('--dir', '-D', type=str, default=dir, help='Data directory. Default: %s' % dir)
    parser.add_argument('--set', '-S', type=str, default=dataset, help='Dataset. Default: %s' % dataset)
    parser.add_argument('--test', '-T', type=str, default=testset, help='Test set. Default: %s' % testset)
    parser.add_argument('--file', '-F', type=str, default=logfile, help='File name. Default: %s' % logfile)
    parser.add_argument('--cache', type=str, default='cache', help='Cache directory')

    parser.add_argument('--preset', nargs='?', type=int, const=-1, choices=range(len(presets)), help='run preset') #
    parser.add_argument('--show-presets', action='store_true', help='print available presets') #

    parser.add_argument('--values', nargs='+', help='Data values', choices=avail_values, default=values)
    parser.add_argument('--plot', nargs='+', default=[], help='Plot modules',
                        choices=['input', 'pre', 'window', 'transform', 'persistence', 'tpers', 'post'])
    parser.add_argument('--nroc', default=30, type=int, help='number of points on ROC curves')
    parser.add_argument('--frame', default=None, type=int, help='Frame to show')
    parser.add_argument('--show', action='store_true', help='show plots')
    parser.add_argument('--save', nargs='?', default=None, const='', help='save plots to directory (default: figures)')
    parser.add_argument('--predict', default='kmeans', choices=MODELS.keys(), help='analyze selected modules')
    parser.add_argument('--analyze', nargs='+', default=[], help='Analyze selected modules using prediction method specified by --predict. \
                                                                    override choice with model=predict. \
                                                                    choices: %s' % ', '.join(['input', 'pre', 'persistence', 'tpers', 'post']))

    parser.add_argument('--aplot', nargs='+', default=[], choices=['input', 'pre', 'tpers', 'post'],
                        help='Analyze and plot selected modules using prediction method specified by --predict')

    parser.add_argument('--interact', action='store_true', help='Interactive plot')
    parser.add_argument('--som', action='store_true', help='compare with saved SOM predict')
    parser.add_argument('--lead', type=int, default=10, help='SOM lead (anomaly pending) time')
    parser.add_argument('--streak', type=int, default=3, help='SOM required streak')


    process_choices = ['scale', 'diff', 'power', 'detrend', 'ma']
    process_str = 'processing to apply, in order:\nscale: scale features to [0,1],\n \
                    diff: apply difference transform (derivative) to each feature,\n \
                    power: apply power transfrom to each feature,\n \
                    detrend: detrend each feature,\n \
                    ma: apply moving average. ma=w convolves with 2*2+1 point window,\n \
                    pca: PCA transform. pca=n reduces to n principal compnents.'
    process_parser = parser.add_argument_group('process', 'input data pre/post processing')
    process_parser.add_argument('--pre', nargs='+', default=[], help='pre-%s' % process_str)
    process_parser.add_argument('--post', nargs='+', default=[], help='post-%s' % process_str)

    window_parser = parser.add_argument_group('window', 'processed windowed data')
    window_parser.add_argument('--length', type=int, default=length, help='window length')
    window_parser.add_argument('--overlap', type=int,  default=overlap, help='window overlap')

    transform_parser = parser.add_argument_group('transform', 'transform processed data')
    transform_parser.add_argument('--period', nargs='?', default=None, type=int, const=0,
                            help='raw complex transform period (window length by default)')
    transform_parser.add_argument('--fft', action='store_true', help='apply fft transform')
    transform_parser.add_argument('--torus', action='store_true', help='to clifford d-torus')
    transform_parser.add_argument('--exp', type=float, default=1., help='data exponent before transform')
    transform_parser.add_argument('--abs', action='store_true', help='absolute value before transform')

    persistence_parser = parser.add_argument_group('persist', 'apply persistence to transformed data')
    persistence_parser.add_argument('--dim', default=dim, type=int, help='maximum dimension.')
    persistence_parser.add_argument('--thresh', default=np.inf, type=float, help='rips threshold')
    persistence_parser.add_argument('--nperm', type=int, help='greedy permutations')
    persistence_parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'manhattan', 'cosine'], help='Metric')

    tpers_parser = parser.add_argument_group('tpers', 'analyze total persistence')
    tpers_parser.add_argument('--invert', nargs='+', default=None, type=int, help='Dimsnions to invert.')
    tpers_parser.add_argument('--entropy', action='store_true', help='Compute persistent entropy.')
    tpers_parser.add_argument('--average', action='store_true', help='average persistence.')
    tpers_parser.add_argument('--pmin', type=float, default=0, help='Minimum persistence.')

    return parser
