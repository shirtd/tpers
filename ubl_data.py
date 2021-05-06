from tpers.base import Data, TimeSeriesData

import numpy as np
import os, sys

# Dataset specification
# Available datasets, testsets
AVAIL_DATA = {'RUBiSLogs' : ['cpuleak', 'memleak', 'nethog'],
        'HadoopLogs' : ['cpuhog', 'memleak'],
        'SystemSLogs' : ['bottleneck', 'cpuhog', 'memleak']}
# Available values (shared by all data/test sets)
AVAIL_VALUES = ['CPU_USAGE', 'MEM_USAGE', 'CPU_AVAI', 'MEM_AVAI',
        'NET_IN', 'NET_OUT', 'VBD_OO', 'VBD_RD', 'VBD_WR',
        'LOAD1', 'LOAD5']

# Dataset-specific defaults
# Data directory
DIR = os.path.join('data', 'UBLData')
# Default dataset
DATASET = 'SystemSLogs'
# Default testset
TESTSET = 'cpuhog'
# Name of testdata file
LOGFILE = 'te.log'
# Default values
VALUES = ['CPU_USAGE', 'MEM_USAGE',
            'NET_IN', 'NET_OUT',
            'LOAD1', 'LOAD5']

# System defaults
# Window length
LENGTH = 20
# Window overlap
OVERLAP = 10
# Max dimension
DIM = 3

# Dataset presets
PRESETS = [
'--pre scale pca=4 --nperm 20 --torus \
--predict kmeans --analyze input pre persistence',

'--pre detrend scale pca=4 --nperm 20 --torus \
--predict kmeans --analyze input pre persistence',

'--pre scale pca=4 scale=all --nperm 20 --torus \
--predict kmeans --analyze input pre persistence',

'--pre detrend scale pca=4 scale=all --nperm 20 --torus \
--predict kmeans --analyze input pre persistence'
]

PRESET_DICT = { 'RUBiSLogs' :   {   'cpuleak' :     1,
                                    'nethog' :      0,
                                    'memleak' :     0
                                },
                'SystemSLogs' : {   'cpuhog' :      1,
                                    'bottleneck' :  0,
                                    'memleak' :     0
                                },
                'HadoopLogs' :  {   'cpuhog' :      0,
                                    'memleak' :     0
                                }}

# IO util
KEYS = ['CPU_CAP', 'CPU_USAGE', 'MEM_CAP', 'MEM_USAGE',
        'CPU_AVAI', 'MEM_AVAI', 'NET_IN', 'NET_OUT',
        'VBD_OO', 'VBD_RD', 'VBD_WR', 'LOAD1', 'LOAD5']

KEY_t = {'CPU_CAP' : int, 'CPU_USAGE' : int,
        'MEM_CAP' : int, 'MEM_USAGE' : int,
        'CPU_AVAI' : int, 'MEM_AVAI' : int,
        'NET_IN' : int, 'NET_OUT' : int,
        'VBD_OO' : int, 'VBD_RD' : int, 'VBD_WR' : int,
        'LOAD1' : float, 'LOAD5' : float}

def parse_line(line):
    l = line.replace('\n', '').split(' ')
    dat = {'timestamp' : int(l.pop(0)), 'label' : int(l.pop(-1))}
    return {**dat, **{k : KEY_t[k](v) for k,v in zip(l[:-1:2], l[1::2])}}

def parse_file(fname):
    return [parse_line(l) for l in open(fname, 'r')]

# Input data specification
class InputData(TimeSeriesData):
    module = 'input'
    prev = None
    args = ['dir', 'set', 'test', 'file']
    def __init__(self, directory, dataset, testset, logfile, values=None):
        self.directory, self.dataset, self.testset, self.logfile = directory, dataset, testset, logfile
        self.file_path = os.path.join(directory, dataset, testset, logfile)
        self.input_data = parse_file(self.file_path)
        values = VALUES if values is None else values
        data = np.array([[d[v] for v in values] for d in self.input_data], dtype=float)
        labels = np.array([1 if d['label'] > 0 else 0 for d in self.input_data])
        name, title = '%s-%s' % (dataset, testset), '%s %s' % (dataset, testset)
        hr, mn = int(np.floor(len(data) / 60 / 60)), int(np.floor(len(data) / 60))
        print('\t%s\n\t%d data points (%dh, %dm, %ds)\n\t%d values (%s)\
            ' % (title, len(data), hr, mn - 60*hr, len(data) - 60 * mn, len(values), ', '.join(values)))
        format_values = {'KMeans' : {'marker' : 'o', 'color' : '#D55E00', 'zorder' : 0}, #'#332288'},
                        'SOM' : {'marker' : 'o', 'color' : '#E69F00', 'zorder' : 1}}
        Data.__init__(self, data, labels, name, title, values, 'Raw', format_values=format_values)
