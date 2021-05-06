# from tpers.data import parse_file # , VALUES
from tpers.stats import get_stats, get_roc

import matplotlib.pyplot as plt
import numpy as np
import os


class Data:
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    def __init__(self, data, labels, name, title, values=None, prefix='', format_values=None):
        if values is None:
            values = ['Value %d' % d for d in range(data.shape[-1])]
        self.data, self.labels, self.values  = data, labels, values
        self.name, self.title, self.prefix = name, title, prefix
        self.value_map = {v : i for i,v in enumerate(self.values)}
        self.anomalies = {i for i,l in enumerate(labels) if l}
        self.format_values = {v : {} for v in values} if format_values is None else format_values
        self.figure, self.axis = None, None
        self.figure_roc, self.axis_roc = None, None
    def to_metric(self):
        return self.data
    def __iter__(self):
        yield from zip(self.data, self.labels)
    def __repr__(self):
        return self.title
    def __getitem__(self, i):
        return self.data[i]
    def get_label(self, i):
        return self.labels[i]
    def cache(dir):
        fname = os.path.join(dir, '%s.pkl' % self.name)
        print('saving %s' % fname)
        with open(fname, 'wb') as f:
            pkl.dump(f, self)
    def unpack_predict(self, Z):
        return self.labels, Z
    def plot_roc(self, predict_dict, axis=None):
        if axis is not None:
            for l, stats in predict_dict.items():
                axis.plot(stats['roc'][:,0], stats['roc'][:,1], label=stats['name'], **stats['kwargs'])
            return None, axis
        else:
            if self.figure_roc is not None or self.axis_roc is not None:
                plt.close(self.figure_roc)
            self.figure_roc, self.axis_roc = plt.subplots(1,1,figsize=(8,8))
            self.axis_roc.set_xlim(0,1); self.axis_roc.set_ylim(0,1)
            for l, stats in predict_dict.items():
                label = '%s %s' % (self.prefix, l)
                self.axis_roc.plot(stats['roc'][:,0], stats['roc'][:,1], label=stats['name'], **stats['kwargs'])
            self.figure_roc.suptitle('%s ROC' % self.title, fontsize=8)
            self.axis_roc.set_xlabel('False Positive')
            self.axis_roc.set_ylabel('True Positive')
            self.axis_roc.legend(loc='lower right')
            plt.tight_layout()
            return self.figure_roc, self.axis_roc

class TimeSeriesData(Data):
    def plot(self, data=None, values=None, plot_anom=True, figsize=(16,12), plot_legend=True, make_title=True):
        values = self.values if values is None else values
        data = (self.data if data is None else data)[:,[self.value_map[v] for v in values]]
        if self.figure is not None or self.axis is not None:
            plt.close(self.figure)
        self.figure, self.axis = plt.subplots(len(values)+1,1, sharex=True, figsize=figsize)
        for i, (d, v) in enumerate(zip(data.T, values)):
            self.axis[0].plot(d, color=self.COLORS[i%len(self.COLORS)], label=v, zorder=1)
            self.axis[i+1].plot(d, color=self.COLORS[i%len(self.COLORS)], zorder=1)
            self.axis[i+1].set_ylabel(v)
            self.axis[i+1].autoscale(False)
            self.axis[i+1].set_xlim(-0.05*len(data), 1.05*len(data))
        self.axis[0].autoscale(False)
        self.axis[0].set_xlim(-0.05*len(data), 1.05*len(data))
        if plot_anom:
            lim = (data.max() - 1.05*(data.max() - data.min()), 1.05*data.max())
            lims = list(zip(data.max(0) - 1.05*(data.max(0) - data.min(0)), 1.05*data.max(0)))
            for l in self.anomalies:
                self.axis[0].plot([l,l],lim, ls=':', c='red', alpha=0.25, zorder=0)
                self.axis[0].set_ylim(*lim)
                for i,h in enumerate(lims):
                    self.axis[i+1].plot((l,l), h, ls=':', c='red', alpha=0.25, zorder=0)
                    self.axis[i+1].set_ylim(*h)
        self.axis[-1].set_xlabel('Time')
        self.axis[0].set_ylabel(self.prefix)
        if plot_legend:
            self.axis[0].legend(fontsize=8, ncol=len(values) // 2)
        if make_title:
            self.figure.suptitle('%s %s' % (self.prefix, self.title))
            plt.tight_layout()
        return self.figure, self.axis

class FramedData:
    def __init__(self, raw_labels, frame_indices):
        self.raw_labels = raw_labels
        self.raw_anom = {i for i,y in enumerate(raw_labels) if y}
        self.frame_indices = frame_indices
        self.frame_imap = [{j : i for i,j in enumerate(sorted(idx))} for idx in frame_indices]
        self.frame_anoms = [{imap[i] for i in self.raw_anom.intersection(idx)} for idx,imap in zip(frame_indices,self.frame_imap)]
    def predict_frame(self, i, y):
        return (np.ones if y else np.zeros)(len(self.frame_indices[i]))
    def unpack_predict(self, Z):
        pmap = [{i : [] for i,_ in enumerate(self.raw_labels)} for _ in Z.T]
        for d,Y in enumerate(Z.T):
            for i,y in enumerate(Y):
                for j in self.frame_indices[i]:
                    pmap[d][j].append(y)
        res = np.array([[all(p[i]) for i,_ in enumerate(self.raw_labels)] for p in pmap]).T
        return self.raw_labels, res
#
# class InputData(TimeSeriesData):
#     module = 'input'
#     prev = None
#     args = ['dir', 'set', 'test', 'file']
#     def __init__(self, directory, dataset, testset, logfile, values=None):
#         self.directory, self.dataset, self.testset, self.logfile = directory, dataset, testset, logfile
#         self.file_path = os.path.join(directory, dataset, testset, logfile)
#         self.input_data = parse_file(self.file_path)
#         values = VALUES if values is None else values
#         data = np.array([[d[v] for v in values] for d in self.input_data], dtype=float)
#         labels = np.array([1 if d['label'] > 0 else 0 for d in self.input_data])
#         name, title = '%s-%s' % (dataset, testset), '%s %s' % (dataset, testset)
#         hr, mn = int(np.floor(len(data) / 60 / 60)), int(np.floor(len(data) / 60))
#         print('\t%s\n\t%d data points (%dh, %dm, %ds)\n\t%d values (%s)\
#             ' % (title, len(data), hr, mn - 60*hr, len(data) - 60 * mn, len(values), ', '.join(values)))
#         format_values = {'KMeans' : {'marker' : 'o', 'color' : '#D55E00', 'zorder' : 0}, #'#332288'},
#                         'SOM' : {'marker' : 'o', 'color' : '#E69F00', 'zorder' : 1}}
#         Data.__init__(self, data, labels, name, title, values, 'Raw', format_values=format_values)
