from tpers.base import *
from tpers.util import to_complex, to_torus, rescale, pmap

from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import signal
from tqdm import tqdm
import scipy.fft
import time

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class Process(TimeSeriesData):
    args = ['process']
    def __init__(self, input, process=[]):
        self.input = input
        self.process = process
        data, labels, values = input.data, input.labels, input.values
        name, title = input.name, input.title
        for f_str in process:
            f,farg = f_str.split('=') if '=' in f_str else (f_str, None)
            data, labels, values = getattr(self, f)(data, labels, values, farg)
        if process:
            name = '%s_%s' % (input.name, '-'.join(process))
            title = '%s: %s' % (input.title, ', '.join(process))
        format_values = {'KMeans' : {'marker' : 'v', 'color' : '#009E73', 'zorder' : 1},#
                        'SOM' : {'marker' : 'v', 'color' : '#117733', 'zorder' : 1}} # '#44AA99'}}
        Data.__init__(self, data, labels, name, title, values, 'Processed', format_values=format_values)
    def scale(self, data, labels, values, farg=None):
        if farg is None:
            x = (data - data.min(0)) / (data.max(0) - data.min(0))
        else:
            fargs = farg.split(',') if ',' in farg else [farg]
            if 'all' in fargs and 'min' in fargs:
                x = data - data.min()
            elif 'min' in fargs:
                x = data - data.min(0)
            elif 'all' in fargs:
                x = (data - data.min()) / (data.max() - data.min())
        return x, labels, values
    def diff(self, data, labels, values, farg=None):
        return data[1:] - data[:-1], labels[1:] & labels[:-1], values
    def power(self, data, labels, values, farg=None):
        return PowerTransformer().fit_transform(data), labels, values
    def detrend(self, data, labels, values, farg=None):
        return signal.detrend(data, axis=0), labels, values
    def ma(self, data, labels, values, farg='1'):
        l = int(farg)
        _data = np.vstack([np.convolve(d, np.ones(2*l+1), 'valid') / (2*l+1) for d in data.T]).T
        return np.vstack([data[:l], _data, data[-l:]]), labels, values
    def pca(self, data, labels, values, farg='1'):
        x = PCA(n_components=int(farg)).fit_transform(data)
        return x, labels, ['PCA%d' % i for i in range(int(farg))]
    def std(self, data, labels, values, farg=None):
        return StandardScaler().fit_transform(data), labels, values

class Window(FramedData, TimeSeriesData):
    args = ['length', 'overlap']#, 'step']
    def __init__(self, input, length, overlap):#, step):
        self.input = input
        step = 1
        self.length, self.overlap, self.step = length, overlap, step
        data, labels, frame_indices = self.apply_window(input.data, input.labels, length, overlap, step)
        name, title = input.name, input.title
        if length is not None and length > 1:
            name += '_l%d' % length
            title += ' (length=%d' % length
            if overlap is not None and overlap > 0:
                name += '-o%d' % overlap
                title += ', overlap=%d)' % overlap
            else:
                title += ')'
        Data.__init__(self, data, labels, name, title, input.values, 'Windowed')
        FramedData.__init__(self, input.labels, frame_indices)
    def apply_window(self, data, labels, length, overlap, step):
        d = [data[i:i+length:step] for i in range(0,len(data), step*(length-overlap))]
        l = np.array([int(any(labels[i:i+length:step])) for i in range(0,len(labels), step*(length-overlap))])
        idx = [list(range(i,i+length,step)) for i in range(0,len(data), step*(length-overlap))]
        while len(d[-1]) != len(d[0]):
            d, l, idx = d[:-1], l[:-1], idx[:-1]
        return np.stack(d), l, idx
    def plot(self, frame=None, values=None):
        if frame is not None:
            TimeSeriesData.plot(self, self.data[frame], values, False, (16,12), True, False)
            self.figure.suptitle('%s %s Frame %d' % (self.prefix, self.title, frame))
            plt.tight_layout()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            return self.figure, self.axis

class Transform(FramedData, Data):
    args = ['torus', 'period', 'fft', 'exp', 'abs']
    def __init__(self, input, torus=False, period=None, fft=True, exp=1, absv=False):
        self.input = input
        name, title = input.name, input.title
        data, labels, values = input.data, input.labels, input.values
        self.torus, self.period, self.fft = torus, period, fft
        self.exp, self.abs = exp, absv
        self.torus_figure, self.torus_axis = None, None
        if exp > 1:
            data = data ** exp
            name += '-exp%f' % exp
            title += ' ^ %f' % exp
        if absv:
            data = abs(data)
            name += '-abs'
            title += ' abs'
        self.complex_data = None
        if torus or fft or period is not None:
            if period:
                data = np.stack([to_complex(p, period) for p in data])
                self.complex_data = data
                name += '-p%d' % period
                title = '%s, period=%d)' % (title[:-1], period)
            if fft:
                w = signal.blackman(data.shape[1])
                d = np.tile(np.reshape(w, (-1,1)), data.shape[0])
                d = np.stack([d for  _ in range(data.shape[2])]).T
                data = scipy.fft.fft(d*data, axis=1, norm='ortho')
                self.complex_data = data
                name += '_fft'
                title += ' FFT'
            if torus:
                if self.complex_data is None:
                    self.complex_data = np.stack([to_complex(p) for p in data])
                data = np.stack([to_torus(x) for x in tqdm(self.complex_data, desc='[ Torus transform')])
                name += '_torus'
                title += ' torus'
        print('[ Transformed data shape:', data.shape)
        Data.__init__(self, data, labels, name, title, values, 'Transform')
        FramedData.__init__(self, input.raw_labels, input.frame_indices)
        self.lims = ((1.05*data.real.min(), 1.05*data.real.max()),
                    (1.05*data.real.min(), 1.05*data.real.max()))
    def plot_complex(self, data, values, plot_anom=False):
        values = self.values if values is None else values
        data = (self.complex_data if data is None else data)[:,[self.value_map[v] for v in values]]
        if self.figure is None or self.axis is None:
            self.figure, self.axis = plt.subplots(2, len(values)//2, sharex=True, sharey=True, figsize=(10,7))
        for i, (d, v) in enumerate(zip(data.T, values)):
            ax = self.axis[i%2, i//2] if len(values)//2 > 1 else self.axis[i]
            ax.cla()
            ax.scatter(d.real, d.imag, color=self.COLORS[i], s=10, zorder=1)
            ax.set_title(v)
            ax.set_xlim(self.lims[0])
            ax.set_ylim(self.lims[1])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        return self.figure, self.axis
    def plot_torus(self, data):
        if self.torus_figure is None or self.torus_axis is None:
            self.torus_figure = plt.figure()
            self.torus_axis = plt.subplot(projection='3d')
        else:
            self.torus_axis.cla()
        G = np.meshgrid(np.angle(data[:,0]), np.angle(data[:,1]))
        X = np.meshgrid(abs(data[:,0]), abs(data[:,1]))
        T = np.vstack(((X[0].flatten() + X[1].flatten()*np.cos(G[0].flatten()))*np.cos(G[1].flatten()),
                    (X[0].flatten() + X[1].flatten()*np.cos(G[0].flatten()))*np.sin(G[1].flatten()),
                    X[1].flatten() * np.sin(G[0].flatten()))).T
        self.torus_axis.set_xlim(-1,1)
        self.torus_axis.set_ylim(-1,1)
        self.torus_axis.set_zlim(-1,1)
        self.torus_axis.scatter(T[:,0], T[:,1], T[:,2], s=5, alpha=0.5)
        self.torus_figure.canvas.draw()
        self.torus_figure.canvas.flush_events()
        return self.torus_figure, self.torus_axis
    def plot(self, frame=None, values=None, plot_anom=False):
        if frame is not None:
            if self.period is None and not self.fft and not self.torus:
                return self.input.plot(frame, values)
            self.plot_complex(self.complex_data[frame], values)
            if self.torus:
                self.plot_torus(self.complex_data[frame])
            self.figure.suptitle('%s %s Frame %d' % (self.prefix, self.title, frame))
            plt.tight_layout()
            return self.figure, self.axis
        return self.figure, self.axis
