from tpers.base import *
from tpers.util import rescale, pmap
from tpers.stats import get_stats, get_roc
from tpers.plot import plot_rocs, plot_diagrams

from persim import PersistenceImager
from sklearn.preprocessing import PowerTransformer
from scipy.spatial import distance_matrix
from persim.persistent_entropy import *
import numpy.linalg as la
from ripser import ripser
from tqdm import tqdm


def noinf(D):
    return np.array([[b,d] for b,d in D if d < np.inf])

def get_tpers(D, average=False, maximum=False, pmin=0, pmax=np.inf, count=False):
    ps = [[d - b for b,d in dgm if pmin <= d - b < pmax] for dgm in D]
    if maximum:
        return [max(p) if len(p) else 0 for p in ps]
    if count:
        return [len(p) for p in ps]
    elif average:
        return [sum(p) / len(p) if len(p) else 0 for p in ps]
    return [sum(p) for p in ps]

def pers_curve(Ds, average=False, maximum=False, pmin=0, pmax=np.inf, count=False):
    return np.vstack([get_tpers(D, average, maximum, pmin, pmax, count) for D in Ds])

def tpers_predict(C_raw, invert=None, n=20):
    C = rescale(C_raw)
    if invert is not None:
        for d in invert:
            C[:,d] = 1 - C[:,d]
    T = np.linspace(0,1,n-1)
    Z = np.zeros((C.shape[1], C.shape[0], n))
    for k,t in enumerate(T):
        for d in range(C.shape[1]):
            I = [i for i,p in enumerate(C[:,d]) if p >= t]
            Z[d,I,k] = 1
    return Z


class Persistence(FramedData, Data):
    args = ['dim', 'thresh', 'nperm', 'metric']
    def __init__(self, input, dim, thresh, nperm, metric, verbose=True):
        self.input = input
        self.dim, self.thresh = dim, thresh
        pargs = [dim, thresh, 2, False, False, metric, nperm]
        diagrams = [ripser(np.hstack((d.real,d.imag)), *pargs)['dgms'] for d in tqdm(input.data, desc='[ Persistence')]
        self.lim = max(d if d < np.inf else b for dgms in diagrams for dgm in dgms for b,d in dgm)
        name, title = input.name, input.title
        values = [r'$\mathrm{H}_%d$' % d for d in range(self.dim+1)]
        format_values = {'KMeans' : {'marker' : 's', 'color' : '#56B4E9', 'zorder' : 2}, #'#AA4499'},
                        'SOM' : {'marker' : 's', 'color' : '#882255', 'zorder' : 2}}
        Data.__init__(self, diagrams, input.labels, name, title, values, 'Persistence', format_values=format_values)
        FramedData.__init__(self, input.raw_labels, input.frame_indices)
    def to_metric(self, pixel_size=0.1):
        births = [b for dgms in self.data for dgm in dgms for b,_ in dgm]
        tpers = [d - b for dgms in self.data for dgm in dgms for b,d in dgm if d < np.inf]
        pim = PersistenceImager((min(births),max(max(births),pixel_size)),
                                (min(tpers),max(max(tpers),pixel_size)),
                                pixel_size=pixel_size)
        return np.vstack([np.stack([pim.transform(noinf(d)) for d in D]).ravel() for D in tqdm(self.data, desc='[ Persim')])
    def plot(self, frame=None, lim=None):
        if frame is not None:
            if self.figure is None or self.axis is None:
                self.figure, self.axis = plt.subplots(1,1, figsize=(5,5))
            lim = self.lim if lim is None else lim
            self.axis.cla()
            plot_diagrams(self.axis, self.data[frame], lim)
            self.figure.suptitle('%s %s Frame %d' % (self.prefix, self.title, frame), fontsize=8)
            plt.tight_layout()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            return self.figure, self.axis

class TPers(FramedData, TimeSeriesData):
    args = ['invert', 'entropy', 'average', 'pmin']
    def __init__(self, input, invert=None, entropy=False, average=False, pmin=0):
        self.input = input
        self.invert = invert
        self.entropy = entropy
        self.average = average
        self.cur_frame_plt = []
        labels = input.labels
        name, title = input.name, input.title
        if entropy:
            data = np.array([persistent_entropy(D) for D in input.data])
            name += '_entropy'
            title += ' entropy'
        else:
            data = pers_curve(input.data, average, pmin=pmin)
            if average:
                name += '_avg'
                title += ' average'
        if invert is not None:
            for d in range(data.shape[1]):
                if d in invert:
                    data[:,d] *= -1
            name += 'inv%s' % '-'.join(map(str, invert))
            title += ' d%s inverted' % ','.join(map(str, invert))
        data = np.vstack((data.T, data.sum(1))).T
        values = input.values + ['sum']
        format_it = zip(input.values, ['o', '+', '^', 'd'], self.COLORS)
        format_values = {l : {'marker' : mark, 'color' : color} for l,mark,color in format_it}
        format_values['sum'] = {'ls' : ':', 'color' : 'black'}
        format_values['KMeans'] = {'color' : '#88CCEE', 'marker' : 'x'}
        Data.__init__(self, data, labels, name, title, values, 'Total Persistence', format_values=format_values)
        FramedData.__init__(self, input.raw_labels, input.frame_indices)
    def plot(self, frame=None, plot_sum=True, plot_anom=True):
        if frame is None:
            figure, axis = TimeSeriesData.plot(self, self.data[:,:-1], self.values[:-1], plot_anom, (12,8), False)
            if plot_sum:
                axis[0].plot(self.data[:,-1], c='black', ls=':', label='sum', zorder=2)
            axis[0].legend(fontsize=8)
            return figure, axis
        while self.cur_frame_plt:
            self.cur_frame_plt.pop().remove()
        for d, ax in enumerate(self.axis[1:]):
            kwargs = {'color' : self.COLORS[d], 's' : 50, 'zorder' : 2}
            self.cur_frame_plt.append(ax.scatter([frame], [self.data[frame,d]], **kwargs))
            self.cur_frame_plt.append(self.axis[0].scatter([frame], [self.data[frame,d]], **kwargs))
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        return self.input.plot(frame)
