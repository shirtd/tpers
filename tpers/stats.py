from tpers.util import rescale

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn_som.som import SOM
from sklearn import cluster
import scipy.signal as sig
import numpy.linalg as la
from tqdm import tqdm
import pandas as pd
import numpy as np
import time


fstats = {  'tn' : lambda l,y: not (l or y),
            'tp' : lambda l,y: l and y,
            'fp' : lambda l,y: not l and y,
            'fn' : lambda l,y: l and not y}
#
# def get_stats(L, Y, streak=1, desc='', lead=20):
#     stats = [{k : 0 for k in fstats.keys()} for _ in range(Y.shape[1])]
#     for i, (l,ys) in tqdm(list(enumerate(zip(L, Y))), desc=desc):
#         for k,y in enumerate(ys):
#             for s,f in fstats.items():
#                 if all(f(l,Y[i+t,k]) for t in range(min(streak,len(L)-i)):
#                     stats[k][s] += 1
#                 elif lead > 1 and any(l for l in L[i:min(i+lead,len(L))]):
#                     if s == 'tp' or s == 'fp':
#                         stats[k][s] += 1
#     return stats

def get_stats(L, Y, streak=1, lead=1, desc=''):
    stats = [{k : 0 for k in ['tn','tp','fp','fn']} for _ in range(Y.shape[1])]
    for i, (l,ys) in tqdm(list(enumerate(zip(L, Y))), desc=desc):
        for k,y in enumerate(ys):
            res = all(Y[t,k] for t in range(max(0,i-streak),i+1))
            if res and any(l for l in L[i:min(i+lead,len(L))]):
                stats[k]['tp'] += 1
            elif res:
                stats[k]['fp'] += 1
            elif not res and not l:
                stats[k]['tn'] += 1
            elif not res and l:
                stats[k]['fn'] += 1
    return stats

def confusion(L, Z):
    return dict(zip(('tn','fp','fn','tp'), confusion_matrix(L, Z).ravel()))

def get_accuracy(s):
    return (s['tn'] + s['tp']) / (s['tn'] + s['fp'] + s['fn'] + s['tp'])

def get_precision(s):
    return s['tp'] / (s['tp'] + s['fp'])

def get_tpr(s):
    if s['tp'] + s['fn'] == 0:
        return s['tp'] / 1e-14
    return s['tp'] / (s['tp'] + s['fn'])

def get_fpr(s):
    if s['fp'] + s['tn'] == 0:
        return s['fp'] / 1e-14
    return s['fp'] / (s['fp'] + s['tn'])

def get_auc(s):
    return (get_tpr(s) + (1-get_fpr(s))) / 2

def get_roc(stats):
    return np.array([[1,1]] + [[get_fpr(s), get_tpr(s)] for s in stats] + [[0,0]])

def max_roc(s):
    # return max(s['roc'], key=lambda p: p[1] - p[0])
    # ss = max(s['stats'], key=get_accuracy)
    ss = max(s['stats'], key=get_auc)
    return np.array([get_fpr(ss), get_tpr(ss), get_accuracy(ss), get_precision(ss)])

def format_stats(s):
    return 100*max_roc(s), '%s' % s['name']

def stat_str(res, precision=2):
    pd.set_option('precision', precision)
    stats, headers = zip(*[format_stats(s) for m, S in res.items() for l,s in S.items()])
    return str(pd.DataFrame(stats, headers, ['FPR','TPR','ACC', 'PPV']))

def stat_dict(res):
    return {s['name'] : dict(zip(('fpr','tpr','acc', 'ppv'), max_roc(s))) for m, S in res.items() for l,s in S.items()}


class Predict:
    def __init__(self, n=30, streak=1, lead=1):
        self.n = n
        self.streak = 1
        self.lead = 1
    def get_thresholds(self, x):
        sx = sorted(x)
        return np.array([sx[int(len(x)*p)-1] for p in np.linspace(0.3,0.98,self.n-1)])
        # return np.linspace(x.min(),x.max(),self.n-1)
    def get_predict(self, x):
        T = self.get_thresholds(x)
        Z = np.zeros((len(x), self.n))
        for k,t in enumerate(T):
            Z[[i for i,p in enumerate(x) if p >= t],k] = 1
        return Z
    def run(self, l, d, module):
        x = self.predict_transform(d)
        if len(x.shape) > 1:
            S = [self.get_stats(l, xx, module) for xx in x.T]
            return max(S, key=lambda s: max(tp-fn for fn,tp in s['roc']))
        return self.get_stats(l, x, module)
    def get_stats(self, l, x, module):
        p = self.get_predict(x)
        L, Z = module.unpack_predict(p)
        s = get_stats(L, Z, self.streak, self.lead, '[ %s, %s predict' % (module.prefix, l))
        return {'predict' : p, 'stats' : s, 'roc' : get_roc(s), 'name' : '%s %s' % (module.prefix, l),
                'kwargs' : module.format_values[l] if l in module.format_values else {}}
    def __call__(self, module):
        return {l : self.run(l, d, module) for l, d in zip(module.values, module.data.T)}

class MetricPredict(Predict):
    def __init__(self, name, *args, **kwargs):
        Predict.__init__(self, *args, **kwargs)
        self.name = name
    def __call__(self, module):
        return {self.name : self.run(self.name, module.to_metric(), module)}

class SOMPredict(MetricPredict):
    def __init__(self, train_data, rows=32, columns=32, lr=0.7, sigma=4, epochs=10, n=30, streak=3, lead=10, ma=None):
        MetricPredict.__init__(self, 'SOM', n, streak, lead)
        self.shape = (rows, columns)
        self.lr, self.sigma, self.epochs = lr, sigma, epochs
        self.train_data, self.som = train_data, None
        self.min, self.max = None, None
        self.train(train_data)
    def get_som(self, x):
        som = SOM(self.shape[0], self.shape[1], x.shape[1], self.lr, self.sigma)
        self.min, self.max = x.min(0), x.max(0)
        som.fit(self.rescale(x), epochs=self.epochs)
        return som
    def rescale(self, x):
        return (x - self.min) / (self.max - self.min)
        # return rescale(x)
    def train(self, train_data, validate=3):
        t0 = time.time()
        if validate is None:
            self.som = self.get_som(X)
            return self.som
        best = None
        kf = KFold(n_splits=validate, shuffle=True)
        X, y = train_data.data, train_data.labels
        for i, (tr_idx, te_idx) in enumerate(kf.split(X)):
            som = self.get_som(X[tr_idx])
            p = self.get_predict(self.som_predict(som, X[te_idx]))
            s = get_stats(y[te_idx], p, self.streak, self.lead, '[ Validation round %d' % i)
            acc = max(get_accuracy(ss) for ss in s)
            # acc = max(get_auc(ss) for ss in s)
            if best is None or acc > best[1]:
                best = (som, acc)
        self.time = time.time() - t0
        print('[ Training time %0.4fsec' % self.time)
        self.som = best[0]
        return self.som
    def som_predict(self, som, x):
        grid = np.reshape(som.weights, self.shape + (-1,))
        return np.array([self.area(grid, *som._locations[l]) for l in som.predict(self.rescale(x))])
    def predict_transform(self, x):
        return self.som_predict(self.som, x)
    def area(self, G, i, j):
        return sum([la.norm(G[i,j] - G[i-1,j],1) if i > 0 else 0,
                    la.norm(G[i,j] - G[i+1,j],1) if i < self.shape[0]-1 else 0,
                    la.norm(G[i,j] - G[i,j-1],1) if j > 0 else 0,
                    la.norm(G[i,j] - G[i,j+1],1) if j < self.shape[0]-1 else 0])
    def get_thresholds(self, x):
        sx = sorted(x)
        return np.array([sx[int(len(x)*p)-1] for p in np.linspace(0.5,0.98,self.n-1)])
    def __call__(self, module):
        return {'SOM' : self.run('SOM', module.to_metric(), module)}

class ThresholdPredict(Predict):
    def predict_transform(self, x):
        return x

class KMeansPredict(MetricPredict):
    def __init__(self, *args, **kwargs):
        MetricPredict.__init__(self, 'KMeans', *args, **kwargs)
    def predict_transform(self, x):
        return cluster.KMeans(n_clusters=2, random_state=0).fit_transform(x) # .min(1)
    def run(self, l, d, module):
        x = self.predict_transform(d)
        S = [self.get_stats(l, xx, module) for xx in x.T]
        return max(S, key=lambda s: max(tp-fp for fp,tp in s['roc']))

class MinKMeansPredict(MetricPredict):
    def __init__(self, *args, **kwargs):
        MetricPredict.__init__(self, 'MinKMeans', *args, **kwargs)
    def predict_transform(self, x):
        return cluster.KMeans(n_clusters=2, random_state=0).fit_transform(x).min(1)

class MaxKMeansPredict(MetricPredict):
    def __init__(self, *args, **kwargs):
        MetricPredict.__init__(self, 'MaxKMeans', *args, **kwargs)
    def predict_transform(self, x):
        return cluster.KMeans(n_clusters=2, random_state=0).fit_transform(x).max(1)

MODELS = {'threshold' : ThresholdPredict, 'SOM' : SOMPredict, 'kmeans' : KMeansPredict,
                        'minkmeans' : MinKMeansPredict, 'maxkmeans' : MaxKMeansPredict}

class PeakPredict(Predict):
    def predict_transform(self, x):
        p = sig.find_peaks(x)[0]
        h = sig.peak_prominences(x,p)[0]
        y = np.zeros(len(x))
        y[p] = h
        return y

# class ARIMAPredict(Predict):
#     def predict_transform(self, X, jump_start=3):
#         logX = X
#         Y = np.zeros(len(X))
#         for t in range(jump_start, len(X)-1):
#             model = ARIMA(logX[:t], order=(1,1,0), enforce_invertibility=False, enforce_stationarity=False)
#             y = 10 ** (model.fit().forecast()[0] - 1)
#             y = model.fit().forecast()[0]
#             Y[t+1] = abs(y - X[t+1])
#         return Y
