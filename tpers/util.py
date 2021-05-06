from multiprocessing import Pool, cpu_count
from itertools import combinations
from sklearn.manifold import MDS
from functools import partial
from tqdm import tqdm
import numpy as np
import sys


def dmat_mds(dmat, dims=2, **kwargs):
    return MDS(dissimilarity='precomputed', n_components=dims, **kwargs).fit_transform(dmat)

def pmap(fun, x, *args, **kw):
# def pmap(fun, x, max_cores=None, *args, **kw):
    pool = Pool()
    f = partial(fun, *args, **kw)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y


def rescale(x):
    return (x - x.min(0)) / (x.max(0) - x.min(0))

def to_complex(x, period=None):
    period = len(x) if (period is None or not period) else period
    return np.array([p * np.exp(1j*2*np.pi*(i % period)/period) for i,p in enumerate(x)])

def to_torus(X):
    G = np.meshgrid(*[np.angle(y) for y in X.T])
    R = np.meshgrid(*[abs(y) for y in X.T])
    return np.vstack([(r*np.exp(1j*g)).flatten() for r,g in zip(R,G)]).T

def get_lim(dgms):
    return max(d if d < np.inf else b for dgm in dgms for b,d in dgm)
