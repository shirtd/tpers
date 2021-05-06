import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from tpers.util import get_lim
import numpy as np
import os

def init_diagram(axis, lim):
    axis.plot([0, 1.2*lim], [0,1.2*lim], c='black', alpha=0.5, zorder=1)
    axis.plot([0, lim], [lim,lim], c='black', ls=':', alpha=0.5, zorder=1)
    axis.plot([lim, lim], [lim, 1.2*lim], c='black', ls=':', alpha=0.5, zorder=1)

def lim_dgm(dgm, lim):
    return np.array([[b, d if d < np.inf else 1.2*lim] for b,d in dgm])

def plot_diagrams(axis, dgms, lim=None, init=True):
    if lim is None:
        lim = get_lim(dgms)
    if init:
        init_diagram(axis, lim)
    elems = []
    for dim, dgm in enumerate(dgms):
        if len(dgm):
            d = lim_dgm(dgm, lim)
            elems += [axis.scatter(d[:,0], d[:,1], s=5, zorder=2, label='%dD' % dim)]
        else:
            elems += [axis.scatter([], [], s=5, zorder=2, label='%dD' % dim)]
    return lim, elems

def plot_rips3(axis, rips, thresh=np.inf, edge_color='red'):
    axis.cla()
    for f in (axis.set_xlim, axis.set_ylim, axis.set_zlim):
        f(rips.points.min(), rips.points.max())
    P_plt = axis.scatter(rips.points[:,0], rips.points[:,1], rips.points[:,2], c='black', s=10, zorder=3)
    edges_plt = []
    edges = rips.points[[e for e in rips[1] if e.data['distance'] <= thresh]]
    if len(edges):
        for e in edges:
            edges_plt += axis.plot(e[:,0], e[:,1], e[:,2], c=edge_color, alpha=0.5)
    verts = [rips.points[list(t)] for t in rips[2] if t.data['distance'] <= thresh]
    poly_plt = None
    if len(verts):
        collection = Poly3DCollection(verts, alpha=0.1)
        poly_plt = axis.add_collection3d(collection)
    return P_plt, edges_plt, poly_plt

def plot_anom(axis, P, anom, **kwargs):
    I, J = list(set(range(len(P)))-set(anom)), list(anom)
    return [axis.scatter(P[I,0], P[I,1], **kwargs),
            axis.scatter(P[J,0], P[J,1], **kwargs)]

def plot_rocs(figure, axis, title, tpers_roc, ubl_roc=None, tpers_sum_roc=None, marker_cycle=['o', '+', '^', 'D', '*']):
    for d,ro in enumerate(tpers_roc):
        axis.plot(ro[:,0], ro[:,1], marker=marker_cycle[d%len(marker_cycle)], label='%d-tpers' % d)
    if tpers_sum_roc is not None:
        axis.plot(tpers_sum_roc[:,0], tpers_sum_roc[:,1], c='black', ls=':', label='tpers sum')
    if ubl_roc is not None:
        axis.plot(ubl_roc[:,0], ubl_roc[:,1], marker='x', label='UBL')
    figure.suptitle('ROC %s' % title)
    axis.set_xlabel('False Positive')
    axis.set_ylabel('True Positive')
    axis.legend()
    plt.tight_layout()

def plot_tpers(figure, axis, title, C=None, anom=set(), Csum=None):
    if C is not None:
        for d,c in enumerate(C.T):
            axis.plot(c, label='%d-tpers' % d, zorder=1)
    if Csum is not None:
        axis.plot(Csum, c='black', ls=':', label='tpers sum', zorder=2)
    for l in anom:
        axis.plot([l,l],[0,1], ls=':', c='red', alpha=0.5, zorder=0)
    axis.legend()
    figure.suptitle('Total persistence %s' % title)
    axis.set_xlabel('Time')
    axis.set_ylabel('Total Persistence')
    plt.tight_layout()

def save_plot(dir, prefix, name, dpi=500):
    fname = os.path.join(dir,'%s_%s.png' % (prefix, name))
    print('saving %s' % fname)
    plt.savefig(fname, dpi=dpi)

def plot_histo(axis, L, histo, name='', ymax=None, stat='count'):
    if ymax is not None:
        ax.set_ylim(0, ymax)
    for dim, h in enumerate(histo):
        axis.plot(L[1:] - 1 / len(L), h, label='H%d' % dim)
    fig.suptitle('TPers histogram %s' % name)
    ax.set_xlabel('TPers')
    ax.set_ylabel(stat)
    ax.legend()
    plt.tight_layout()
