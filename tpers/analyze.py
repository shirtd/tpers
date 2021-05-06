from tpers.base import *
from tpers.process import Process, Window, Transform
from tpers.tda import Persistence, TPers
from tpers.stats import *

import time
import sys


def plot_rocs(figure, axis, title, rocs, marker_cycle=['o', '+', '^', 'D', '*']):
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

def parse_analyze(analyze, predict, *args, **kwargs):
    if '=' in analyze:
        m,p = analyze.split('=')
        predict = p if p in MODELS else predict
    return MODELS[predict](*args, **kwargs)

def savefig(name, dir='figures', fig=None, dpi=300):
    fname = os.path.join(dir, '%s.png' % name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(' - Saving %s' % fname)
    fig.savefig(fname, dpi=dpi)


class Pipeline:
    module_sequence = [('pre', Process, ['pre']),
                        ('window' , Window, Window.args),
                        ('transform' , Transform, Transform.args),
                        ('persistence' , Persistence, Persistence.args),
                        ('tpers', TPers, TPers.args),
                        ('post', Process, ['post'])]
    marker_cycle=['o', '+', '^', 'D', '*']
    def __init__(self, args):
        self.args = args
        pers = {'transform', 'persistence', 'tpers', 'post'}
        todo = set(args.plot).union(args.analyze)
        if not args.interact and not pers.intersection(todo):
            self.module_sequence = self.module_sequence[:1]
        self.interact = args.interact
    def __call__(self, input_data):
        data, prev = {'input' : input_data}, input_data
        for k, module, argl in self.module_sequence:
            args = [getattr(self.args, s) for s in argl]
            data[k] = module(prev, *args)
            prev = data[k]
        return data
    def get_predict(self, *args, **kwargs):
        return {m : parse_analyze(m, self.args.predict,*args, **kwargs) for m in self.args.analyze}
    def analyze(self, predict, data, axis=None):
        res = {}
        for m, p in predict.items():
            mm = m.split('=')[0] if '=' in m else m
            res[m] = p(data[mm])
            data[mm].plot_roc(res[m], axis)
        return res
    def plot(self, data, plot=None, frame=None):
        plot = self.args.plot if plot is None else plot
        frame = self.args.frame if frame is None else frame
        for m in plot:
            if isinstance(data[m], FramedData):
                fig, ax = data[m].plot(frame)
                if self.args.save is not None:
                    savefig(data[m].name, self.args.save, fig)
            else:
                fig, ax = data[m].plot()
                if self.args.save is not None:
                    savefig(data[m].name, self.args.save, fig)

class TPersInteract(TPers):
    def __init__(self, data_dict, plot, save=None):
        self.input, self.data_dict, inp = data_dict['persistence'], data_dict, data_dict['post']
        Data.__init__(self, inp.data, inp.labels, inp.name, inp.title, inp.values, 'Processed TPers', format_values=inp.format_values)
        FramedData.__init__(self, data_dict['tpers'].raw_labels, data_dict['tpers'].frame_indices)
        self.cur_frame_plt = []
        self.plot()
        if save is not None:
            savefig(self.name, save, self.figure)
        plt.show(block=False)
        self.frame_modules = {m for m,f in data_dict.items() if isinstance(f, FramedData)}
        self.cur_framed_data = {l : data_dict[l] for l in self.frame_modules.intersection(plot)}
        self.n_frames = len(self.data)
        self.mouse_cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.release_cid = self.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.key_cid = self.figure.canvas.mpl_connect('key_press_event', self.onpress)
        self.last_frame, self.press_time = -1, None
        if any(m in plot for m in self.frame_modules):
            frame = input('[ Plot frame (0-%d): ' % self.n_frames)
            while frame:
                self.plot_frame(int(frame))
                frame = input('[ Plot frame (0-%d): ' % self.n_frames)
        else:
            input('[ Exit ]')
    def plot_frame(self, frame):
        if frame < self.n_frames:
            self.last_frame = frame
            for l, module in self.cur_framed_data.items():
                module.plot(frame)
            self.plot(frame)
            plt.show(block=False)
        else:
            print(' ! Invalid frame')
    def onclick(self, event):
        self.press_time = time.time()
    def onrelease(self, event):
        if (any(event.inaxes == ax for ax in self.axis)
                and self.press_time is not None and time.time() - self.press_time < 0.5):
            frame = min(max(int(np.round(event.xdata)),0), self.n_frames-1)
            sys.stdout.write('%d\n[ Plot frame (0-%d): ' % (frame, self.n_frames))
            sys.stdout.flush()
            self.plot_frame(frame)
            self.figure.canvas.manager.window.activateWindow()
            self.figure.canvas.manager.window.raise_()
    def onpress(self, event):
        if event.key == 'right':
            frame = (self.last_frame+1) % self.n_frames
        elif event.key == 'left':
            frame = (self.last_frame-1) % self.n_frames
        else:
            return
        sys.stdout.write('%d\n[ Plot frame (0-%d): ' % (frame, self.n_frames))
        sys.stdout.flush()
        self.plot_frame(frame)
        self.figure.canvas.manager.window.activateWindow()
        self.figure.canvas.manager.window.raise_()
