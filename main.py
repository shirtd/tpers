#!/usr/bin/env /Users/kirk/anaconda3/envs/py38/bin/python

from tpers.args import get_parser
from tpers.analyze import *
# from tpers.presets import *
from tpers.util import *

import os, sys, json,time
import pickle as pkl

from ubl_data import *

if __name__ == "__main__":
    parser = get_parser(AVAIL_DATA, AVAIL_VALUES,
                        DIR, DATASET, TESTSET,
                        LOGFILE, VALUES,
                        LENGTH, OVERLAP, DIM,
                        PRESETS, PRESET_DICT)
    
    args = parser.parse_args()

    if args.show_presets:
        print('\n'.join(['[ %d ]\t%s' % (i,p) for i,p in enumerate(PRESETS)]))
        sys.exit(1)

    info_dict = {'command' : ' '.join(sys.argv)}
    if args.preset is not None:
        if args.preset == -1:
            args.preset = PRESET_DICT[args.set][args.test]
        info_dict['preset'] = PRESETS[args.preset]
        print('[ Preset %d ]\n\t%s' % (args.preset, PRESETS[args.preset]))
        args, _args = parser.parse_args(PRESETS[args.preset].split()), args.__dict__
        args.analyze, args.som = args.analyze + _args['analyze'], _args['som']
        args.interact, args.plot = _args['interact'], _args['plot']
        args.show, args.save = _args['show'], _args['save']
        args.set, args.test = _args['set'], _args['test']
        if _args['nperm'] is not None:
            args.nperm = _args['nperm']

    if args.aplot:
        args.plot = args.aplot
        args.analyze = args.aplot

    if args.save is not None and not args.save:
        args.save = os.path.join('figures', args.set, args.test)

    model = Pipeline(args)

    print('[ Test Data ]')
    input_data = InputData(args.dir, args.set, args.test, args.file, args.values)
    t0 = time.time()
    data = model(input_data)
    info_dict['time'] = {'persist' :  time.time() - t0}
    model.plot(data)

    if args.analyze or args.som:
        predict = model.get_predict(n=args.nroc)
        if args.som:
            fsom = os.path.join(args.cache, 'som_%s-%s.pkl' % (args.set, args.test))
            if os.path.exists(fsom):
                print('[ loading %s' % fsom)
                with open(fsom, 'rb') as f:
                    predict['SOM'] = pkl.load(f)
                    predict['SOM'].n = args.nroc
                    predict['SOM'].streak = args.streak
                    predict['SOM'].lead = args.lead
                    data['SOM'] = input_data

        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.01)
        res = model.analyze(predict, data, ax)
        fig.suptitle('%s ROC' % input_data.title)
        ax.set_xlabel('False Positive')
        ax.set_ylabel('True Positive')
        ax.legend(loc='lower right')
        plt.tight_layout()
        stats = stat_str(res)
        print('[ Stats ]\n\t%s' % stats.replace('\n', '\n\t'))

    if args.save is not None:
        name = '%s%s' % (input_data.name, '' if _args['preset'] is None else '-preset%d' % _args['preset'])
        savefig('%s_roc' % name, args.save, fig)
        if args.analyze:
            fstats = os.path.join(args.save, '%s_stats.txt' % name)
            print(' - Writing %s' % fstats)
            with open(fstats, 'w') as f:
                f.write(stats)
            info_dict['stats'] = stat_dict(res)
            finfo = os.path.join(args.save, '%s_info.json' % name)
            print(' - Writing %s' % finfo)
            with open(finfo, 'w') as f:
                json.dump(info_dict, f, indent=4)

    if args.interact:
        interact = TPersInteract(data, args.plot, args.save)

    if args.show:
        plt.show(block=False)
        input('[ Exit ]')
