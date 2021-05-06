#!/usr/bin/env /Users/kirk/anaconda3/envs/py38/bin/python

from tpers.args import get_parser
from tpers.analyze import *
from tpers.util import *

import pickle as pkl
import os, sys

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model = Pipeline(args)

    print('[ Test Data ]')
    input_data = InputData(args.dir, args.set, args.test, args.file, args.values)

    print('[ Train Data ]')
    train_input_data = InputData(args.dir, args.set, args.test, 'tr.log', args.values)
    predict = {'input' : SOMPredict(train_input_data, n=args.nroc)}
    data = {'input' : input_data}

    fsom = os.path.join(args.cache, 'som_%s-%s.pkl' % (args.set, args.test))
    if os.path.exists(fsom):
        print('[ loading %s' % fsom)
        with open(fsom, 'rb') as f:
            predict['saved'] = pkl.load(f)
            predict['saved'].n = args.nroc
            predict['saved'].streak = args.streak
            predict['saved'].lead = args.lead

            data['saved'] = InputData(args.dir, args.set, args.test, args.file, args.values)
            data['saved'].format_values['SOM']['color'] = '#332288'
            data['saved'].prefix = 'Saved'


    fig, ax = plt.subplots(1,1,figsize=(8,8))
    res = model.analyze(predict, data, ax)
    fig.suptitle('%s ROC' % input_data.title, fontsize=8)
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show(block=False)
    stats = stat_str(res)
    print('[ Stats ]\n\t%s' % stats.replace('\n', '\n\t'))

    if not (os.path.exists(fsom) and input('[ Save? ') in {'', 'no', 'n'}):
        if not os.path.exists(args.cache):
            os.mkdir(args.cache)
        print('[ writing %s' % fsom)
        with open(fsom, 'wb') as f:
            pkl.dump(predict['input'], f)
    else:
        input('[ Exit ]')
