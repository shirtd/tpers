# TPers: Exploring Persistent Homology for Time-series with Total Persistence

## Installation

Running

    pip install -r requirements.txt

might work.
The only dependencies other than the usual (numpy, scipy, sklearn, matplotlib, pandas, [tqdm](https://pypi.org/project/tqdm/)) are [ripser](https://pypi.org/project/ripser/) for fast persistence computations, [persim](https://pypi.org/project/persim/) for computing persistence images (also persistence landscapes and bottleneck distance), and [sklearn-som](https://pypi.org/project/sklearn-som/).

## System Specification

The dataset should be specified and included in `main.py`.
The UBL data is included (?), and specified in `ubl_data.py`.
At this time the user is required to specify

#### Available Data
* `AVAIL_DATA`: Dictionary of available data/test sets,
* `AVAIL_VALUES`: List of available values (features) for each data point/

#### Data Defaults
* `DIR`: Directory containing data,
* `DATASET`: Default dataset,
* `TESTSET`:Default test set,
* `LOGFILE`:Default test file,
* `VALUES`: Default values (features).

#### Program Defaults
* `LENGTH`: Transformation window length,
* `OVERLAP`: Transformation window overlap,
* `DIM`: Max persistence dimension,

#### Presets
* `PRESETS`: List of argument presets,
* `PRESET_DICT`: Data/test set presets (when `--preset` is passed without argument).

#### InputData class (todo)
Definition of an `InputData` object that extends `TimeSeriesData` and contains raw input data for a given data/test set.

## Usage

To do a simple viz of raw data from the default data/test set run

    ./main.py --plot input --show

To run and view (but not save) the default data/test set run

    ./main.py --preset --interact

Interaction may not work (I think you have to set your matplotlib backend to TkAgg, or something).
Try clicking on the TPers plot.
To replicate the results detailed in the report (saving to `figures/{DATASET}/{TESTSET}` call

  ./main.py --preset --som --plot input pre tpers --save --set {DATASET} --test {TESTSET}

For example,

    ./main.py --preset --som --plot input pre tpers --save --set SystemSLogs --test cpuhog

is the default behavior.
Running the bash script

    ./mkfigs.sh

will run presets on all data/test sets in the `data` directory, generating the figures included in the report (hopefully).

#### A Note

The --som flag attempts to load a `.pkl` file containing a pre-trained self-organizing map (SOM) for the specified data/test set.
I don't know if `.pkl` files will survive, new ones can be trained by running

    ./mksom.py --set {DATASET} --test {TESTSET}

The script trains a model using the training data (`tr.log`) file for the specified data/test set, tests it on the corresponding test data set (`te.log`), and plots the results against the existing model in `cache/som_{DATASET}-{TESTSET}.pkl`, if available.
Pass anything (other than `n` or `no`) to override the existing model.
If no model exists it just saves it.

### Arguments

There are a number of module-specific arguments that can be tested with the default behavior on the data/test set.
A few presets are provided if you want to ignore all the options and run what works, but theres a lot to play with here.

#### Processing
Passing

    --pre A B C

will run operations A, B, and C in order.
Default behavior (`--preset 0`) is

    --pre scale pca=4


Available operations are as follows:
* `scale`: Min-Max scaling on features independently
* `scale=min`: Min scaling only on features independently,
* `scale=all`: Min-Max scaling on the whole dataset (min and max of all entries),
* `scale=min,all`: Min scaling on the whole dataset (min of all entries),
* `diff`: apply difference transform (discrete derivative) to each feature independently,
* `power`: apply power transform to each feature independently,
* `detrend`: detrend each feature independently,
* `ma`: apply moving average to each feature independently. `ma=w` convolves with 2*2+1 point window,
* `pca`: PCA transform. `pca=n` reduces to `n` principal compnents.

The same operations can be applied to the total persistence curve by passing them as arguments to `--post`.
Note that this has no effect on `kmeans` prediction on `persistence`, but does affect prediction via `threshold` on `tpers` (see below).

#### Window

* `--length {n}`: set the window length to `n`,
* `--overlap {w}`: set the window overlap to `w`,

#### Transform

If none of `--period`, `--fft`, or `--torus` are passed persistence will be run on raw windowed data.

* `--period {t}`: Period of cycle in the complex plane. Equal to the length of the window if passed without argument. If passed with argument and `--torus`, the complex data will be period specified will be provided to the torus transform (untested).
* `--fft`: Run Fourier transform on each frame with blackmann window. If passed with `--torus` the complex frequency domain output of the Fourier transform will be passed as phase and amplitude to the torus transform (super cool, kinda works... sometimes).
* `--torus`: Apply torus transform (__warning__ don't do torus transform on more than two values/features without setting `--nperm` less than ~50, usually 20 works).
* `--exp {p}`: Apply `p` as an exponent to all data, for fun. Executed before all other transforms.
* `--abs`: Take the absolute value of all data, for science. Executed before all other transforms.

#### Persistence

The persistence computation is carried out using Persistence is carried out using [ripser](https://pypi.org/project/ripser/).
The following arguments are passed to `ripser` for each frame.

* `--dim {d}`: Maximum rips/persistence dimension,
* `--thresh {t}`: Maximum distance to compute in the Rips complex,
* `--nperm {n}`: Number of greedy permutations. Probably safe to set to 20 for all applications. __Huge__ speedup., type=int, help='greedy permutations')
* `--metric {euclidean, manhattan, cosine}`: Metric for Rips computation. Default: `euclidean`.


#### Total Persistence (almost depricated)
* `--invert {d,...}`: Invert provided dimensions (multiply by -1. Inverted in the sum),
* `--entropy`: Compute [persistent entropy](https://persim.scikit-tda.org/en/latest/notebooks/Persistence%20barcode%20measure.html), for science (and fun),
* `--average`: Compute average total persistence in each dimension,
* `--pmin {m}`: Only include diagram features with total persistence at least `m`.

#### Program Arguments

Ugh. Just run

    ./main.py -h

It's the same thing.

* `--data`: Print available data/test sets,
* `--dir {DATA_DIR}`: Data directory. Default: `./data`,
* `--set {DATASET}`: Dataset. Default: `SystemSLogs.log`,
* `--test {TESTSET}`: Test set. Default: `cpuhog.log`,
* `--file {LOGFILE}`: File name. Default: `te.log`,
* `--cache {CACHE}`: Cache directory. Currently only for SOM models. Default `./cache`,
* `--preset {?i}`: Preset to run. Default for dataset provided if passed without argument,
* `--show-presets`: Print available presets,
* `--values {COLUMN_NAME1 COLUMN_NAME2 ...}`: Data values (features) to use,
* `--plot {input, pre, window, transform, persistence, tpers', post}`: Modules to plot,
* `--nroc {n}`: Number of points on ROC curve,
* `--frame {f}`: Frame to plot (if `window` or `transform` passed to plot). For saving purposes. __Warning__ untested,
* `--show`: Show plot, otherwise it will just quit (if neither `--save` nor `--interact` is passed),
* `--save`: Save plots to directory. Default: `./figures/{DATASET}/{TESTSET}`)
* `--predict {threshold,SOM,kmeans,minkmeans,maxkmeans}`: Don't pass `SOM`. min/max kmeans are dumb. Threshold is just prediction by thresholding each feature.
* `--analyze {input, pre, persistence, tpers, post}`: Modules to analyze. Pass `--analyze {MODULE}={PREDICT}` to override prediction type passed by `--predict` for a given module,
* `--aplot {input, pre, persistence, tpers, post}`: Analyze and plot module,
* `--interact`: Interact with terminal module. Useful for viewing data from individual frames for "framed" modules passed to `--plot` such as `transform`, `window`, and `persistence`. Default behavior is to plot persistence diagram. Very useful with `transform` passed to `--plot`.
* `--som`: Compare with saved SOM model (in cache),
* `--lead {W}`: SOM predict lead (anomaly pending) time. Default: 10,
* `--streak {s}`: Streak of anomalies required to raise SOM alarm. Default: 3

# GLHF
