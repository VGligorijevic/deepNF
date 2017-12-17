import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
from keras.models import Model, load_model
from sklearn.preprocessing import minmax_scale
from validation import cross_validation, temporal_holdout
from deepNF import build_MDA, build_AE
from keras.callbacks import EarlyStopping
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import sys


def read_params(fname):
    params = {}
    fR = open(fname, 'r')
    for line in fR:
        print line.strip()
        key, val = line.strip().split('=')
        key = str(key)
        val = str(val)
        if key == 'select_arch':
            params[key] = map(int, val.strip('[]').split(','))
        else:
            params[key] = str(val)
    print "###############################################################"
    print
    print
    fR.close()

    return params


# Main code starts here
params = read_params(sys.argv[1])

mark = params['mark']   # {--all or --no}
org = params['org']   # {yeast or human}
valid_type = params['valid_type']  # {cv or th}
model_type = params['model_type']  # {mda}
ofile_keywords = params['ofile_keywords']  # {example: 'final_res'}

models_path = params['models_path']  # directory with models
results_path = params['results_path']  # directotry with results
select_arch = params['select_arch']  # a number 1-6 (see below)
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])
n_trials = int(params['n_trials'])  # number of cv trials
K = params['K']  # number of prop
alpha = params['alpha']  # propagation parameter


# all possible combinations for architectures
if model_type == 'mda':
    if org == 'yeast':
        arch = {1: [600],
                2: [6*2000, 600, 6*2000],
                3: [6*2000, 6*1000, 600, 6*1000, 6*2000],
                4: [6*2000, 6*1000, 6*500, 600, 6*500, 6*1000, 6*2000],
                5: [6*2000, 6*1200, 6*800, 600, 6*800, 6*1200, 6*2000],
                6: [6*2000, 6*1200, 6*800, 6*400, 600, 6*400, 6*800, 6*1200, 6*2000]
                }
    elif org == 'human':
        arch = {1: [1200],
                2: [6*2500, 1200, 6*2500],
                3: [6*2500, 6*1500, 1200, 6*1500, 6*2500],
                4: [6*2500, 6*1500, 6*1000, 1200, 6*1000, 6*1500, 6*2500],
                5: [6*2500, 6*2000, 6*1000, 1200, 6*1000, 6*2000, 6*2500],
                6: [6*2500, 6*2000, 6*1500, 6*1000, 1200, 6*1000, 6*1500, 6*2000, 6*2500]
                }
    else:
        print "### Wrong organism!"
elif model_type == 'ae':
    if org == 'yeast':
        arch = {1: [600],
                2: [2000, 600, 2000],
                3: [2000, 1000, 600, 1000, 2000],
                4: [2000, 1000, 800, 600, 800, 1000, 2000]
                }
    elif org == 'human':
        arch = {1: [1200],
                2: [2500, 1200, 2500],
                3: [2500, 1500, 1200, 1500, 2500],
                4: [2500, 1500, 1000, 1200, 1000, 1500, 2500]
                }
    else:
        print "### Wrong organism!"
else:
    print "### Wrong model type!"


arch = dict((key, a) for key, a in arch.iteritems() if key in select_arch)

# measures
measures = ['m-aupr_avg', 'm-aupr_std', 'M-aupr_avg', 'M-aupr_std',
            'F1_avg', 'F1_std', 'acc_avg', 'acc_std']

# annotation parameters
if org == 'yeast':
    if valid_type == 'cv':
        annot = ['level1', 'level2', 'level3']
    elif valid_type == 'th':
        annot = ['MF', 'BP', 'CC']
    else:
        print "### Wrong valid_type!"
elif org == 'human':
    if valid_type == 'cv':
        annot = ['bp_1', 'bp_2', 'bp_3',
                 'mf_1', 'mf_2', 'mf_3',
                 'cc_1', 'cc_2', 'cc_3']
    elif valid_type == 'th':
        annot = ['MF', 'BP', 'CC']
    else:
        print "### Wrong valid_type!"
else:
    print "### Wrong organism!"


# load GO annotations
if valid_type == 'cv':
    GO = sio.loadmat('./annotations/' + org + '_annotations.mat')
elif valid_type == 'th':
    Annot = sio.loadmat('./annotations/' + org + '_annot_temporal_holdout.mat', squeeze_me=True)
    fRead = open('./annotations/th_trials_' + org + '.pckl', 'rb')
    Bootstrap = pickle.load(fRead)
    fRead.close()
else:
    print "### Wrong validation type!"


# load networks
Nets = []
input_dims = []
for i in range(1, 7):
    print "### [%d] Loading network..." % (i)
    N = sio.loadmat('./annotations/' + org + '_net_' + str(i) + '_K' + K + '_alpha' + alpha + '.mat', squeeze_me=True)
    Net = N['Net'].todense()
    Nets.append(minmax_scale(Net))
    input_dims.append(Net.shape[1])


# Training MDA/AE
model_names = []
if model_type == 'mda':
    for a in arch:
        print "### [Model] Running for architecture: ", arch[a]
        model_name = org + '_' + model_type.upper() + '_arch_' + str(a) + '_' + ofile_keywords + '.h5'
        if mark == '--all':
            model = build_MDA(input_dims, arch[a])
            history = model.fit(Nets, Nets, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1,
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2)])
            mid_model = Model(inputs=model.input,
                              outputs=model.get_layer('middle_layer').output)
            mid_model.save(models_path + model_name)

            # Export figure: loss vs epochs (history)
            plt.plot(history.history['loss'], 'o-')
            plt.plot(history.history['val_loss'], 'o-')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(models_path + model_name + '_loss.png', bbox_inches='tight')
        model_names.append(model_name)
elif model_type == 'ae':
    for a in arch:
        print "### [Model] Running for architecture: ", arch[a]
        for i in range(0, len(Nets)):
            print "### [Model 1] Running for network: ", i
            model_name = org + '_net_' + str(i) + '_AE_arch_' + str(a) + '_' + ofile_keywords + '.h5'
            if mark == '--all':
                model = build_AE(input_dims[i], arch[a])
                history = model.fit(Nets[i], Nets[i], epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1,
                                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2)])
                model.save(models_path + model_name)

                # Export figure: loss vs epochs (history)
                plt.plot(history.history['loss'], 'o-')
                plt.plot(history.history['val_loss'], 'o-')
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.savefig(models_path + model_name + '_loss.png', bbox_inches='tight')

else:
    print "### Wrong model type!"

filename = results_path + 'deepNF_' + model_type + '_' + ofile_keywords + '_' + valid_type + '_performance_' + org + '.txt'
fout = open(filename, 'w')

# Training SVM
if model_type == 'mda':
    for model_name in model_names:
        print "### Running for: %s" % (model_name)
        fout.write(model_name)
        fout.write('\n')
        my_file = Path(models_path + model_name)
        if my_file.exists():
            mid_model = load_model(models_path + model_name)
        else:
            print "### Model % s does not exist. Use 'mark=--all' to generate models." % (model_name)
            break
        mid_model = load_model(models_path + model_name)
        features = mid_model.predict(Nets)
        features = minmax_scale(features)
        sio.savemat(models_path + model_name + '_features.mat', {'features': features})
        for level in annot:
            print "### Running for level: %s" % (level)
            if valid_type == 'cv':
                perf = cross_validation(features, GO[level],
                                        n_trials=n_trials,
                                        fname=results_path + model_name + '_' + level + '_' + valid_type + '_performance_trials.txt')
                fout.write('%s' % (level))
                for m in measures:
                    fout.write(' %0.5f' % (perf[m]))
                fout.write('\n')
            else:
                perf = temporal_holdout(features,
                                        Annot['GO'][level].tolist(),
                                        Annot['indx'][level].tolist(),
                                        Bootstrap[level],
                                        results_path + model_name + '_' + level + '_th_performance_trials.txt',
                                        goterms=Annot['labels'][level].tolist(),
                                        go_fname=results_path + model_name + '_' + level + '_th_performance_GOterms.txt')
                fout.write('%s ' % (level))
                for m in measures:
                    fout.write('%0.5f ' % (perf[m]))
                fout.write('\n')
elif model_type == 'ae':
    for a in arch:
        print "### [Model] Running for architecture: ", arch[a]
        for i in range(0, len(Nets)):
            print "### [Model] Running for network: ", i
            model_name = org + '_net_' + str(i) + '_AE_arch_' + str(a) + '_' + ofile_keywords + '.h5'
            fout.write(model_name)
            fout.write('\n')
            model = load_model(models_path + model_name)
            mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)
            features = mid_model.predict(Nets[i])
            features = minmax_scale(features)
            for level in annot:
                print "### Running for level: %s" % (level)
                if valid_type == 'cv':
                    perf = cross_validation(features, GO[level],
                                            n_trials=n_trials,
                                            fname=results_path + model_name + '_' + level + '_' + valid_type + '_performance_trials.txt')
                    fout.write('%s ' % (level))
                    for m in measures:
                        fout.write('%0.5f ' % (perf[m]))
                    fout.write('\n')
                else:
                    perf = temporal_holdout(features,
                                            Annot['GO'][level].tolist(),
                                            Annot['indx'][level].tolist(),
                                            Bootstrap[level],
                                            results_path + model_name + '_' + level + '_th_performance_trials.txt',
                                            goterms=Annot['labels'][level].tolist(),
                                            go_fname=results_path + model_name + '_' + level + '_th_performance_GOterms.txt')
                    fout.write('%s ' % (level))
                    for m in measures:
                        fout.write('%0.5f ' % (perf[m]))
                    fout.write('\n')
else:
    print "### Wrong model type!"

fout.close()
