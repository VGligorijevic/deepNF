from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, scale
from validation import cross_validation, temporal_holdout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from deepNF import build_MDA, build_AE, build_MDA2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as Path
import scipy.io as sio
import numpy as np
import pickle
import sys


def read_params(fname):
    params = {}
    fR = open(fname, 'r')
    for line in fR:
        print (line.strip())
        key, val = line.strip().split('=')
        key = str(key.strip())
        val = str(val.strip())
        if key == 'select_arch' or key == 'select_nets':
            params[key] = map(int, val.strip('[]').split(','))
        else:
            params[key] = str(val)
    print ("###############################################################")
    print
    print
    fR.close()

    return params


def build_model(X, input_dims, arch, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=64):
    if mtype == 'mda':
        model = build_MDA(input_dims, arch)
        # model = build_MDA2(input_dims, arch)
    elif mtype == 'ae':
        model = build_AE(input_dims[0], arch)
    else:
        print ("### Wrong model.")
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test_noisy, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)])
    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

    return mid_model, history


# ### Main code starts here
params = read_params(sys.argv[1])


org = params['org']   # {yeast or human}
valid_type = params['valid_type']  # {cv or th}
model_type = params['model_type']  # {mda or ae}
ofile_keywords = params['ofile_keywords']  # {example: 'final_res'}

models_path = params['models_path']  # directory with models
results_path = params['results_path']  # directotry with results
select_arch = params['select_arch']  # a number 1-6 (see below)
select_nets = params['select_nets']  # a number 1-6 (see below)
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])
nf = float(params['noise_factor'])  # nf > 0 for denoising AE/MDA
n_trials = int(params['n_trials'])  # number of cv trials
K = params['K']  # number of propagations
alpha = params['alpha']  # propagation parameter


# all possible combinations for architectures
arch = {}
arch['mda'] = {}
arch['mda']['yeast'] = {}
arch['mda']['human'] = {}

arch['mda']['yeast'] = {1: [600],
                        2: [6*2000, 600, 6*2000],
                        3: [6*2000, 6*1000, 600, 6*1000, 6*2000],
                        4: [6*2000, 6*1000, 6*500, 600, 6*500, 6*1000, 6*2000],
                        5: [6*2000, 6*1200, 6*800, 600, 6*800, 6*1200, 6*2000],
                        6: [6*2000, 6*1200, 6*800, 6*400, 600, 6*400, 6*800, 6*1200, 6*2000]
                        }

arch['mda']['human'] = {1: [1200],
                        2: [6*2500, 1200, 6*2500],
                        3: [6*2500, 6*1500, 1200, 6*1500, 6*2500],
                        4: [6*2500, 6*1500, 6*1000, 1200, 6*1000, 6*1500, 6*2500],
                        5: [6*2500, 6*2000, 6*1000, 1200, 6*1000, 6*2000, 6*2500],
                        6: [6*2500, 6*2000, 6*1500, 6*1000, 1200, 6*1000, 6*1500, 6*2000, 6*2500]
                        }
arch['ae'] = {}
arch['ae']['yeast'] = {}
arch['ae']['human'] = {}

arch['ae']['yeast'] = {1: [600],
                       2: [2000, 600, 2000],
                       3: [2000, 1000, 600, 1000, 2000],
                       4: [2000, 1000, 800, 600, 800, 1000, 2000]
                       }

arch['ae']['human'] = {1: [1200],
                       2: [2500, 1200, 2500],
                       3: [2500, 1500, 1200, 1500, 2500],
                       4: [2500, 1500, 1000, 1200, 1000, 1500, 2500]
                       }

# annotation parameters
annot = {}
annot['yeast'] = {}
annot['human'] = {}
annot['yeast']['cv'] = ['level1', 'level2', 'level3']
annot['yeast']['th'] = ['MF', 'BP', 'CC']
annot['human']['th'] = ['MF', 'BP', 'CC']
annot['human']['cv'] = ['bp_1', 'bp_2', 'bp_3',
                        'mf_1', 'mf_2', 'mf_3',
                        'cc_1', 'cc_2', 'cc_3']

annot = annot[org][valid_type]

# load GO annotations
if valid_type == 'cv':
    GO = sio.loadmat('./annotations/' + org + '_annotations.mat')
elif valid_type == 'th':
    Annot = sio.loadmat('./annotations/' + org + '_annot_temporal_holdout.mat', squeeze_me=True)
    # Bootstrap = pickle.load(open('./annotations/th_trials_' + org + '.pckl', 'rb'))
else:
    print ("### Wrong validation type!")


# load PPMI matrices
Nets = []
input_dims = []
for i in select_nets:
    print ("### [%d] Loading network..." % (i))
    N = sio.loadmat('./annotations/' + org + '_net_' + str(i) + '_K' + K + '_alpha' + alpha + '.mat', squeeze_me=True)
    Net = N['Net'].todense()
    print ("Net %d, NNZ=%d \n" % (i, np.count_nonzero(Net)))
    Nets.append(minmax_scale(Net))
    input_dims.append(Net.shape[1])


# Training MDA/AE
model_names = []
for a in select_arch:
    print ("### [%s] Running for architecture: %s" % (model_type, str(arch[model_type][org][a])))
    model_name = org + '_' + 'deepNF_' + model_type.upper() + '_arch_' + str(a) + '_' + ofile_keywords + '.h5'
    if not Path.isfile(models_path + model_name):
        mid_model, history = build_model(Nets, input_dims, arch[model_type][org][a], nf, 1.0, model_type, epochs, batch_size)
        # save model
        mid_model.save(models_path + model_name)
        with open(models_path + model_name.split('.')[0] + '_history.pckl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # Export figure: loss vs epochs (history)
        plt.figure()
        plt.plot(history.history['loss'], '.-')
        plt.plot(history.history['val_loss'], '.-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(models_path + model_name + '_loss.png', bbox_inches='tight')
    model_names.append(model_name)


# Training SVM
for model_name in model_names:
    print ("### Running for: %s" % (model_name))
    fout = open(results_path + model_name.split('.')[0] + '_' + valid_type + '.txt', 'w')
    fout.write('### %s\n' % (model_name))
    fout.write('\n')
    if Path.isfile(models_path + model_name):
        mid_model = load_model(models_path + model_name)
    else:
        print ("### Model % s does not exist. Check the 'models_path' directory.\n" % (model_name))
        break
    mid_model = load_model(models_path + model_name)
    features = mid_model.predict(Nets)
    features = minmax_scale(features)
    sio.savemat(models_path + model_name.split('.')[0] + '_features.mat', {'features': features})
    for level in annot:
        print ("### Running for level: %s" % (level))
        if valid_type == 'cv':
            perf = cross_validation(features, GO[level],
                                    n_trials=n_trials)
        else:
            perf = temporal_holdout(features,
                                    Annot['GO'][level].tolist(),
                                    Annot['indx'][level].tolist(),
                                    Annot['labels'][level].tolist())
            fout.write('### %s goterms:\n' % (level))
            fout.write('GO_id, AUPRs\n')
            for goid in perf['pr_goterms']:
                fout.write('%s' % (goid))
                for pr in perf['pr_goterms'][goid]:
                    fout.write(' %0.5f' % (pr))
                fout.write('\n')
            fout.write('\n')
        fout.write('### %s trials:\n' % (level))
        fout.write('aupr[micro], aupr[macro], F_max, accuracy\n')
        avg_micro = 0.0
        for ii in range(0, len(perf['fmax'])):
            fout.write('%0.5f %0.5f %0.5f %0.5f\n' % (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['fmax'][ii], perf['acc'][ii]))
            avg_micro += perf['pr_micro'][ii]
        fout.write('\n')
        avg_micro /= len(perf['fmax'])
        print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
        print
    fout.close()
