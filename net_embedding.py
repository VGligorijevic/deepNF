from preprocessing import load_networks, RWR, PPMI_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from deepNF import build_MDA, build_AE

from keras.models import Model
from keras.callbacks import EarlyStopping

import numpy as np
import argparse
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def build_model(X, input_dims, arch, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=64):
    if mtype == 'mda':
        model = build_MDA(input_dims, arch)
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
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', type=str, default='mda', help="Deep autoencoder model. Possible: 'mda': for multiple nets;  'ae': for single net.")
    parser.add_argument('--models_path', type=str, default='./test_models/', help="Saving models.")
    parser.add_argument('--results_path', type=str, default='./test_results/', help="Saving results.")
    parser.add_argument('--hidden_dims', type=int, default=[1000, 500, 1000], nargs='+', help="Model architecture configuration.")
    parser.add_argument('--nets', type=str, default=['example_net_1.txt', 'example_net_2.txt'], nargs='+', help="Network files (edgelist format: i, j, w_ij).")
    parser.add_argument('--epochs', type=int, default=80, help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--noise_factor', type=float, default=0.5, help="Noise factor for denoising AE/MDA.")

    args = parser.parse_args()
    print (args)

    model_type = args.model_type
    models_path = args.models_path
    results_path = args.results_path
    hidden_dims = args.hidden_dims
    nets = args.nets
    epochs = args.epochs
    batch_size = args.batch_size
    nf = args.noise_factor

    # construct PPMI matrices
    Nets = load_networks(nets)
    input_dims = []
    for i in range(len(Nets)):
        #  Random Walk wih Restarts
        print ("[%d] Computing RWR profile..." % (i))
        Nets[i] = RWR(Nets[i])
        #  PPMI matrix
        print ("[%d] Computing PPMI matrix..." % (i))
        Nets[i] = minmax_scale(PPMI_matrix(Nets[i]))
        print ("Net %d, NNZ=%d \n" % (i, np.count_nonzero(Nets[i])))
        input_dims.append(Nets[i].shape[1])

    # Training MDA/AE
    print ("### [%s] Running for architecture: %s" % (model_type, str(hidden_dims)))
    model_name = 'deepNF_' + model_type.upper() + '_arch_' + '-'.join(list(map(str, hidden_dims))) + '.h5'
    mid_model, history = build_model(Nets, input_dims, hidden_dims, nf, 1.0, model_type, epochs, batch_size)

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
    plt.savefig(models_path + model_name.split('.')[0] + '_loss.png', bbox_inches='tight')

    # Export features (node embeddings)
    features = mid_model.predict(Nets)
    features = minmax_scale(features)
    pickle.dump(features, open(results_path + model_name.split('.')[0] + '_features.pckl', 'wb'))
