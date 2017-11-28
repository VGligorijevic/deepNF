import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold
from scipy.stats import sem as std
from sklearn.metrics.pairwise import rbf_kernel


def real_AUPR(label, score):
    """Computing real AUPR . By Vlad and Meet"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0

    return pr, f


def ml_split(y):
    """Split annotations"""
    kf = KFold(n_splits=5, shuffle=True)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def evaluate_performance(y_test, y_score, y_pred):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    perf["M-aupr"] /= n

    # Compute micro-averaged AUPR
    perf["m-aupr"], _ = real_AUPR(y_test, y_score)

    # Computes accuracy
    perf['acc'] = accuracy_score(y_test, y_pred)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')

    return perf


def temporal_holdout(X, y, indx, bootstrap, fname, goterms=None, go_fname=None):
    """Perform temporal holdout validation"""

    X_train = X[indx['train'].tolist()]
    X_test = X[indx['test'].tolist()]
    X_valid = X[indx['valid'].tolist()]
    y_train = y['train'].tolist()
    y_test = y['test'].tolist()
    y_valid = y['valid'].tolist()
    if goterms is not None:
        goterms = goterms['terms'].tolist()

    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    gamma_range = 10.**np.arange(-3, 1)

    # pre-generating kernels
    print "### Pregenerating kernels..."
    K_rbf_train = {}
    K_rbf_test = {}
    K_rbf_valid = {}
    for gamma in gamma_range:
        K_rbf_train[gamma] = rbf_kernel(X_train, gamma=gamma)
        K_rbf_test[gamma] = rbf_kernel(X_test, X_train, gamma=gamma)
        K_rbf_valid[gamma] = rbf_kernel(X_valid, X_train, gamma=gamma)
    print "### Done."
    print "Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0])

    # parameter fitting
    C_opt = None
    gamma_opt = None
    max_aupr = 0
    for C in C_range:
        for gamma in gamma_range:
            # Multi-label classification
            clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                              probability=False), n_jobs=-1)
            clf.fit(K_rbf_train[gamma], y_train)
            y_score_valid = clf.decision_function(K_rbf_valid[gamma])
            y_pred_valid = clf.predict(K_rbf_valid[gamma])
            perf = evaluate_performance(y_valid,
                                        y_score_valid,
                                        y_pred_valid)
            micro_aupr = perf['m-aupr']
            print "### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, micro_aupr)
            if micro_aupr > max_aupr:
                C_opt = C
                gamma_opt = gamma
                max_aupr = micro_aupr
    print "### Optimal parameters: "
    print "C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt)
    print "### Train dataset: AUPR = %0.3f" % (max_aupr)
    print
    print "### Computing performance on test dataset..."
    clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                      probability=False), n_jobs=-1)
    clf.fit(K_rbf_train[gamma_opt], y_train)

    # Compute performance on test set
    y_score = clf.decision_function(K_rbf_test[gamma_opt])
    y_pred = clf.predict(K_rbf_test[gamma_opt])

    # performance measures for bootstrapping
    perf = dict()
    pr_micro = []
    pr_macro = []
    fmax = []
    acc = []

    # individual goterms
    pr_goterms = {}
    for i in range(0, len(goterms)):
        pr_goterms[goterms[i]] = []

    for ind in bootstrap:
        perf_ind = evaluate_performance(y_test[ind],
                                        y_score[ind],
                                        y_pred[ind])
        pr_micro.append(perf_ind['m-aupr'])
        pr_macro.append(perf_ind['M-aupr'])
        fmax.append(perf_ind['F1'])
        acc.append(perf_ind['acc'])
        for i in range(0, len(goterms)):
            pr_goterms[goterms[i]].append(perf_ind[i])

    perf['m-aupr_avg'] = np.mean(pr_micro)
    perf['m-aupr_std'] = std(pr_micro)
    perf['M-aupr_avg'] = np.mean(pr_macro)
    perf['M-aupr_std'] = std(pr_macro)
    perf['F1_avg'] = np.mean(fmax)
    perf['F1_std'] = std(fmax)
    perf['acc_avg'] = np.mean(acc)
    perf['acc_std'] = std(acc)

    # trials
    fout = open(fname, 'w')
    print>>fout, "aupr[micro], aupr[macro], F_max, accuracy"
    for it in range(0, len(bootstrap)):
        print>>fout, pr_micro[it], pr_macro[it], fmax[it], acc[it]
    fout.close()

    # write performance on individual GO terms
    if go_fname is not None:
        fout = open(go_fname, 'wb')
        print>>fout, "GO_id, AUPRs"
        for i in range(0, len(goterms)):
            print>>fout, goterms[i], sum(y_train[:, i])/float(y_train.shape[0]),
            for pr in pr_goterms[goterms[i]]:
                print>>fout, pr,
            print>>fout
        fout.close()

    return perf


def cross_validation(X, y, n_trials=5, trial_splits=None, fname=None):
    """Perform model selection via 5-fold cross validation"""
    # filter samples with no annotations
    del_rid = np.where(y.sum(axis=1) == 0)[0]
    y = np.delete(y, del_rid, axis=0)
    X = np.delete(X, del_rid, axis=0)

    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    gamma_range = 10.**np.arange(-3, 1)

    # pre-generating kernels
    print "### Pregenerating kernels..."
    K_rbf = {}
    for gamma in gamma_range:
        K_rbf[gamma] = rbf_kernel(X, gamma=gamma)
    print "### Done."

    # performance measures
    perf = dict()
    pr_micro = []
    pr_macro = []
    fmax = []
    acc = []

    if trial_splits is None:
        # shuffle and split training and test sets
        trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
        ss = trials.split(X)
        trial_splits = []
        for train_idx, test_idx in ss:
            trial_splits.append((train_idx, test_idx))

    it = 0
    for jj in range(0, n_trials):
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
        it += 1
        y_train = y[train_idx]
        y_test = y[test_idx]
        print "### [Trial %d] Perfom cross validation...." % (it)
        print "Train samples=%d; #Test samples=%d" % (y_train.shape[0],
                                                      y_test.shape[0])
        # setup for neasted cross-validation
        splits = ml_split(y_train)

        # parameter fitting
        C_opt = None
        gamma_opt = None
        max_aupr = 0
        for C in C_range:
            for gamma in gamma_range:
                # Multi-label classification
                cv_results = []
                for train, valid in splits:
                    clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                                      probability=False), n_jobs=-1)
                    K_train = K_rbf[gamma][train_idx[train], :][:, train_idx[train]]
                    K_valid = K_rbf[gamma][train_idx[valid], :][:, train_idx[train]]
                    y_train_t = y_train[train]
                    y_train_v = y_train[valid]
                    y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                    y_pred_valid = np.zeros_like(y_train_v)
                    idx = np.where(y_train_t.sum(axis=0) > 0)[0]
                    clf.fit(K_train, y_train_t[:, idx])
                    y_score_valid[:, idx] = clf.decision_function(K_valid)
                    y_pred_valid[:, idx] = clf.predict(K_valid)
                    perf_cv = evaluate_performance(y_train_v,
                                                   y_score_valid,
                                                   y_pred_valid)
                    cv_results.append(perf_cv['m-aupr'])
                cv_aupr = np.median(cv_results)
                print "### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, cv_aupr)
                if cv_aupr > max_aupr:
                    C_opt = C
                    gamma_opt = gamma
                    max_aupr = cv_aupr
        print "### Optimal parameters: "
        print "C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt)
        print "### Train dataset: AUPR = %0.3f" % (max_aupr)
        print
        print "### Using full training data..."
        clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                          probability=False), n_jobs=-1)
        y_score = np.zeros(y_test.shape, dtype=float)
        y_pred = np.zeros_like(y_test)
        idx = np.where(y_train.sum(axis=0) > 0)[0]
        clf.fit(K_rbf[gamma_opt][train_idx, :][:, train_idx], y_train[:, idx])

        # Compute performance on test set
        y_score[:, idx] = clf.decision_function(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        y_pred[:, idx] = clf.predict(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        perf_trial = evaluate_performance(y_test, y_score, y_pred)
        pr_micro.append(perf_trial['m-aupr'])
        pr_macro.append(perf_trial['M-aupr'])
        fmax.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print "### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['m-aupr'], perf_trial['M-aupr'], perf_trial['F1'], perf_trial['acc'])
        print
        print
    perf['m-aupr_avg'] = np.mean(pr_micro)
    perf['m-aupr_std'] = std(pr_micro)
    perf['M-aupr_avg'] = np.mean(pr_macro)
    perf['M-aupr_std'] = std(pr_macro)
    perf['F1_avg'] = np.mean(fmax)
    perf['F1_std'] = std(fmax)
    perf['acc_avg'] = np.mean(acc)
    perf['acc_std'] = std(acc)

    if fname is not None:
        fout = open(fname, 'w')
        print>>fout, "aupr[micro], aupr[macro], F_max, accuracy"
        for ii in range(0, n_trials):
            print>>fout, pr_micro[ii], pr_macro[ii], fmax[ii], acc[ii]
        fout.close()

    return perf
