"""
Utils 

"""

import math


def float_to_int(x):
    # int becomes float when loaded by json
    if not isinstance(x, float):
        return x
    elif math.isnan(x) or math.isinf(x):
        return x
    else:
        if int(x) == x:
            return int(x)
        else:
            return x


from typing import Callable
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class CrossValid(object):
    def __init__(self, xx, yy, data_split='cv', n_fold=10, shuffle=True, random_state=None, is_clf=False):
        # Must instantiate with data and split methods.
        self.xx = np.asarray(xx)
        self.yy = np.asarray(yy)
        self.split = data_split
        self.n_fold = n_fold
        self.shuffle = shuffle
        self.seed = random_state
        self.is_clf = is_clf


    def _train(self, mdl, para):
        # only return loss value(s). 
        if isinstance(para['loss_func'], Callable):
            para['loss_func'] = [para['loss_func']]
        if self.split == 'cv' or self.split == 'cross_valid':
            assert self.n_fold > 1, "Num_fold must > 1 for cross validation."
            loss = self._cross_valid(mdl, para)
        elif self.split == 'random':
            loss = self._rand_split(mdl, para)
        return loss

    def _cross_valid(self, mdl, para):
        loss = np.zeros(len(para['loss_func']))
        if self.shuffle:
            kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_splits=self.n_fold, shuffle=False)
        
        for train_idx, test_idx in kf.split(self.xx):
            x_train = self.xx[train_idx]
            y_train = self.yy[train_idx]
            x_test = self.xx[test_idx]
            y_test = self.yy[test_idx]

            try:  # with or without eval_set
                mdl.fit(x_train, y_train,
                        eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                        **para['fit_params'])
            except AttributeError:
                mdl.fit(x_train, y_train)

            try:  # clf or reg
                pred = mdl.predict_proba(x_test)
            except AttributeError:
                pred = mdl.predict(x_test)

            if self.is_clf:
                loss_list = [f(y_test, pred[:, 1]) for f in para['loss_func']]
                loss += np.array(loss_list) / self.n_fold
            else:
                loss_list = [f(y_test, pred) for f in para['loss_func']]
                loss += np.array(loss_list) / self.n_fold
        if loss.size == 1:
            loss = float(loss)
        return loss

    def _rand_split(self, mdl, para):
        loss = 0
        # (by default 90-10% split. Edit later if anything else needed. )
        for seed_incre in range(self.n_fold):
            seed = self.seed + seed_incre
            x_train, x_test, y_train, y_test = train_test_split(self.xx, self.yy, test_size=0.1, random_state=seed)
            try:
                mdl.fit(x_train, y_train,
                        eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                        **para['fit_params'])
            except AttributeError:
                mdl.fit(x_train, y_train)

            pred = mdl.predict(x_test)
            if self.is_clf:
                loss_list = [f(y_test, pred[:, 1]) for f in para['loss_func']]
                loss += np.array(loss_list) / self.n_fold
            else:
                loss_list = [f(y_test, pred) for f in para['loss_func']]
                loss += np.array(loss_list) / self.n_fold
        if loss.size == 1:
            loss = float(loss)
        return loss
