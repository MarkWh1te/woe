# -*- coding:utf-8 -*-
__author__ = 'slundberg'

# this creates an sklearn API style wrapper
from tqdm import tqdm
from .feature_process import proc_woe_continuous, proc_woe_discrete, woe_trans

class WOEEncoder():
    def __init__(self):
        pass
    
    def fit(self, X, y, continuous_features=None):
        """ Learn a WoE transform from a dataset and binary label.
        """
        
        # auto-detect continuous features
        if continuous_features is None:
            continuous_features = []
            for i in range(len(X.columns)):
                if str(X.dtypes[i]).startswith("float"):
                    continuous_features.append(X.columns[i])
        self.continuous_features = continuous_features
        
        # set the target in the joint dataset
        assert "target" not in X.columns, "'target' is a reserved name, and can't be a feature name!"
        joint_data = X.copy()
        joint_data["target"] = y
        
        # compute some parameters
        self.dataset_len = X.shape[0]
        self.min_sample = int(self.dataset_len * 0.05)
        self.global_bt = sum(y)
        self.global_gt = self.dataset_len - self.global_bt
            
        # build transformations
        self.rst = []
        for c in tqdm(X.columns):
            
            # continuous features
            if c in self.continuous_features:
                joint_data.loc[joint_data[c].isnull(), (c)] = -1 # fill null
                self.rst.append(proc_woe_continuous(
                    joint_data, c, self.global_bt,
                    self.global_gt, self.min_sample, alpha=0.05, silent=True
                ))
                
            # discrete features
            else:
                joint_data.loc[joint_data[c].isnull(), (c)] = 'missing' # fill null
                self.rst.append(proc_woe_discrete(
                    joint_data, c, self.global_bt, self.global_gt,
                    self.min_sample, alpha=0.05, silent=True
                ))
    
    def transform(self, X):
        """ Apply the learned WoE transform to a new dataset.
        """
        
        X_new = X.copy()
        for c in X.columns:
            if c in self.continuous_features:
                X_new.loc[X_new[c].isnull(), (c)] = -1 # fill null
            else:
                X_new.loc[X_new[c].isnull(), (c)] = 'missing' # fill null

        # training dataset WoE Transformation
        for r in self.rst:
            X_new[r.var_name] = woe_trans(X_new[r.var_name], r)
        
        return X_new

        