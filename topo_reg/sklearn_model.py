
"""
Return sklearn models initialized with given params.

"""


import importlib
from typing_extensions import Literal
import sklearn
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

def get_arbitary_model(model_module_name: str, model_params: dict) -> sklearn.base.BaseEstimator:
    """
    Returns arbitary scikit-learn model obj, constructed with model_params.
    Must use the full name. 
    Example:
        mdl3 = get_arbitary_model("RandomForestClassifier", "sklearn.ensemble")
    """
    import_module, model_name = model_module_name.rsplit(".", 1)
    model_class = getattr(importlib.import_module(import_module), model_name)
    model = model_class(**model_params)  # Instantiates the model
    return model


def get_model(model_name: Literal['rf_reg', 'rf_clf', 'xgb_reg', 'xgb_clf', 'lgb_ref', 'lgb_clf', 
                                    'svm_reg', 'svm_clf', 'ridge', 'lin_reg', 'lr', 'gpr',
                                    ], 
            model_params: dict) -> sklearn.base.BaseEstimator:
    f = getattr(models, model_name)  # construction func.
    mdl = f(model_params)  # not ** params
    return mdl


class models():
    """Provide construction functions by names."""
    def __init__(self) -> None:
        pass

    def rf_reg(para):
        mdl = RFR(**para)
        return mdl

    def rf_clf(para):
        mdl = RFC(**para)
        return mdl

    def xgb_reg(para):
        mdl = xgb.XGBRegressor(**para)
        return mdl

    def xgb_clf(para):
        mdl = xgb.XGBClassifier(**para)
        return mdl

    def lgb_ref(para):
        mdl = lgb.LGBMRegressor(**para)
        return mdl

    def lgb_clf(para):
        mdl = lgb.LGBMClassifier(**para)
        return mdl

    def svm_reg(para):
        mdl = SVR(**para)
        return mdl

    def svm_clf(para):
        mdl = SVC(probability=True, **para)
        return mdl

    def ridge(para):
        mdl = Ridge(**para)
        return mdl

    def lin_reg(para):
        mdl = LinearRegression(**para)
        return mdl

    def lr(para):
        mdl = LinearRegression(**para)
        return mdl

    def gpr(para):
        mdl = GPR(**para)
        return mdl


if __name__ == "__main__":
    # test code
    f = getattr(models, 'rf_clf')
    param = {'n_estimators': 10, 'n_jobs': -1}
    mdl = f(param)
    print(mdl)

    mdl2 = get_model('rf_reg', param)
    print(mdl2)

    mdl3 = get_arbitary_model("sklearn.ensemble.RandomForestClassifier", {})
    print(mdl3)

    print(models.__dict__.keys())
