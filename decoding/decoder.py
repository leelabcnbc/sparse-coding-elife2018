from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score

def _parse_cv_obj(X, y, y_group, seed):
    # if y_group is not None, then use LeaveOneGroupOut
    # else, I will use StratifiedKFold
    if y_group is not None:
        assert seed is None
        cv_obj = LeaveOneGroupOut().split(X, y, y_group)
    else:
        raise RuntimeError
        # TODO: for now, I just use 5 folds. If needed, I can always use more and further refactor the program.
        #cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)

    return cv_obj


def _parse_classifier_params(params):
    name, param_kw, additional_params = params
    fancy_classification = False
    if name == 'NearestCentroid':
        classifier = NearestCentroid(**param_kw)
    else:
        raise ValueError(f'unsupported classifier {name}')

    return name, classifier, fancy_classification, param_kw, additional_params

def classification_wrapper_one_Xy(X, y, params, seed, y_group=None, n_jobs=-1, label_noise_seed=None,
                                  label_noise_level=None):
    # this only returns prediction.
    assert len(X) == len(y)
    cv_obj = _parse_cv_obj(X, y, y_group, seed)

    # hack to change y. remember to do this before doing cv object. otherwise, stratified CV won't work.

    if label_noise_level is not None:
        assert y.dtype == np.float64
        rng_state_this = np.random.RandomState(seed=label_noise_seed)
        y = y + label_noise_level * rng_state_this.randn(y.size)

    # then get classifier params.
    name, classifier, fancy_classification, param_kw, additional_params = _parse_classifier_params(params)

    # then classify, return prediction results.
    y_pred = _classifier_return_pred(X, y, cv_obj, name, classifier, fancy_classification, param_kw, additional_params,
                                     n_jobs)
    return y_pred

def _classifier_return_pred(X, y, cv_obj, name, classifier, fancy_classification, param_kw, additional_params, n_jobs):
    if not fancy_classification:
        y_pred = cross_val_predict(classifier, X, y, cv=cv_obj, n_jobs=n_jobs)
    else:
        raise NotImplementedError
    assert y.shape == y_pred.shape
    return y_pred
