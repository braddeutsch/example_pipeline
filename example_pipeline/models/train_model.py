from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from features.build_features import featurize_cols


def make_pipeline(features):
    """
    build a pipeline that includes feature building, modeling, and gridsearch
    :param features: list of features, e.g. ['age', 'gender', ...]
    :return: pipeline object
    """

    base_pipeline = Pipeline([('featurize', featurize_cols(features)), ('gbf', GradientBoostingClassifier())])

    # define cross-validation scheme
    cv_folds = StratifiedKFold(n_splits=3)

    # do a grid search to find best model
    param_grid = dict(gbf__n_estimators=[5, 20, 50], gbf__subsample=[.6, .8], gbf__min_samples_leaf=[1, 3])

    # define grid search object
    pipeline_out = GridSearchCV(base_pipeline, param_grid=param_grid, cv=cv_folds, scoring='accuracy')

    return pipeline_out
