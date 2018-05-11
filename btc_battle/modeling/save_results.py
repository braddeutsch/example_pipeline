import os
from sklearn.externals import joblib


def save_results(_path, _model, _report):
    """
    Save model object, pipeline info, and classification report to a file specified
    """

    # create directory if ndef
    if not os.path.exists(_path):
        os.makedirs(_path)

    # save model
    joblib.dump(_model, _path + 'model.pkl')

    # Prepare to save pipeline info
    s = ''
    for k in _model.get_params():
        s += str(k) + ': ' + str(_model.get_params()[k]) + '\n'

    # what to save
    output_list = ['Classification report:\n' + _report + '\n\n',
                   'Pipeline:\n' + s]

    # Save each item in the list
    with open(_path + 'results.txt', 'w') as f:
        for item in output_list:
            f.writelines(item)

    return
