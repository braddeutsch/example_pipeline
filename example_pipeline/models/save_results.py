import os
from sklearn.externals import joblib


def save_results(_path, _model, _report):
    if not os.path.exists(_path):
        os.makedirs(_path)

    # save model
    joblib.dump(_model, _path + 'model.pkl')

    # Prepare to save pipeline info
    s = ''
    for k in _model.get_params():
        s += str(k) + ': ' + str(_model.get_params()[k]) + '\n'

    output_list = ['Classification report:\n' + _report + '\n\n',
                   'Pipeline:\n' + s]

    # Save
    with open(_path + 'results.txt', 'w') as f:
        for item in output_list:
            f.writelines(item)

    return
