import os
from scipy.io import loadmat, savemat
from scipy.stats import zscore
from scipy.signal import detrend


class PREPROCESS_DATA:
    def __init__(self, data):
        self.data = data

    def detrend_data(self):
        new_data = detrend(self.data, axis=-1, type='linear')
        return new_data

    def Zscore(self):
        new_data = zscore(self.data, axis=-1)
        return new_data

    def preprocess_data(self):
        new_data = detrend(self.data, axis=-1, type='linear')
        new_data = zscore(new_data, axis=-1)
        return new_data[:, :, 500:]


def make_new_data(subject):
    new_data = PREPROCESS_DATA
    new_sj = {'X_train': new_data.preprocess_data(subject['X_train']),
              'X_test': new_data.preprocess_data(subject['X_test']),
              'y_train': subject['y_train'], 'y_test': subject['y_test'],
              'subject_num': subject['subject_num']}

    return new_sj


if __name__ == '__main__':
    # if nessessary, run this file to output the preprocessed data else just run train.py

    PATH = ""
    subject_num = 1
    print(os.getcwd())

    # load raw file, data before preprocessing

    sj = loadmat(os.path.join(PATH, 'subject%d_raw.mat' % subject_num))

    # plot_data(sj['X_train'][0, 0], "before data preprocessing")

    # preproces data
    new_sj = make_new_data(sj)
    print(sj['y_train'], '\n', sj['y_test'], sj['subject_num'])

    # plot_data(new_sj['X_train'][0, 0], "after data preprocessing")

    # savemat(os.path.join(PATH, "raw/subject%d.mat" % subject_num), new_sj)
    savemat(os.path.join(PATH, "pre_processed_subject%d.mat" % subject_num), new_sj)
