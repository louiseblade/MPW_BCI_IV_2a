import numpy as np
import os
import scipy.io as sio
class Read_Data:
    def __init__(self, data_path, label_path, Type, subject_num):
        self.subject = subject_num
        self.path = data_path
        self.Type = Type # T or E
        self.true_labels = label_path
        #data func
        self.raw = self.data()['s'].T
        self.events_type = self.data()['etyp'].T
        self.events_position = self.data()['epos'].T
        self.events_duration = self.data()['edur'].T
        self.artifacts = self.data()['artifacts'].T
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}
        self.others_type = {276: 'eyes_open', 277: 'eyes_close', 768: 'trial_start', 1023: 'rejected_trial', 1072: 'Eye_movements', 32766: 'new_run' }
        self.channel_name = {'Fz': 0, 'C3': 7, 'Cz': 9, 'C4': 11, 'Pz': 19}

    def data(self):

        data = np.load(os.path.join(self.path, "A0%d%s.npz" % (self.subject, self.Type)))
        return data

    def count_mi_trial(self):
        c_left, c_right, c_foot, c_tongue = 0, 0, 0, 0
        for i in self.events_type[0]:
            if i == 769:
                c_left += 1
            elif i == 770:
                c_right += 1
            elif i == 771:
                c_foot += 1
            elif i == 772:
                c_tongue += 1
            else:
                continue
        return c_left, c_right, c_foot, c_tongue

    def artifact_trial(self):
        for i, j in enumerate(self.artifacts[0]):
            if j == 1:
                yield i


    def get_trial_from_ch(self, channel):
        # print("duration\n", self.events_duration)

        starttrial_events = self.events_type == 768
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials, classes = [], []
        for index in idxs:
            try:
                TYPE = self.events_type[0][index+1]
                CLASS = self.mi_types[TYPE]
                start = self.events_position[0][index]
                stop = start + self.events_duration[0][index] - 375  # always + 1875 if MI - 375
                trial = self.raw[channel, start:stop]

            except KeyError:
                TYPE = self.events_type[0][index + 2]
                CLASS = self.mi_types[TYPE]
                start = self.events_position[0][index]
                stop = start + self.events_duration[0][index] - 375  # always + 1875 if MI - 375
                trial = self.raw[channel, start:stop]

            trials.append(trial)
            classes.append(CLASS)

        return trials, classes

    def stack_trial_in_ch(self, *args): #input a list of channel
        channels = list(args)
        list_channel = []
        for ch in channels[0]:

            trial, _ = self.get_trial_from_ch(ch)

            list_channel.append(trial)
        return np.stack(list_channel, axis=-1) #shape(288 trials, 1500 datapoints, 22 channels)

    def Load_truelabel(self):
        lb = sio.loadmat(os.path.join(self.true_labels, "A0%d%s.mat" % (self.subject, self.Type)))
        y = np.array(lb["classlabel"]).T[0] - 1
        return y

def create_single_subject_raw_data(X_train, X_test, y_train, y_test, subject_num, save_to_files=False):

    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "subject_num": subject_num}

    if save_to_files:
        sio.savemat("subject%d_raw.mat" % subject_num, data)
    return data


if __name__ == '__main__':
    data = "data_npz"
    label = "true_labels"
    subject_num = 1
    sj_train, sj_test = Read_Data(data, label, Type="T", subject_num=subject_num), Read_Data(data, label, Type="E", subject_num=subject_num)

    X_train = np.nan_to_num((sj_train.stack_trial_in_ch([i for i in range(22)]))).transpose([0, 2, 1])
    X_test = np.nan_to_num((sj_test.stack_trial_in_ch([j for j in range(22)]))).transpose([0, 2, 1])

    y_train, y_test = sj_train.Load_truelabel(), sj_test.Load_truelabel()

    # Note that test file has no label, therefore load true label is necessary
    print("subject", subject_num)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # to save data to file, set save_to_files
    create_single_subject_raw_data(X_train, X_test, y_train, y_test, subject_num=subject_num, save_to_files=False)

