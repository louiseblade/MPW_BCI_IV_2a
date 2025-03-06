import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import itertools
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model, save_model

from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Add, Activation, SeparableConv2D, SeparableConv1D, \
    LSTM, Input, BatchNormalization, LeakyReLU, Dense, Flatten, Dropout, GlobalAveragePooling1D, GaussianNoise, \
    Conv1D, Conv2D, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D, TimeDistributed, AveragePooling1D, \
    SpatialDropout2D

from tensorflow.keras import backend as K
from sklearn.utils import shuffle


def concat_data(*args):
    sj = []
    for a in args:
        sj = np.concatenate(a, axis=0)
    return sj


def generate_subject(subject_list, Type="T"):
    """
    Returns a list of dictionaries. Each dict corresponds to one subject, e.g.:
        {
          "id": <subject_number>,
          "X": <numpy array of shape (trials, channels, samples)>,
          "y": <numpy array of shape (trials, num_classes)>
        }
    """
    from Make_data import Read_Data
    from preprocessing import PREPROCESS_DATA
    from tensorflow.keras.utils import to_categorical

    all_subjects = []

    for sj_num in subject_list:
        data = "data_npz"
        label = "true_labels"

        # Read the raw data
        sj_train = Read_Data(data, label, Type=Type, subject_num=sj_num)

        # X_raw shape: (trials, channels, samples)
        X_raw = np.nan_to_num(sj_train.stack_trial_in_ch([i for i in range(22)]))
        # If your code uses transpose [0,2,1], do it here if needed
        X_raw = X_raw.transpose([0, 2, 1])  # shape now: (trials, samples, channels)

        # process data
        X_proc = PREPROCESS_DATA.preprocess_data(X_raw)
        y = to_categorical(sj_train.Load_truelabel())

        print(f"Subject {sj_num}: X.shape={X_proc.shape}, y.shape={y.shape}")

        # Store in a dictionary
        subject_data = {
            "id": sj_num,
            "X": X_proc,
            "y": y
        }

        all_subjects.append(subject_data)

    return all_subjects


def conv2D(Type, X_train, n_filters, dropoutRate=0.5, nb_classes=4):
    n_electrodes = X_train.shape[1]
    n_times = X_train.shape[2]
    if Type == "shallow":
        print("shallow conv2D")
        input_main = Input(X_train.shape[1:])
        block1 = Conv2D(40, (1, 13),
                        input_shape=(n_electrodes, n_times, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(40, (n_electrodes, 1), use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(dropoutRate)(block1)
        flatten = Flatten()(block1)
        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5), activation="softmax")(softmax)

        return Model(inputs=input_main, outputs=dense)
    else:
        print("Deep conv2D")
        input_main = Input(X_train.shape[1:])

        block1 = Conv2D(n_filters, (1, 5),
                        input_shape=(n_electrodes, n_times, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(n_filters, (X_train.shape[1], 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(0.5)(block1)

        block2 = Conv2D(n_filters * 2, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(0.5)(block2)

        block3 = Conv2D(n_filters * 4, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(0.5)(block3)

        block4 = Conv2D(n_filters * 8, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)

        flatten = Flatten()(block4)

        flatten = Dropout(0.5)(flatten)
        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=input_main, outputs=softmax)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def generate_sublist(model_list, k):
    import itertools as its
    for c in its.combinations(model_list, k):
        print(list(c), '  , ', list(filter(lambda a: a not in list(c), model_list)))

def generate_loo_lists(available_subjects):
    """Generate train/val combinations for Leave-One-Subject-Out (LOSO)"""
    n = len(available_subjects)
    train_list = []
    val_list = []

    # For each possible cross-sj level (1 to n-1 subjects left out)
    for k in range(1, n):  # Can't leave out all subjects
        # Generate combinations of training subjects (n-k subjects)
        train_combos = list(itertools.combinations(available_subjects, n - k))
        # Get corresponding validation subjects (k subjects)
        val_combos = [list(set(available_subjects) - set(combo)) for combo in train_combos]

        train_list.append([list(combo) for combo in train_combos])
        val_list.append(val_combos)

    return train_list, val_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train BCI IV 2a project.")
    parser.add_argument(
        "--test_subjects",
        nargs="+",
        type=int,
        default=[1,],
        help="One or more subject IDs to exclude from training. "
             "Default is 1, e.g.: --test_subjects 1 2 3"
    )
    args = parser.parse_args()

    for test_subject in args.test_subjects:
        subject_Number = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Remove test subject from available subjects
        available_subjects = [s for s in subject_Number if s != test_subject]


        print("Test subject: ", test_subject)

        # Remove test subject from available subjects
        available_subjects = [s for s in subject_Number if s != test_subject]

        # Generate dynamic train/val lists ,or you should simply just use train_test_split, train different models and aggregate the results later
        # We specifically use 7 train - 1 val split for this example because we want each subject contribute equally to the training set
        train_list, val_list = generate_loo_lists(available_subjects)

        # Generate subject data (only for available subjects)
        subjects = generate_subject(available_subjects)

        # Training parameters
        verbose, n_filters, num_classes, TYPE = 2, 32, 4, "Deep"
        epochs, batch_size = 1000, 128

        # cross-sj level 1: 7 train - 1 val
        # cross-sj level 2: 6 train - 2 val
        # cross-sj level 3: 5 train - 3 val
        # cross-sj level 4: 4 train - 4 val
        # cross-sj level 5: 3 train - 5 val
        # cross-sj level 6: 2 train - 6 val
        # cross-sj level 7: 1 train - 7 val
        cross_sj_levels = [1]

        for cross_sj_level in cross_sj_levels:

            # Get all combinations for this cross-sj level
            level_train_combos = train_list[cross_sj_level-1]
            level_val_combos = val_list[cross_sj_level-1]
            print(f"Total combinations: {len(level_train_combos)}")
            print(f"Train subjects: {level_train_combos}")
            print(f"Val subjects: {level_val_combos}")
            # Train each combination in this level
            for combo_idx, (train_subjs, val_subjs) in enumerate(zip(level_train_combos, level_val_combos)):
                print(f"Training combo {combo_idx + 1}: Train on {train_subjs}, Validate on {val_subjs}")

                # PREPARE DATA
                X_train_list, y_train_list = [], []
                for s in train_subjs:
                    idx = available_subjects.index(s)  # find index of subject ID s
                    X_train_list.append(subjects[idx]["X"])
                    y_train_list.append(subjects[idx]["y"])

                X_train = concat_data(X_train_list)
                y_train = concat_data(y_train_list)

                X_val_list, y_val_list = [], []
                for s in val_subjs:
                    idx = available_subjects.index(s)
                    X_val_list.append(subjects[idx]["X"])
                    y_val_list.append(subjects[idx]["y"])

                X_val = concat_data(X_val_list)
                y_val = concat_data(y_val_list)

                # Shuffle data
                X_train, y_train = shuffle(X_train, y_train)  # 2016 samples of Training for 1 model
                X_val, y_val = shuffle(X_val, y_val)  # 288 samples of Validating for 1 model
                X_train, X_val = X_train[..., np.newaxis], X_val[..., np.newaxis]

                # Model setup
                opt = optimizers.Adam(learning_rate=1e-4)
                model = conv2D(Type=TYPE, X_train=X_train, n_filters=n_filters)

                # File path setup
                FILEPATH = f'models/subject_{test_subject}/cross_sj_{cross_sj_level}_combo_{combo_idx+1}.h5'
                checkpoint = ModelCheckpoint(filepath=FILEPATH, monitor='val_accuracy',
                                             save_best_only=True, mode='max', verbose=verbose)

                # Train model
                model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                    epochs=epochs, batch_size=batch_size,
                                    verbose=verbose, callbacks=[checkpoint])

                # Clean up
                del model, history
                K.clear_session()

        print("Training complete! Test subject was completely excluded from all training/validation sets.")
