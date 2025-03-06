import os
import sys
import importlib.util
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
import numpy as np

from itertools import combinations
from tensorflow.keras.models import load_model
from train_model import generate_subject
class minimum_possible_weight:
    def __init__(self, *preds, last_weight_only=False):
        self.predictions = list(preds[0])
        self.lwo = last_weight_only

    def generated_sublist(self):
        """when last_weight_only is False, it will generate all possible combinations of the predictions\
        If we have 8 predictions from 8 models, this will generate weights between for not only 8 models but also 7, 6, 5, 4, 3, 2 models
        for example, in case of two models we have weight between models [1, 2] ,[1, 3], [1,4]...[1, 8], [2, 3], [2, 4]...[2, 8], [3, 4]...[3, 8], [4, 5]...[4, 8], [5, 6]...[5, 8], [6, 7], [6, 8], [7, 8]
        then three models [1,2,3], [1,2,4]...[1,2,8], [1,3,4]...[1,3,8], [1,4,5]...[1,4,8], [1,5,6]...[1,5,8], [1,6,7]...[1,6,8], [1,7,8], [2,3,4]...[2,3,8], [2,4,5]...[2,4,8], [2,5,6]...[2,5,8], [2,6,7]...[2,6,8],
        [2,7,8], [3,4,5]...[3,4,8], [3,5,6]...[3,5,8], [3,6,7]...[3,6,8], [3,7,8], [4,5,6]...[4,5,8], [4,6,7]...[4,6,8], [4,7,8], [5,6,7]...[5,6,8], [5,7,8], [6,7,8]...
        and finally  8 models [1,2,3,4,5,6,7,8] which is the weight for all models.
        If last_weight_only is True, it will only generate weight between models [1,2,3,4,5,6,7,8] as shown in Equation (11) in the paper. The result in the paper follow this setting."""

        sublists, i = [], 0
        C = len(self.predictions)
        while i <= len(self.predictions):
            temp = [list(x) for x in combinations(self.predictions, i)]
            sublists.extend(temp)
            i += 1

        final_sublist = sublists[C + 1:]

        return final_sublist

    def calculate_minimum_weight(self, k):

        if self.lwo:
            last_sub_list = self.generated_sublist()[-1]
            Weights = k * np.sum(np.array(last_sub_list)) / np.sum(np.array(last_sub_list) ** 2)
        else:
            _Weights = []
            for pred in self.generated_sublist():
                weight = k * np.sum(np.array(pred)) / np.sum(np.array(pred) ** 2)
                _Weights.append(weight)

            Weights = np.sum(_Weights)

        return Weights





def Averaging_prediction(models, data):
    predicted_list = []
    for md in models:
        predicted_list.append(md.predict(data))

    new_pred = np.mean(predicted_list, axis=0)
    return new_pred

def PER_CLASS_PROBABILITY(models, data):
    predicted_list = []

    for md in models:
        predicted_list.append(md.predict(data))  # (num_model, trials, classes)

    # overall mean
    mean_list = []
    for i in range(np.array(predicted_list).shape[0]):
        mean_list.append(np.mean(predicted_list[i], axis=0))

    Confidence_score = np.mean(mean_list, axis=0, dtype=np.float64)

    return Confidence_score

# calculate minimum possible weight to reach the per-class probability
def min_weight_possible(models, data, per_class_probability=np.ones(4), lwo=False):
    """per_class_probability is set to 1 by default, which mean confidence is equal for all classes.
     In MPW, this value is calculated by the PER_CLASS_PROBABILITY function, which is the average of the model predictions over all trials."""

    predicted_list = []
    for md in models:
        try:
            predicted_list.append(md.predict(data))  # (num_model, trials, classes)
        except:  # avoid none model
            continue

    trial, num_class = data.shape[0], 4

    predicted_list = np.array(predicted_list)
    NEW_PRED = np.zeros((trial, num_class))

    # min weight cost
    for tr in range(trial):
        for cl in range(num_class):
            W = minimum_possible_weight(predicted_list[:, tr, cl], last_weight_only=lwo)
            NEW_PRED[tr, cl] = W.calculate_minimum_weight(k=per_class_probability[cl])

    return NEW_PRED


if __name__ == '__main__':
    import argparse

    # 1. Setup argument parser
    parser = argparse.ArgumentParser(
        description="Compute MPW results for one or more test subjects."
    )
    parser.add_argument(
        "--test_subjects",
        nargs="+",
        type=int,
        default=[1],
        help="Subject IDs to evaluate with MPW. Default: [1]. Example: --test_subjects 1 2 3"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="E",
        help="Which data type to load with generate_subject (T or E). Default: E."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Path to the folder that contains subject_x subfolders of .h5 models."
    )
    parser.add_argument("--lwo", dest="lwo", action="store_true", help="Enable last weight only.")
    parser.add_argument("--no-lwo", dest="lwo", action="store_false", help="Full Weight properties")
    parser.set_defaults(lwo=True)
    args = parser.parse_args()

    # 2. Loop over each requested subject and run the original logic
    for subj in args.test_subjects:
        print(f"\n=== Evaluating MPW for subject {subj} ===")

        # original code logic:
        test_data = generate_subject([subj], Type=args.data_type)
        X_test, y_test = test_data[0]['X'], test_data[0]['y']

        MODEL_PATH = os.path.join(args.models_dir, f"subject_{subj}")
        models = []
        for model_file in os.listdir(MODEL_PATH):
            if model_file.endswith(".h5"):
                full_path = os.path.join(MODEL_PATH, model_file)
                model = load_model(full_path)
                models.append(model)

        pcp = PER_CLASS_PROBABILITY(models, data=X_test)
        result_pred = min_weight_possible(models, data=X_test, per_class_probability=pcp, lwo=args.lwo)

        # MPW: we do argmin on each trial
        mpw_preds = np.argmin(result_pred, axis=1)
        # ground truth
        y_true = np.argmax(y_test, axis=1)

        # note: your code used np.amax(MPW), but that was a bit odd. Possibly you meant just MPW * 100.
        MPW_acc = accuracy_score(mpw_preds, y_true) * 100.0

        # average ensemble: argmax of the average
        avg_preds = np.argmax(Averaging_prediction(models, data=X_test), axis=1)
        AVG_acc = accuracy_score(avg_preds, y_true) * 100.0

        print(f"Subject {subj} MPW accuracy:       {MPW_acc:.2f}%")
        print(f"Subject {subj} Averaging accuracy: {AVG_acc:.2f}%")



