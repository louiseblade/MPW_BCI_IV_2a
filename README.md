for PhysionetMI , please visit https://github.com/louiseblade/MPW_PhysioNetMI

BCI Competition IV 2a Project
=============================

This repository provides scripts for training and evaluating EEG classification models for the BCI Competition IV 2a dataset. 
Follow the steps below to get started.

1. Installation & Setup
------------------------
1. Clone this repository and navigate into its folder:
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
   ```

2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset (e.g., A01T.npz , A01E.npz) is correctly placed in `data_npz/` and  your True label is (e.g. A01T.mat , A01E.mat)`true_labels/`. 
   If necessary, generate the data using:
   ```bash
   python Make_data.py
   ```


   ```

2. Training Models
-------------------
Run `train.py` to train models while excluding a given subject as the test subject.

### Train for a single subject exclusion:
```bash
python train_model.py --test_subjects 1 


```
This excludes subject 1 from training, training on all other subjects and saving models in `models/subject_1/`.

### Train for multiple subjects:
```bash
python train_model.py --test_subjects 2 5 7 
```
This runs separate training sessions where subject 2, then 5, then 7 are treated as test subjects.

If `--test_subjects` is omitted, it defaults to `[1]`.


3. Evaluating with MPW (and Simple Averaging)
------------------------
After training, use `MPW.py` to evaluate the models.

### Evaluate MPW for a single subject:
```bash
python MPW.py --test_subjects 1 --data_type E  --lwo
```
This loads the trained models from `models/subject_1/` and computes MPW accuracy and Averaging accuracy.

data type might also set to T since the leave-out subject data were never used.(However result on the paper were using Evaluation session )

For full weight properties use --no-lwo instead

```bash
python MPW.py --test_subjects 1 --data_type E  --no-lwo
```


### Evaluate MPW for multiple subjects:
```bash
python MPW.py --test_subjects 2 3 9 --data_type E --lwo
```
This sequentially evaluates subjects 2, 3, and 9.

- `--models_dir MyCustomModelsDir`: Loads models from a custom directory.

4. Other Scripts
-----------------
- `Make_data.py`: Reads raw `.npz` files and generates data arrays.
- `preprocessing.py`: Applies detrending and Z-scoring to EEG signals.


Reminder: Custom models might be used. HOWEVER,ALL MODELS SHOULD BE HOMOGENEOUS. (This might yield unexpected results or, in the worst-case scenario, a decline in performance)
