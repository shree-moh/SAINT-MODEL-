from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from pretrainmodel import SAINT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    return PD, PF, balance, FIR

# Function to evaluate classifier
def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD, PF, balance, FIR = calculate_metrics(y_test, y_pred)
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)
    print('Confusion Matrix:', cm)
    print('Length of y_test:', len(y_test))
    print('Length of y_pred:', len(y_pred))

# CSV file path
csv_file_path = "EQ.csv"

# Read data
df = pd.read_csv(csv_file_path)
X = df.drop(columns=['class'])
y = df['class']

# Set up K-layer cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store metrics
pd_list = []
pf_list = []
bal_list = []
fir_list = []

# Perform K-layer cross-validation


for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Model instantiation, training, and evaluation for this fold
    model = RandomForestClassifier()  # Example model instantiation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)  # Obtain predictions for validation set

    # Calculate metrics for this fold
    PD, PF, balance, FIR = calculate_metrics(y_val, y_pred)
    # Pre-processing
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_normalized, y_train)
    # Store metrics for this fold
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)



    # Define cat_dims before using it
    cat_dims = []  # Initialize as an empty list

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # ...

        # Append values to cat_dims
        cat_dims = np.append(np.array(cat_dims), np.array([2])).astype(int)

        # ...

    cat_dims = np.append(np.array(cat_dims), np.array([2])).astype(
        int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification

    # Further code follows...


    # Define and assign a value to con_idxs before using it
    con_idxs = [0, 1, 3, 5]  # Replace with the appropriate list of continuous feature indices

    pretrainmodel = SAINT(
        num_continuous=5,
        num_categories=10,
        dim=8,
        dim_out=1,
        depth=1,
        heads=4,
        attn_dropout=0,
        ff_dropout=0.8,
        mlp_hidden_mults=(4, 2),
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col',
        final_mlp_style='common',
        y_dim=2,
        categories=10  # Make sure to replace this with the correct argument
    )

# Assuming you have a separate test set for final evaluation
y_test_final = [1, 0, 1, 1, 0]  # Replace this with your actual test data labels
y_pred_final = [1, 1, 1, 0, 0]  # Replace this with your actual predictions for the test set
classifier_eval(y_test_final, y_pred_final)

# Calculating average metrics
avg_PD = sum(pd_list) / len(pd_list) if len(pd_list) > 0 else 'No values in pd_list'
avg_PF = sum(pf_list) / len(pf_list) if len(pf_list) > 0 else 'No values in pf_list'
avg_balance = sum(bal_list) / len(bal_list) if len(bal_list) > 0 else 'No values in bal_list'
avg_FIR = sum(fir_list) / len(fir_list) if len(fir_list) > 0 else 'No values in fir_list'

# Print or use the average metrics as needed
print('Average PD:', avg_PD)
print('Average PF:', avg_PF)
print('Average balance:', avg_balance)
print('Average FIR:', avg_FIR)