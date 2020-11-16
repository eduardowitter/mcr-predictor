#!/usr/bin/python3 -u
'''
Research Question 3 implementation: evaluation of performance considering different training and
test sets.
'''

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve
from sklearn.metrics import auc, precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from dataset import Dataset

####################################################################################################
#                                                                                                  #
# Configuration for Research Questions 3:                                                          #
# - Path to file with all-time dataset in CSV format                                               #
# - Name of training and test sets used                                                            #
# - File where results are written                                                                 #
# - Undersampling rate                                                                             #
#                                                                                                  #
####################################################################################################

DATASET_CSV_PATH = 'raw_dataset.csv'

OUTPUT_CSV_FILE_NAME = 'rq3_participation_results.csv'

TARGET_COLUMN_NAME = 'Target-HaveParticipation'

# undersampling rate: 25% of the majority class (highest F1 score in average and best case)
UNDERSAMPLING_RATES = 0.25

# all features (F1-F12)
EXECUTIONS_FEATURES = ['F1-ChangedLOC', 'F2-SameTeam', 'F3-SameLocation', 'F4-IsMaintainer',
                       'F5-ModuleReviewXp', 'F6-ModuleModifXp', 'F7-TeamReviewXp',
                       'F8-TeamModifXp', 'F9-FileReviewXp', 'F10-FileModifXp',
                       'F11-OngoingAsAuthor', 'F12-OngoingAsReviewer']

# each execution has its own training and test set
EXECUTIONS = ['T1', 'T2', 'T3', 'T4', 'T5',
              'S1', 'S2', 'S3', 'S4', 'S5',
              'N1', 'N2', 'N3', 'N4', 'N5',
              'Y1', 'Y2', 'Y3', 'Y4', 'Y5']

####################################################################################################
#                                                                                                  #
# Run Research Question 3 tests.                                                                   #
# Results are written to both terminal and output CSV file.                                        #
#                                                                                                  #
####################################################################################################

output = open(OUTPUT_CSV_FILE_NAME, 'w+')
csv_header = "Execution,Model,AUPRC,AUROC,Accuracy,F1,Precision,Recall"
output.write(csv_header + "\n")


# Load training and test sets
dataset = Dataset(DATASET_CSV_PATH)


# For each combination of features...
for execution in EXECUTIONS:
    print('\n\n' + '=' * 100)
    print('{}: {}'.format(execution, ', '.join(EXECUTIONS_FEATURES)))

    training_df, test_df = dataset.get_training_and_test_by_exec_name(execution)

    # traning and test sets
    exec_training_features = training_df.reindex(columns=EXECUTIONS_FEATURES)
    exec_training_target = training_df[TARGET_COLUMN_NAME]
    exec_test_features = test_df.reindex(columns=EXECUTIONS_FEATURES)
    exec_test_target = test_df[TARGET_COLUMN_NAME]

    base_learners = {
        'LinearSVC': LinearSVC(random_state=0, loss='squared_hinge', penalty='l1', C=1e-2, max_iter=1000, fit_intercept=True, dual=False, tol=1e-4),
        'Random Forest': RandomForestClassifier(random_state=0, n_estimators=300, min_samples_leaf=10, criterion='gini', max_features=None, max_depth=None),
        'Neural Network': MLPClassifier(random_state=0, activation='tanh', solver='adam', alpha=1e-2, beta_1=0.7, beta_2=0.9, max_iter=1000, hidden_layer_sizes=100),
        'Logistic Regression': LogisticRegression(random_state=0, penalty='l1', C=1e-3, max_iter=100, solver='saga', fit_intercept=True, dual=False, tol=1e-4),
    }

    print("Execution  Model                 AUPRC    AUROC    ACC    F1     Precision  Recall")

    # For each base learner...
    for learner_name, learner in base_learners.items():

        # undersample majority class, which is "False"
        counter = Counter(exec_training_target)
        true_count = counter[True]
        false_count = int(counter[False] * UNDERSAMPLING_RATES)
        undersampler = RandomUnderSampler(
            sampling_strategy={True: true_count, False: false_count})

        # build pipeline
        steps = [('under', undersampler), ('scale', StandardScaler()), ('learner', learner)]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(exec_training_features, exec_training_target)

        # prediction
        predicted = pipeline.predict(exec_test_features)

        # evaluation
        acc = accuracy_score(exec_test_target, predicted)
        precision, recall, f1, _ = precision_recall_fscore_support(
            exec_test_target, predicted, average='binary')

        if hasattr(pipeline, "predict_proba"):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                exec_test_target, pipeline.predict_proba(exec_test_features)[:, 1])
        else:
            false_positive_rate, true_positive_rate, _ = roc_curve(
                exec_test_target, pipeline['learner']._predict_proba_lr(exec_test_features)[:, 1])
        auroc = auc(false_positive_rate, true_positive_rate)

        # precision-recall AUC
        precision_, recall_, _ = precision_recall_curve(exec_test_target, predicted)
        auprc = auc(precision_, recall_)

        output.write("{}, {:20s}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
            execution, learner_name, auprc, auroc, acc,
            f1, precision, recall))
        print("{}         {:20s}  {:.3f}    {:.3f}    {:.3f}  {:.3f}  {:.3f}      {:.3f}".format(
            execution, learner_name, auprc, auroc, acc,
            f1, precision, recall))

output.close()
print("\n\nOutput written to: {}".format(OUTPUT_CSV_FILE_NAME))
