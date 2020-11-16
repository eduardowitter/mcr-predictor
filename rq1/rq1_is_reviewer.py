####################################################################################################
#                                                                                                  #
# Research Question 1 implementation: evaluation of different undersampling rates, feature sets,   #
# and different learners for the prediction of reviewer participation.                             #
#                                                                                                  #
#    Example: python3 -u is_reviewer.py                                                            #
#                                                                                                  #
####################################################################################################

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import sys
sys.path.append('../lib')
from dataset import Dataset

####################################################################################################
#                                                                                                  #
# Configuration for Research Questions 1:                                                          #
# - Path to file with all-time dataset in CSV format                                               #
# - Name of training and test sets used                                                            #
# - Features used for each of the five executions                                                  #
# - Target Name: Target-IsReviewer                                                                 #
# - Evaluated undersampling rates for the majority class (Target-IsReviewer=FALSE)                 #
# - File where results are written                                                                 #
#                                                                                                  #
# In this Research Question, we evaluate how the prediction for Target-IsReviewer is               #
# affected by different sets of features, different undersampling rates and different learners.    #
#                                                                                                  #
####################################################################################################

DATASET_CSV_PATH = '../raw_dataset.csv'

OUTPUT_CSV_FILE_NAME = 'rq1_is_reviewer_results.csv'

TRAINING_SET_NAME = 'training-Y1'

TEST_SET_NAME = 'test-T1a'

TARGET_COLUMN_NAME = 'Target-IsReviewer'

UNDERSAMPLING_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]

EXECUTIONS_FEATURES = {
    'LOC': ['F1-ChangedLOC'],

    'Baseline': ['F9-FileReviewXp'],

    'CO': ['F4-IsMaintainer', 'F5-ModuleReviewXp', 'F6-ModuleModifXp', 'F9-FileReviewXp', 'F10-FileModifXp'],

    'WL': ['F11-OngoingAsAuthor', 'F12-OngoingAsReviewer'],

    'TS': ['F2-SameTeam', 'F3-SameLocation', 'F7-TeamReviewXp', 'F8-TeamModifXp'],

    'CO+TS+WL': ['F2-SameTeam', 'F3-SameLocation', 'F4-IsMaintainer', 'F5-ModuleReviewXp',
                 'F6-ModuleModifXp', 'F7-TeamReviewXp', 'F8-TeamModifXp', 'F9-FileReviewXp',
                 'F10-FileModifXp', 'F11-OngoingAsAuthor', 'F12-OngoingAsReviewer'],

    'All': ['F1-ChangedLOC', 'F2-SameTeam', 'F3-SameLocation', 'F4-IsMaintainer',
            'F5-ModuleReviewXp', 'F6-ModuleModifXp', 'F7-TeamReviewXp', 'F8-TeamModifXp',
            'F9-FileReviewXp', 'F10-FileModifXp', 'F11-OngoingAsAuthor', 'F12-OngoingAsReviewer'],
}

####################################################################################################
#                                                                                                  #
# Run Research Question 1 tests.                                                                   #
# Results are written to both terminal and output CSV file.                                        #
#                                                                                                  #
####################################################################################################

# Load training and test sets
dataset = Dataset(DATASET_CSV_PATH)
training_df, test_df = dataset.get_training_and_test_by_name(TRAINING_SET_NAME, TEST_SET_NAME)

output = open(OUTPUT_CSV_FILE_NAME, 'w+')
csv_header = "Execution,Sampling,Model,AUPRC,AUROC,Accuracy,F1,Precision,Recall"
output.write(csv_header + "\n")


# For each combination of features...
for execution, features in EXECUTIONS_FEATURES.items():
    print('\n\n' + '=' * 100)
    print('{}: {}'.format(execution, ', '.join(features)))

    # traning and test sets
    exec_training_features = training_df.reindex(columns=features)
    exec_training_target = training_df[TARGET_COLUMN_NAME]
    exec_test_features = test_df.reindex(columns=features)
    exec_test_target = test_df[TARGET_COLUMN_NAME]

    base_learners = {
        'LinearSVC': LinearSVC(random_state=0, loss='squared_hinge', penalty='l1', C=1e-2, max_iter=1000, fit_intercept=True, dual=False, tol=1e-4),
        'Random Forest': RandomForestClassifier(random_state=0, n_estimators=300, min_samples_leaf=10, criterion='gini', max_features=None, max_depth=None),
        'Logistic Regression': LogisticRegression(random_state=0, penalty='l1', C=1e-3, max_iter=100, solver='saga', fit_intercept=True, dual=False, tol=1e-4),
    }

    print("{:14s}  {:>8s}  {:20s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}".format(
        'Execution', 'Sampling', 'Model', 'AUPRC', 'AUROC', 'ACC', 'F1', 'Prec.', 'Recall'))

    # For each undersampling rate...
    for undersampling_rate in UNDERSAMPLING_RATES:

        # For each base learner...
        for learner_name, learner in base_learners.items():

            # undersample majority class, which is "False"
            counter = Counter(exec_training_target)
            true_count = counter[True]
            false_count = int(counter[False] * undersampling_rate)
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
                exec_test_target, predicted, average='binary', zero_division=0)

            if hasattr(pipeline, "predict_proba"):
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    exec_test_target, pipeline.predict_proba(exec_test_features)[:, 1])
            else:
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    exec_test_target, pipeline['learner']._predict_proba_lr(exec_test_features)[:, 1])

            auroc = auc(false_positive_rate, true_positive_rate)

            # precision-recall AUC
            if precision == 0.0 and recall == 0.0 and f1 == 0.0:
                f1 = 'ND'
                auprc = 'ND'
            else:
                precision_, recall_, _ = precision_recall_curve(exec_test_target, predicted)
                f1 = '{:.3f}'.format(f1)
                auprc = '{:.3f}'.format(auc(precision_, recall_))

            output.write("{},{},{},{},{:.3f},{:.3f},{},{:.3f},{:.3f}\n".format(
                execution, str(undersampling_rate*100) + "%", learner_name, auprc, auroc, acc,
                f1, precision, recall))
            print("{:14s}  {:>8s}  {:20s}  {:>5}  {:>.3f}  {:>.3f}  {:>5}  {:>.3f}  {:>.3f}".format(
                execution, str(undersampling_rate*100) + "%", learner_name, auprc, auroc, acc,
                f1, precision, recall))
            output.flush()

output.close()
print("\n\nOutput written to: {}".format(OUTPUT_CSV_FILE_NAME))
