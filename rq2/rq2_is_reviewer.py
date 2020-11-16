#!/usr/bin/python3 -u
'''
Research Question 2:
Run Recursive Feature Elimination (RFE) for the prediction of 'Target-IsReviewer'
'''

from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import PredefinedSplit
from imblearn.under_sampling import RandomUnderSampler
from dataset import Dataset

####################################################################################################
#                                                                                                  #
# Configuration for Research Questions 2:                                                          #
# - Path to file with all-time dataset in CSV format                                               #
# - Name of training and test sets used                                                            #
# - Target Name:  Target-IsReviewer                                                                #
#                                                                                                  #
####################################################################################################

DATASET_CSV_PATH = 'raw_dataset.csv'

TRAINING_SET_NAME = 'training-Y1'

TEST_SET_NAME = 'test-T1a'

TARGET_COLUMN_NAME = 'Target-IsReviewer'

UNDERSAMPLING_RATE = 0.25

# all features (F1-F12)
EXECUTIONS_FEATURES = ['F1-ChangedLOC', 'F2-SameTeam', 'F3-SameLocation', 'F4-IsMaintainer',
                       'F5-ModuleReviewXp', 'F6-ModuleModifXp', 'F7-TeamReviewXp',
                       'F8-TeamModifXp', 'F9-FileReviewXp', 'F10-FileModifXp',
                       'F11-OngoingAsAuthor', 'F12-OngoingAsReviewer']

BASE_LEARNERS = {
    'Random Forest': RandomForestClassifier(random_state=0, n_estimators=300, min_samples_leaf=10, criterion='gini', max_features=None, max_depth=None),
}

####################################################################################################
#                                                                                                  #
# Run Research Question 2 tests.                                                                   #
# Results are written to terminal only.                                                            #
#                                                                                                  #
####################################################################################################

# Load training and test sets
dataset = Dataset(DATASET_CSV_PATH)
training_df, test_df = dataset.get_training_and_test_by_name(TRAINING_SET_NAME, TEST_SET_NAME)
exec_training_features = training_df.reindex(columns=EXECUTIONS_FEATURES)
exec_training_target = training_df[TARGET_COLUMN_NAME]
exec_test_features = test_df.reindex(columns=EXECUTIONS_FEATURES)
exec_test_target = test_df[TARGET_COLUMN_NAME]

# build undersampler for majority class, which is "False"
counter = Counter(exec_training_target)
true_count = counter[True]
false_count = int(counter[False] * UNDERSAMPLING_RATE)
undersampler = RandomUnderSampler(
    sampling_strategy={True: true_count, False: false_count})

# build scaler
scaler = StandardScaler()
scaler.fit(exec_training_features)

# scale and undersample
exec_training_features = scaler.transform(exec_training_features)
exec_test_features = scaler.transform(exec_test_features)
exec_training_features, exec_training_target = undersampler.fit_resample(
    exec_training_features, exec_training_target)

# configure training and test splitting for RFECV
training_and_test_features = np.concatenate((exec_training_features, exec_test_features), axis=0)
training_and_test_target = np.concatenate((exec_training_target, exec_test_target), axis=0)
test_fold = [-1] * len(exec_training_target) + [0] * len(exec_test_target)
ps = PredefinedSplit(test_fold)

# Test for each base learner
for learner_name, learner in BASE_LEARNERS.items():
    print('\n\nComputing RFE with cross validation for {} ...'.format(learner_name))
    rfe = RFECV(learner, min_features_to_select=1, verbose=10, cv=ps, scoring='f1')
    fit = rfe.fit(training_and_test_features, training_and_test_target)
    print("\tNum Features: %d" % fit.n_features_)
    print("\tSelected Features: %s" % fit.support_)
    print("\tFeature Ranking: %s" % fit.ranking_)
    try:
        print("\tImportance: {}".format(fit.estimator_.feature_importances_))
    except AttributeError:
        print("\tImportance: Not found")
    except RuntimeError:
        print("\tImportance: Not found")
    print("\tGrid: %s" % str(fit.grid_scores_))
