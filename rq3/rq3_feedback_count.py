#!/usr/bin/python3 -u
'''
Research Question 3 implementation: evaluation of performance considering different training and
test sets.
'''

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
#from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from imblearn.pipeline import Pipeline
from scipy import stats
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

OUTPUT_CSV_FILE_NAME = 'rq3_comments_results.csv'

TARGET_COLUMN_NAME = 'Target-CommentsCount'

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
csv_header = "Execution,Model,RMSE,R2,Pearson Correlation,Spearman Correlation"
output.write(csv_header + "\n")


# Load training and test sets
dataset = Dataset(DATASET_CSV_PATH)


# For each combination of features...
for execution in EXECUTIONS:
    print('\n\n' + '=' * 100)
    print('{}: {}'.format(execution, ', '.join(EXECUTIONS_FEATURES)))

    training_df, test_df = dataset.get_training_and_test_by_exec_name(
        execution, only_with_participation=True)

    # traning and test sets
    exec_training_features = training_df.reindex(columns=EXECUTIONS_FEATURES)
    exec_training_target = training_df[TARGET_COLUMN_NAME]
    exec_test_features = test_df.reindex(columns=EXECUTIONS_FEATURES)
    exec_test_target = test_df[TARGET_COLUMN_NAME]

    base_learners = {
        'kNN': KNeighborsRegressor(n_neighbors=100, metric='minkowski', p=1, weights='uniform', algorithm='auto'),
        'Linear Regression': Ridge(alpha=10, solver='auto'),
#        'Neural Network': MLPRegressor(random_state=0, activation='relu', solver='adam',
#                                       alpha=0.0001, max_iter=200),
        'Random Forest': RandomForestRegressor(random_state=0, n_estimators=150, min_samples_leaf=10, criterion='mse', max_features=None, max_depth=None),
    }

    print("Execution       Model                 RMSE   R2     Pearson Correlation / p-value     Spearman Correlation / p-value")

    # For each base learner...
    for learner_name, learner in base_learners.items():

        # build pipeline
        steps = [('scale', StandardScaler()), ('learner', learner)]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(exec_training_features, exec_training_target)

        # prediction
        predicted = pipeline.predict(exec_test_features)

        # evaluation
        r2 = r2_score(exec_test_target, predicted)
        rmse = mean_squared_error(exec_test_target, predicted)
        pearson_corr, pvalue = stats.pearsonr(exec_test_target, predicted)
        spearmanr, spr_pvalue = stats.spearmanr(exec_test_target, predicted)

        output.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
            execution, learner_name, rmse, r2, pearson_corr, spearmanr))
        print("{}     {:20s}  {:.3f}  {:.3f}  {:.3f}/{:.5f}  {:.3f}/{:.5f}".format(
            execution, learner_name, rmse, r2, pearson_corr, pvalue, spearmanr, spr_pvalue))

output.close()
print("\n\nOutput written to: {}".format(OUTPUT_CSV_FILE_NAME))
