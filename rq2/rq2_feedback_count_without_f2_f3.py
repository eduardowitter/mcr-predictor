#!/usr/bin/python3 -u
'''
Research Question 2 : test predictions without features removed after RFECV
'''

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from imblearn.pipeline import Pipeline
from scipy import stats
from dataset import Dataset

####################################################################################################
#                                                                                                  #
# Configuration for Research Questions 2 test:                                                     #
# - Path to file with all-time dataset in CSV format                                               #
# - Name of training and test sets used                                                            #
# - Target Name:  Target-FeedbackCount                                                             #
# - File where results are written                                                                 #
#                                                                                                  #
# In this Research Question, we evaluate how the prediction for Target-FeedbackCount is            #
# affected by removing features with low rankings in RFECV.                                        #
#                                                                                                  #
####################################################################################################

DATASET_CSV_PATH = 'raw_dataset.csv'

OUTPUT_CSV_FILE_NAME = 'rq2_feedback_count_without_f2_f3_results.csv'

TRAINING_SET_NAME = 'training-Y1'

TEST_SET_NAME = 'test-T1a'

TARGET_COLUMN_NAME = 'Target-FeedbackCount'

EXECUTIONS_FEATURES = ['F1-ChangedLOC', 'F4-IsMaintainer',
                       'F5-ModuleReviewXp', 'F6-ModuleModifXp', 'F7-TeamReviewXp',
                       'F8-TeamModifXp', 'F9-FileReviewXp', 'F10-FileModifXp',
                       'F11-OngoingAsAuthor', 'F12-OngoingAsReviewer']

EXECUTION_NAME = 'RQ2-FeedbackCount-NoF2F3'


####################################################################################################
#                                                                                                  #
# Run Research Question 1 tests.                                                                   #
# Results are written to both terminal and output CSV file.                                        #
#                                                                                                  #
####################################################################################################

output = open(OUTPUT_CSV_FILE_NAME, 'w+')
csv_header = "Execution,Model,RMSE,R2,Pearson Correlation"
output.write(csv_header + "\n")


# Load training and test sets
dataset = Dataset(DATASET_CSV_PATH)
training_df, test_df = dataset.get_training_and_test_by_name(
    TRAINING_SET_NAME, TEST_SET_NAME, only_with_participation=True)


print('\n\n' + '=' * 100)
print('{}: {}'.format(EXECUTION_NAME, ', '.join(EXECUTIONS_FEATURES)))

# traning and test sets
exec_training_features = training_df.reindex(columns=EXECUTIONS_FEATURES)
exec_training_target = training_df[TARGET_COLUMN_NAME]
exec_test_features = test_df.reindex(columns=EXECUTIONS_FEATURES)
exec_test_target = test_df[TARGET_COLUMN_NAME]

base_learners = {
    'kNN': KNeighborsRegressor(n_neighbors=100, metric='minkowski', p=1, weights='uniform', algorithm='auto'),
    'Linear Regression': Ridge(alpha=10, solver='auto'),
    'Neural Network': MLPRegressor(random_state=0, activation='relu', solver='adam',
                                   alpha=0.0001, max_iter=200),
    'Random Forest': RandomForestRegressor(random_state=0, n_estimators=150, min_samples_leaf=10, criterion='mse', max_features=None, max_depth=None),
}

print("Execution               Model                 RMSE   R2     Pearson Correlation / p-value Spearman Correlation / p-value")
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
        EXECUTION_NAME, learner_name, rmse, r2, pearson_corr, spearmanr))
    print("{}     {:20s}  {:.3f}  {:.3f}  {:.3f}/{:.5f}  {:.3f}/{:.5f}".format(
        EXECUTION_NAME, learner_name, rmse, r2, pearson_corr, pvalue, spearmanr, spr_pvalue))


output.close()
print("\n\nOutput written to: {}".format(OUTPUT_CSV_FILE_NAME))
