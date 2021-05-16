# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/01/27        #
# # # # # # # # # # # # # # # # #

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
from termcolor import colored
from prettytable import PrettyTable
import os

Dataset = 'Datasets/CrisisLexT26/Similarity_TF-IDF_Top10.csv'
df = pd.read_csv(Dataset, encoding='utf-8', lineterminator='\n')

# encoding labels automatically
df['Label'] = df['Label'].astype('category')
df['Label'] = df['Label'].cat.codes

encoding = {
    'building_collapse': 0,
    'crash': 1,
    'earthquake': 2,
    'explosion': 3,
    'flood': 4,
    'haze': 5,
    'meteor': 6,
    'shoot': 7,
    'typhoon': 8,
    'wildfire': 9,
}

dataset = {
    'building_collapse': df[df.Label == encoding['building_collapse']],
    'crash': df[df.Label == encoding['crash']],
    'earthquake': df[df.Label == encoding['earthquake']],
    'explosion': df[df.Label == encoding['explosion']],
    'flood': df[df.Label == encoding['flood']],
    'haze': df[df.Label == encoding['haze']],
    'meteor': df[df.Label == encoding['meteor']],
    'shoot': df[df.Label == encoding['shoot']],
    'typhoon': df[df.Label == encoding['typhoon']],
    'wildfire': df[df.Label == encoding['wildfire']],
}


def evaluation(model, X_test, y_test, classifier_name, subClassifier_name=''):
    accuracy = model.score(X_test, y_test)
    if subClassifier_name:
        model_name = classifier_name + ' ' + subClassifier_name
    else:
        model_name = classifier_name
    print(colored('Accuracy ' + model_name + ':', 'cyan'), accuracy)
    # precision recall fscore
    predicted = model.predict(X_test)
    precision, recall, fscore, support = score(y_test, predicted)
    x = PrettyTable()
    # x.field_names = [model_name, "Precision", "Recall", "FScore", "Support"]
    x.field_names = [model_name, "Precision", "Recall", "FScore"]
    for index, key in enumerate(dataset):
        # x.add_row([key, str(int(precision[index] * 10000) / 100) + '%', str(int(recall[index] * 10000) / 100) + '%',
        #            str(int(fscore[index] * 10000) / 100) + '%', support[index]])
        x.add_row([key, str(int(precision[index] * 10000) / 100) + '%', str(int(recall[index] * 10000) / 100) + '%',
                   str(int(fscore[index] * 10000) / 100) + '%'])
    print(x)
    x.add_row(
        ['average', str(int(np.mean(precision) * 10000) / 100) + '%', str(int(np.mean(recall) * 10000) / 100) + '%',
         str(int(np.mean(fscore) * 10000) / 100) + '%'])
    print("\n".join(x.get_string().splitlines()[-2:]))
    return accuracy, precision, recall, fscore, support


def save_model(model, accuracy, precision, recall, fscore, support, classifier_name, subClassifier_name=''):
    path = 'Datasets/CrisisLexT26/Models/'
    MYDIR = path + classifier_name
    # If folder doesn't exist, then create it.
    if not os.path.isdir(MYDIR):
        os.makedirs(MYDIR)
        print(colored('created folder:', 'yellow'), MYDIR)
    if subClassifier_name:
        # filename = subClassifier_name + '_' + str(int(accuracy * 10000) / 100) + '.model'
        filename = subClassifier_name + '.model'
        accuracy_filename = subClassifier_name + '.txt'
        model_name = classifier_name + ' ' + subClassifier_name
    else:
        # filename = classifier_name + '_' + str(int(accuracy * 10000) / 100) + '.model'
        filename = classifier_name + '.model'
        accuracy_filename = classifier_name + '.txt'
        model_name = classifier_name
    pickle.dump(model, open(MYDIR + '/' + filename, 'wb'))
    print(colored('model saved:', 'green'), filename, end='\n\n\n')

    f = open(MYDIR + '/' + accuracy_filename, "w")
    f.write('Accuracy ' + model_name + ': ' + str(accuracy) + '\n')
    x = PrettyTable()
    # x.field_names = [model_name, "Precision", "Recall", "FScore", "Support"]
    x.field_names = [model_name, "Precision", "Recall", "FScore"]
    for index, key in enumerate(dataset):
        x.add_row([key, str(int(precision[index] * 10000) / 100) + '%', str(int(recall[index] * 10000) / 100) + '%',
                   str(int(fscore[index] * 10000) / 100) + '%'])
    f.write(str(x) + '\n')
    x.add_row(
        ['average', str(int(np.mean(precision) * 10000) / 100) + '%', str(int(np.mean(recall) * 10000) / 100) + '%',
         str(int(np.mean(fscore) * 10000) / 100) + '%'])
    f.write("\n".join(x.get_string().splitlines()[-2:]) + '\n')
    f.close()


# shuffle & slice
# dataset = {
#     'building_collapse': dataset['building_collapse'].sample(frac=1)[:1000],
#     'crash': dataset['crash'].sample(frac=1)[:1000],
#     'earthquake': dataset['earthquake'].sample(frac=1)[:1000],
#     'explosion': dataset['explosion'].sample(frac=1)[:1000],
#     'flood': dataset['flood'].sample(frac=1)[:1000],
#     'haze': dataset['haze'].sample(frac=1)[:1000],
#     'meteor': dataset['meteor'].sample(frac=1)[:1000],
#     'shoot': dataset['shoot'].sample(frac=1)[:1000],
#     'typhoon': dataset['typhoon'].sample(frac=1)[:1000],
#     'wildfire': dataset['wildfire'].sample(frac=1)[:1000],
# }

X_train = pd.DataFrame([])
y_train = pd.DataFrame([])
X_test = pd.DataFrame([])
y_test = pd.DataFrame([])
X_all = pd.DataFrame([])
y_all = pd.DataFrame([])

# train 80%, test 20%
# for key in dataset:
#     print(colored(key, 'blue'), ':', len(dataset[key]))
#     percent_80 = round(len(dataset[key]) * .80)
#     X_train = pd.concat([X_train, dataset[key].iloc[:percent_80, :10]], ignore_index=True)
#     y_train = pd.concat([y_train, dataset[key].iloc[:percent_80, 10]], ignore_index=True)
#     X_test = pd.concat([X_test, dataset[key].iloc[percent_80:, :10]], ignore_index=True)
#     y_test = pd.concat([y_test, dataset[key].iloc[percent_80:, 10]], ignore_index=True)
#
# # reshape column vector to row vector
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

# K-fold cross validation
for key in dataset:
    print(colored(key, 'blue'), ':', len(dataset[key]))
    X_all = pd.concat([X_all, dataset[key].iloc[:, :10]], ignore_index=True)
    y_all = pd.concat([y_all, dataset[key].iloc[:, 10]], ignore_index=True)

y_all = y_all.values.ravel()

kf10 = KFold(n_splits=10, shuffle=True)
kfold_accuracy = np.zeros((11, 10))
kfold_precision = np.zeros((11, 10, 10))  # 11 classifier, 10 fold, 10 class
kfold_recall = np.zeros((11, 10, 10))
kfold_fscore = np.zeros((11, 10, 10))
fold_num = 0
all_test_index = []
for train_index, test_index in kf10.split(X_all):
    print(colored('Fold:', 'cyan'), fold_num)
    all_test_index += [test_index]

    X_train = X_all.iloc[train_index]
    y_train = np.take(y_all, train_index)
    X_test = X_all.iloc[test_index]
    y_test = np.take(y_all, test_index)

    # SVM
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(linear, X_test, y_test, 'SVM', 'Linear')
    kfold_accuracy[0, fold_num] = accuracy
    kfold_precision[0, fold_num] = precision
    kfold_recall[0, fold_num] = recall
    kfold_fscore[0, fold_num] = fscore

    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(rbf, X_test, y_test, 'SVM', 'RBF')
    kfold_accuracy[1, fold_num] = accuracy
    kfold_precision[1, fold_num] = precision
    kfold_recall[1, fold_num] = recall
    kfold_fscore[1, fold_num] = fscore

    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(poly, X_test, y_test, 'SVM', 'Polynomial')
    kfold_accuracy[2, fold_num] = accuracy
    kfold_precision[2, fold_num] = precision
    kfold_recall[2, fold_num] = recall
    kfold_fscore[2, fold_num] = fscore

    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(sig, X_test, y_test, 'SVM', 'Sigmoid')
    kfold_accuracy[3, fold_num] = accuracy
    kfold_precision[3, fold_num] = precision
    kfold_recall[3, fold_num] = recall
    kfold_fscore[3, fold_num] = fscore

    # KNN
    # Calculating error for K values between 1 and 40
    error = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
    n_neighbors = error.index(min(error)) + 1  # = min index
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(KNN, X_test, y_test, 'KNN',
                                                              str(n_neighbors) + '_neighbors')
    kfold_accuracy[4, fold_num] = accuracy
    kfold_precision[4, fold_num] = precision
    kfold_recall[4, fold_num] = recall
    kfold_fscore[4, fold_num] = fscore
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    # Naive Bayes
    GNB = GaussianNB().fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(GNB, X_test, y_test, 'Naive_Bayes', 'Gaussian')
    kfold_accuracy[5, fold_num] = accuracy
    kfold_precision[5, fold_num] = precision
    kfold_recall[5, fold_num] = recall
    kfold_fscore[5, fold_num] = fscore
    # MNB = MultinomialNB().fit(X_train, y_train)
    # accuracy = evaluation(MNB, X_test, y_test, 'Naive_Bayes', 'Multinomial')

    # Decision Tree
    DTC = DecisionTreeClassifier().fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(DTC, X_test, y_test, 'DecisionTree')
    kfold_accuracy[6, fold_num] = accuracy
    kfold_precision[6, fold_num] = precision
    kfold_recall[6, fold_num] = recall
    kfold_fscore[6, fold_num] = fscore

    # Random Forest
    regressor = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(regressor, X_test, y_test, 'RandomForest')
    kfold_accuracy[7, fold_num] = accuracy
    kfold_precision[7, fold_num] = precision
    kfold_recall[7, fold_num] = recall
    kfold_fscore[7, fold_num] = fscore

    # Gradient Boosting
    # gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2,
    #                                     random_state=0).fit(X_train, y_train)
    # accuracy = evaluation(gb_clf, X_test, y_test, 'GradientBoosting')
    # save_model(gb_clf, accuracy, 'GradientBoosting')
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    error = []
    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2,
                                            random_state=0).fit(X_train, y_train)
        pred_i = gb_clf.predict(X_test)
        error.append(np.mean(pred_i != y_test))
    learning_rate = error.index(min(error))  # = min index
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=lr_list[learning_rate], max_features=2,
                                        max_depth=2,
                                        random_state=0).fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(gb_clf, X_test, y_test, 'GradientBoosting',
                                                              'LearningRate_' + str(lr_list[learning_rate]))
    kfold_accuracy[8, fold_num] = accuracy
    kfold_precision[8, fold_num] = precision
    kfold_recall[8, fold_num] = recall
    kfold_fscore[8, fold_num] = fscore

    # XGBoost
    xgb_clf = XGBClassifier().fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(xgb_clf, X_test, y_test, 'XGBoost')
    kfold_accuracy[9, fold_num] = accuracy
    kfold_precision[9, fold_num] = precision
    kfold_recall[9, fold_num] = recall
    kfold_fscore[9, fold_num] = fscore

    # MLP
    MLP = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", random_state=1).fit(X_train, y_train)
    accuracy, precision, recall, fscore, support = evaluation(MLP, X_test, y_test, 'MLP')
    kfold_accuracy[10, fold_num] = accuracy
    kfold_precision[10, fold_num] = precision
    kfold_recall[10, fold_num] = recall
    kfold_fscore[10, fold_num] = fscore

    fold_num += 1

x = PrettyTable()
x.field_names = ['classifier', 'Accuracy', 'Precision', 'Recall', 'FScore']
x.add_row([colored('SVM Linear:', 'magenta'), str(np.mean(kfold_accuracy[0])), str(np.mean(kfold_precision[0])),
           str(np.mean(kfold_recall[0])), str(np.mean(kfold_fscore[0]))])
x.add_row([colored('SVM RBF:', 'magenta'), str(np.mean(kfold_accuracy[1])), str(np.mean(kfold_precision[1])),
           str(np.mean(kfold_recall[1])), str(np.mean(kfold_fscore[1]))])
x.add_row([colored('SVM Polynomial:', 'magenta'), str(np.mean(kfold_accuracy[2])), str(np.mean(kfold_precision[2])),
           str(np.mean(kfold_recall[2])), str(np.mean(kfold_fscore[2]))])
x.add_row([colored('SVM Sigmoid:', 'magenta'), str(np.mean(kfold_accuracy[3])), str(np.mean(kfold_precision[3])),
           str(np.mean(kfold_recall[3])), str(np.mean(kfold_fscore[3]))])
x.add_row([colored('KNN:', 'magenta'), str(np.mean(kfold_accuracy[4])), str(np.mean(kfold_precision[4])),
           str(np.mean(kfold_recall[4])), str(np.mean(kfold_fscore[4]))])
x.add_row(
    [colored('Naive Bayes Gaussian:', 'magenta'), str(np.mean(kfold_accuracy[5])), str(np.mean(kfold_precision[5])),
     str(np.mean(kfold_recall[5])), str(np.mean(kfold_fscore[5]))])
x.add_row([colored('Decision Tree:', 'magenta'), str(np.mean(kfold_accuracy[6])), str(np.mean(kfold_precision[6])),
           str(np.mean(kfold_recall[6])), str(np.mean(kfold_fscore[6]))])
x.add_row([colored('Random Forest:', 'magenta'), str(np.mean(kfold_accuracy[7])), str(np.mean(kfold_precision[7])),
           str(np.mean(kfold_recall[7])), str(np.mean(kfold_fscore[7]))])
x.add_row([colored('Gradient Boosting:', 'magenta'), str(np.mean(kfold_accuracy[8])), str(np.mean(kfold_precision[8])),
           str(np.mean(kfold_recall[8])), str(np.mean(kfold_fscore[8]))])
x.add_row([colored('XGBoost:', 'magenta'), str(np.mean(kfold_accuracy[9])), str(np.mean(kfold_precision[9])),
           str(np.mean(kfold_recall[9])), str(np.mean(kfold_fscore[9]))])
x.add_row([colored('MLP:', 'magenta'), str(np.mean(kfold_accuracy[10])), str(np.mean(kfold_precision[10])),
           str(np.mean(kfold_recall[10])), str(np.mean(kfold_fscore[10]))])
print(x, end='\n')

model_names = ['SVM Linear', 'SVM RBF', 'SVM Polynomial', 'SVM Sigmoid', 'KNN', 'Naive Bayes Gaussian', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'MLP']
for idx, model_name in enumerate(model_names):
    print(colored(model_name + ':', 'cyan'))
    x = PrettyTable()
    x.field_names = [model_name, 'Precision', 'Recall', 'FScore']
    for index, key in enumerate(dataset):

        x.add_row([key, str(int(np.mean(kfold_precision[idx, :, index]) * 10000) / 100) + '%', str(int(np.mean(kfold_recall[idx, :, index]) * 10000) / 100) + '%',
                   str(int(np.mean(kfold_fscore[idx, :, index]) * 10000) / 100) + '%'])
    print(x)

all_test_index = pd.DataFrame(all_test_index)
all_test_index.to_csv('test_index_top_10.csv', index=False)

print(colored('Done!', 'green'))
