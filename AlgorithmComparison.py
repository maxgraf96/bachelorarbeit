# Compare Algorithms
import glob

import joblib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import tqdm

# load dataset
import util

X = joblib.load(util.ext_hdd_path + "data/mfcc_trn.joblib")
Y = joblib.load(util.ext_hdd_path + "data/mfcc_lbls.joblib")

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = [
    ('LR', LogisticRegression()),
    ('SGD', SGDClassifier()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB())
]

names = []
# evaluate each model in turn
if len(glob.glob("alg_comparison.joblib")) < 1:
    results = []
    scoring = 'accuracy'
    for name, model in models:
        print("Processing model ", name)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("Processing done")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    joblib.dump(results, "alg_comparison.joblib")

else:
    for name, model in models:
        names.append(name)
    results = joblib.load("alg_comparison.joblib")
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.xlabel("Classifiers")
plt.ylabel("Mean accuracy")
plt.show()

print(results)
