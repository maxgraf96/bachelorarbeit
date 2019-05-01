# Copyright (c) 2019 Max Graf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Compare Algorithms
import glob

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

import util

# Load dataset
from features import MFCC

if len(glob.glob(util.data_path + "mfcc_trn_gtzan.joblib")) < 1:
    X, Y = MFCC.calculate_mfccs(util.data_path + "speech/gtzan", util.data_path + "music/gtzan", max_duration=1800)
    joblib.dump(X, util.data_path + "mfcc_trn_gtzan.joblib")
    joblib.dump(X, util.data_path + "mfcc_lbls_gtzan.joblib")
else:
    X = joblib.load(util.data_path + "mfcc_trn_gtzan.joblib")
    Y = joblib.load(util.data_path + "mfcc_lbls_gtzan.joblib")

# Prepare configuration for cross validation
seed = 7

# Prepare models
models = [
    ('LR', LogisticRegression()),
    ('SGD', SGDClassifier()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('PC', Perceptron()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=5))
]

names = []
# evaluate each model in turn
test_scores = []
if len(glob.glob("alg_comparison.joblib")) < 1:
    results = []
    scoring = 'accuracy'
    for name, model in models:
        print("Processing model ", name)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_validate(model, X, Y, cv=kfold, scoring=scoring)
        print("Processing done")
        results.append(cv_results)
        current_score = cv_results["test_score"]
        names.append(name)
        msg = "%s: %f (%f)" % (name, current_score.mean(), current_score.std())
        print(msg)
        test_scores.append(current_score)

    joblib.dump(results, "alg_comparison.joblib")

else:
    for name, model in models:
        names.append(name)
    results = joblib.load("alg_comparison.joblib")
    for i in range(len(results)):
        result = results[i]
        current_score = result["test_score"]

        print(names[i])
        print("Average test score:", round(current_score.mean(), 3), "std: ", round(current_score.std(), 3))
        print("Average fitting time (per set):", round(np.mean(result["fit_time"]), 2))
        print("Average score time (per set):", round(np.mean(result["score_time"]), 2))
        print()
        test_scores.append(current_score)

# Boxplot algorithm comparison
fig = plt.figure()
plt.title('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.grid(True)
plt.boxplot(test_scores)
ax.set_xticklabels(names)
plt.xlabel("Classifier")
plt.ylabel("Mean accuracy")
# plt.show()
plt.savefig("alg_comparison.png")