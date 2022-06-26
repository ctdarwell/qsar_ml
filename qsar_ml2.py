import urllib.request, os
import zipfile
import pandas as pd
import lazypredict
import matplotlib.pyplot as plt
import seaborn as sns

#Calculate feature for both positive and negative classes + combines the two classes + merge with class labels
pos = 'train_po_cdhit.txt'
neg = 'train_ne_cdhit.txt'

aac_pos_neg = pd.read_csv(f"aac_{pos.replace('.txt','')}_{neg.replace('.txt','')}.csv", header=0)
dpc_pos_neg = pd.read_csv(f"dpc_{pos.replace('.txt','')}_{neg.replace('.txt','')}.csv", header=0)


#1st work on aac
aac_X = aac_pos_neg.drop('class', axis=1)
aac_y = aac_pos_neg['class'].copy()
aac_y = aac_y.map({"positive": 1, "negative": 0}) 

dpc_X = dpc_pos_neg.drop('class', axis=1)
dpc_y = dpc_pos_neg['class'].copy()
dpc_y = dpc_y.map({"positive": 1, "negative": 0})


# Feature selection (Variance threshold)
from sklearn.feature_selection import VarianceThreshold

fs = VarianceThreshold(threshold=0.1)
fs.fit_transform(aac_X)
#X2.shape
aac_X2 = aac_X.loc[:, fs.get_support()]
#print(aac_X2)

# Data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(aac_X2, aac_y, test_size=0.2, random_state=42, stratify=aac_y)


# Import libraries
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

# Load dataset
#X = feature.drop('class', axis=1)
#y = feature['class'].copy()

# Data split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42, stratify=y)

# Defines and builds the lazyclassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=matthews_corrcoef)
models_train, predictions_train = clf.fit(X_train, X_train, y_train, y_train)
models_test, predictions_test = clf.fit(X_train, X_test, y_train, y_test)

models_train.to_csv('models_train.csv', index=False)


# Plot of Accuracy
plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=models_train.index, x="Accuracy", data=models_train)
ax.set(xlim=(0, 1))
plt.savefig('Accuracy.pdf')

# Plot of MCC
plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=models_train.index, x="matthews_corrcoef", data=models_train)
ax.set(xlim=(0, 1))
plt.savefig('matthews_corrcoef.pdf')


#Random Forest - looks best

# Build random forest model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)


# Simplest and quickest way to obtain the model performance (Accuracy)
rf.score(X_test,y_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

# Matthew Correlation Coefficient
from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_test_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)


# Classification report
from sklearn.metrics import classification_report
model_report = classification_report(y_train, y_train_pred, target_names=['positive','negative'])
f = open('model_report.txt','w')
f.writelines(model_report) 
f.close()

# ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

plot_roc_curve(rf, X_test, y_test)  
plt.savefig('roc_curve_test.pdf')

plot_roc_curve(rf, X_train, y_train)  
plt.savefig('roc_curve_train.pdf')


#Feature importance
# Retrieve feature importance from the RF model
importance = pd.Series(rf.feature_importances_, name = 'Gini')

# Retrieve feature names
feature_names = pd.Series(aac_X2.columns, name = 'Feature')


# Combine feature names and Gini values into a Dataframe
df = pd.concat([feature_names, importance], axis=1, names=['Feature', 'Gini'])


# Plot of feature importance
import matplotlib.pyplot as plt
import seaborn as sns

df_sorted = df.sort_values('Gini', ascending=False)[:20] # Sort by Gini in descending order; Showing only the top 20 results

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x = 'Gini', y = 'Feature', data = df_sorted)
plt.xlabel("Feature Importance")
plt.savefig('feat_imp.pdf')

print("*** NOW REPEAT FOR DPC ***")
