''' Below code is to apply multiple sklearn algorithms .
    Some individual trials are done before, whose results are not reported
    Grid Search SVC section performs the Analysis with SVC classifier
    Grid Search Decision Tree section performs the Analysis with Decision Tree classifier
    Grid Search Random Forest section performs the Analysis with Random Forest classifier
    Gaussian Naive Bayes section implements the sklearn Gaussian Naive Bayes learner'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree.export import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import pickle
import timeit
import numpy as np
import json
from sklearn.metrics import classification_report


''' Change the source path and use the tf_idf_master.pkl downloading from google storage here'''

source="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\tfidf_master.pkl"
df=pd.read_pickle(source)
df=df.drop(df.columns[0], axis=1)
# df_spam=df[df['Spam-Label']==1]
# df_nspam=df[df['Spam-Label']==0]

X = df.drop(['Spam-Label','Document'],axis=1)
Y = df['Spam-Label']
X_train, X_test, y_train, y_test = train_test_split(X, Y,stratify=Y,test_size=0.20)
y_test1=y_test.to_frame()
y_test1.reset_index(inplace=True) # Resets the index, makes factor a column
y_test1.drop("index",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
y_train1=y_train.to_frame()
y_train1.reset_index(inplace=True) # Resets the index, makes factor a column
y_train1.drop("index",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True

########### Single Decision Tree Classifier - Results not reported for this one#################

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

fn=list(df.drop(['Spam-Label','Document'],axis=1))
cn=["Not Spam",'Spam']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=220,)


tree.export_graphviz(dt,
                     out_file="tree.dot",
                     feature_names = fn,class_names=cn,
                     filled = True)
# Run the below command in cmd line in the working directory where tree.dot got created to get the final image
# dot -Tpng tree.dot -o tree.png

#On test data
y_pred =dt.predict(X_test)
prediction=pd.DataFrame(y_pred,columns=["Predicted"])
confusion_matrix(y_test1['Spam-Label'],prediction['Predicted'])
pd.crosstab(y_test1['Spam-Label'],prediction['Predicted'], rownames=['True'], colnames=['Predicted'])
accuracy_score(y_test1['Spam-Label'],prediction['Predicted'],normalize=True)

#On train data

y_pred =dt.predict(X_train)
prediction=pd.DataFrame(y_pred,columns=["Predicted"])

confusion_matrix(y_train1['Spam-Label'],prediction['Predicted'])
accuracy_score(y_train1['Spam-Label'],prediction['Predicted'],normalize=True)
prediction['Actual']= y_train1
prediction.to_csv("C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\prediction.csv",sep=',',index=False)

################# Non Bootstrapped Random Forest Classifier----Results not reported for this one#################

start=timeit.timeit()
acc={}
depths=[1,5,9,13,17,21,25,30,35,40,45,50]

for d in depths:
    model = RandomForestClassifier(n_estimators=500,
                                   bootstrap = True,max_depth=d,
                                   max_features = 'sqrt')
    model.fit(X_train, y_train)
    # Actual class predictions
    rf_predictions = model.predict(X_test)
    # Probabilities for each class
    rf_probs = model.predict_proba(X_test)[:, 1]

    prediction=pd.DataFrame(rf_predictions,columns=["Predicted"])

    pd.crosstab(y_test1['Spam-Label'],prediction['Predicted'], rownames=['True'], colnames=['Predicted'])
    # cm.iloc[0,0]
    confusion_matrix(y_test1['Spam-Label'], prediction['Predicted'])
    acc[d]=accuracy_score(y_test1['Spam-Label'],prediction['Predicted'],normalize=True)
accuracy_tab = pd.DataFrame(list(acc.items()),columns = ['depth','accuracy'])
end = timeit.timeit()
print(end - start)

################# Single SVC classifier , results not reported based on this############################################
start=timeit.timeit()

from sklearn.svm import SVC
clf = SVC(gamma=10.0,kernel='rbf',C=10.0)
clf.fit(X_train, y_train)
SVC_pred=clf.predict(X_test)
prediction=pd.DataFrame(SVC_pred,columns=["Predicted"])
confusion_matrix(y_test1['Spam-Label'],prediction['Predicted'])
pd.crosstab(y_test1['Spam-Label'],prediction['Predicted'], rownames=['True'], colnames=['Predicted'])
acc=accuracy_score(y_test1['Spam-Label'],prediction['Predicted'],normalize=True)
acc
from sklearn.metrics import f1_score

f1_score(y_test1['Spam-Label'], prediction['Predicted'], average='macro')


################# GRID SEARCH WITH SVC - These results are reported ###################################################
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
import time
start_time = time.time()

bg_svc=BaggingClassifier(SVC(),n_estimators = 20, max_features = 0.5,n_jobs=-1)
_param= {'base_estimator__kernel':('linear', 'poly'), 'base_estimator__C':[0.1,1,10,100]}
_scoring = ['accuracy', 'roc_auc']
clf_svc = GridSearchCV(estimator=bg_svc,param_grid=_param, scoring =_scoring,refit='roc_auc')
clf_svc.fit(X_train, y_train)
results =pd.DataFrame (clf_svc.cv_results_)
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\output_SVC.csv"
results.to_csv(out)
print("--- %s seconds ---" % (time.time() - start_time))
""" Based on the grid search output results , the based paramters considering both AUC-ROC score and Accuracy Ranking
    are following :   
    C=10,
    kernel=linear,
Using these hyperparameters iterating over different sample sizes
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

classifier=['_SVC']
metric=['accuracy','auc_roc','f1-score']
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\{}.csv".format(classifier[0])
out1 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\accuracy.csv"
out2 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\f1.csv"
out3 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\auc.csv"

test_sizes=[0.1,0.15,0.2,0.25]
accuracy={}
f1={}
auc={}
cf={}
max_features=[0.5,0.75,1.0]
for t in test_sizes:
    for i in max_features:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=t)
        y_test1 = y_test.to_frame()
        y_test1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_test1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        y_train1 = y_train.to_frame()
        y_train1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_train1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True

        bg_svc = BaggingClassifier(SVC(C=10,kernel='linear'), n_estimators=20, max_features=i, n_jobs=-1)
        bg_svc.fit(X_train, y_train)
        y_train_pred=bg_svc.predict(X_train)
        prediction_trn=pd.DataFrame(y_train_pred,columns=["Predicted"])
        # on training data
        # cf[(i,t,'train')]= pd.crosstab(y_train1['Spam-Label'],prediction_trn['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'train')] = accuracy_score(y_train1['Spam-Label'],prediction_trn['Predicted'],normalize=True)
        f1[(i,t, 'train')]=f1_score(y_train1['Spam-Label'],prediction_trn['Predicted'])
        auc[(i,t,'train')] = roc_auc_score(y_train1['Spam-Label'],prediction_trn['Predicted'],average='macro')

        #on test data
        y_test_pred = bg_svc.predict(X_test)
        prediction_tst=pd.DataFrame(y_test_pred,columns=["Predicted"])
        # cf[(i,t,'test')]= pd.crosstab(y_test1['Spam-Label'],prediction_tst['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'test')] =accuracy_score(y_test1['Spam-Label'],prediction_tst['Predicted'],normalize=True)
        f1[(i,t,'test')]=f1_score(y_test1['Spam-Label'],prediction_tst['Predicted'])
        auc[(i,t,'test')] = roc_auc_score(y_test1['Spam-Label'],prediction_tst['Predicted'],average='macro')
        accuracy1 = pd.DataFrame([(*k, v) for k, v in accuracy.items()])
        accuracy1['Model'] = classifier[0]
        accuracy1['Metric'] = metric[0]
        accuracy1.to_csv(out1)

        auc1 = pd.DataFrame([(*k, v) for k, v in auc.items()])
        auc1['Model'] = classifier[0]
        auc1['Metric'] = metric[1]
        auc1.to_csv(out2)

        f1_ = pd.DataFrame([(*k, v) for k, v in f1.items()])
        f1_['Model'] = classifier[0]
        f1_['Metric'] = metric[2]
        f1_.to_csv(out3)

        print("done with feature {} and test size {}".format(i,t))

# accuracy
# accuracy1=pd.DataFrame([(*k, v) for k, v in accuracy.items()])
# accuracy1['Model']=classifier[0]
# accuracy1['Metric']=metric[0]
# accuracy1.to_csv(out,mode='a')
# auc1=pd.DataFrame([(*k, v) for k, v in auc.items()],columns=['Estimators','test-size','sample','value'])
# auc1['Model']=classifier[0]
# auc1['Metric']=metric[1]
# auc1.to_csv(out,mode='a')
# f1_=pd.DataFrame([(*k, v) for k, v in f1.items()],columns=['Estimators','test-size','sample','value'])
# f1_['Model']=classifier[0]
# f1_['Metric']=metric[2]
# f1_.to_csv(out,mode='a')

print(classification_report(y_test1['Spam-Label'],prediction['Predicted']))






################# GRID SEARCH WITH DECISION TREE ####################################################
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

import time
start_time = time.time()

bg_DT=BaggingClassifier(DecisionTreeClassifier(),n_estimators = 20, max_features = 0.5,n_jobs=-1)

_param= {'base_estimator__max_depth':[5, 10, 15, 20, 25,30,40,50], 'base_estimator__max_features':('sqrt', 'log2')}
_scoring = ['accuracy', 'roc_auc']
clf_dt = GridSearchCV(estimator=bg_DT,param_grid=_param, scoring =_scoring,refit='roc_auc')
clf_dt.fit(X_train, y_train)
best_tree=clf_dt.best_estimator_
# best_tree.
fn=list(df.drop(['Spam-Label','Document'],axis=1))
cn=["Not Spam",'Spam']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=220,)

tree.export_graphviz(best_tree,
                     out_file="tree.dot",
                     feature_names = fn,class_names=cn,
                     filled = True)

# Run the below command in cmd line in the working directory where tree.dot got created to get the final image
# dot -Tpng best_tree.dot -o btree.png

results_DT =pd.DataFrame (clf_dt.cv_results_)
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\output_DT.csv"
results_DT.to_csv(out)
print("--- %s seconds ---" % (time.time() - start_time))

""" Based on the grid search output results , the based paramters considering both AUC-ROC score and Accuracy Ranking
    are following :   
    max_depth=25,
    max_features=sqrt,
Using these hyperparameters iterating over different sample sizes
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

classifier=['_DT']
metric=['accuracy','auc_roc','f1-score']
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\{}.csv".format(classifier[0])
out1 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\dt_accuracy.csv"
out2 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\dt_f1.csv"
out3 = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\dt_auc.csv"

test_sizes=[0.1,0.15,0.2,0.25]
accuracy={}
f1={}
auc={}
cf={}
max_features=[0.5,0.75,1.0]
for t in test_sizes:
    for i in max_features:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=t)
        y_test1 = y_test.to_frame()
        y_test1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_test1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        y_train1 = y_train.to_frame()
        y_train1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_train1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True

        bg_DT= BaggingClassifier(DecisionTreeClassifier(max_depth=25,max_features='sqrt'), n_estimators=20, max_features=i, n_jobs=-1)
        bg_DT.fit(X_train, y_train)
        y_train_pred=bg_DT.predict(X_train)
        prediction_trn=pd.DataFrame(y_train_pred,columns=["Predicted"])
        # on training data
        # cf[(i,t,'train')]= pd.crosstab(y_train1['Spam-Label'],prediction_trn['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'train')] = accuracy_score(y_train1['Spam-Label'],prediction_trn['Predicted'],normalize=True)
        f1[(i,t, 'train')]=f1_score(y_train1['Spam-Label'],prediction_trn['Predicted'])
        auc[(i,t,'train')] = roc_auc_score(y_train1['Spam-Label'],prediction_trn['Predicted'],average='macro')
        cf[(i, t, 'train')] = pd.crosstab(y_train1['Spam-Label'], prediction_trn['Predicted'], rownames=['True'],
                                          colnames=['Predicted'])


        #on test data
        y_test_pred = bg_DT.predict(X_test)
        prediction_tst=pd.DataFrame(y_test_pred,columns=["Predicted"])
        # cf[(i,t,'test')]= pd.crosstab(y_test1['Spam-Label'],prediction_tst['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'test')] =accuracy_score(y_test1['Spam-Label'],prediction_tst['Predicted'],normalize=True)
        f1[(i,t,'test')]=f1_score(y_test1['Spam-Label'],prediction_tst['Predicted'])
        auc[(i,t,'test')] = roc_auc_score(y_test1['Spam-Label'],prediction_tst['Predicted'],average='macro')
        accuracy1 = pd.DataFrame([(*k, v) for k, v in accuracy.items()])
        accuracy1['Model'] = classifier[0]
        accuracy1['Metric'] = metric[0]
        accuracy1.to_csv(out1)

        auc1 = pd.DataFrame([(*k, v) for k, v in auc.items()])
        auc1['Model'] = classifier[0]
        auc1['Metric'] = metric[1]
        auc1.to_csv(out2)

        f1_ = pd.DataFrame([(*k, v) for k, v in f1.items()])
        f1_['Model'] = classifier[0]
        f1_['Metric'] = metric[2]
        f1_.to_csv(out3)
        cf[(i, t, 'test')] = pd.crosstab(y_test1['Spam-Label'], prediction_tst['Predicted'], rownames=['True'],
                                         colnames=['Predicted'])

        print("done with feature {} and test size {}".format(i,t))

# cf_out = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\dt_cf.csv"
cf.keys()
cf[(1.0, 0.2, 'test')]



################# GRID SEARCH WITH RANDOM FOREST - Results are reported ################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import time
start_time = time.time()
_param= {'max_depth':[5,10,20,30,40,50,75], 'n_estimators':[50,100,250,500,1000],'max_features':('sqrt', 'log2'),'bootstrap': [True]}


_scoring = ['accuracy', 'roc_auc']
clf_rf = GridSearchCV(estimator=RandomForestClassifier(),param_grid=_param, scoring =_scoring,refit='roc_auc')
clf_rf.fit(X_train, y_train)

results_RF =pd.DataFrame (clf_rf.cv_results_)
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\output_RF.csv"
results_RF.to_csv(out)
print("--- %s seconds ---" % (time.time() - start_time))

""" Based on the grid search output results , the based paramters considering both AUC-ROC score and Accuracy Ranking
    are following :   
    max_depth=50,
    n_estimators=500,
    max_features='sqrt'
Using these hyperparameters iterating over different sample sizes
"""
classifier=['RF']
metric=['accuracy','auc_roc','f1-score']
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\{}.csv".format(classifier[0])
print(out)
test_sizes=[0.1,0.15,0.2,0.25]
accuracy={}
f1={}
auc={}
cf={}
ccp_alpha=[0.0,0.01,0.1,1.0,10.0]
for t in test_sizes:
    for i in ccp_alpha:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=t)
        y_test1 = y_test.to_frame()
        y_test1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_test1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        y_train1 = y_train.to_frame()
        y_train1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_train1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        rf=RandomForestClassifier(n_estimators=500,max_depth=50,max_features='sqrt',bootstrap=True,ccp_alpha=i)
        rf.fit(X_train, y_train)
        y_train_pred=rf.predict(X_train)
        prediction_trn=pd.DataFrame(y_train_pred,columns=["Predicted"])
        # on training data
        cf[(i,t,'train')]= pd.crosstab(y_train1['Spam-Label'],prediction_trn['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'train')] = accuracy_score(y_train1['Spam-Label'],prediction_trn['Predicted'],normalize=True)
        f1[(i,t, 'train')]=f1_score(y_train1['Spam-Label'],prediction_trn['Predicted'])
        auc[(i,t,'train')] = roc_auc_score(y_train1['Spam-Label'],prediction_trn['Predicted'],average='macro')

        #on test data
        y_test_pred = rf.predict(X_test)
        prediction_tst=pd.DataFrame(y_test_pred,columns=["Predicted"])
        cf[(i,t,'test')]= pd.crosstab(y_test1['Spam-Label'],prediction_tst['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'test')] =accuracy_score(y_test1['Spam-Label'],prediction_tst['Predicted'],normalize=True)
        f1[(i,t,'test')]=f1_score(y_test1['Spam-Label'],prediction_tst['Predicted'])
        auc[(i,t,'test')] = roc_auc_score(y_test1['Spam-Label'],prediction_tst['Predicted'],average='macro')


accuracy1=pd.DataFrame([(*k, v) for k, v in accuracy.items()],columns=['ccp_alpha','test-size','sample','value'])
accuracy1['Model']=classifier[0]
accuracy1['Metric']=metric[0]
accuracy1.to_csv(out,mode='a')
auc1=pd.DataFrame([(*k, v) for k, v in auc.items()],columns=['ccp_alpha','test-size','sample','value'])
auc1['Model']=classifier[0]
auc1['Metric']=metric[1]
auc1.to_csv(out,mode='a')
f1_=pd.DataFrame([(*k, v) for k, v in f1.items()],columns=['ccp_alpha','test-size','sample','value'])
f1_['Model']=classifier[0]
f1_['Metric']=metric[2]
f1_.to_csv(out,mode='a')



# ############################ Gaussian Naive Bayes - Results are reported##############################################
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
classifier=['GNB']
metric=['accuracy','auc_roc','f1-score']
out="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\{}.csv".format(classifier[0])
print(out)
estimators=[10,20,30]
test_sizes=[0.1,0.15,0.2,0.25]
accuracy={}
f1={}
auc={}
cf={}
for i in estimators:
    for t in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=t)
        y_test1 = y_test.to_frame()
        y_test1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_test1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        y_train1 = y_train.to_frame()
        y_train1.reset_index(inplace=True)  # Resets the index, makes factor a column
        y_train1.drop("index", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
        bg_GNB=BaggingClassifier(GaussianNB(),n_estimators = i, max_features = 1.0,n_jobs=-1)
        bg_GNB.fit(X_train, y_train)
        y_train_pred=bg_GNB.predict(X_train)
        prediction_trn=pd.DataFrame(y_train_pred,columns=["Predicted"])
        # on training data
        cf[(i,t,'train')]= pd.crosstab(y_train1['Spam-Label'],prediction_trn['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i, t,'train')] = accuracy_score(y_train1['Spam-Label'],prediction_trn['Predicted'],normalize=True)
        f1[(i,t, 'train')]=f1_score(y_train1['Spam-Label'],prediction_trn['Predicted'])
        auc[(i, t,'train')] = roc_auc_score(y_train1['Spam-Label'],prediction_trn['Predicted'],average='macro')

        #on test data
        y_test_pred = bg_GNB.predict(X_test)
        prediction_tst=pd.DataFrame(y_test_pred,columns=["Predicted"])
        cf[(i, t,'test')]= pd.crosstab(y_test1['Spam-Label'],prediction_tst['Predicted'], rownames=['True'], colnames=['Predicted'])
        accuracy[(i,t,'test')] =accuracy_score(y_test1['Spam-Label'],prediction_tst['Predicted'],normalize=True)
        f1[(i,t,'test')]=f1_score(y_test1['Spam-Label'],prediction_tst['Predicted'])
        auc[(i,t,'test')] = roc_auc_score(y_test1['Spam-Label'],prediction_tst['Predicted'],average='macro')

accuracy=pd.DataFrame([(*k, v) for k, v in accuracy.items()],columns=['Estimators','test-size','sample','value'])
accuracy['Model']=classifier[0]
accuracy['Metric']=metric[0]
accuracy.to_csv(out,mode='a')
auc=pd.DataFrame([(*k, v) for k, v in auc.items()],columns=['Estimators','test-size','sample','value'])
auc['Model']=classifier[0]
auc['Metric']=metric[1]
auc.to_csv(out,mode='a')
f1=pd.DataFrame([(*k, v) for k, v in f1.items()],columns=['Estimators','test-size','sample','value'])
f1['Model']=classifier[0]
f1['Metric']=metric[2]
f1.to_csv(out,mode='a')


############# IF NEEDED #####################
# save the model to disk
output="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\clf_rf.sav"
pickle.dump(clf_rf, open(output, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(output, 'rb'))
y_pred=loaded_model.predict(X_test)
# result = loaded_model.score(X_test, Y_test)

# ----------------------------------------

