import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, roc_curve, auc
import matplotlib.pyplot as plt

def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    return pd.DataFrame(scaled_train_data, columns=train_data.columns), pd.DataFrame(scaled_test_data, columns=test_data.columns)
 
def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc=False):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
 
    best_model = grid_search.best_estimator_
 
    print(f"\nResults for {model.__class__.__name__} - Cross Validation:")
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
 
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
 
        fold_model = model.__class__(**best_model.get_params())
        fold_model.fit(X_train_fold, y_train_fold)
        predictions = fold_model.predict(X_val_fold)
 
        fpr, tpr, _ = roc_curve(y_val_fold, fold_model.predict_proba(X_val_fold)[:, 1])
        roc_auc = auc(fpr, tpr)
 
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {fold} (AUC = {roc_auc:.2f})')
 
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)
 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model.__class__.__name__} (Cross Validation)')
    plt.legend(loc='lower right')
 
    if save_auroc:
        plt.savefig(f"{model.__class__.__name__}_CrossValidation_ROC.pdf")
    plt.show()
    plt.close()
 
    predictions = best_model.predict(X_test)
 
    print("\nResults for", model.__class__.__name__, "- Testing:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
 
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)
    print("Precision for Class 0:", precision[0])
    print("Recall for Class 0:", recall[0])
    print("F1-Score for Class 0:", f1[0])
 
    print("Precision for Class 1:", precision[1])
    print("Recall for Class 1:", recall[1])
    print("F1-Score for Class 1:", f1[1])
 
    print("\nMatthews Correlation Coefficient:", matthews_corrcoef(y_test, predictions))

def svm_classifier(X_train, y_train, X_test, y_test, cv, save_auroc=False):
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.01, 0.1, 1, 10, 100, 1000]
    }
 
    svm_model = SVC(probability=True)
    train_and_evaluate(svm_model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc)
    
def random_forest_classifier(X_train, y_train, X_test, y_test, cv, save_auroc=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8, 16, 20]
    }
 
    rf_model = RandomForestClassifier()
    train_and_evaluate(rf_model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc)
    
def adaboost_classifier(X_train, y_train, X_test, y_test, cv, save_auroc=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
 
    adaboost_model = AdaBoostClassifier()
    train_and_evaluate(adaboost_model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc)
    
def xgboost_classifier(X_train, y_train, X_test, y_test, cv, save_auroc=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
 
    xgb_model = XGBClassifier()
    train_and_evaluate(xgb_model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc)
    
def extratrees_classifier(X_train, y_train, X_test, y_test, cv, save_auroc=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
 
    extratrees_model = ExtraTreesClassifier()
    train_and_evaluate(extratrees_model, param_grid, X_train, y_train, X_test, y_test, cv, save_auroc)

    
training_data = pd.read_csv("/home/users/ntu/leiz0003/DBPMoD/train_features_withPBDeepFRI.csv", index_col=0)
testing_data = pd.read_csv("/home/users/ntu/leiz0003/DBPMoD/test_features_withPBDeepFRI.csv", index_col=0)

X_train_data = training_data.drop('Label', axis=1)
y_train_data = training_data['Label']
X_test_data = testing_data.drop('Label', axis=1)
y_test_data = testing_data['Label']

scaled_X_train, scaled_X_test = scale_data(X_train_data, X_test_data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_classifier(scaled_X_train, y_train_data, scaled_X_test, y_test_data, cv, save_auroc=True)
random_forest_classifier(scaled_X_train, y_train_data, scaled_X_test, y_test_data, cv, save_auroc=True)
adaboost_classifier(scaled_X_train, y_train_data, scaled_X_test, y_test_data, cv, save_auroc=True)
xgboost_classifier(scaled_X_train, y_train_data, scaled_X_test, y_test_data, cv, save_auroc=True)
extratrees_classifier(scaled_X_train, y_train_data, scaled_X_test, y_test_data, cv, save_auroc=True)