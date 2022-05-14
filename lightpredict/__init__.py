import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn libraries
from sklearn.model_selection import train_test_split

from sklearn.metrics import (r2_score, 
    mean_squared_error, 
    mean_absolute_error, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    roc_curve)

from sklearn.linear_model import (LinearRegression,
                                  LogisticRegression,
                                  LogisticRegressionCV,
                                  RidgeClassifier,
                                  RidgeClassifierCV,
                                  ElasticNet,
                                  ElasticNetCV,
                                  LinearRegression,
                                  Ridge,
                                  RidgeCV,
                                  Lasso,
                                  LassoCV,
                                 SGDClassifier,
                                 SGDRegressor)

from sklearn.ensemble import (AdaBoostClassifier,
                              AdaBoostRegressor,
                             BaggingClassifier,
                              BaggingRegressor,
                             ExtraTreesClassifier,
                              ExtraTreesRegressor,
                             GradientBoostingClassifier,
                              GradientBoostingRegressor,
                             RandomForestClassifier,
                             RandomForestRegressor)

from sklearn.naive_bayes import (BernoulliNB,
                                GaussianNB)

from sklearn.svm import (LinearSVC,
                        NuSVC,
                        SVC,
                        SVR)

from sklearn.tree import (DecisionTreeClassifier, 
    DecisionTreeRegressor, 
    ExtraTreeClassifier, 
    ExtraTreeRegressor)

from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor)

from sklearn.svm import LinearSVC

from xgboost import (XGBClassifier, XGBRegressor)
from lightgbm import (LGBMClassifier, LGBMRegressor)

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# for printing tables and UI
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

# for hyperparams tuning
import optuna
from optuna import trial
import contextlib

console = Console()



class LightRegressor:
    '''
        This module helps to quickly give an idea about which model will be suitable for regression on the dataset.
        It gives the list of most commonly used Scikit-learn models trained on the training dataset.
    '''

    def __init__(self):
        self.REGRESSORS = [
        ('AdaBoostRegressor',AdaBoostRegressor()), 
        ('BaggingRegressor',BaggingRegressor()),
        ('DecisionTreeRegressor',DecisionTreeRegressor()),
        ('ElasticNet',ElasticNet()),
        ('ElasticNetCV',ElasticNetCV()),
        ('ExtraTreeRegressor',ExtraTreeRegressor()),
        ('GradientBoostingRegressor',GradientBoostingRegressor()),
        ('KNeighborsRegressor',KNeighborsRegressor()),
        ('Lasso',Lasso()),
        ('LassoCV',LassoCV()),
        ('LinearRegression',LinearRegression()),
        ('RandomForestRegressor',RandomForestRegressor()),
        ('Ridge',Ridge()),
        ('RidgeCV',RidgeCV()),
        ('SGDRegressor',SGDRegressor()),
        ('SVR', SVR()),
        ('XGBRegressor',XGBRegressor()),
        ('LGBMRegressor',LGBMRegressor())
    ]
        

    def fit(self, x_train, x_test, y_train, y_test, rounds=3):
        '''
            This function fits the regression algorithms to x_train and y_train and prints their result on x_test and y_test.

            Parameters:
            ================================================
            x_train: Training vectors in the form of an array
            x_test: Testing vectors in the form of an array
            y_train: Training (labeled) vectors in the form of an array
            y_test: Testing (labeled) vectors in the form of an array
            rounds: The desired decimal to round the resulting scores to
            ================================================
        '''

        md = Markdown("# Regression Models")
        console.print(md)
        
        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Model",  justify="center")
        table.add_column("r2 score",  justify="center")
        table.add_column("RMSE",  justify="center")
        table.add_column("MAE",  justify="center")
        
        for i in range(len(self.REGRESSORS)):
                fi = self.REGRESSORS[i][1]
                fi.fit(x_train, y_train)

                pred = fi.predict(x_test)

                table.add_row(f'[bold]{self.REGRESSORS[i][0]}', 
                              f'{round(r2_score(y_test, pred), rounds)}',
                              f'{round(mean_squared_error(y_test, pred, squared=False), rounds)}',
                              f'{round(mean_absolute_error(y_test, pred), rounds)}')
        
        console.print(table, justify='center')



class LightClassifier:
    '''
        This module helps to quickly give an idea about which model will be suitable for classification on the dataset.
        It gives the list of most commonly used Scikit-learn models trained on the training dataset.
    '''

    def __init__(self):
        self.CLASSIFIERS = [
                ('AdaBoostClassifier', AdaBoostClassifier()), 
                ('BaggingClassifier',BaggingClassifier()),
                ('BernoulliNB',BernoulliNB()),
                ('DecisionTree',DecisionTreeClassifier()),
                ('ExtraTree',ExtraTreeClassifier()),
                ('GaussianNB',GaussianNB()),
                ('KNeighbors',KNeighborsClassifier()),
                ('LinearSVC',LinearSVC()),
                ('LogisticReg',LogisticRegression()),
                ('LogisticReg CV',LogisticRegressionCV()),
                ('NuSVC',NuSVC(probability=True)),
                ('RandomForest',RandomForestClassifier()),
                ('RidgeClassifier',LogisticRegression(penalty='l2')),
                ('RidgeClassifierCV',LogisticRegressionCV(penalty='l2')),
                ('SGDClassifier',SGDClassifier()),
                ('SVC', SVC(probability=True)),
                ('XGBoost', XGBClassifier(verbosity=0)),
                ('Lightgbm', LGBMClassifier())
            ]

        self.models = [key for key, value in self.CLASSIFIERS]
        self.scores = []


    def fit(self, x_train, x_test, y_train, y_test, rounds=3, plot=False):
        '''
            This function fits the classification algorithms to x_train and y_train and prints their result on x_test and y_test.

            Parameters:
            ================================================
            x_train: Training vectors in the form of an array
            x_test: Testing vectors in the form of an array
            y_train: Training (labeled) vectors in the form of an array
            y_test: Testing (labeled) vectors in the form of an array
            rounds: The desired decimal to round the resulting scores to
            plot: To plot the accuracy scores of the models for visual comparison
            ================================================
        '''
        md = Markdown("# Classification Models")
        console.print(md)
        
        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Model",  justify="center")
        table.add_column("Accuracy score",  justify="center")
        table.add_column("f1-score",  justify="center")
        table.add_column("ROC-AUC",  justify="center")
        table.add_column("Precision score",  justify="center")
        table.add_column("Recall score",  justify="center")
        

        ACCURACY = {'Model': self.models, 'Accuracy': self.scores}

        for i in range(len(self.CLASSIFIERS)):
            if self.CLASSIFIERS[i][0] in ['LinearSVC', 'SGDClassifier']:
                fi = self.CLASSIFIERS[i][1]
                fi.fit(x_train, y_train)
                pred = fi.predict(x_test)
                self.scores.append(round(accuracy_score(y_test, pred), rounds))
                table.add_row(f'[bold]{self.CLASSIFIERS[i][0]}', 
                          f'{round(accuracy_score(y_test, pred), rounds)}', 
                          f'{round(f1_score(y_test, pred), rounds)}',
                          f'N/A',
                          f'{round(precision_score(y_test, pred), rounds)}',
                          f'{round(recall_score(y_test, pred), rounds)}')
            
            else:
                fi = self.CLASSIFIERS[i][1]
                fi.fit(x_train, y_train)

                pred = fi.predict(x_test)     
                self.scores.append(round(accuracy_score(y_test, pred), rounds))       
                proba = fi.predict_proba(x_test)[:, 1]

                table.add_row(f'[bold]{self.CLASSIFIERS[i][0]}', 
                              f'{round(accuracy_score(y_test, pred), rounds)}', 
                              f'{round(f1_score(y_test, pred), rounds)}',
                              f'{round(roc_auc_score(y_test, proba), rounds)}',
                              f'{round(precision_score(y_test, pred), rounds)}',
                              f'{round(recall_score(y_test, pred), rounds)}')
        
        console.print(table)

        if plot:
            data = pd.DataFrame.from_dict(ACCURACY)
            plt.figure(figsize=[15, 10])
            sns.barplot(data=data, x='Model', y='Accuracy')
            plt.xticks(rotation=90)
    

    def roc_auc_curves(self, x_train, x_test, y_train, y_test):
        '''
            This function plots the roc_auc curve for the defined classification algorithms.

            Parameters:
            ================================================
            x_train: Training vectors in the form of an array
            x_test: Testing vectors in the form of an array
            y_train: Training (labeled) vectors in the form of an array
            y_test: Testing (labeled) vectors in the form of an array
            ================================================
        '''
        fig = plt.figure(figsize=[15, 10])
        classifiers = self.CLASSIFIERS.copy()
        del classifiers[7]
        del classifiers[13]
        
        titles = []
        for x in classifiers:
            titles.append(x[0])
        for i in range(len(classifiers)):
            fi = classifiers[i][1]
            fi.fit(x_train, y_train)
            pred = fi.predict(x_test)
            pred_prob = fi.predict_proba(x_test)
            fpr, tpr, thresh1 = roc_curve(y_test, pred_prob[:,1])
            auc_score = roc_auc_score(y_test, pred_prob[:, 1])
            # roc curve for tpr = fpr 
            random_probs = [0 for i in range(len(y_test))]
            p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
            ax = fig.add_subplot(3, 6, i+1)
            ax.plot(fpr, tpr)
            ax.plot(p_fpr, p_tpr)
            ax.set_title(f'{titles[i]}: {round(auc_score, 3)}')
            fig.tight_layout()
            
            
    def optimize(self, x_train, x_test, y_train, y_test, trials=5, plot=False):
        '''
            This function optimizes the classification models using OPTUNA on x_train and y_train and prints their result on x_test and y_test.

            Parameters:
            ================================================
            x_train: Training vectors in the form of an array
            x_test: Testing vectors in the form of an array
            y_train: Training (labeled) vectors in the form of an array
            y_test: Testing (labeled) vectors in the form of an array
            trials: The no. of rounds to optimize the model for
            plot: To plot the results summary of Optuna optimization
            ================================================
        '''
        x_train, x_test, y_train, y_test = x_train, x_test, y_train, y_test
        

        def abc_tuner(trial):
            abc_params = {
                'n_estimators': trial.suggest_int("n_estimators", 2, 150),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.0005, 1.0)
            }
            
            model = AdaBoostClassifier(**abc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def bgc_tuner(trial):
            bgc_params = {
                'n_estimators': trial.suggest_int('n_estimators', 2, 200),
                'max_samples': trial.suggest_int('max_samples', 1, 100)
            }
            
            model = BaggingClassifier(**bgc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def dtc_tuner(trial):
            dtc_params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
                'min_weight_fraction_leaf': trial.suggest_uniform('min_weight_fraction_leaf', 0.0, 0.5),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            
            model = DecisionTreeClassifier(**dtc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def etc_tuner(trial):
            etc_params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6)
            }
            
            model = ExtraTreeClassifier(**etc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def knc_tuner(trial):
            knc_params = {
                'n_neighbors': trial.suggest_int("n_neighbors", 2, 25)
            }
            
            model = KNeighborsClassifier(**knc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def rfc_tuner(trial):
            rfc_params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6),
                'n_estimators': trial.suggest_int("n_estimators", 2, 150),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10)
            }
            
            model = RandomForestClassifier(**rfc_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def xgb_tuner(trial):
            xgb_params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6),
                'n_estimators': trial.suggest_int("n_estimators", 1, 150),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.0000001, 1),
                'gamma': trial.suggest_uniform('gamma', 0.0000001, 1),
                'subsample': trial.suggest_uniform('subsample', 0.0001, 1.0)
            }
            
            model = XGBClassifier(**xgb_params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
        

        def lgbm_tuner(trial):
            lgb_params = {
                "boosting_type": "gbdt",
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
            }
            
            model = LGBMClassifier(**lgb_params, verbose=-1)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            score = accuracy_score(y_test, pred)
            return score
            
        funcs = [
            ('AdaBoost', abc_tuner),
            ('BaggingClassifier', bgc_tuner),
            ('Decision Tree', dtc_tuner),
            ('Extra Trees', etc_tuner),
            ('KNeighbors', knc_tuner),
            ('RandomForest', rfc_tuner),
            ('XGBClassifier', xgb_tuner),
            ('LightGBM', lgbm_tuner)
        ]
        
        console.print(f'[bold] Optimizing models...')
        
        md = Markdown("# Optimized Models & Scores")
        console.print(md)
        
        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Model",  justify="center")
        table.add_column("Best Score",  justify="center")
        table.add_column("Best Params", justify="left")
        
        
        for i in range(len(funcs)):
            with contextlib.redirect_stdout(None):
                study = optuna.create_study(direction='maximize')
                study.optimize(funcs[i][1], n_trials=10)
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            table.add_row(f'[bold]{funcs[i][0]}', 
                              f'{round(study.best_value, 3)}', 
                              f'{study.best_params}')
                
            if plot:
                optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.title(f'{funcs[i][0]}')
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.title(f'{funcs[i][0]}')
                
        console.print(table) 