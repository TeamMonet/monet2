# -*- coding: utf-8 -*-
"""
@author: alban.petit / sabrina.chaouche
"""

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score
from myPreprocessing import Preprocessor
from myBestParams import BestParams
import pickle

""" /!\ /!\ Outdated sklearn version on codalab (0.17) /!\ /!\ """
submission = 0   #Mettre à 1 si on veut calculer une soumission, 0 sinon
if submission == 0: #On va importer l'interface seulement si on en a besoin pour ne pas alourdir
    from myData_manager import DataManager
        
      #Ici, on définit notre Classifier
class Classifier(BaseEstimator):
    def __init__(self):
        if submission == 0:
            self.params = BestParams()
            self.estimators = self.params.bestParamsRFC()
            forest = RandomForestClassifier(n_estimators=self.estimators[0], max_features=self.estimators[1])
        else:
            forest = RandomForestClassifier(n_estimators=200, max_features='auto')
        self.clf = Pipeline([('preproc',Preprocessor(forest)),('class',forest)])

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return ['negative','positive']

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self =      pickle.load(open(path + '_model.pickle'))
        return self



        
        #Partie qui s'exécute seulement si on exécute la classe
        
if __name__=="__main__":
    #On importe DataManager au cas où il n'est pas encore été importé
    from myData_manager import DataManager
    input_dir="../public_data"
    output_dir="../res"      
    basename = 'hiva'
    
        #On initialise DataManager et le classifier  
    C = Classifier()
    D = DataManager([],[])
    
        #On ajuste le modèle et on fait les prédictions sur différents datasets
    Y_train_data = D.data['Y_train']
    print "Fitting"
    C.fit(D.data['X_train'], Y_train_data)
    print "Predicting on training data"
    Y_train_proba_pred = C.predict_proba(D.data['X_train'])
    Y_train_pred = C.predict(D.data['X_train'])
    print "Predicting on validation data"
    Y_valid_pred = C.predict(D.data['X_valid'])
    print "Predicting on test data"
    Y_test_pred = C.predict(D.data['X_test'])
    
        #On calcule l'accuracy et le score de cross-validation
    print "Calculating the accuracy"
    class_acc = accuracy_score(Y_train_data, Y_train_pred)
    print "Calculating cross-validation score"
    class_cross = cross_val_score(C, D.data['X_train'], Y_train_data, cv=5, scoring='accuracy')
    class_mean = class_cross.mean()*0.8+class_acc*0.2
    
        #On affiche les scores calculés et une matrice de confusion brute
    print "Classifier accuracy on training dataset :"
    print class_acc
    print "\n Cross-validation accuracy :"
    print class_cross.mean()
    print "\n Raw confusion matrix :"
    print confusion_matrix(Y_train_data,Y_train_pred)
        
        #On appelle le DataManager pour des statistiques plus détaillées et une matrice de confusion graphique
    D = DataManager(Y_train_pred,Y_train_data)
    D.afficheTableau()
    D.plot_confusion_matrix()
    