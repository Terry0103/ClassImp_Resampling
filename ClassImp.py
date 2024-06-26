import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
import math
from typing import Literal, Dict
from dataclasses import dataclass, field

class ClassImpurity:
    '''
    ## A k-nearest neighbor-based measure for estimating the degree of class impurity of an instance in supervised tasks
    n_neighbors : int, defalut = 3
        Number of neighbors to use by default for kneighbors

    kNN : sklearn.neighbors.NearestNeighbors, default = None
        Unsupervised learner for implementing neighbor searches which from sklear.neighbors
    ---    
    ## Artributes
    class_impurity : numpy.array
        The fitted class impurity.
    ---
    ## Example
    >>> from ClassImp_Resampling.ClassImp import ClassImpurity
    >>> import pandas as pd
    >>> test = pd.DataFrame([[1, 2, 1], [1, 0, 1], [10, 4, 0], [10, 0, 0], [10, 2, 0], [1, 4, 1], [10, 4, 0], [10, 4, 0], [10, 4, 0], [10, 4, 0]])
    
    >>> cls_imp = ClassImpurity()
    >>> cls_imp.fit_class_impurity(X = test.iloc[:, 0:-1], Y = test.iloc[:, -1])
    >>> print(cls_imp.class_impurity)
    ... [[0.25 ]
    ... [0.375]
    ... [0.125]
    ... [0.25 ]
    ... [0.125]
    ... [0.375]
    ... [0.125]
    ... [0.125]
    ... [0.125]
    ... [0.125]]
    '''
    def __init__(self, n_neighbors : int = 3, kNN : NearestNeighbors = None):
        self.n_neighbors = n_neighbors
        self.kNN = kNN
        class_impurity: np.ndarray = field(init = False)
        
    def fit_class_impurity(self, X: any = None, Y: any = None) -> dict:
        '''
        Fit the class impurity from given dataset, `X` and `Y`.
        ## Parameters
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape
            The target of training data.
        '''
        X: np.array = np.array(X)
        Y: np.array = np.array(Y)
        # self.__recog_minority__(Y)

        # if there's no trained kNN model is given, we trained a new one.
        if self.kNN is None:
            self.kNN = NearestNeighbors()

        self.kNN = self.kNN.set_params(**{'n_neighbors' : X.shape[0],}) # Euclidean distance
        # KNN = KNN.set_params(**{'n_neighbors' : data.shape[0], \ # Mahalanobi's distance
        # 'metric' : 'mahalanobis', 'metric_params' : {'VI' : np.linalg.inv(np.cov(data.iloc[:, 0:-1].T))}})

        if '_fit_X' not in self.kNN.__dict__:
            self.kNN.fit(X = X)

        return self.__compute_class_impurity(X, Y)

    # TODO: if it's possible, for loop should be transformed to matrix form.
    def __compute_relative_density(self, X: any, Y: any) -> np.array:
        '''
        Computing relative density among the k-nearest neighbors of the instances
        '''
        _distances, _indices = self.kNN.kneighbors(X)

        ## matrix transform test
        

        # the reason of put {[:, 1:]} here is to exclude the original point
        HON_dist = np.where(Y[_indices] == np.tile(Y, (_indices.shape[0], 1)).T, _distances, 0)[:, 1:]
        HEN_dist = np.where(Y[_indices] != np.tile(Y, (_indices.shape[0], 1)).T, _distances, 0)[:, 1:]

        HON_indices = np.transpose(np.nonzero(HON_dist))
        HEN_indices = np.transpose(np.nonzero(HEN_dist))

        # relative_density = np.empty([1, 1], dtype = np.int8)
        relative_density = np.ones([X.shape[0], 1], dtype = np.int8)

        for i in range(X.shape[0]):
            sum_HONK_dist = HON_dist[i, HON_indices[np.where(HON_indices[:, 0] == i)[0], 1][0: self.n_neighbors]].sum()
            sum_HENK_dist = HEN_dist[i, HEN_indices[np.where(HEN_indices[:, 0] == i)[0], 1][0: self.n_neighbors]].sum()

            # relative_density = np.vstack((relative_density, sum_HENK_dist / sum_HONK_dist))
            relative_density[i, 0] = sum_HENK_dist / sum_HONK_dist

        # relative_density = relative_density[1:, :]

        del HON_dist, HEN_dist, HON_indices, HEN_indices,sum_HONK_dist , sum_HENK_dist, _distances, _indices
        return relative_density

    def __compute_kNN_ratio(self, X, Y) -> np.array:
        '''
        Computing the class proportion among the k-nearest neighbors of the instances
        '''
        _distances, _indices = self.kNN.kneighbors(X)
        kNN_ratio = np.where(Y[_indices] == np.tile(Y, (_indices.shape[0], 1)).T, 1, 0)[:, 1:4].sum(axis = 1) / self.n_neighbors
        kNN_ratio = kNN_ratio.reshape((kNN_ratio.shape[0], 1))

        del _distances, _indices
        return kNN_ratio

    def __compute_class_impurity(self, X, Y) -> np.array:
        '''
        Aggregaing k-NN ratio and relative density into `class impurity`.
        '''

        # Computing class purity
        RD = self.__compute_relative_density(X, Y)
        kNN_ratio = self.__compute_kNN_ratio(X, Y)
        purity = RD * kNN_ratio
        del RD, kNN_ratio

        # Preventing zero division
        # TODO: Instead of a constant, a more adaptive value should be assign to the zero purity
        # for reducing the extream value when scaling class impurity
        purity[np.where(purity == 0)[0], :]+=0.001 

        # Normalize and tranform class purity into class impurity (class by class)
        for label in range(len(np.unique(Y))):
            label_id = np.where(Y == label)[0]
            purity[label_id, :] = purity[label_id, :] ** (-1)

            sum_purity = purity[label_id, :].sum()
            purity[label_id, :] = purity[label_id, :] / sum_purity

        self.class_impurity = purity
        del purity, label_id, sum_purity
        return self

    def __recog_minority__(self, Y):
        '''
        Recognizing the minority class in the training data.
        '''
        labels, counts = np.unique(Y, return_counts = True)

        label_index = np.where(counts == min(counts))
        self.minority_class_label = labels[label_index][0]
        # G = sum(counts) - counts[label_index]
        del labels, counts, label_index 
    
    def __str__(self) -> str:
        return f'ClassImpurity(n_neighbors={self.n_neighbors}, kNN={self.kNN}, class_impurity={self.class_impurity}' 


class IHOT(ClassImpurity):
    '''
    ## A novel class impurity-based hybrid resampling technique for imbalanced classification problem
    ---
    n_neighbors : int, defalut = 3
        Parameter of the k-nearest neighbor algorithm.

    classifier : any, default = None
        A classifier used to search the best classification performance
        None: scikit-learn package is needed.

    kNN : sklearn.neighbors.NearestNeighbors, default = None
        A k-NN model that used to calculate class impurity.

    optimization : Literal, default = 'best'
        The optimization method when searching the dataset that has best classfication performance.

    max_saturation : int, default = 3
        The maximun number of time that classification performance saturated. When max_saturation is reached \
              the algorithm will terminate and return the best_balanced_data.

    metric: Any, default = sklearn.metric.roc_auc_score
        The evaluation metric which is used in optimal dataset searching.
    ---    
    ## Artributes

    best_balanced_data : tuple
        The dataset that preprocessed by IHOT algorithm.

    ---
    ## Example
    >>> from RDBU import IHOT
    >>> import pandas as pd
    >>> test = pd.DataFrame([[1, 2, 1], [1, 0, 1], [10, 4, 0], [10, 0, 0], [10, 2, 0], [1, 4, 1], [10, 4, 0], [10, 4, 0], [10, 4, 0], [10, 4, 0]])

    // Suppose the last column is the label of test dataset.
    >>> X, Y = data.iloc[:, 0:-1], data.iloc[:, -1]
    >>> Y.value_counts()
    >>> 0.0    7
        1.0    3

    >>> classifier = DecisionTreeClassifier()
    >>> resampler = IHOT(n_neighbors = 3,
                        classifier = classifier)

    X, Y = resampler.fit_resample(X, Y)
    Y.value_counts()
    >>> 0.0    7
        1.0    7

    '''
    # classifier: any = None
    # optimization: Literal['best', 'saturation'] = 'best'
    # max_saturation: int = 3
    # __saturation_count__: int = field(init = False, default_factory = 0)
    # best_score: float = field(init = False, default_factory = 0)
    # best_balanced_data: any = field(init = False, default_factory = None)
    
    def __init__(self, 
                 n_neighbors : int = 3, 
                 classifier: any = None, 
                 kNN: any = None, 
                 optimization : Literal['best', 'saturation'] = 'best', 
                 max_saturation :int = 3,
                 metric: any = roc_auc_score):
        # super().__init__(n_neighbors, kNN) # Inherit all element from class `ClassImpurity`
        self.n_neighbors: int = n_neighbors
        self.kNN = kNN
        self.classifier = classifier
        self.optimization = optimization
        self.max_saturation = max_saturation
        self.metric = metric

        self.__saturation_count__: int = 0
        self.best_score: int = 0
        self.best_balanced_data: any = None
        self.scores: list[float] = list()
        
    def fit_resample(self, X, Y) -> tuple:
        '''
        Fit `IHOT` algorithm
        '''
        # Initializing
        X = np.array(X)
        Y = np.array(Y)
        self.__recog_minority__(Y)

        # If kNN model is not given, Euclidean distance is used when fitting k-NN of instances.
        if self.kNN == None:
            self.kNN = NearestNeighbors()
        

        # Hybrid resampling
        X, Y = self.__undersampling__(X, Y)
        X, Y = self.__oversampling__(X, Y)

        return X, Y
    
    def __undersampling__(self, X, Y) -> tuple:
        '''
        According to the relationship between a majority class instance to its' nearest neighbor, the majority class instances\
        are adaptively undersampling until there is no the nearest neighbor of any majority class instance belongs to minroity class.
        '''

        stop = False
        while stop == False:
            
            # For each majority class instances
            majority_id = np.where(Y != self.minority_class_label)[0]

            # self.kNN = self.kNN.set_params(**{'n_neighbors' : X.shape[0]})
            self.kNN.fit(X = X)
            _distance, _indices = self.kNN.kneighbors(X = X, n_neighbors = X.shape[0])
            # If the nearest neighbor of any majority class instance belong to minroty class, then undersampling
            if np.where(Y[_indices[majority_id, 1]] == self.minority_class_label, 1, 0).sum() != 0:
                self.fit_class_impurity(X, Y)
                class_impurity = self.class_impurity
                sorted_majority_id = sorted(majority_id, key = lambda x: class_impurity[x], reverse = True)
                remove_maj_list = sorted_majority_id[:math.ceil(len(sorted_majority_id) * 0.05)]
            else:
                break

            # Until no need undersampling or the number of instances from classes are balanced.
            if (len(remove_maj_list) > 0):
                X = np.delete(X, obj = remove_maj_list, axis = 0)
                Y = np.delete(Y, obj = remove_maj_list, axis = 0)
            else:
                stop = True

            u, c = np.unique(Y, return_counts = True)
            if abs(c[0] - c[1]) < 5:
                del u, c
                break

            self.class_impurity = None

        # End while
        return X, Y
    
    def __oversampling__(self, X, Y)-> tuple:
        '''
        Based on the class impurity of minority class instances to orderly and adaptively oversampling. \
              And, finally return the dataset that has the best classification performance.
        '''
        # Initializing the oversampling
        self.classifier.fit(X, Y)
        pred = self.classifier.predict(X)
        self.best_score =  self.metric(Y, pred)
        self.best_balanced_data = X, Y

        X_prime = X
        Y_prime = Y
        
        # Re-fitting k-NN since the dataset shape is changed after undersampling
        self.kNN.fit(X = X)
        _distance, _indices = self.kNN.kneighbors(X = X, n_neighbors = X.shape[0])

        # Sorting minority class instances by thier class impurity
        self.fit_class_impurity(X, Y)
        class_impurity = self.class_impurity
        minority_id = np.where(Y == self.minority_class_label)[0]
        sorted_minority_id = sorted(minority_id, key = lambda x: class_impurity[x], reverse = True)
        G = max(np.unique(Y, return_counts = True)[1]) - min(np.unique(Y, return_counts = True)[1])

        for min_instance in minority_id:

            # Adaptively assigning synthetic size for each minority class instance
            g = math.ceil(G * class_impurity[min_instance])

            # If the nearest neighbor of a minority class instance belongs to minority, then SMOTE, else sampling from a MVN.
            if Y[_indices[min_instance, 1]] == self.minority_class_label:

                # SMOTE
                root_index = min_instance 
                candi_index = np.random.choice(
                    self.__minority_SMOTE_candidate(X, Y, min_instance),
                    size = g,
                    replace = True,
                    )

                x = X[root_index , :]
                root = np.tile(x, (g, 1))
                candi = X[candi_index, :]
                beta = np.random.rand(g, 1)
                # x_new = x_i + (x_i - x_j) * beta
                new_instances = root + (candi - root) * beta                
            else:

                # Sampling from a multi-variants normal(MVN) distribution.
                mean = X[min_instance] + (X[_indices[min_instance, 1]] - X[min_instance]) * float(np.random.rand(1))
                kNN_id = _indices[min_instance, 1:][np.where(Y[_indices[min_instance, 1:]] == Y[min_instance])[0]][:self.n_neighbors]
                covariance = np.cov(X[kNN_id, :].T)
                new_instances = np.random.multivariate_normal(mean = mean, cov = covariance, size = g)

            # Updating current dataset
            X_prime = np.vstack((X_prime, new_instances))
            Y_prime = np.hstack((Y_prime, np.ones(new_instances.shape[0])))

            # Fitting classifier by current dataset
            self.classifier.fit(X_prime, Y_prime)
            pred = self.classifier.predict(X_prime)
            score = self.metric(Y_prime, pred)
            self.scores.append(score)

            # If classification performance is improved, the best dataset is updated by the current dataset.
            if self.best_score < score:
                self.best_score = score
                self.best_balanced_data = X_prime, Y_prime
                self.__saturation_count__ = 0
            else:
                self.__saturation_count__+=1

            # If optimization method is saturation
            # ,and classification performance saturated, the process is terminated and the best dataset is returned.
            if (self.optimization == 'saturation') and (self.__saturation_count__ >= self.max_saturation):
                self.__saturation_count__ = 0
                return self.best_balanced_data
            
        self.__saturation_count__ = 0
        return self.best_balanced_data
    
    def __minority_SMOTE_candidate(self, X: any, Y: any, id: int) -> np.array:
        '''
        Finding the interpolating candidates of minority class instances which the nearest neighbor belongs to minority class. \
        Candidates are the minority class instances among the k-NN and closer than the nearest majority class neighbor of a minoeirty class instance.
        '''
        _distance, _indices = self.kNN.kneighbors(X = X, n_neighbors = X.shape[0])
        final_candi = np.where(Y[_indices[id, 1:]] != self.minority_class_label)[0][0]
        
        return _indices[id, 1:][np.where(Y[_indices[id, 1:]] == Y[id])[0]][: final_candi]
    
    def get_params(self) -> None:
        return {}

    def set_params(self, **parameters : Dict[str, any]) -> None:
        for para_name in parameters.keys():
            self.__dict__[para_name] = parameters[para_name]
        return self
    

    def __str__(self) -> str:
        return f'IHOT(n_neighbors={self.n_neighbors}, kNN={self.kNN}, classifier={self.classifier}, optimization={self.optimization}, max_saturation={self.max_saturation}' 
