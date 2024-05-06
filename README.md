# A novel class impurity-based hybrid resampling method for imbalanced classification problem

## Introduction
---
This is a new hybrid sampling (combined oversampling and undersampling) technique for imbalanced binary classification problem. The most important idea of this is that we provide a new instance assessment metric, `class impurity`, to more precisely sampling the instnaces. The provided `class impurity` is a instance metric that simultaneously consider the `relative density` and the `relationship of nearest neighboirs`, which is a useful assessment when selective sampling method is applied to the instances. Moreover, this algorithm also considers the classification performance during sampling procedure, for this, when fitting the resampler a classifier should be given.

##
> [!TIP]
> Guide of using the algorithm
```
git clone https://github.com/Terry0103/ClassImp_Resampling.git
pip install -r ./ClassImp_Resampling/requirements.txt
```

```
from ClassImp_Resampling.ClassImp import IHOT
import pandas as pd
test = pd.DataFrame([[1, 2, 1], [1, 0, 1], [10, 4, 0], [10, 0, 0], [10, 2, 0], [1, 4, 1], [10, 4, 0], [10, 4, 0], [10, 4, 0], [10, 4, 0]])

# Suppose the last column is the label of test dataset.
X, Y = data.iloc[:, 0:-1], data.iloc[:, -1]
Y.value_counts()
... 0.0    7
... 1.0    3

classifier = DecisionTreeClassifier()
resampler = IHOT(n_neighbors = 3,
                classifier = classifier)

X, Y = resampler.fit_resample(X, Y)
Y.value_counts()
... 0.0    7
... 1.0    7
```
