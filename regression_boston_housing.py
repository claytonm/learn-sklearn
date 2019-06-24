import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

def get_boston_data() -> pd.DataFrame:
    # load boston boston data into pandas DataFrame
    dt = datasets.load_boston()
    df = pd.DataFrame(dt.data)
    df.columns = dt.feature_names
    df['MEDVAL'] = dt.target
    return df, dt.DESCR

boston, descr = get_boston_data()

# stratified random sample based on the RM variable
from sklearn.model_selection import StratifiedShuffleSplit
# label records according to RM quintile
boston["RM_cat"] = pd.qcut(boston['RM'], 5, labels=False)
# sample according to RM quintile
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(boston, boston["RM_cat"]):
    strat_train_set = boston.iloc[train_index]
    strat_test_set = boston.iloc[test_index]

# remove RM_cat column from test/training sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop('RM_cat', axis=1, inplace=True)

# copy training data for data exploration
boston = strat_train_set.copy()


# find variables with highest correlation to MEDVAL
corr_high = corr_matrix['MEDVAL'][corr_matrix['MEDVAL'].abs() > 0.5]
# MEDVAL is truncated at 50
# consider dropping these data points

boston = strat_train_set.drop('MEDVAL', axis=1)
boston_labels = strat_train_set['MEDVAL'].copy()

def chas_cats():
    """
    transform CHAS into a string and then 1-hot encode
    """
    chas_cat = pd.DataFrame(pd.cut(boston['CHAS'], bins=2, labels=False))
    cat_encoder = OneHotEncoder()
    chas_cat_1hot = cat_encoder.fit_transform(chas_cat)
    return chas_cat_1hot

chas_cat_1hot = chas_cats()
