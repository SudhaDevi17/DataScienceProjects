import numpy as np
import pandas as pd


def get_categorical_columns(dataframe):
    cat_columns = list(set(dataframe.columns) - set(dataframe._get_numeric_data().columns))
    return dataframe[cat_columns]


def feature_processing(raw_train):
    # drop id column
    raw_train.drop(['Id'], axis=1, inplace=True)

    # filter out columns with missing values more than 80%
    columns_subset = raw_train.columns[raw_train.isnull().mean() < 0.8]
    raw_train = raw_train[columns_subset]

    # process cateogorical data
    raw_train['FireplaceQu'] = raw_train['FireplaceQu'].fillna('missing')
    cols = ['GarageQual',
            'GarageFinish',
            'GarageType',
            'BsmtFinType2',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtCond',
            'BsmtQual',
            'MasVnrType',
            'Electrical',
            'GarageCond']

    raw_train[cols] = raw_train[cols].fillna(raw_train.mode().iloc[0])
    # process numeric data
    numeric_train = raw_train._get_numeric_data().fillna(raw_train._get_numeric_data().mean())
    cat_train = get_categorical_columns(raw_train)
    # combine th data
    dataset = pd.concat([numeric_train, cat_train.reindex(numeric_train.index)], axis=1)
    print(dataset.isnull().shape)

    # based on feature importance restricting the column selection to top few
    features_selected = ['LotArea',
                         'MoSold',
                         'GarageArea',
                         'BsmtFinSF1',
                         'TotalBsmtSF',
                         'BsmtUnfSF',
                         'WoodDeckSF',
                         'YearRemodAdd',
                         'GrLivArea',
                         '1stFlrSF',
                         'LotFrontage']
    return dataset[features_selected]


# feature engineering

def implement_label_encoding(train_dataset):
    # implement label encoding
    cols_to_labelencode = [
        'KitchenAbvGr',
        'BsmtHalfBath',
        'HalfBath',
        'BsmtFullBath',
        'FullBath',
        'Fireplaces',
        'GarageCars',
        'BedroomAbvGr']

    for column in cols_to_labelencode:
        train_dataset[column] = train_dataset[column].astype('category').cat.codes

    return train_dataset


class Node:

    def __init__(self, x, y, idxs, min_leaf=20):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.rows = len(idxs)
        self.columns = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.variable_split()

    def variable_split(self):
        for c in range(self.columns):
            self.best_split(c)
        if self.is_a_leaf: return
        x = self.split_a_col
        left_array = np.nonzero(x <= self.node_split_val)[0]
        right_array = np.nonzero(x > self.node_split_val)[0]
        self.left_node = Node(self.x, self.y, self.idxs[left_array], self.min_leaf)
        self.right_node = Node(self.x, self.y, self.idxs[right_array], self.min_leaf)

    def best_split(self, var_idx):

        x = self.x.values[self.idxs, var_idx]

        # for categorical data
        if self.x.iloc[:, var_idx].dtype == 'O':

            # count frequency and calculate scores
            unique_vals = list(set(x))
            for v in unique_vals:
                # seperate the num of cat and non-cat
                in_cat = np.array([val == v for val in x])
                not_in_cat = np.array([val != v for val in x])
                split_score = self.variance_score(in_cat, not_in_cat)
                self.update_scores(split_score, v, var_idx)

        # for numerical data
        else:
            for r in range(self.rows):
                left_node = x <= x[r]
                right_node = x > x[r]
                if right_node.sum() < self.min_leaf or left_node.sum() < self.min_leaf: continue

                curr_score = self.variance_score(left_node, right_node)
                self.update_scores(curr_score, x[r], var_idx)

    def update_scores(self, current_score, nodeplit_val, var_idx):

        if current_score < self.score:
            self.var_idx = var_idx
            self.score = current_score
            self.node_split_val = nodeplit_val

    def variance_score(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    @property
    def split_a_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_a_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        return np.array([self.predict_instance(xi) for xi in x])

    def predict_instance(self, xi):
        if self.is_a_leaf: return self.val
        node = self.left_node if xi[self.var_idx] <= self.node_split_val else self.right_node
        return node.predict_instance(xi)


class DecisionTree:

    def train(self, path):
        data = pd.read_csv(path)

        X, y, min_leaf = data.loc[:, data.columns != 'SalePrice'], data['SalePrice'], 20
        X = feature_processing(X)

        # process train data
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self

    def predict(self, path):
        # feature processing test
        X = pd.read_csv(path)
        X = feature_processing(X)
        return self.dtree.predict(X.values)
