import numpy as np
import logging
import tensorflow as tf
import requests
import os
import urllib

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def test():
    logging.debug("tf data test start")
    evens = np.arange(0, 100, step=2, dtype=np.int32)
    evens_label = np.zeros(50, dtype=np.int32)
    odds = np.arange(1, 100, step=2, dtype=np.int32)
    odds_label = np.ones(50, dtype=np.int32)
    features = np.concatenate([evens, odds])
    labels = np.concatenate([evens_label, odds_label])

    data = tf.data.Dataset.from_tensor_slices((features, labels))
    # data = data.shuffle(100)
    data = data.batch(5)
    data = data.prefetch(2)
    for batch_x, batch_y in data.take(5):
        print(batch_x, batch_y)
    data_numpy = list(data.take(4).as_numpy_iterator())

    pass


def downData(url: str = None, filePath: str = None):
    if url == None or filePath == None:
        raise ValueError("url or filepath is None!")
    else:
        if os.path.exists(filePath):
            logging.debug(filePath + " is exist")
        else:
            # headers = {'Connection':'close'}
            # url_data = requests.get(url, headers=headers)
            # print(url_data.status_code, url_data.headers)
            # with open(filePath, 'wb') as fp:
            #    fp.write(url_data.content)

            # urllib.urlretrieve(url, filePath)
            logging.debug(url + " download end")
    pass


class FieldHandler(object):
    def __init__(self, train_file_path, test_file_path=None, category_columns=[], continuation_columns=[]):
        """
        train_file_path : 训练数据文件
        test_file_path ： 测试数据文件
        category_columns : 离散数据维度
        continuation_columns ：连续数据维度
        """
        self.train_file_path = None
        self.test_file_path = None
        self.feature_nums = 0
        self.field_dict = {}

        self.category_columns = category_columns
        self.continuation_columns = continuation_columns

        if not isinstance(train_file_path, str):
            raise ValueError("rain file path must str")
        if os.path.exists(train_file_path):
            self.train_file_path = train_file_path
        else:
            raise OSError("train file path isn't exists!")
        if test_file_path:
            if os.path.exists(test_file_path):
                self.test_file_path = test_file_path
            else:
                raise OSError("test file path isn't exists!")
        self.read_data()
        self.df[category_columns].fillna("-1", inplace=True)

        self.build_filed_dict()
        print(self.field_dict, self.feature_nums)
        self.build_standard_scaler()
        print(self.field_dict, self.feature_nums)
        self.field_nums = len(self.category_columns + self.continuation_columns)

    def build_filed_dict(self):
        for column in self.df.columns:
            print(column)
            if column in self.category_columns:
                cv = self.df[column].unique()
                self.field_dict[column] = dict(zip(cv, range(self.feature_nums, self.feature_nums + len(cv))))
                self.feature_nums += len(cv)
            else:
                self.field_dict[column] = self.feature_nums
                self.feature_nums += 1

    def read_data(self):
        if self.train_file_path and self.test_file_path:

            train_df = pd.read_csv(self.train_file_path)[self.category_columns + self.continuation_columns]
            test_df = pd.read_csv(self.test_file_path)[self.category_columns + self.continuation_columns]
            self.df = pd.concat([train_df, test_df])
        else:
            self.df = pd.read_csv(self.train_file_path)[self.category_columns + self.continuation_columns]
            print("train data true,,,,,test data flase")

    def build_standard_scaler(self):
        if self.continuation_columns:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(self.df[self.continuation_columns].values)
        else:
            self.standard_scaler = None


def transformation_data(file_path: str, field_hander: FieldHandler, label=None):
    """
    file_path : 训练数据文件路径
    field_hander ： 文件数据句柄
    lable : 标签维度
    """
    df_v = pd.read_csv(file_path)
    if label:
        if label in df_v.columns:
            labels = df_v[[label]].values.astype("float32")
            print(labels[0:10])
        else:
            raise KeyError(f'label "{label}" isn\'t exists')

    df_v = df_v[field_hander.category_columns + field_hander.continuation_columns]
    df_v[field_hander.category_columns].fillna("-1", inplace=True)
    df_v[field_hander.continuation_columns].fillna(-999, inplace=True)

    if field_hander.standard_scaler:
        df_v[field_hander.continuation_columns] = field_hander.standard_scaler.transform(
            df_v[field_hander.continuation_columns].values)
    df_i = df_v.copy()

    for column in df_v.columns:
        print(field_hander.field_dict[column], type(df_i[column]), column)
        if column in field_hander.category_columns:
            df_i[column] = df_i[column].map(field_hander.field_dict[column])
            df_v[column] = 1
        else:
            df_i[column] = field_hander.field_dict[column]

    df_v = df_v.values.astype("float32")
    df_i = df_i.values.astype("int32")

    features = {
        "df_i": df_i,
        "df_v": df_v
    }

    if label:
        return features, labels
    return features, None


def dataGenerate(path="./Dataset/train.csv"):
    df = pd.read_csv(path)
    df = df[['Pclass', "Sex", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
    class_columns = ['Pclass', "Sex", "SibSp", "Parch", "Embarked"]
    continuous_columns = ['Fare']
    train_x = df.drop('Survived', axis=1)
    train_y = df['Survived'].values
    train_x = train_x.fillna("-1")
    le = LabelEncoder()
    oht = OneHotEncoder()
    files_dict = {}
    s = 0
    for index, column in enumerate(class_columns):
        try:
            train_x[column] = le.fit_transform(train_x[column])
        except:
            pass
        ont_x = oht.fit_transform(train_x[column].values.reshape(-1, 1)).toarray()
        for i in range(ont_x.shape[1]):
            files_dict[s] = index
            s += 1
        if index == 0:
            x_t = ont_x
        else:
            x_t = np.hstack((x_t, ont_x))
    x_t = np.hstack((x_t, train_x[continuous_columns].values.reshape(-1, 1)))
    files_dict[s] = index + 1

    return x_t.astype("float32"), train_y.reshape(-1, 1).astype("float32"), files_dict


def createTrainInputFN(features, label, batch_size=3, num_epochs=10):
    # print(features[0:10], label[0:10], features.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features, label))
    # print("dataset top 2,,,,,,",list(dataset.as_numpy_iterator())[0:2])
    dataset = dataset.shuffle(100, reshuffle_each_iteration=False)
    # print("shuffle top 2,,,,,",list(dataset.as_numpy_iterator())[0:2])
    dataset = dataset.repeat(num_epochs)
    # print("repeat top 2,,,,,", list(dataset.as_numpy_iterator())[0:2])
    dataset = dataset.batch(batch_size=batch_size)
    # print("batch top 2,,,,,,", list(dataset.as_numpy_iterator())[0:2])

    return dataset.as_numpy_iterator()


class HParams(object):
    def __init__(self
                 , model="fm"
                 , opt_type="Adam"
                 , threshold=0.5
                 , loss_type="log_loss"
                 , use_deep=True
                 , layers=[30, 30]
                 , lr=0.01
                 , fm_output_keep_dropout=0.9
                 , line_output_keep_dropout=0.9
                 , deep_input_keep_dropout=0.9
                 , deep_output_keep_dropout=0.9
                 , deep_mid_keep_dropout=0.8
                 , embedding_size=3
                 , use_batch_normal=False
                 , batch_size=64
                 , epoches=100
                 , activation='relu'
                 , seed=20
                 , field_nums=0
                 , feature_nums=0):
        self.model = model
        self.opt_type = opt_type
        self.threshold = threshold
        self.loss_type = loss_type
        self.use_deep = use_deep
        self.layers = layers
        self.lr = lr
        self.fm_output_keep_dropout = fm_output_keep_dropout
        self.line_output_keep_dropout = line_output_keep_dropout
        self.deep_input_keep_dropout = deep_input_keep_dropout
        self.deep_output_keep_dropout = deep_output_keep_dropout
        self.deep_mid_keep_dropout = deep_mid_keep_dropout
        self.embedding_size = embedding_size
        self.use_batch_normal = use_batch_normal
        self.batch_size = batch_size
        self.epoches = epoches
        self.activation = activation
        self.seed = seed
        self.field_nums = field_nums
        self.feature_nums = feature_nums


if __name__ == "__main__":
    test()