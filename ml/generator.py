import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=4, n_classes=9, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index:
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = indexes

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size, 4, 19, 1))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample = self.x[ID]
            valid_cols = [*range(len(sample))]
            if len(sample) < 4:
                _valid_col = valid_cols
                while len(valid_cols) < 4:
                    np.random.shuffle(_valid_col)
                    valid_cols.append(_valid_col)

            for idx in range(4):
                x[i, idx, :, :] = np.expand_dims(sample[valid_cols[idx]], -1)

            # Store class
            y[i] = self.y[ID]

        return x, y


def read_data(file):
    df = pd.read_csv(file)
    df.fillna(0, inplace=True)
    df.max()
    # len(df[df['Type'] != "Blank_PCR"])
    x, y = extract_samples(df)

    return x, y


def extract_samples(df):
    x = list()
    y = list()
    x_new, y_new = None, None
    current_counter = np.inf
    for index, row in df.iterrows():
        print(row["replicate_value"])
        new_counter = row["replicate_value"]
        if new_counter <= current_counter:
            if x_new:
                x.append(x_new)
            if y_new:
                y.append(y_new)
            x_new = list([row[1:-1].to_list()])
            y_new = row[0]
        else:
            x_new.append(row[1:-1].to_list())
        current_counter = new_counter

    y.append(y_new)
    x.append(x_new)
    return x, y


def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
    # return
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    label_encoder = preprocessing.LabelEncoder()
    x, y = read_data(file='../data/dataset_single_ann.csv')
    label_encoder.fit(y)
    x_train, y_train, x_test, y_test = split_train_test(x, y)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    print(max([len(xr) for xr in x]))
    dd = DataGenerator(x_train, y_train)
    pass
