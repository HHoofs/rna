from copy import deepcopy
from random import random

import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from random import sample


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, encoder: preprocessing.LabelEncoder, batch_size=4, n_classes=9, shuffle=True, cut_off=None):
        'Initialization'
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.encoder = encoder
        self.n_classes = n_classes
        self.conc = "single"
        self.shuffle = shuffle
        self.indexes = None
        self.cut_off = cut_off
        self.on_epoch_end()
        self.mixture = {}
        self.single = {}
        self._split_data()

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
        x = np.zeros((self.batch_size, 19, 1))
        y = np.zeros((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            mode = random.choices(['single', 'augment', 'mixture'], [.3, .3 ,.3])
            if mode == "single":
                sample_type = sample(self.encoder.classes_, 1)
                sample = random.choice(self.single[sample_type])

            if mode == "mixture":
                sample_type = sample(list(self.mixture.keys()), 1)
                sample = random.choice(self.mixture[sample_type])

            if mode == "augment":
                sample_types = sample(self.encoder.classes_, 3)
                samples = [random.choice(self.single[sample_type]) for sample_type in sample_types]
                if self.conc == "single":
                    sel_samples = [conv_to_2d(np.array(random.choice(replicates))) for replicates in samples]
                if self.conc == "avg":
                    sel_samples = [conv_to_2d(np.mean(conv_to_2d(np.array(replicates)), 0)) for replicates in samples]
                fin_sample = conv_to_2d(np.sum(conv_to_2d(np.array(sel_samples)), 0))
            else:
                sample_types = sample_type[0].split("+")
                if self.conc == "single":
                    fin_sample = conv_to_2d(np.array(random.choice(sample)))
                elif self.conc == "avg":
                    fin_sample = conv_to_2d(np.mean(conv_to_2d(np.array(sample)), 0))

            x[i, :, :] = fin_sample

            # Store class
            for sample_type in sample_types:
                class_idx = self.encoder.transform(sample_type)
                y[i, class_idx] = 1

        if self.cut_off:
            x = x > self.cut_off
            x = x.astype(int)

        return x, y

    def _split_data(self):
        for x, y in zip(self.x, self.y):
            if "+" in y:
                if y not in self.mixture:
                    self.mixture[y] = [x]
                else:
                    self.mixture[y].append(x)
            else:
                if y not in self.single:
                    self.single[y] = [x]
                else:
                    self.single[y].append(x)

def read_data(file, include_blanks=False):
    df = pd.read_csv(file, sep=";")
    df.fillna(0, inplace=True)
    # df.max()
    if not include_blanks:
        df = df[df['type'] != "Blank_PCR"]
    x, y = extract_samples(df)

    return x, y


def extract_samples(df):
    x = list()
    y = list()
    x_new, y_new = None, None
    current_counter = np.inf
    for index, row in df.iterrows():
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

    x.append(x_new)
    y.append(y_new)
    return x, y


def conv_to_2d(samples_array):
    if len(samples_array.shape) == 1:
        samples_array = np.expand_dims(samples_array, -1)
    return samples_array

def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1)
    # return
    return x_train, y_train, x_test, y_test


def generate_data(include_blanks: bool=False, include_mixtures: bool=False):
    label_encoder = preprocessing.LabelEncoder()
    x_single, y_single = read_data(file='data/data_single.csv', include_blanks=include_blanks)
    label_encoder.fit(y_single)
    x_single_train, y_single_train, x_single_test, y_single_test = split_train_test(x_single, y_single)

    if include_mixtures:
        x_mix, y_mix = read_data(file='data/data_mixture.csv', include_blanks=include_blanks)
        x_mix_train, y_mix_train, x_mix_test, y_mix_test = split_train_test(x_mix, y_mix)

        return x_single_train + x_mix_train, \
               list(y_single_train) + list(y_mix_train), \
               x_single_test + x_mix_test, \
               list(y_single_test) + list(y_mix_test), \
               label_encoder
    else:
        return x_single_train, \
               list(y_single_train), \
               x_single_test, \
               list(y_single_test), \
               label_encoder


if __name__ == '__main__':
    label_encoder = preprocessing.LabelEncoder()
    x, y = read_data(file='../data/data_single.csv')
    label_encoder.fit(y)
    x2, y2 = read_data(file='../data/data_mixture.csv')
    x_train, y_train, x_test, y_test = split_train_test(x2, y2)
    y_train = [label_encoder.transform(y_mixt.split("+")) for y_mixt in y_train]
    y_test = [label_encoder.transform(y_mixt.split("+")) for y_mixt in y_test]



    print(max([len(xr) for xr in x]))
    dd = DataGenerator(x_train, y_train)
    pass
