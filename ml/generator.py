from random import sample, choice, choices, seed, shuffle
from typing import Dict, Iterable

import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, encoder: preprocessing.LabelEncoder, n_features: int = 19,
                 sampling: Dict[str, int] = None,
                 batch_size: int = 1, batches_per_epoch: int = 1,
                 shuffle=True, cut_off=None):
        'Initialization'
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.x = x
        self.y = y
        self.encoder = encoder
        self.n_features = n_features
        self.classes = list(encoder.classes_)
        self.n_classes = len(encoder.classes_)
        self.conc = "single"
        self.shuffle = shuffle
        self.indexes = list()
        self.cut_off = cut_off
        self.mixture = {}
        self.single = {}
        sampling_def = {"single": 1, "mixture": 1, "augment": 0}
        if sampling:
            assert sampling.keys() == sampling_def.keys(), \
                "Keys of sampling dict should (only) contain 'single', 'mixture', and 'augment'"
            assert all([isinstance(sampling_weight, int) or isinstance(sampling_weight, float) for
                        sampling_weight in sampling.values()]), "henk"
            self.sampling = sampling
        else:
            self.sampling = sampling_def
        self._split_data()
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        return self.batches_per_epoch

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index:
        :return:
        """
        # Find list of IDs
        list_ids_temp = [*range(self.batch_size)]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        self.first = True
        seed(a=None, version=2)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size, self.n_features, 1))
        y = np.zeros((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i in list_IDs_temp:
            # select a mode for the generation of the data
            mode = choices(['single', 'augment', 'mixture'],
                           [self.sampling['single'], self.sampling['augment'], self.sampling['mixture']])[0]
            if mode == "single":
                fin_sample, sample_types = self._generate_single_sample()

            elif mode == "mixture":
                fin_sample, sample_types = self._generate_mixture_sample()

            else:
                fin_sample, sample_types = self._generate_augmented_sample()

            x[i, :, 0] = fin_sample

            # Store class
            for sample_type_idx in self.encoder.transform(sample_types):
                y[i, sample_type_idx] = 1

        if self.cut_off:
            x = x > self.cut_off
            x = x.astype(int)
        else:
            x = x / 1000

        if self.first:
            self.first = False

        return x, y

    def _generate_mixture_sample(self):
        sample_type = sample(list(self.mixture.keys()), 1)
        samples = choice(self.mixture[sample_type[0]])
        sample_types = sample_type[0].split("+")
        if self.conc == "single":
            fin_sample = samples[choice(range(samples.shape[0])), :] if len(samples.shape) == 2 else samples
        else:
            fin_sample = np.mean(samples, 0) if len(samples.shape) == 2 else samples
        return fin_sample, sample_types

    def _generate_single_sample(self):
        sample_type = sample(self.classes, 1)
        samples = choice(self.single[sample_type[0]])
        sample_types = sample_type
        if self.conc == "single":
            fin_sample = samples[choice(range(samples.shape[0])), :] if len(samples.shape) == 2 else samples
        else:
            fin_sample = np.mean(samples, 0) if len(samples.shape) == 2 else samples
        return fin_sample, sample_types

    def _generate_augmented_sample(self):
        sample_types = sample(self.classes, 2)
        samples = [choice(self.single[sample_type]) for sample_type in sample_types]
        if self.conc == "single":
            sel_samples = [replicates[choice(range(replicates.shape[0])), :] if len(replicates.shape) == 2 else
                           replicates for replicates in samples]
        if self.conc == "avg":
            sel_samples = [np.mean(np.array(replicates), 0) for replicates in samples]
        fin_sample = np.sum(sel_samples, 0)
        return fin_sample, sample_types

    def _split_data(self):
        for x, y in zip(self.x, self.y):
            if "+" in y:
                if y not in self.mixture:
                    self.mixture[y] = [x]
                else:
                    self.mixture[y].append(x)
                self.indexes.append(['mixture', y, len(self.mixture[y]) - 1])
            else:
                if y not in self.single:
                    self.single[y] = [x]
                else:
                    self.single[y].append(x)
                self.indexes.append(['single', y, len(self.single[y]) - 1])


class EvalGenerator(DataGenerator):
    'Generates data for Keras'
    def __init__(self, x, y, encoder: preprocessing.LabelEncoder, n_features: int = 19,
                 batch_size: int = 1, shuffle: bool = False, cut_off: int = None):
        'Initialization'
        self.x = x
        self.y = y
        self.encoder = encoder
        self.n_features = n_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = len(encoder.classes_)
        self.conc = "avg"
        self.cut_off = cut_off
        self.mixture = {}
        self.single = {}
        self.indexes = list()
        self._split_data()
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        return len(self.y)

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index:
        :return:
        """
        # Find list of IDs
        list_id_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # list_id_temp = self.indexes[index]

        # Generate data
        X, y = self.__data_generation(list_id_temp)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size, self.n_features, 1))
        y = np.zeros((self.batch_size, self.n_classes), dtype=int)

        for i, sample_info  in enumerate(list_IDs_temp):
            sample_group, sample_types, index = sample_info

            if sample_group == "single":
                samples = self.single[sample_types][index]

            if sample_group == "mixture":
                samples = self.mixture[sample_types][index]

            fin_sample = np.mean(samples, 0) if len(samples.shape) == 2 else samples

            x[i, :, 0] = fin_sample

            sample_types = sample_types.split("+")

            # Store class
            for sample_type_idx in self.encoder.transform(sample_types):
                y[i, sample_type_idx] = 1

        if self.cut_off:
            x = x > self.cut_off
            x = x.astype(int)
        else:
            x = x / 1000

        return x, y


def read_data(file: str, include_blanks: bool = False) -> (list, list):
    """
    read in data from csv file into pandas and convert it to samples and classes

    :param file: string of file location
    :param include_blanks: boolean to indicate if blanks should be included in the samples (and classes)
    :return: the samples (x) and corresponding classes (y)
    """
    # read data
    df = pd.read_csv(file, sep=";")
    # fill missings with 0
    df.fillna(0, inplace=True)
    # if blanks should not be included remove them from the data
    if not include_blanks:
        df = df[df['type'] != "Blank_PCR"]
    # extract samples and classes
    x, y = extract_samples(df)

    return x, y


def extract_samples(df: pd.DataFrame) -> (list, list):
    """
    Extract the samples and classes from a dataframe

    :param df: a pandas dataframe
    :return: the samples (x) and corresponding classes (y)
    """
    # init x and y
    x = list()
    y = list()
    # init samples and classes
    x_new, y_new = None, None
    # init counter
    current_counter = np.inf
    # iterrate over all rows
    for index, row in df.iterrows():
        # get new counter
        new_counter = row["replicate_value"]
        # if new counter is below current counter new sample is initiated
        if new_counter <= current_counter:
            # catch first iteration
            if x_new:
                # append sample (as array) to x
                x.append(np.array(x_new))
            if y_new:
                # append class to y
                y.append(y_new)
            # retrieve sample and class
            x_new = list([row[1:-1]])
            y_new = row[0]
        else:
            # retrieve sample (as class is already determined)
            x_new.append(row[1:-1])
        # update counter
        current_counter = new_counter

    # add last sample and class
    x.append(np.array(x_new))
    y.append(y_new)

    return x, y


def split_train_test(x, y) -> (list, list, list, list):
    """
    split the data into a train and test, stratifying for the classes

    :param x: samples
    :param y: classes
    :return: train samples, test samples, train classes, test classes
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1, random_state=42)

    return x_train, y_train, x_test, y_test


def generate_data(include_blanks: bool = False, include_mixtures: bool = False):
    label_encoder = preprocessing.LabelEncoder()
    x_single, y_single = read_data(file='data/dataset_single_ann.csv', include_blanks=include_blanks)
    label_encoder.fit(y_single)
    x_single_train, y_single_train, x_single_test, y_single_test = split_train_test(x_single, y_single)

    if include_mixtures:
        x_mix, y_mix = read_data(file='data/dataset_mixture_ann.csv', include_blanks=include_blanks)
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
