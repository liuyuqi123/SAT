"""
Load roll out datasets.
"""

import os
import sys

import numpy as np
import pickle
import shutil
import matplotlib.pyplot as plt


def load_single_pickle_dataset(file_path):
    """
    Load single dataset from dataset file path.

    :param file_path:
    :return:
    """

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def load_pickle_datasets(folder_path):
    """
    Load pickle datasets from a folder.
    """

    name_list = os.listdir(folder_path)
    # todo fix sorting method
    name_list.sort()

    # data_dict_list = []
    conc_data_dict = {}

    for dataset_name in name_list:
        dataset_path = os.path.join(folder_path, dataset_name)
        data_dict = load_single_pickle_dataset(dataset_path)
        # data_dict_list.append(data_dict)

        # init dict if empty
        if not conc_data_dict:
            for key, item in data_dict.items():
                conc_data_dict[key] = item
            continue

        # concatenate data according to dict keys
        for key, item in data_dict.items():
            conc_data_dict[key] += item  # item is List

    return conc_data_dict


def load_single_dataset(file_path):
    """
    Load single npz dataset with given file path.
    """

    dataset = np.load(file_path, allow_pickle=True)

    # # todo add args to set output dict keys
    # print(
    #     'Dataset: ', '\n',
    #     file_path, '\n',
    #     'Available keys: ', dataset.files,
    # )

    batch = dict()  # data stored as dict
    for key in dataset.files:
        batch[key] = dataset[key]

    # print('dataset is loaded.')

    return batch


def load_multiple_datasets(parent_folder):
    """
    Load all datasets stored in given folder.

    :param parent_folder: path of the folder
    :return: concatenate datasets into a batch of single dict
    """

    name_list = os.listdir(parent_folder)
    # name_list.sort()

    name_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))

    merge_batch = {}

    for dataset_name in name_list:

        dataset_path = os.path.join(parent_folder, dataset_name)

        # todo load file in sub-folders recursively
        # skip if not file
        if not os.path.isfile(dataset_path):
            continue

        batch = load_single_dataset(dataset_path)

        # todo fix this, merge dataset into single batch
        #  ref on safety layer class method

        # initialization
        if not merge_batch:
            for k, v in batch.items():
                merge_batch[k] = v
            continue

        # concatenate array into same batch
        for k, v in batch.items():
            _data = merge_batch[k]

            merge_batch[k] = np.concatenate((_data, v,), axis=0,)

            # debug
            # print('')

    return merge_batch


def run_load_dataset():
    """
    Load datasets to debug.
    """
    # source dataset path
    file_path = './data_collection/output/merge/left/failure/dataset_0.npz'

    # target dataset path
    file_path = ''

    batch = load_single_dataset(file_path)

    print('')


def load_batch_plot_ttc_dist():
    """
    Load datasets and plot ttc and normed ttc distribution.
    """
    folder_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_normed_new/left/failure'

    batch = load_multiple_datasets(folder_path)

    plt.figure()
    # plt.figure(figsize=(50, 10))

    # plt.suptitle('Traffic Flow Distribution of {}. \n total veh num={}'.format(key, total_vehicle_num))  # , y=0.98)

    axe = plt.subplot(3, 1, 1)
    axe.set_title('original ttc')
    axe.set_xlabel('s')
    axe.set_ylabel('pdf')
    plt.hist(batch['c_2'], density=True, bins=150)

    axe = plt.subplot(3, 1, 2)
    axe.set_title('clip ttc (0, 20)')
    axe.set_xlabel('s')
    axe.set_ylabel('pdf')

    data = np.clip(batch['c_2'], 0, 20)
    plt.hist(data, density=True, bins=150)

    # axe.set_xscale('log', base=100)

    axe = plt.subplot(3, 1, 3)
    axe.set_title('normed ttc A=10, dx=1')
    axe.set_xlabel('normed value')
    axe.set_ylabel('pdf')

    plt.hist(batch['c_2_norm'], density=True, bins=150)

    plt.tight_layout()
    plt.show()

    print('')


def main():
    """
    Some major usages.
    """

    # ==========  load pickle dataset  ==========
    # file_path = '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/2021-09-23-Time16-06-44/full_data/left/failure/full_0.pkl'
    # data_dict = load_single_pickle_dataset(file_path)

    # folder_path = '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/2021-09-23-Time16-06-44/full_data/left/failure'
    # data_dict_list = load_pickle_datasets(folder_path)


    # ==================================================
    # ==========  load npz dataset  ==========
    #
    # file_path = '/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/original_state/left/hybrid/dataset_0.npz'
    # batch = load_single_dataset(file_path)

    folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_rename/straight/failure'
    dataset_list = load_multiple_datasets(folder)

    print('')


if __name__ == '__main__':

    # main()

    # rename_and_merge_datasets()

    run_load_dataset()

    # load_batch_plot_ttc_dist()
