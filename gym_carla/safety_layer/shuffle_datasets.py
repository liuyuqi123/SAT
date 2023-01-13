"""
Methods for shuffling datasets for multi-task training.


if self.route_option == 'left':
    task_code = [1, 0, 0, 1]
elif self.route_option == 'right':
    task_code = [0, 0, 1, 1]
elif self.route_option == 'straight':
    task_code = [0, 1, 0, 1]

"""

import os
import sys

import numpy as np
import pickle
import shutil
import random

from gym_carla.safety_layer.load_datasets import load_single_dataset
from gym_carla.safety_layer.data_analysis import tan_normalize, plot_hist, save_dataset


def load_multi_task_datasets(dataset_path_list):
    """
    Load datasets stored in a 2-level folder structure.

    :param dataset_path_list: list of the dataset file path
    :return: concatenate datasets into a batch of single dict
    """
    # # original, sort by index number
    # name_list = os.listdir(parent_folder)
    # name_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))

    merge_batch = {}

    for dataset_path in dataset_path_list:

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


def add_norm_to_batch(
        batch,
        plot_dist_fig=True,
        save_fig=False,
):
    """
    Append task code into batch data

    This method is inherited from normalize_ttc method.
    Process ttc using normalization function.

    Optional to save it into new dataset.
    """
    # original value, tangent normed value, tangent with quantile normed value
    key_list = [
        ('c_1', 'c_1_norm', 'c_1_q_norm'),
        ('c_2', 'c_2_norm', 'c_2_q_norm'),
        ('c_1_next', 'c_1_norm_next', 'c_1_q_norm_next'),
        ('c_2_next', 'c_2_norm_next', 'c_2_q_norm_next'),
    ]

    for k, add_key_1, add_key_2 in key_list:

        ttc_array = batch[k]

        # =====  use tangent function to norm ttc value  =====
        # # use quantile as the factor
        # q_value = np.quantile(a=ttc_array, q=.8)
        # use the fixed factor
        q_value = 4.

        ttc_norm = tan_normalize(ttc_array)
        # normed using tangent function with a scale factor
        ttc_q_norm = tan_normalize(ttc_array, q_value)

        # append data to the batch
        batch[add_key_1] = ttc_norm
        batch[add_key_2] = ttc_q_norm

        # plot hist figure
        if plot_dist_fig:
            plot_hist(
                dict_data={
                    'original ttc': ttc_array,
                    'ttc_norm': ttc_norm,
                    'ttc_q_norm': ttc_q_norm,
                },
                save=save_fig,
                legend=['original', 'tangent', '0.8 quantile: '+str(round(q_value, 2))]
            )

        print('')

    return batch


def shuffle_datasets():
    """
    Shuffle datasets for the multi-task safety layer.

    Procedure:
     - load datasets into batch from all merge_folder
     - add task code to the data batch according to the route option
     - shuffle data batch and save into the merged datasets

    """
    # # # debug
    # parent_folder = './data_collection/output/debug_merge_multi_task'
    # output_path = './data_collection/output/debug_merge_normed_multitask'

    # use relative path
    parent_folder = './data_collection/output/merge'
    # add normed  ttc_2 into datasets
    output_path = './data_collection/output/merge_normed_multitask'

    dataset_path_list = []

    route_list = os.listdir(parent_folder)
    for route in route_list:
        # route_folder = os.path.join(parent_folder, route)

        # # only use the failure datasets
        # # desired: category_list = ['failure', 'hybrid', 'success']
        # category_list = os.listdir(route_folder)
        category_list = ['failure']

        for category in category_list:
            # datasets
            datasets_folder = os.path.join(parent_folder, route, category)
            datasets_list = os.listdir(datasets_folder)

            # append dataset file path
            for file_name in datasets_list:
                file_path = os.path.join(datasets_folder, file_name)
                dataset_path_list.append(file_path)

            # --------

            # # sub-folder list
            # sub_folder_list = os.listdir(parent_folder)
            #
            # name_list = []
            # for sub_folder in sub_folder_list:
            #     sub_folder_path = os.path.join(parent_folder, sub_folder)
            #     file_list = os.listdir(sub_folder_path)
            #     name_list = name_list + file_list if name_list else file_list
            #
            # random.shuffle(name_list)

            # -----

    random.shuffle(dataset_path_list)

    # put all data to single batch
    batch = load_multi_task_datasets(dataset_path_list)

    # load and process the data
    normed_batch = add_norm_to_batch(
        batch,
        plot_dist_fig=False,
        save_fig=False,
    )

    # todo shuffle the batch
    index_list = list(range(np.shape(normed_batch['action'])[0]))
    random.shuffle(index_list)

    normed_batch['task_code'] = np.squeeze(normed_batch['task_code'])

    # shuffle and save to a new batch
    full_batch = {}

    for key, item in normed_batch.items():

        # init a full-zero array
        data_shape = np.shape(item)
        data_batch = np.zeros(data_shape)
        print('In processing {}...'.format(key))

        for i, index in enumerate(index_list):
            data_batch[i, :] = item[index, :]

        full_batch[key] = data_batch
        print('Data element {} is processed.'.format(key))

    # restore task code to its original dimension
    full_batch['task_code'] = full_batch['task_code'][:, np.newaxis, :]
    print('')

    # save the datasets
    dataset_size = int(1e4)
    save_dataset(
        data_dict=normed_batch,
        dataset_size=int(dataset_size),
        folder_path=output_path,
    )

    print('done')


if __name__ == '__main__':

    shuffle_datasets()

