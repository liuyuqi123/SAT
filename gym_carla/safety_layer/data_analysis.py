"""
Consider normalization of the TTC value for comparison.

"""

import os

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from gym_carla.safety_layer.load_datasets import load_multiple_datasets, load_single_dataset


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


def dumpclean(obj):
    """
    Print data in a dict recursively.

    :param obj:
    :return:
    """
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k)
                dumpclean(v)
            else:
                print('%s : %s' % (k, v))
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print(v)
    else:
        print(obj)


def tan_normalize(ttc, dx=1.):
    """
    Normalization using tangent function.

    :param ttc:
    :param dx:
    :return:
    """
    # value clip
    ttc = np.clip(ttc, 0., float("inf"))

    norm_ttc = 2 / np.pi * np.arctan(ttc / dx)

    norm_ttc = np.clip(norm_ttc, 0., 1.)

    return norm_ttc


def normalize_ttc(
        dataset_folder,
        dataset_size=1e4,
        save_data=False,
        output_path=None,
        plot_dist_fig=True,
        save_fig=False,
):
    """
    Process ttc using normalization function.

    Optional to save it into new dataset.
    """

    # all data stored into single dict
    batch = load_multiple_datasets(dataset_folder)

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

    # save to datasets
    if save_data:
        # output_path path will be created by save_dataset method
        if not output_path:
            output_path = os.path.join(dataset_folder, 'normalized')

        # set dataset size through input args
        # dataset_size = int(1e4)

        save_dataset(
            data_dict=batch,
            dataset_size=int(dataset_size),
            folder_path=output_path,
        )

    print('')


def save_dataset(data_dict, dataset_size: int, folder_path):
    """
    Save the dataset with desired dataset size.
    """

    os.makedirs(folder_path, exist_ok=True)

    dict_len = data_dict['action'].shape[0]
    dataset_number = int(dict_len // dataset_size)
    remainder = int(dict_len % dataset_size)

    # index refers time scale of the dataset
    for index in range(dataset_number):
        # update all key from data_dict
        save_dict = dict()
        for k, v in data_dict.items():
            # clip the buffer until filled position
            save_dict[k] = v[index: int(index+dataset_size), :]  # 1st dim is step length

        # save the dataset
        path = os.path.join(folder_path, 'dataset_' + str(index) + '.npz')
        np.savez(path, **save_dict)
        print('dataset {} is saved.'.format(index))

    # todo fix bugs
    # if there are remainder
    if remainder >= 1:
        save_dict = dict()

        for k, v in data_dict.items():
            # clip the buffer until filled position
            save_dict[k] = v[int(-1*remainder):, :]  # 1st dim is step length
            print('')

        # sample randomly from original datasets to fill the remainder dataset
        additional_index = np.random.choice(dict_len, int(dataset_size - remainder))
        for i in additional_index:
            for k, v in data_dict.items():
                # # debug
                # _original = save_dict[k]

                _add = v[i, :][np.newaxis, :]
                save_dict[k] = np.concatenate((save_dict[k], _add), axis=0)

                # _re = save_dict[k]
                # print('')

        # save the filled remainder dataset
        path = os.path.join(folder_path, 'dataset_' + str(dataset_number) + '.npz')
        np.savez(path, **save_dict)

        print('dataset {} is saved.'.format(dataset_number))

    print('')


def plot_hist(dict_data, save=False, legend: list = None):
    """
    todo merge this method into util script

    Plot hist figure of 1-dim array data.
    """

    plt.figure()
    # plt.figure(figsize=(50, 10))

    index = int(1)
    for k, v in dict_data.items():

        # plt.suptitle('Traffic Flow Distribution of {}. \n total veh num={}'.format(key, total_vehicle_num))  # , y=0.98)

        axe = plt.subplot(len(dict_data), 1, index)
        axe.set_title(k+' Distribution, '+legend[index-1])
        axe.set_xlabel(k)
        axe.set_ylabel('pdf')
        # if legend:
        #     _legend = legend[index-1]
        #     axe.text(1, 1, _legend)
        # test_v = np.array(v)
        plt.hist(np.array(v), density=True, bins=150)

        index += 1

    plt.tight_layout()
    plt.show()

    if save:
        save_path = os.path.join('./ttc_distribution/', TIMESTAMP)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, k + '.png'))

    print('')


def run():
    """
    1. load datasets into single batch
     ref on SafetyLayer class method

    2. use numpy.quantile() to calculate p quantile

    """

    # debug
    dataset_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/original_state/left/failure'

    # all data stored into single dict
    batch = load_multiple_datasets(dataset_folder)

    key_list = [
        'c_1',
        'c_2',

        # normalize ttc with tangent function
        # 'c_1_norm',
        # 'c_2_norm',
    ]

    result = {}

    for ttc in key_list:
        result[ttc] = {
            'mean': None,
            'var': None,
            'quantile': {},
        }
        ttc_array = batch[ttc]

        # plot hist figure
        plot_hist(
            dict_data={ttc: ttc_array},
            save=False,
        )

        # mean
        mean = np.mean(ttc_array)
        result[ttc]['mean'] = mean

        # variance
        var = np.var(ttc_array)
        result[ttc]['var'] = var

        # quantile of TTC distribution
        for p in [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]:
            q_value = np.quantile(
                a=ttc_array,
                q=p,
            )

            result[ttc]['quantile']['p='+str(p)] = q_value
            print('')

    dumpclean(result)

    output_path = './ttc_distribution.pkl'

    with open(output_path, "wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    print('')


def load_ttc_distributin(path = './ttc_distribution.pkl'):
    """"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    dumpclean(data)

    return data


def test_merge_result():
    """"""
    path_1 = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/hybrid/dataset_0.npz'
    dataset_1 = load_single_dataset(path_1)
    path_2 = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/hybrid/hybrid_with_norm/dataset_0.npz'
    dataset_2 = load_single_dataset(path_2)

    print('')


def calculate_ttc_quantile():
    """
    Since forgetting to save the quantile value of the ttc distribution, we have to re-calculate it again...
    """
    # parent folder is the merge_normed
    parent_folder = './data_collection/output/merge'
    route_list = os.listdir(parent_folder)

    result_dict = {'left': {}, 'straight': {}, 'right': {}}

    for route in route_list:
        route_folder = os.path.join(parent_folder, route)
        category_list = os.listdir(route_folder)

        # desired: category_list = ['failure', 'hybrid', 'success']
        for category in category_list:

            datasets_folder = os.path.join(parent_folder, route, category)

            # store all data into single dict
            batch = load_multiple_datasets(datasets_folder)

            # original value, tangent normed value, tangent with quantile normed value
            key_list = [
                ('c_1', 'c_1_norm', 'c_1_q_norm'),
                ('c_2', 'c_2_norm', 'c_2_q_norm'),
                # ('c_1_next', 'c_1_norm_next', 'c_1_q_norm_next'),
                # ('c_2_next', 'c_2_norm_next', 'c_2_q_norm_next'),
            ]

            for k, add_key_1, add_key_2 in key_list:

                # original ttc value in ndarray
                ttc_array = batch[k]

                # calculate the quantile point
                q = 0.8
                q_value = np.quantile(a=ttc_array, q=q)  # 0.8 is the critical parameter, determined manually

                print('route: {}, category: {}, {} quantile: {}'.format(route, category, q, q_value))

                # norm with tangent function
                ttc_norm = tan_normalize(ttc_array)
                # normed using tangent function with a scale factor
                ttc_q_norm = tan_normalize(ttc_array, q_value)

                # # compare data with the batch
                # original_ttc_norm = batch[add_key_1]
                # original_ttc_q_norm = batch[add_key_2]

                print('')

                # plot hist figure
                plot_hist(
                    dict_data={
                        'original ttc': ttc_array,
                        'ttc_norm': ttc_norm,
                        'ttc_q_norm': ttc_q_norm,
                    },
                    save=False,
                    legend=['original', 'tangent', '0.8 quantile: '+str(round(q_value, 2))]
                )

                # # debug, compare with last time result
                # plot_hist(
                #     dict_data={
                #         'original ttc': ttc_array,
                #
                #         'ttc_norm': ttc_norm,
                #         'ttc_q_norm': ttc_q_norm,
                #
                #         'original_ttc_norm': original_ttc_norm,
                #         # 'original_ttc_q_norm': original_ttc_q_norm,
                #     },
                #     save=False,
                #     legend=['original', 'tangent', '0.8 quantile: '+str(round(q_value, 2))]
                # )

                result_dict[route][category] = q_value

                print('')

            print('')

    # save the result into a json file
    output_folder = './data_collection/output/merge_normed'
    with open(os.path.join(output_folder, 'quantile.json'), 'w') as fp:
        json.dump(result_dict, fp, indent=2)

    print('')


if __name__ == '__main__':

    # run()

    # test_merge_result()

    #
    calculate_ttc_quantile()
