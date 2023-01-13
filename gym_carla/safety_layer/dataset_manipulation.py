"""
Some methods to manipulate and regroup the datasets.
"""

import os
import sys

import numpy as np
import pickle
import shutil
from interval import Interval

from gym_carla.safety_layer.data_analysis import save_dataset, plot_hist

from gym_carla.safety_layer.load_datasets import load_multiple_datasets, load_single_dataset


def atan_norm(x, A=1., dx=1.):
    """
    Normalization using tangent function.

    y = A * 2/pi * arctan(x/dx)
    """
    # value clip
    x = np.clip(x, 0., float("inf"))

    y = A * 2 / np.pi * np.arctan(x / dx)

    y = np.clip(y, 0., float("inf"))

    return y


def normalize_ttc(
        dataset_folder,
        dataset_size=1e4,
        save_data=False,
        output_path=None,
        plot_dist_fig=True,
        save_fig=False,
):
    """
    This method is fixed based on the original version.

    Process ttc using normalization function.

    Optional to save it into new dataset.
    """

    # all data stored into single dict
    batch = load_multiple_datasets(dataset_folder)

    index_list = []
    original_ttc_array = batch['c_2']
    original_ttc_next_array = batch['c_2_next']

    zoom = Interval(0., 5.)

    # the whole batch needs to be filtered
    for i in range(batch['c_2'].shape[0]):

        a = original_ttc_array[i]
        b = original_ttc_next_array[i, 0]

        # keep starting ttc less than 5.
        if original_ttc_array[i] <= 5.:
            if original_ttc_next_array[i, 0] in zoom:
                index_list.append(i)

        print('')

    # todo add a debug dataset to test this part
    # generate the new batch
    new_batch = {}
    for key, data in batch.items():
        #
        new_data = data[index_list, :]
        new_batch[key] = new_data
        print('')

    print('')

    # original value, tangent normed value, tangent with quantile normed value
    # only use c_2
    key_list = [
        # ('c_1', 'c_1_norm', 'c_1_q_norm'),
        # ('c_2', 'c_2_norm', 'c_2_q_norm'),
        # ('c_1_next', 'c_1_norm_next', 'c_1_q_norm_next'),
        # ('c_2_next', 'c_2_norm_next', 'c_2_q_norm_next'),

        ('c_2', 'c'),
        ('c_2_next', 'c_next'),
    ]

    # for k, add_key_1, add_key_2 in key_list:
    for k, add_key in key_list:

        # ttc_array = batch[k]

        # =====  use tangent function to norm ttc value  =====
        # # use quantile as the factor
        # q_value = np.quantile(a=ttc_array, q=.8)

        # # normed using tangent function with a scale factor
        # ttc_q_norm = tan_normalize(ttc_array, q_value)

        # # fix the atan norm function
        # A, dx = 10., 1.
        # ttc_norm = atan_norm(ttc_array, A=A, dx=dx)

        # # append data to the batch
        # batch[add_key_1] = ttc_norm
        # batch[add_key_2] = ttc_q_norm

        # batch[add_key] = ttc_norm

        # ===================================
        ttc_array = new_batch[k]

        # new linear transformation
        # env.simulator_timestep_length = 0.05
        step_length = 0.05

        alpha = 6.
        beta = .9
        fixed_c_array = alpha * step_length - beta * ttc_array

        print('')

        # batch[add_key] = fixed_c_array
        new_batch[add_key] = fixed_c_array


        # plot hist figure
        if plot_dist_fig:
            plot_hist(
                dict_data={
                    'original ttc': original_ttc_array,
                    'clipped ttc': ttc_array,
                    'constraint value': fixed_c_array,
                },
                save=save_fig,
                legend=['original ttc', 'clipped ttc', 'constraint']
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
            data_dict=new_batch,
            dataset_size=int(dataset_size),
            folder_path=output_path,
        )

    print('')


# ====================  some developing methods  ====================
def copy_datasets(parent_folder):
    """"""

    sub_folder_list = os.listdir(parent_folder)

    for path in sub_folder_list:
        if path == 'copy':
            continue
        # src path
        src_path = os.path.join(parent_folder, path, 'original_state')
        # dst path
        dst_path = os.path.join(parent_folder, 'copy', path, 'original_state')

        shutil.copytree(src_path, dst_path)

        print('')


def create_folder_tree(parent_folder):
    """
    A temp method for datasets copy.
    """

    sub_folder_list = os.listdir(parent_folder)

    for path in sub_folder_list:
        new_path = os.path.join(parent_folder, 'copy', path)
        os.makedirs(new_path, exist_ok=True)

        print('')

    print('')


def run_copy_datasets():
    """"""
    # path = '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/n_1e2_size_1e4/test'

    path_list = [
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/n_1e2_size_1e4',
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/size_1e4',
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/size_1e5',
    ]

    for path in path_list:
        copy_datasets(path)

    print('')

    path_list = [
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/n_1e2_size_1e4',
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/size_1e4',
        '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/size_1e5',
    ]

    for path in path_list:
        create_folder_tree(path)

    print('')


def regroup_datasets(parent_folder, output_path):
    """"""

    sub_folder_list = os.listdir(parent_folder)

    count_dict = {
        'left': 0,
        'straight': 0,
        'right': 0,
    }

    for datetime in sub_folder_list:

        sub_path = os.path.join(parent_folder, datetime, 'original_state')

        # route
        _route = os.listdir(sub_path)[0]

        if _route in ['straight', 'straight_0']:
            route = 'straight'
            count_dict['straight'] += 1
        elif _route == 'left':
            route = 'left'
            count_dict['left'] += 1
        elif _route == 'right':
            route = 'right'
            count_dict['right'] += 1

        else:
            raise ValueError('Wrong route option')

        category_list = ['failure', 'hybrid', 'success']
        for category in category_list:

            src_folder = os.path.join(sub_path, _route, category)
            datasets_list = os.listdir(src_folder)

            # todo test this line
            # sort datasets list by index
            datasets_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))

            for dataset in datasets_list:
                # src path refers the path of single dataset
                src_path = os.path.join(sub_path, _route, category, dataset)

                # dst path
                dst_folder = os.path.join(
                    output_path,
                    route,
                    category,
                    datetime,
                )
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, dataset)

                # folder
                # shutil.copytree(src_path, dst_path)
                # file
                shutil.copy(src_path, dst_path)

                print('')


def re_calculate_ttc_norm():
    """
    Calculate normed ttc value through merged datasets.
    """
    # # test
    # parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge/test'
    # output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_normed/test'

    # =====
    # # test
    # parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/debug_train'

    # original dataset
    parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge'
    parent_folder = './data_collection/output/merge'

    # # get normed ttc data and save to new dataset file
    # output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_normed_q4'

    # # use new atan norm function
    # output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_normed_new'

    # clip data and linear transform
    output_parent_path = './data_collection/output/merge_clipped'

    route_list = os.listdir(parent_folder)
    for route in route_list:
        route_folder = os.path.join(parent_folder, route)
        category_list = os.listdir(route_folder)

        # desired: category_list = ['failure', 'hybrid', 'success']
        for category in category_list:
            datasets_folder = os.path.join(parent_folder, route, category)
            output_path = os.path.join(output_parent_path, route, category)
            os.makedirs(output_path, exist_ok=True)

            normalize_ttc(
                dataset_folder=datasets_folder,
                dataset_size=1e4,  # fix dataset_size to 1e4
                save_data=True,
                output_path=output_path,
                plot_dist_fig=False,  # True, False
                save_fig=False,
            )

            print('')

        print('')

    print('')


def merge_datasets():
    """
    Merge regrouped datasets into same folder.

    use save_dataset method.
    """
    # source and target folder
    parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/todo/regroup_1e4'
    output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/1e4'

    route_list = os.listdir(parent_folder)
    for route in route_list:

        # category_list = ['failure', 'hybrid', 'success']
        for category in ['success', 'failure', 'hybrid']:
            datetime_list = os.listdir(os.path.join(parent_folder, route, category))

            for datetime in datetime_list:
                sub_folder_path = os.path.join(parent_folder, route, category, datetime)
                output_path = os.path.join(output_parent_path, route, category, datetime)
                os.makedirs(output_path, exist_ok=True)

                #
                datasets_list = os.listdir(sub_folder_path)

                normalize_ttc(
                    dataset_folder=sub_folder_path,
                    dataset_size=1e4,  # fix dataset_size to 1e4
                    save_data=True,
                    output_path=output_path,
                    plot_dist_fig=True,
                    save_fig=False,
                )

                print('')

        print('')

    print('')


def rename_datasets(parent_folder, target_folder):
    """
    Rename datasets of a 2-level tree structure.

     - parent folder
         - sub folder 1
            - datasets
         - sub folder 2
            - datasets
         ...    ...    ...

    """
    dataset_index = 0
    folder_list = os.listdir(parent_folder)
    folder_list.sort()  # sort by datetime

    for folder in folder_list:
        sub_folder_path = os.path.join(parent_folder, folder)
        # dataset file list
        sub_folder_list = os.listdir(sub_folder_path)
        # sort file with number
        sub_folder_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))

        for file in sub_folder_list:
            abs_file_path = os.path.join(sub_folder_path, file)
            # todo improve file check
            if not os.path.isfile(abs_file_path):
                print('A wrong file is found: {}'.format(abs_file_path))
                continue

            # rename file with new index
            new_file_name = 'dataset_' + str(int(dataset_index)) + '.' + file.split('.')[-1]

            os.makedirs(target_folder, exist_ok=True)
            new_file_path = os.path.join(target_folder, new_file_name)

            shutil.copy(abs_file_path, new_file_path)

            dataset_index += 1

        print('Data files in {} are regrouped.'.format(sub_folder_path))

    print('All data files in {} are regrouped. {} in total.'.format(parent_folder, dataset_index))


def rename_and_merge_datasets():
    """
    Rename and merge the datasets form regroup folder.
    """
    # datasets source
    parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/todo/regroup_merge'
    # target
    target_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge'

    route_list = os.listdir(parent_folder)  # ['left', 'straight', 'right']
    for route in route_list:
        route_folder = os.path.join(parent_folder, route)
        category_list = os.listdir(route_folder)

        for category in category_list:
            category_folder = os.path.join(route_folder, category)
            # target folder
            dst_folder = os.path.join(target_folder, route, category)

            rename_datasets(
                parent_folder=category_folder,
                target_folder=dst_folder,
            )

            print('')

        print('')

    print('')


def process_datasets():
    """
    todo original method, this method is split into 2 methods.

    Re-size the 1e5 datasets.

    normed ttc is added into datasets as well.
    """
    # 1e4 datasets
    parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/todo/regroup_1e4'
    output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/1e4'

    # # 1e5 datasets
    # parent_folder = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/todo/regroup_1e5'
    # output_parent_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/1e5'

    route_list = os.listdir(parent_folder)
    for route in route_list:

        # category_list = ['failure', 'hybrid', 'success']
        for category in ['success', 'failure', 'hybrid']:
            datetime_list = os.listdir(os.path.join(parent_folder, route, category))

            for datetime in datetime_list:
                sub_folder_path = os.path.join(parent_folder, route, category, datetime)
                output_path = os.path.join(output_parent_path, route, category, datetime)
                os.makedirs(output_path, exist_ok=True)

                normalize_ttc(
                    dataset_folder=sub_folder_path,
                    dataset_size=1e4,  # fix dataset_size to 1e4
                    save_data=True,
                    output_path=output_path,
                    plot_dist_fig=True,
                    save_fig=False,
                )

                print('')

        print('')

    print('')


def run():
    """"""
    # # debug
    # test_parent_folder = './data_collection/todo/test'
    #
    # # 1e4 datasets
    # parent_folder = './data_collection/todo/size_1e4'

    # 1e5 datasets
    parent_folder = './data_collection/todo/size_1e5'

    # ===============  output path  ===============
    # output_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output'

    # 1e5 outputs
    output_path = './data_collection/todo/regroup_1e5'

    for r in ['left', 'right', 'straight']:
        _output_path = os.path.join(output_path, r)
        os.makedirs(_output_path, exist_ok=True)

    regroup_datasets(parent_folder, output_path)


if __name__ == '__main__':

    # run()

    # process_datasets()

    # rename_and_merge_datasets()

    # calculate normed ttc value and save to new datasets
    re_calculate_ttc_norm()
