"""
Compare result of random policy rolling out with and without safety layer.

"""
import numpy as np
import matplotlib.pyplot as plt


# # todo capsule method
# def plot_errorbar(data):
#     """"""
#     pass


# # ===============  original lines  ===============
# # without safety
# data_1 = np.array([
#     [365, 341, 425, 423, 320],
# ]) / 1000
#
# # with safety
# data_2 = np.array([
#     [387, 326, 420, 427, 328],
# ]) / 1000
#
# # fixed safety
# data_3 = np.array([
#     [388, 412, 383, 398, 390],
# ]) / 1000
#
# mean_1 = np.mean(data_1)
# std_1 = np.std(data_1)
#
# mean_2 = np.mean(data_2)
# std_2 = np.std(data_2)
#
# mean_3 = np.mean(data_3)
# std_3 = np.std(data_3)


# merge data
data_total = np.array([
    [365, 341, 425, 423, 320],
    [387, 326, 420, 427, 328],
    [388, 412, 383, 398, 390],
    [388, 364, 430, 436, 344],
    [390, 346, 405, 410, 355],
]) / 1000

label_list = (
    'random',
    'Normed C',
    'fixed C \n seed=0',
    'fixed C \n seed=0:4',
    'fixed C \n hybrid data \n seed=0:4',
)

# total data group
data_number = data_total.shape[0]

# mean list
mean_list = list(np.mean(data_total, axis=1))
print('mean: {}'.format(mean_list))

# std list
std_list = list(np.std(data_total, axis=1))
print('std: {}'.format(std_list))

fig = plt.figure()

errorbar = plt.errorbar(list(range(1, data_number+1)), mean_list, std_list, fmt='o', capsize=5.)
plt.grid()

plt.xlim((0, data_number+1))
plt.xticks(list(range(1, data_number+1)), label_list)

# plt.ylim((0, data_number+1))
plt.ylabel('Success Rate')

plt.title('Comparison on random policy.')

# # original
# plt.text(1, 0.425, '%.2f%%' % (100*mean_list[0]), ha='center')
# plt.text(4, 0.425, '%.2f%%' % (100*mean_list[3]), ha='center')

for i in range(data_number):
    plt.text(i+1, 0.425, '%.2f%%' % (100 * mean_list[i]), ha='center')

plt.show()

print('')

