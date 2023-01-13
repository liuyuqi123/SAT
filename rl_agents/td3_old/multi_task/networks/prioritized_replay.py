import numpy as np
import random
import pickle

class SumTree(object):
    """
        This SumTree code is modified version of Morvan Zhou:
        https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py"""

    data_pointer = 0

    # initialise tree with all nodes = 0 and data with all values =0

    def __init__(self, capacity):

        self.capacity = capacity
        # number of leaf nodes that contains experiences
        # generate the tree with all nodes values =0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity

        self.tree = np.zeros(
            2 * capacity - 1)  # was initally np.zeroes, but after making memory_size>pretain_length, it had to be adjusted

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            # overwrite
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # change = new priority - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through the tree // update whole tree
        while tree_index != 0:
            """
                        Here we want to access the line above
                        THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
                            0
                           / \
                          1   2
                         / \ / \
                        3  4 5  [6]
                        If we are in leaf at index 6, we updated the priority score
                        We need then to update index 2 node
                        So tree_index = (tree_index - 1) // 2
                        tree_index = (6-1)//2
                        tree_index = 2 (because // round the result)
                        """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        # here we get the leaf_index, priority value of that leaf and experience associated with that index
        """
                        Tree structure and array storage:
                        Tree index:
                             0         -> storing priority sum
                            / \
                          1     2
                         / \   / \
                        3   4 5   6    -> storing priority for experiences
                        Array type for storing:
                        [0,1,2,3,4,5,6]
                        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        # data_index is the child index that we want ro get the data from, leaf index is it's parent ndex
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):
    """
    This SumTree code is modified version of:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, pretrain_length):  # init arg action_size is deprecated
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """

        self.tree = SumTree(capacity)
        self.pretrain_length = pretrain_length

        # # possible actions is deprecated
        # self.action_size = action_size
        # self.possible_actions = np.identity(self.action_size, dtype=int).tolist()

        # hyper-parameters
        self.absolute_error_upper = 1.  # clipped abs error
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001

    def store(self, experience):
        """
         Store a new experience in the tree with max_priority
         When training the priority is to be ajusted according with the prediction error
        """
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # use minimum priority =1
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        # create sample array to contain minibatch
        memory_b = []
        if n > self.tree.capacity:
            print("Sample number more than capacity")
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # calc the priority segment, divide Range into n ranges
        priority_segment = self.tree.total_priority / n

        # increase PER_b each time we sample a minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # calc max_Weights
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # print(np.min(self.tree.tree[-self.tree.capacity:]))
        # print(self.tree.total_priority)
        # print("pmin =" , p_min)
        max_weight = (p_min * n) ** (-self.PER_b)
        # print("max weight =" ,max_weight)

        for i in range(n):
            # A value is uniformly sampled from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            # print("priority =", priority)

            sampling_probabilities = priority / self.tree.total_priority
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            # print("weights =", b_ISWeights[i,0])
            # print(b_ISWeights.shape) shape(64,1)

            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def fill_memory(self, env):
        '''
        Fill memory with experiences.

        If autopilot == True experience is given by autopilot, otherwise agent takes random actions.
        '''

        print("Start to fill memory")

        state = env.reset()

        for i in range(self.pretrain_length):
            if i % 500 == 0:
                print(i, "steps of experience are stored.")

            action = np.array([random.random(), random.random()])
            # target_speed = map_action(action)
            # next_state, reward, done, _ = env.step(target_speed)

            next_state, reward, done, _ = env.step(action)
            experience = state, action, reward, next_state, done

            # print(action)

            self.store(experience)

            if done:
                state = env.reset()
            else:
                state = next_state

        print('Finished filling memory. %s experiences stored.' % self.pretrain_length)

    @staticmethod
    def save_memory(filename, object):
        """"""
        handle = open(filename, "wb")
        pickle.dump(object, handle)

    @staticmethod
    def load_memory(filename):
        """"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
