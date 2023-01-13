"""
Pure TD3 deployed the multi-task setting.

Developed based on pure td3 network.
"""

import tensorflow as tf
import numpy as np
import random
from .utils import OU


class MtActorNetwork:

    def __init__(self, config, net_name):

        self.config = config
        self.state_size = self.config.state_size_with_task_code
        self.action_size = self.config.action_size

        if self.config.use_lra_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.config.lra,
                global_step=self.global_step,
                decay_steps=self.config.decay_steps,
                decay_rate=self.config.decay_rate,
                staircase=True
            )
        else:
            self.learning_rate = self.config.lra

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.tau = self.config.tau

        self.state_inputs, self.actor_variables, self.action = self.build_actor_network(net_name)
        self.state_inputs_target, self.actor_variables_target, self.action_target = self.build_actor_network('actor_target')

        self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_gradients")
        self.actor_gradients = tf.compat.v1.gradients(self.action, self.actor_variables, -self.action_gradients)
        self.update_target_op = [
            self.actor_variables_target[i].assign(
                tf.multiply(self.actor_variables[i], self.tau) + tf.multiply(self.actor_variables_target[i], 1 - self.tau)
            ) for i in range(len(self.actor_variables))
        ]

        if self.config.use_lra_decay:
            self.optimize = self.optimizer.apply_gradients(
                zip(self.actor_gradients, self.actor_variables),
                global_step=self.global_step
            )
        else:
            self.optimize = self.optimizer.apply_gradients(
                zip(self.actor_gradients, self.actor_variables),
                # global_step=self.global_step
            )

    def split_input(self, state_with_task_code):
        """
        state shape:
        [batch_size, ego_feature_num + npc_feature_num*npc_num + task_code]

        [batch_size, 38]
        """
        vehicle_state = state_with_task_code[:, 0:self.config.ego_feature_num + self.config.npc_num*self.config.npc_feature_num]  # Dims: batch, (ego+npcs)features

        # original line, rank=3, for attention module
        # Dims: batch, 1, features
        # ego_state = tf.reshape(vehicle_state[:, 0:self.config.ego_feature_num], [-1, 1, self.config.ego_feature_num])

        # without ego state encoder
        # Dims: (batch, features)
        ego_state = tf.reshape(
            vehicle_state[:, 0:self.config.ego_feature_num],
            [-1, self.config.ego_feature_num]
        )

        npc_state = tf.reshape(
            vehicle_state[:, self.config.ego_feature_num:],
            [-1, self.config.npc_num, self.config.npc_feature_num]
        )  # Dims: batch, entities, features

        task_code = state_with_task_code[:, -self.config.task_code_size:]  # Dims: batch, len(task code)

        return ego_state, npc_state, task_code

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            ego_state, npc_state, task_code = self.split_input(state_inputs)

            # calculate action
            encoder_1 = tf.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="encoder_1"
            )
            encoder_2 = tf.layers.dense(
                inputs=encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="encoder_2"
            )
            concat = tf.concat([encoder_2[:, i] for i in range(5)], axis=1, name="concat")

            # task code encoder
            task_encoder = tf.layers.dense(
                inputs=task_code,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="task_encoder",
            )

            fc_1 = tf.concat([ego_state, concat, task_encoder], axis=1, name="fc_1")
            fc_2 = tf.layers.dense(
                inputs=fc_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="fc_2",
            )
            # action output
            action_1 = tf.layers.dense(
                inputs=fc_2,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_1",
            )
            action_2 = tf.layers.dense(
                inputs=action_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_2",
            )
            speed_up = tf.layers.dense(
                inputs=action_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="speed_up",
            )
            slow_down = tf.layers.dense(
                inputs=action_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="slow_down",
            )
            action = tf.concat([speed_up, slow_down], axis=1, name="action")

        actor_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return state_inputs, actor_variables, tf.squeeze(action)

    def get_action(self, sess, state):

        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))

        action = sess.run(
            self.action,
            feed_dict={self.state_inputs: state}
        )

        return action

    def get_action_noise(self, sess, state, rate=1):

        rate = np.clip(rate, 0., 1.)
        action = self.get_action(sess, state)
        # print("original action: ", action)

        speed_up_noised = action[0] + OU(action[0], mu=0.6, theta=0.15, sigma=0.3) * rate
        slow_down_noised = action[1] + OU(action[1], mu=0.2, theta=0.15, sigma=0.05) * rate
        action_noise = np.squeeze(np.array([np.clip(speed_up_noised, 0.01, 0.99), np.clip(slow_down_noised, 0.01, 0.99)]))
        # print("noised action: ", action_noise)

        return action_noise

    def get_action_target(self, sess, state):
        target_noise = 0.01
        action_target = sess.run(self.action_target, feed_dict={
                                    self.state_inputs_target: state
                                })
        action_target_smoothing = np.clip(action_target + np.random.rand(2)*target_noise, 0.01, 0.99)
        return action_target_smoothing

    def train(self, sess, state, action_gradients):
        sess.run(
            self.optimize,
            feed_dict={
                self.state_inputs: state,
                self.action_gradients: action_gradients
            }
        )

    def update_target(self, sess):
        sess.run(self.update_target_op)


class MtCriticNetwork:

    def __init__(self, config, net_name):
        self.config = config
        self.state_size = self.config.state_size_with_task_code
        self.action_size = self.config.action_size

        if self.config.use_lrc_decay:
            self.global_step_1 = tf.Variable(0, trainable=False)
            self.global_step_2 = tf.Variable(0, trainable=False)

            self.learning_rate_1 = tf.train.exponential_decay(
                self.config.lrc,
                global_step=self.global_step_1,
                decay_steps=self.config.decay_steps,
                decay_rate=self.config.decay_rate,
                staircase=True
            )

            self.learning_rate_2 = tf.train.exponential_decay(
                self.config.lrc,
                global_step=self.global_step_2,
                decay_steps=self.config.decay_steps,
                decay_rate=self.config.decay_rate,
                staircase=True
            )
        else:
            self.learning_rate_1 = self.config.lrc
            self.learning_rate_2 = self.config.lrc

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate_1)
        self.optimizer_2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate_2)

        self.tau = self.config.tau

        self.state_inputs, self.action, self.critic_variables, self.q_value = self.build_critic_network(net_name)
        self.state_inputs_target, self.action_target, self.critic_variables_target, self.q_value_target = \
            self.build_critic_network(net_name + "_target")

        # # original td3
        # self.target = tf.compat.v1.placeholder(tf.float32, [None])
        # need to set the dim to 4 for multi-task
        self.target = tf.compat.v1.placeholder(tf.float32, [None, self.config.task_code_size])

        self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.absolute_errors = tf.abs(self.target - self.q_value)  # for updating sumtree
        self.loss = tf.reduce_mean(self.ISWeights * tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        self.loss_2 = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        #self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_variables])
        #self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + 0.01*self.l2_loss + 0*self.ISWeights

        if self.config.use_lrc_decay:
            self.optimize = self.optimizer.minimize(
                self.loss,
                global_step=self.global_step_1
            )
            self.optimize_2 = self.optimizer_2.minimize(
                self.loss_2,
                global_step=self.global_step_2
            )
        else:
            self.optimize = self.optimizer.minimize(
                self.loss,
                # global_step=self.global_step_1
            )
            self.optimize_2 = self.optimizer_2.minimize(
                self.loss_2,
                # global_step=self.global_step_2
            )

        self.update_target_op = [self.critic_variables_target[i].assign(tf.multiply(self.critic_variables[i], self.tau) + tf.multiply(self.critic_variables_target[i], 1 - self.tau)) for i in range(len(self.critic_variables))]
        self.action_gradients = tf.gradients(self.q_value, self.action)

    # original lines from pure td3
    # def split_input(self, state): # state:[batch, 31]
    #     ego_state = tf.reshape(state[: , 0:self.config.ego_feature_num], [-1, self.config.ego_feature_num])
    #     npc_state = tf.reshape(state[: , self.config.ego_feature_num:], [-1, self.config.npc_num, self.config.npc_feature_num])
    #     return ego_state, npc_state

    def split_input(self, state_with_task_code):
        """
        state shape:
        [batch_size, ego_feature_num + npc_feature_num * npc_num + task_code]

        [batch_size, 38]
        """
        vehicle_state = state_with_task_code[:, 0:self.config.ego_feature_num + self.config.npc_num*self.config.npc_feature_num]  # Dims: batch, (ego+npcs)features

        # # same issue as actor NN
        # # original line, Dims: (batch, 1, features)
        # ego_state = tf.reshape(vehicle_state[:, 0:self.config.ego_feature_num], [-1, 1, self.config.ego_feature_num])

        # Dims: batch, 1, features
        ego_state = tf.reshape(
            vehicle_state[:, 0:self.config.ego_feature_num],
            [-1, self.config.ego_feature_num]
        )

        npc_state = tf.reshape(vehicle_state[:, self.config.ego_feature_num:], [-1, self.config.npc_num, self.config.npc_feature_num]) # Dims: batch, entities, features

        task_code = state_with_task_code[:, -self.config.task_code_size:]  # Dims: batch, len(task code)

        return ego_state, npc_state, task_code

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            action_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_inputs")
            ego_state, npc_state, task_code = self.split_input(state_inputs)
            # calculate q-value
            encoder_1 = tf.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="encoder_1"
            )
            encoder_2 = tf.layers.dense(
                inputs=encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="encoder_2"
            )
            concat = tf.concat([encoder_2[:, i] for i in range(5)], axis=1, name="concat")
            # task code encoder
            task_encoder = tf.layers.dense(
                inputs=task_code,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="task_encoder"
            )

            # original lines in td3
            # fc_1 = tf.concat([ego_state, concat], axis=1, name="fc_1")

            # fixed version, same as in sumo experiments
            fc_1 = tf.concat([ego_state, concat, task_encoder], axis=1, name="fc_1")
            fc_2 = tf.layers.dense(
                inputs=fc_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="fc_2",
            )

            # state+action merge
            action_fc = tf.layers.dense(
                inputs=action_inputs,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_fc",
            )

            merge = tf.concat([fc_2, action_fc], axis=1, name="merge")
            merge_fc = tf.layers.dense(
                inputs=merge,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="merge_fc"
            )

            # original lines from td3
            # q value output
            # q_value = tf.layers.dense(
            #     inputs=merge_fc,
            #     units=1, activation=None,
            #     kernel_initializer=tf.variance_scaling_initializer(),
            #     name="q_value",
            # )

            # todo check units value
            q_value = tf.layers.dense(
                inputs=merge_fc,
                units=self.config.task_code_size,
                activation=None,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="q_value",
            )

        critic_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return state_inputs, action_inputs, critic_variables, tf.squeeze(q_value)

    def get_q_value_target(self, sess, state, action):
        return sess.run(
            self.q_value_target,
            feed_dict={
                self.state_inputs_target: state,
                self.action_target: action
            }
        )

    def get_gradients(self, sess, state, action):
        return sess.run(
            self.action_gradients,
            feed_dict={
                self.state_inputs: state,
                self.action: action
            }
        )

    def train(self, sess, state, action, target, ISWeights):
        _, _, loss, absolute_errors = sess.run(
            [self.optimize, self.optimize_2, self.loss, self.absolute_errors],
            feed_dict={
                self.state_inputs: state,
                self.action: action,
                self.target: target,
                self.ISWeights: ISWeights
            }
        )

        return loss, absolute_errors

    def update_target(self, sess):
        sess.run(self.update_target_op)
