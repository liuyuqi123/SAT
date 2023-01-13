"""
TD3 deployed with social attention and multi-task.
"""

import numpy as np
import random
import tensorflow as tf
from .utils import OU


class SocMtActorNetwork:

    def __init__(self, config, name):

        self.config = config
        self.state_size = self.config.state_size_with_att_mask_and_task_code
        self.action_size = self.config.action_size
        
        self.features_per_head = 64
        self.feature_head = 1

        self.tau = self.config.tau

        if self.config.use_lra_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.config.lra,
                global_step=self.global_step,
                decay_steps=self.config.decay_steps,
                decay_rate=self.config.decay_rate,
                staircase=True,
            )
        else:
            self.learning_rate = self.config.lra

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.state_inputs, self.actor_variables, self.action, self.att_matrix = self.build_actor_network(name)
        self.state_inputs_target, self.actor_variables_target, self.action_target, self.att_matrix_target = \
            self.build_actor_network(name + "_target")

        self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_gradients")
        self.actor_gradients = tf.compat.v1.gradients(self.action, self.actor_variables, -self.action_gradients)

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

        self.update_target_op = [self.actor_variables_target[i].assign(
            tf.multiply(self.actor_variables[i], self.tau) + tf.multiply(self.actor_variables_target[i], 1 - self.tau)
        ) for i in range(len(self.actor_variables))]

    def split_input(self, state_size_with_att_mask_and_task_code):
        """
        state:
        [batch, ego_feature_num + npc_feature_num * npc_num + attention_mask + task_code]

        :param state_size_with_att_mask_and_task_code:
        :return:
        """
        vehicle_state = state_size_with_att_mask_and_task_code[:, 0:self.config.ego_feature_num + self.config.npc_num * self.config.npc_feature_num]  # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(vehicle_state[:, 0:self.config.ego_feature_num], [-1, 1, self.config.ego_feature_num])  # Dims: batch, 1, features
        npc_state = tf.reshape(vehicle_state[:, self.config.ego_feature_num:], [-1, self.config.npc_num, self.config.npc_feature_num])  # Dims: batch, entities, features

        vehicle_mask = state_size_with_att_mask_and_task_code[:, -(self.config.vehicle_mask_size + self.config.task_code_size):-self.config.task_code_size]  # Dims: batch, len(mask)
        vehicle_mask = vehicle_mask < 0.5

        task_code = state_size_with_att_mask_and_task_code[:, -self.config.task_code_size:]  # Dims: batch, len(task code)

        return ego_state, npc_state, vehicle_mask, task_code

    def attention(self, query, key, value, vehicle_mask):
        """
        Compute a Scaled Dot Product Attention.
        :param query: size: batch, head, 1 (ego-entity), features
        :param key:  size: batch, head, entities, features
        :param value: size: batch, head, entities, features
        :param vehicle_mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
        :return: the attention softmax(QK^T/sqrt(dk))V
        """
        d_k = self.features_per_head
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / np.sqrt(d_k)
        mask_constant = scores * 0 + -1e9
        if vehicle_mask is not None:
            scores = tf.where(vehicle_mask, mask_constant, scores)
        p_attn = tf.nn.softmax(scores, dim=-1)
        att_output = tf.matmul(p_attn, value)

        return att_output, p_attn

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            ego_state, npc_state, vehicle_mask, task_code = self.split_input(state_inputs)
            # vehicle encoders
            ego_encoder_1 = tf.layers.dense(
                inputs=ego_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.orthogonal_initializer(),
                name="ego_encoder_1"
            )
            ego_encoder_2 = tf.layers.dense(
                inputs=ego_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.orthogonal_initializer(),
                name="ego_encoder_2"
            )  # Dims: batch, 1, 64
            npc_encoder_1 = tf.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.orthogonal_initializer(),
                name="npc_encoder_1"
            )
            npc_encoder_2 = tf.layers.dense(
                inputs=npc_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.orthogonal_initializer(),
                name="npc_encoder_2"
            )  # Dims: batch, entities, 64

            vehicle_encoders_output = tf.concat([ego_encoder_2, npc_encoder_2], axis=1)  # Dims: batch, npcs_entities + 1, 64

            # social attention module
            # q,k,v vectors
            query_ego = tf.layers.dense(
                inputs=ego_encoder_2,
                units=self.feature_head * self.features_per_head,
                use_bias=None,
                kernel_initializer=tf.orthogonal_initializer(),
                name="query_ego"
            )
            key_all = tf.layers.dense(
                inputs=vehicle_encoders_output,
                units=self.feature_head * self.features_per_head,
                use_bias=None,
                kernel_initializer=tf.orthogonal_initializer(),
                name="key_all"
            )
            value_all = tf.layers.dense(
                inputs=vehicle_encoders_output,
                units=self.feature_head * self.features_per_head,
                use_bias=None,
                kernel_initializer=tf.orthogonal_initializer(),
                name="value_all"
            )
            # dimensions: Batch, entity, head, feature_per_head
            query_ego = tf.reshape(query_ego, [-1, 1, self.feature_head, self.features_per_head])
            key_all = tf.reshape(key_all, [-1, self.config.npc_num + 1, self.feature_head, self.features_per_head])
            value_all = tf.reshape(value_all, [-1, self.config.npc_num + 1, self.feature_head, self.features_per_head])
            # dimensions: Batch, head, entity, feature_per_head
            query_ego = tf.transpose(query_ego, perm=[0, 2, 1, 3])
            key_all = tf.transpose(key_all, perm=[0, 2, 1, 3])
            value_all = tf.transpose(value_all, perm=[0, 2, 1, 3])
            vehicle_mask = tf.reshape(vehicle_mask, [-1, 1, 1, self.config.vehicle_mask_size])
            vehicle_mask = tf.tile(vehicle_mask, [1, self.feature_head, 1, 1])
            # attention mechanism and its outcome
            att_result, att_matrix = self.attention(query_ego, key_all, value_all, vehicle_mask)
            att_result = tf.reshape(att_result, [-1, self.features_per_head * self.feature_head], name = 'att_result')
            att_matrix = tf.identity(att_matrix, name="att_matrix")
            att_result_encoder = tf.layers.dense(
                inputs=att_result,
                units=128,
                use_bias=None,
                kernel_initializer=tf.orthogonal_initializer(),
                name="att_result_encoder"
            )
            # task code encoder
            task_encoder = tf.layers.dense(
                inputs=task_code,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="task_encoder"
            )
            # concat
            att_and_task = tf.concat([att_result_encoder, task_encoder], axis=1, name="att_and_task")
            # decoders
            decoder_1 = tf.layers.dense(
                inputs=att_and_task,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="decoder_1"
            )
            decoder_2 = tf.layers.dense(
                inputs=decoder_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="decoder_2"
            )
            # action output
            action_positive = tf.layers.dense(
                inputs=decoder_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_positive"
            )
            action_negative = tf.layers.dense(
                inputs=decoder_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_negative"
            )

            action = tf.concat([action_positive, action_negative], axis=1, name="action")

        actor_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return state_inputs, actor_variables, tf.squeeze(action), att_matrix

    def get_attention_matrix(self, sess, state):
        """
        todo is this method for visualize the matrix distribution???
        :param sess:
        :param state:
        :return:
        """
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        attention_matrix = sess.run(
            self.attention_matrix,
            feed_dict={self.state_inputs: state}
        )

        return attention_matrix

    def get_action(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))

        action = sess.run(
            self.action, feed_dict={self.state_inputs: state}
        )

        return action

    def get_action_noise(self, sess, state, rate=1):
        """"""
        # norm to [0, 1]
        rate = np.clip(rate, 0., 1.)
        # get original action
        action = self.get_action(sess, state)

        # add noise to the original action
        action_positive_noised = action[0] + OU(action[0], mu=0.6, theta=0.15, sigma=0.3) * rate
        action_negative_noised = action[1] + OU(action[1], mu=0.2, theta=0.15, sigma=0.05) * rate
        action_noise = np.squeeze(np.array([np.clip(action_positive_noised, 0.01, 0.99), np.clip(action_negative_noised, 0.01, 0.99)]))

        return action_noise

    def get_action_target(self, sess, state):
        target_noise = 0.01
        action_target = sess.run(
            self.action_target,
            feed_dict={self.state_inputs_target: state}
        )
        action_target_smoothing = np.clip(action_target + np.random.rand(self.action_size)*target_noise, 0.01, 0.99)

        return action_target_smoothing

    def train(self, sess, state, action_gradients):
        """
        todo add global step to debug
        """
        sess.run(
            self.optimize,
            feed_dict={
                self.state_inputs: state,
                self.action_gradients: action_gradients
            }
        )

    def update_target(self, sess):
        sess.run(self.update_target_op)


class SocMtCriticNetwork:

    def __init__(self, config, name):

        self.config = config
        self.state_size = self.config.state_size_with_att_mask_and_task_code
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

        self.state_inputs, self.action, self.critic_variables, self.q_value = self.build_critic_network(name)
        self.state_inputs_target, self.action_target, self.critic_variables_target, self.q_value_target = self.build_critic_network(name + "_target")

        self.target = tf.compat.v1.placeholder(tf.float32, [None, self.config.task_code_size])
        self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.absolute_errors = tf.abs(self.target - self.q_value)  # for updating sumtree
        self.action_gradients = tf.gradients(self.q_value, self.action)

        self.loss = tf.reduce_mean(self.ISWeights * tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        self.loss_2 = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))

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

    def split_input(self, state_size_with_att_mask_and_task_code):
        """"""
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask]
        vehicle_state = state_size_with_att_mask_and_task_code[:, 0:self.config.ego_feature_num + self.config.npc_num*self.config.npc_feature_num]  # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(vehicle_state[: , 0:self.config.ego_feature_num], [-1, 1, self.config.ego_feature_num])  # Dims: batch, 1, features
        npc_state = tf.reshape(vehicle_state[: , self.config.ego_feature_num:], [-1, self.config.npc_num, self.config.npc_feature_num])  # Dims: batch, entities, features

        vehicle_mask = state_size_with_att_mask_and_task_code[:, -(self.config.vehicle_mask_size + self.config.task_code_size):-self.config.task_code_size]  # Dims: batch, len(mask)
        vehicle_mask = vehicle_mask < 0.5
        task_code = state_size_with_att_mask_and_task_code[:, -self.config.task_code_size:]  # Dims: batch, len(task code)

        return ego_state, npc_state, vehicle_mask, task_code

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            action_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_inputs")
            ego_state, npc_state, _, task_code = self.split_input(state_inputs)
            ego_state = tf.squeeze(ego_state, axis=1)
            # vehicle state encoders
            ego_encoder_1 = tf.layers.dense(
                inputs=ego_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="ego_encoder_1"
            )
            ego_encoder_2 = tf.layers.dense(
                inputs=ego_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="ego_encoder_2"
            )
            npc_encoder_1 = tf.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="npc_encoder_1"
            )
            npc_encoder_2 = tf.layers.dense(
                inputs=npc_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="npc_encoder_2"
            )
            # task code encoder
            task_encoder = tf.layers.dense(
                inputs=task_code,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="task_encoder"
            )
            
            # converge state encoders
            concat_1 = tf.concat([npc_encoder_2[:, i] for i in range(5)], axis=1, name="concat_1")
            concat_2 = tf.concat([task_encoder, ego_encoder_2, concat_1], axis=1, name="concat_2")
            concat_fc = tf.layers.dense(
                inputs=concat_2,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="concat_fc"
            )

            # state+action merge
            action_fc = tf.layers.dense(
                inputs=action_inputs,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="action_fc"
            )
            
            # decoder
            converge = tf.concat([concat_fc, action_fc], axis=1, name="converge")
            decoder = tf.layers.dense(
                inputs=converge,
                units=256, activation=tf.nn.tanh,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="decoder"
            )
            # q value output
            q_value = tf.layers.dense(
                inputs=decoder,
                units=self.config.task_code_size,
                activation=None,
                kernel_initializer=tf.variance_scaling_initializer(),
                name="q_value"
            )

        critic_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return state_inputs, action_inputs, critic_variables, tf.squeeze(q_value)

    def get_q_value_target(self, sess, state, action):
        return sess.run(self.q_value_target, feed_dict={
            self.state_inputs_target: state,
            self.action_target: action
        })

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
