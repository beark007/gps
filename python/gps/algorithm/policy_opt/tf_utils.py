import tensorflow as tf
import numpy as np
import time


def check_list_and_convert(the_object):
    if isinstance(the_object, list):
        return the_object
    return [the_object]


class TfMap:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""

    def __init__(self, input_tensor, target_output_tensor,
                 precision_tensor, output_op, loss_op, var_lists, fp=None):
        self.input_tensor = input_tensor
        self.target_output_tensor = target_output_tensor
        self.precision_tensor = precision_tensor
        self.output_op = output_op
        self.loss_op = loss_op
        self.img_feat_op = fp

        self.var_lists = var_lists

    @classmethod
    def init_from_lists(cls, inputs, outputs, loss, var_lists, fp=None):
        inputs = check_list_and_convert(inputs)
        outputs = check_list_and_convert(outputs)
        loss = check_list_and_convert(loss)
        var_lists = check_list_and_convert(var_lists)
        if len(inputs) < 3:  # pad for the constructor if needed.
            inputs += [None]*(3 - len(inputs))
        return cls(inputs[0], inputs[1], inputs[2], outputs[0], loss[0], var_lists, fp=fp)

    def get_var_lists_tensor(self):
        return self.var_lists

    def get_input_tensor(self):
        return self.input_tensor

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_target_output_tensor(self):
        return self.target_output_tensor

    def set_target_output_tensor(self, target_output_tensor):
        self.target_output_tensor = target_output_tensor

    def get_precision_tensor(self):
        return self.precision_tensor

    def set_precision_tensor(self, precision_tensor):
        self.precision_tensor = precision_tensor

    def get_output_op(self):
        return self.output_op

    def get_feature_op(self):
        return self.img_feat_op

    def set_output_op(self, output_op):
        self.output_op = output_op

    def get_loss_op(self):
        return self.loss_op

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op

class TfMap_withPre:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""

    def __init__(self, input_tensor, target_output_tensor, target_output_tensor_pre,
                 precision_tensor, output_op, loss_op, fp=None):
        self.input_tensor = input_tensor
        self.target_output_tensor = target_output_tensor
        self.target_output_tensor_pre = target_output_tensor_pre
        self.precision_tensor = precision_tensor
        self.output_op = output_op
        self.loss_op = loss_op
        self.img_feat_op = fp

    @classmethod
    def init_from_lists(cls, inputs, outputs, loss, fp=None):
        inputs = check_list_and_convert(inputs)
        outputs = check_list_and_convert(outputs)
        loss = check_list_and_convert(loss)
        if len(inputs) < 3:  # pad for the constructor if needed.
            inputs += [None]*(3 - len(inputs))
        return cls(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], loss[0], fp=fp)

    def get_input_tensor(self):
        return self.input_tensor

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_target_output_tensor(self):
        return self.target_output_tensor

    def get_target_output_tensor_pre(self):
        return self.target_output_tensor_pre

    def set_target_output_tensor(self, target_output_tensor):
        self.target_output_tensor = target_output_tensor

    def get_precision_tensor(self):
        return self.precision_tensor

    def set_precision_tensor(self, precision_tensor):
        self.precision_tensor = precision_tensor

    def get_output_op(self):
        return self.output_op

    def get_feature_op(self):
        return self.img_feat_op

    def set_output_op(self, output_op):
        self.output_op = output_op

    def get_loss_op(self):
        return self.loss_op

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op


class TfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self, loss_scalar, solver_name='adam', base_lr=None, lr_policy=None,
                 momentum=None, weight_decay=None, fc_vars=None,
                 last_conv_vars=None, vars_to_opt=None, summery=None):
        self.base_lr = base_lr
        self.lr_policy = lr_policy
        self.momentum = momentum
        self.solver_name = solver_name
        self.loss_scalar = loss_scalar

        self.summery = summery

        if self.lr_policy != 'fixed':
            raise NotImplementedError('learning rate policies other than fixed are not implemented')

        self.weight_decay = weight_decay
        """ not exactly known with this part"""
        if weight_decay is not None:
            if vars_to_opt is None:
                trainable_vars = tf.trainable_variables()
            else:
                trainable_vars = vars_to_opt
            loss_with_reg = self.loss_scalar
            for var in trainable_vars:
                loss_with_reg += self.weight_decay*tf.nn.l2_loss(var)
            self.loss_scalar = loss_with_reg

        self.solver_op = self.get_solver_op()
        if fc_vars is not None:
            self.fc_vars = fc_vars
            self.last_conv_vars = last_conv_vars
            self.fc_solver_op = self.get_solver_op(var_list=fc_vars)

    def get_solver_op(self, var_list=None, loss=None):
        solver_string = self.solver_name.lower()
        if var_list is None:
            var_list = tf.trainable_variables()
        if loss is None:
            loss = self.loss_scalar


        if solver_string == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.base_lr,
                                          beta1=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.base_lr,
                                             decay=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.base_lr,
                                              momentum=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.base_lr,
                                             initial_accumulator_value=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.base_lr).minimize(loss, var_list=var_list)
        else:
            raise NotImplementedError("Please select a valid optimizer.")


    def get_last_conv_values(self, sess, feed_dict, num_values, batch_size):
        i = 0
        values = []
        while i < num_values:
            batch_dict = {}
            start = i
            end = min(i+batch_size, num_values)
            for k,v in feed_dict.iteritems():
                batch_dict[k] = v[start:end]
            batch_vals = sess.run(self.last_conv_vars, batch_dict)
            values.append(batch_vals)
            i = end
        i = 0
        final_value = np.concatenate([values[i] for i in range(len(values))])
        return final_value

    def get_var_values(self, sess, var, feed_dict, num_values, batch_size):
        i = 0
        values = []
        while i < num_values:
            batch_dict = {}
            start = i
            end = min(i+batch_size, num_values)
            for k,v in feed_dict.iteritems():
                batch_dict[k] = v[start:end]
            batch_vals = sess.run(var, batch_dict)
            values.append(batch_vals)
            i = end
        i = 0
        final_values = np.concatenate([values[i] for i in range(len(values))])
        return final_values

    def update_loss(self,fisher_info=None, var_lists=None, var_lists_old=None, lam=None):
        """
        update loss when encounter a new condition
        """
        if fisher_info:
            if not hasattr(self, "ewc_loss"):
                self.ewc_loss = self.loss_scalar
            else:
                self.ewc_loss = self.loss_scalar
                for num_task in range(len(fisher_info)):
                    for v in range(len(var_lists)):
                        self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(fisher_info[num_task][v].astype(np.float32),
                                                                 tf.square(var_lists[v] - var_lists_old[v])))
        else:
            self.ewc_loss = self.loss_scalar

        """ update the train loss"""

    def restore(self, sess, var_list):
        self.star_vars = []
        for v in range(len(var_list)):
            self.star_vars.append(var_list[v].eval())

        if hasattr(self, "star_vars"):
            for v in range(len(var_list)):
                sess.run(var_list[v].assign(self.star_vars[v]))

    def reset_solver_op(self, loss, var_lists):
        """ using Adam as default optimizer"""
        return tf.train.AdamOptimizer(learning_rate=self.base_lr,
                                      beta1=self.momentum).minimize(loss, var_list=var_lists)


    def __call__(self, feed_dict, sess, device_string="/cpu:0", use_fc_solver=False):
        with tf.device(device_string):
            if use_fc_solver:
                # loss = sess.run([self.loss_scalar, self.fc_solver_op], feed_dict)
                loss = sess.run([self.ewc_loss, self.fc_solver_op, self.summery], feed_dict)
                # with sess.as_default:
                #     self.solver_op.run(session=sess, feed_dict=feed_dict)
                #     loss = sess.run([self.ewc_loss, self.summery], feed_dict)
                print('using fc solver')
            else:
                # loss = sess.run([self.loss_scalar, self.solver_op], feed_dict)
                loss = sess.run([self.ewc_loss, self.solver_op, self.summery], feed_dict)
                # with sess.as_default:
                #     self.solver_op.run(feed_dict=feed_dict)
                #     loss = sess.run([self.ewc_loss, self.summery], feed_dict)
                # print('not using fc solver')
            return loss[0], loss[1]

