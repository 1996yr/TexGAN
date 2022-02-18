import os
import tensorflow as tf
from framework.modules.module_base import module_base
from exp_config.config import Config


class solver_base(module_base):
    def __init__(self, var_scope):
        module_base.__init__(self)
        self.var_scope = var_scope

    def build_graph(self, **kwargs):
        loss_list = kwargs['loss_list']
        var_prefix_list = kwargs['var_prefix']
        lr_in = kwargs['lr_in']
        solver_param = kwargs['params']
        reuse = kwargs['reuse']


        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        tf_vars = tf.trainable_variables()
        var_list = []
        var_filename = ''
        for var_prefix in var_prefix_list:
            var_filename = var_filename + var_prefix + '-'
            if var_prefix != '':
                var_list = var_list + [var for var in tf_vars if var_prefix in var.name]
            else:
                var_list = tf_vars

        f = open(os.path.join(Config.global_cfg.folder_cfg.log_dir, var_filename[0:-1]+'.txt'), 'w')
        for var in var_list:
            f.write(var.name + ' ' * (80 - len(var.name)) + str(var) + '\r\n')
        f.close()

        assert len(var_list) > 0

        if len(loss_list) == 1:
            loss_full_list = loss_list[0]
        else:
            loss_full_list = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _loss = sum([_term[g_id] for _term in loss_list])
                loss_full_list.append(_loss)
        train_op, grads_and_vars = self.construct_solver(loss_full_list, var_list, solver_param, lr_in, reuse)
        self.public_ops['train_op'] = train_op
        self.public_ops['grads_and_vars'] = grads_and_vars


    def average_gradient(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, var in grad_and_vars:
                grads.append(g)
                # with tf.name_scope('average_grad'):
                #     expanded_g = tf.expand_dims(g, 0)
                #     grads.append(expanded_g)
            # grad = tf.concat(grads, axis=0)
            # grad = tf.reduce_mean(grad, axis=0)
            with tf.name_scope('average_grad'):
                try:
                    grad = sum(grads) / len(Config.global_cfg.meta_cfg.gpu_list)
                except:
                    print(grads)
                    exit()

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def sum_gradient(self, tower_grads):
        sum_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, var in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_sum(grad, axis=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            sum_grads.append(grad_and_var)
        return sum_grads


    def construct_solver(self, _loss_on_each_gpu, _var_list, solver_param, lr_in, reuse):
        solver_param_list = solver_param.split('-')
        solver_type = solver_param_list[0]
        with tf.variable_scope(self.var_scope, reuse=reuse):
            # with tf.device('/cpu:0'):
            if solver_type == 'adam':
                beta1, beta2 = float(solver_param_list[2]), float(solver_param_list[3])
                self.tf_solver = tf.train.AdamOptimizer(learning_rate=lr_in,
                                            beta1 = beta1,
                                            beta2 = beta2)

            tower_grads = []
            for g_id, _loss in enumerate(_loss_on_each_gpu):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    grad_tower = self.tf_solver.compute_gradients(_loss, _var_list)
                    tower_grads.append(grad_tower)

            # with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grad_avg = self.average_gradient(tower_grads)
                apply_gradient_op = self.tf_solver.apply_gradients(grad_avg)
            return apply_gradient_op, grad_avg