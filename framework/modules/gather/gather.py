import tensorflow as tf
import numpy as np
from exp_config.config import Config
from framework.modules.module_base import module_base
from framework.modules.codebase.layers import denormalizeInput
from framework.modules.codebase.visualization import tileImage



@tf.custom_gradient
def weight_grad_identity(x, weight_map):
    def grad(dy1):
        weight_grad = dy1 * weight_map * Config.gather_cfg.normal_weight
        return weight_grad, None
    return tf.identity(x), grad

class gather6chart_mix(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']
        tex_generator = kwargs['tex_generator']

        chart_synbatch_all_gpu = tex_generator.public_ops['fake_chart6view_synbatch']
        gather_map_synbatch_all_gpu = data_loader.public_ops['input_gather_map_synbatch']
        if phase == 'train':
            chart_realbatch_all_gpu = tex_generator.public_ops['fake_chart6view_realbatch']
            gather_map_realbatch_all_gpu = data_loader.public_ops['input_gather_map_realbatch']
            if Config.gather_cfg.normal_weight > 0.0:
                weight_map_synbatch_all_gpu = data_loader.public_ops['input_weight_map_synbatch']
            else:
                weight_map_synbatch_all_gpu = []
            if Config.loss.smooth_loss > 0.0:
                smooth_gather_all_gpu = data_loader.public_ops['input_smooth_gather_map_synbatch']
            else:
                smooth_gather_all_gpu = []
        else:
            weight_map_synbatch_all_gpu = None
            chart_realbatch_all_gpu = None
            gather_map_realbatch_all_gpu = None
            smooth_gather_all_gpu = None

        if phase == 'train':
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_group_per_gpu = Config.dataloader_cfg.batch_group // len(gpu_list)
            assert batch_group_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_group
        elif phase == 'test':
            gpu_list = [0]
            batch_group_per_gpu = 1
        else:
            raise NotImplementedError


        fake_img_recons_realbatch_all_gpu = []
        fake_img_recons_synbatch_all_gpu = []
        smooth_gather_img_synbatch_all_gpu = []

        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):

                chart_synbatch = chart_synbatch_all_gpu[g_id]
                gather_map_synbatch = gather_map_synbatch_all_gpu[g_id]
                fake_img_synbatch = self.bilinear_gather_chart6view(chart_synbatch, gather_map_synbatch, phase)
                if phase == 'train' and Config.gather_cfg.normal_weight > 0.0:
                    fake_img_synbatch = weight_grad_identity(fake_img_synbatch, weight_map_synbatch_all_gpu[g_id])
                fake_img_recons_synbatch_all_gpu.append(fake_img_synbatch)
                if phase == 'train':
                    chart_realbatch = chart_realbatch_all_gpu[g_id]
                    gather_map_realbatch = gather_map_realbatch_all_gpu[g_id]

                    fake_img_realbatch = self.bilinear_gather_chart6view(chart_realbatch, gather_map_realbatch, phase)


                    if Config.loss.smooth_loss > 0.0:
                        smooth_gather = smooth_gather_all_gpu[g_id]
                        smooth_gather_img_synbatch = self.bilinear_gather_chart6view(chart_synbatch, smooth_gather, phase)
                        smooth_gather_img_synbatch_all_gpu.append(smooth_gather_img_synbatch)
                    fake_img_recons_realbatch_all_gpu.append(fake_img_realbatch)



        self.public_ops['fake_img_recons_synbatch'] = fake_img_recons_synbatch_all_gpu
        if phase == 'train':
            self.public_ops['fake_img_recons_realbatch'] = fake_img_recons_realbatch_all_gpu
            if Config.loss.smooth_loss > 0.0:
                self.public_ops['smooth_gather_img_recons_synbatch'] = smooth_gather_img_synbatch_all_gpu


        if phase == 'train':
            fake_recons = []
            for g_id in range(len(gpu_list)):
                for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_d_synbatch):
                    fake_recons.append(tf.clip_by_value(denormalizeInput(fake_img_recons_synbatch_all_gpu[g_id][i]), 0.0, 1.0))
            self.add_summary(tileImage(tf.stack(fake_recons), nCol=Config.global_cfg.exp_cfg.n_view_d_synbatch),
                             'train', 'image', 'fake_recons')

            if Config.loss.smooth_loss > 0.0:
                smooth_recons = []
                for g_id in range(len(gpu_list)):
                    for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_d_synbatch):
                        smooth_recons.append(tf.clip_by_value(denormalizeInput(smooth_gather_img_synbatch_all_gpu[g_id][i]), 0.0, 1.0))
                self.add_summary(tileImage(tf.stack(smooth_recons), nCol=Config.global_cfg.exp_cfg.n_view_d_synbatch),
                                 'train', 'image', 'smooth_recons')


    def bilinear_gather_chart6view(self, input_chart, gather_map, phase):
        if phase == 'train':
            group_num = input_chart.shape[0] // Config.global_cfg.exp_cfg.n_view_g
        elif phase == 'test':
            group_num = 1
        else:
            raise NotImplementedError
        result_list = []
        # print('group_num', group_num)
        if Config.global_cfg.exp_cfg.random_background:
            bg_color = tf.random_uniform(shape=[1, 1, 1, 1], minval=-0.9, maxval=-0.1, name='random_background_syn')
        else:
            bg_color = tf.constant(shape=[1, 1, 1, 1], value=-1, dtype=tf.float32)
        if phase == 'train':
            for i in range(group_num):
                chart_list = []
                for chart_id in range(Config.global_cfg.exp_cfg.n_view_g):
                    chart_list.append(input_chart[chart_id + i * Config.global_cfg.exp_cfg.n_view_g, ::, ::, ::])
                chart = tf.concat(chart_list, axis=1) # (128, n_view * 128, 3)
                if Config.global_cfg.exp_cfg.random_background:
                    chart = chart - bg_color[0]
                else:
                    chart = chart - bg_color[0]
                gather_n_img = gather_map.shape[0] // group_num
                u = gather_map[i * gather_n_img: (i+1)*gather_n_img, ::, ::, 0:1]
                v = gather_map[i * gather_n_img: (i+1)*gather_n_img, ::, ::, 1:2]

                u_floor_float, u_ceil_float = tf.floor(u), tf.ceil(u)
                u_floor_int, u_ceil_int = tf.cast(u_floor_float, tf.int32), tf.cast(u_ceil_float, tf.int32)

                v_floor_float, v_ceil_float = tf.floor(v), tf.ceil(v)
                v_floor_int, v_ceil_int = tf.cast(v_floor_float, tf.int32), tf.cast(v_ceil_float, tf.int32)
                uv_int_00 = tf.concat([u_floor_int, v_floor_int], axis=3)
                uv_int_10 = tf.concat([u_ceil_int, v_floor_int], axis=3)
                uv_int_01 = tf.concat([u_floor_int, v_ceil_int], axis=3)
                uv_int_11 = tf.concat([u_ceil_int, v_ceil_int], axis=3)
                fake_img0_0 = tf.gather_nd(indices=uv_int_00, params=chart)
                fake_img1_0 = tf.gather_nd(indices=uv_int_10, params=chart)
                fake_img0_1 = tf.gather_nd(indices=uv_int_01, params=chart)
                fake_img1_1 = tf.gather_nd(indices=uv_int_11, params=chart)
                w_epsilon = 1e-10
                w0_0 = (u_ceil_float - u) * (v_ceil_float - v) + w_epsilon
                w1_0 = (u - u_floor_float) * (v_ceil_float - v) + w_epsilon
                w0_1 = (u_ceil_float - u) * (v - v_floor_float) + w_epsilon
                w1_1 = (u - u_floor_float) * (v - v_floor_float) + w_epsilon
                w_sum = (u_ceil_float - u_floor_float) * (v_ceil_float - v_floor_float) + 4 * w_epsilon
                w0_0 = w0_0 / w_sum
                w1_0 = w1_0 / w_sum
                w0_1 = w0_1 / w_sum
                w1_1 = w1_1 / w_sum
                fake_img = w0_0 * fake_img0_0 + w1_0 * fake_img1_0 + w0_1 * fake_img0_1 + w1_1 * fake_img1_1
                fake_img = fake_img + bg_color
                result_list.append(fake_img)
            result = tf.concat(result_list, axis=0)
        elif phase == 'test':
            chart_list = []
            for chart_id in range(Config.global_cfg.exp_cfg.n_view_g):
                chart_list.append(input_chart[chart_id, ::, ::, ::])
            chart = tf.concat(chart_list, axis=1)  # (128, n_view * 128, 3)
            if Config.global_cfg.exp_cfg.random_background:
                chart = chart - bg_color[0]
            else:
                chart = chart - bg_color[0]

            u = gather_map[::, ::, ::, 0:1]
            v = gather_map[::, ::, ::, 1:2]

            u_floor_float, u_ceil_float = tf.floor(u), tf.ceil(u)
            u_floor_int, u_ceil_int = tf.cast(u_floor_float, tf.int32), tf.cast(u_ceil_float, tf.int32)

            v_floor_float, v_ceil_float = tf.floor(v), tf.ceil(v)
            v_floor_int, v_ceil_int = tf.cast(v_floor_float, tf.int32), tf.cast(v_ceil_float, tf.int32)
            uv_int_00 = tf.concat([u_floor_int, v_floor_int], axis=3)
            uv_int_10 = tf.concat([u_ceil_int, v_floor_int], axis=3)
            uv_int_01 = tf.concat([u_floor_int, v_ceil_int], axis=3)
            uv_int_11 = tf.concat([u_ceil_int, v_ceil_int], axis=3)
            fake_img0_0 = tf.gather_nd(indices=uv_int_00, params=chart)
            fake_img1_0 = tf.gather_nd(indices=uv_int_10, params=chart)
            fake_img0_1 = tf.gather_nd(indices=uv_int_01, params=chart)
            fake_img1_1 = tf.gather_nd(indices=uv_int_11, params=chart)
            w_epsilon = 1e-10
            w0_0 = (u_ceil_float - u) * (v_ceil_float - v) + w_epsilon
            w1_0 = (u - u_floor_float) * (v_ceil_float - v) + w_epsilon
            w0_1 = (u_ceil_float - u) * (v - v_floor_float) + w_epsilon
            w1_1 = (u - u_floor_float) * (v - v_floor_float) + w_epsilon
            w_sum = (u_ceil_float - u_floor_float) * (v_ceil_float - v_floor_float) + 4 * w_epsilon
            w0_0 = w0_0 / w_sum
            w1_0 = w1_0 / w_sum
            w0_1 = w0_1 / w_sum
            w1_1 = w1_1 / w_sum
            fake_img = w0_0 * fake_img0_0 + w1_0 * fake_img1_0 + w0_1 * fake_img0_1 + w1_1 * fake_img1_1
            fake_img = fake_img + bg_color
            result = fake_img
        else:
            raise NotImplementedError
        if Config.global_cfg.exp_cfg.random_shift:
            res = Config.global_cfg.exp_cfg.img_res
            scale = tf.random_uniform(shape=[2], minval=0.7, maxval=0.9, name='rd_scale')
            scale_size_1 = tf.floor(res * scale[0])
            scale_size_2 = tf.floor(res * scale[1])
            result = tf.image.resize(result, size=[scale_size_1, scale_size_2])
            # padding = tf.Variable(initial_value=0, shape=[4, 2])
            shift_1 = tf.floor(tf.random_uniform(shape=[1], minval=0, maxval=(res-scale_size_1)))
            shift_2 = tf.floor(tf.random_uniform(shape=[1], minval=0, maxval=(res - scale_size_2)))
            padding = [[0, 0], [shift_1[0], res-shift_1[0]-scale_size_1], [shift_2[0], res-shift_2[0]-scale_size_2], [0, 0]]
            result = tf.pad(result, paddings=padding, constant_values=bg_color[0, 0, 0, 0])
            result.set_shape([result.shape[0], res, res, result.shape[3]])
        return result