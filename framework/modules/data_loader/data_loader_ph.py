# create data loader based on place holder which will be fed if you want to use sess.run()
from framework.modules.module_base import module_base
from exp_config.config import Config
import tensorflow as tf
from framework.modules.codebase.layers import denormalizeInput, normalizeInput
from framework.modules.codebase.visualization import tileImage

class data_loader_npy(module_base):
    def __init__(self):
        module_base.__init__(self)
        self.tf_dataset_real_batch = dict()
        self.tf_dataset_syn_batch = dict()
        self.n_view_g = Config.global_cfg.exp_cfg.n_view_g
        self.n_view_d_syn = Config.global_cfg.exp_cfg.n_view_d_synbatch
        self.n_view_d_real = Config.global_cfg.exp_cfg.n_view_d_realbatch
        self.max_shape = None
        self.resolution = None
        self.resolution_log2 = None
        self.cur_minibatch = -1
        self.cur_lod = -1

    def build_graph(self, **kwargs):
        global_data_dict = kwargs['global_data_dict']
        phase = kwargs['phase']
        res_log2_list = [2, 3, 4, 5, 6, 7]
        if phase == 'train':
            # batch_size_g = Config.dataloader_cfg.batch_size_G
            # batch_size_d = Config.dataloader_cfg.batch_size_D
            batch_group = Config.dataloader_cfg.batch_group
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            group_per_gpu = batch_group // len(gpu_list)
            assert group_per_gpu * len(gpu_list) == batch_group
            with tf.name_scope('dataset_{}'.format(phase)):
                batch_realimg_realbatch = tf.placeholder(shape=[batch_group*self.n_view_d_real, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 4], dtype=tf.float32, name='batch_realimg_realbatch')
                batch_realimg_synbatch = tf.placeholder(shape=[batch_group*self.n_view_d_syn, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 4], dtype=tf.float32, name='batch_realimg_synbatch')

                batch_attr_chart_realbatch = tf.placeholder(shape=[batch_group*self.n_view_g, Config.global_cfg.exp_cfg.chart_res, Config.global_cfg.exp_cfg.chart_res, 1], dtype=tf.float32, name='batch_attr_chart_realbatch')
                batch_attr_chart_synbatch = tf.placeholder(shape=[batch_group*self.n_view_g, Config.global_cfg.exp_cfg.chart_res, Config.global_cfg.exp_cfg.chart_res, 1], dtype=tf.float32, name='batch_attr_chart_synbatch')

                batch_gather_map_realbatch = tf.placeholder(shape=[batch_group*self.n_view_d_real, Config.global_cfg.exp_cfg.gather_map_res, Config.global_cfg.exp_cfg.gather_map_res, 2], dtype=tf.float32, name='batch_gather_map_realbatch')
                batch_gather_map_synbatch = tf.placeholder(shape=[batch_group*self.n_view_d_syn, Config.global_cfg.exp_cfg.gather_map_res, Config.global_cfg.exp_cfg.gather_map_res, 2], dtype=tf.float32, name='batch_gather_map_synbatch')
                batch_weight_map_synbatch = tf.placeholder(
                    shape=[batch_group * self.n_view_d_syn, Config.global_cfg.exp_cfg.gather_map_res,
                           Config.global_cfg.exp_cfg.gather_map_res, 1], dtype=tf.float32,
                    name='batch_weight_map_synbatch')

                batch_smooth_gather_map_synbatch = tf.placeholder(shape=[batch_group * self.n_view_d_syn, Config.global_cfg.exp_cfg.gather_map_res, Config.global_cfg.exp_cfg.gather_map_res, 2], dtype=tf.float32, name='batch_smooth_gather_map_synbatch')
                batch_smooth_mask_synbatch = tf.placeholder(shape=[batch_group * self.n_view_d_syn, Config.global_cfg.exp_cfg.gather_map_res, Config.global_cfg.exp_cfg.gather_map_res, 1], dtype=tf.float32, name='batch_smooth_mask_synbatch')

                self.public_ops['realimg_realbatch_ph'] = batch_realimg_realbatch
                self.public_ops['realimg_synbatch_ph'] = batch_realimg_synbatch

                self.public_ops['attrchart_realbatch_ph'] = batch_attr_chart_realbatch
                self.public_ops['attrchart_synbatch_ph'] = batch_attr_chart_synbatch

                self.public_ops['gathermap_realbatch_ph'] = batch_gather_map_realbatch
                self.public_ops['gathermap_synbatch_ph'] = batch_gather_map_synbatch
                self.public_ops['weight_map_synbatch_ph'] = batch_weight_map_synbatch

                self.public_ops['smooth_gather_synbatch_ph'] = batch_smooth_gather_map_synbatch
                self.public_ops['smooth_mask_synbatch_ph'] = batch_smooth_mask_synbatch

                tf_real_realbatch_all_gpu = []
                tf_attr_realbatch_chart_all_gpu = []
                tf_gather_map_realbatch_all_gpu = []

                tf_real_synbatch_all_gpu = []
                tf_attr_synbatch_chart_all_gpu = []
                tf_gather_map_synbatch_all_gpu = []
                tf_weight_map_synbatch_all_gpu = []

                tf_smooth_gather_map_synbatch_all_gpu = []
                tf_smooth_mask_synbatch_all_gpu = []

                for g_id in range(0, len(gpu_list)):
                    with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                        # real batch
                        tf_real_realbatch = batch_realimg_realbatch[g_id * group_per_gpu * self.n_view_d_real:(g_id + 1) * group_per_gpu * self.n_view_d_real]
                        tf_real_realbatch = self.preprocess(tf_real_realbatch)
                        tf_attr_chart_realbatch = batch_attr_chart_realbatch[g_id * group_per_gpu * self.n_view_g:(g_id + 1) * group_per_gpu * self.n_view_g]
                        tf_attr_pyramid_realbatch = self.get_pyramid(tf_attr_chart_realbatch, res_log2_list[-1], res_log2_list[0])
                        # [(N, 128, 128, 1), (N, 64, 64, 1), ... ,(N, 4, 4, 1)]
                        tf_gather_map_realbatch = batch_gather_map_realbatch[g_id * group_per_gpu * self.n_view_d_real:(g_id + 1) * group_per_gpu * self.n_view_d_real]
                        tf_real_realbatch_all_gpu.append(tf_real_realbatch)
                        tf_attr_realbatch_chart_all_gpu.append(tf_attr_pyramid_realbatch)
                        tf_gather_map_realbatch_all_gpu.append(tf_gather_map_realbatch)

                        # syn batch
                        tf_real_synbatch = batch_realimg_synbatch[g_id * group_per_gpu * self.n_view_d_syn:(g_id + 1) * group_per_gpu * self.n_view_d_syn]
                        tf_real_synbatch = self.preprocess(tf_real_synbatch)
                        tf_attr_chart_synbatch = batch_attr_chart_synbatch[g_id * group_per_gpu * self.n_view_g:(g_id + 1) * group_per_gpu * self.n_view_g]
                        tf_attr_pyramid_synbatch = self.get_pyramid(tf_attr_chart_synbatch, res_log2_list[-1], res_log2_list[0])
                        # [(N, 128, 128, 1), (N, 64, 64, 1), ... ,(N, 4, 4, 1)]
                        tf_gather_map_synbatch = batch_gather_map_synbatch[g_id * group_per_gpu * self.n_view_d_syn:(g_id + 1) * group_per_gpu * self.n_view_d_syn]
                        tf_weight_map = batch_weight_map_synbatch[g_id * group_per_gpu * self.n_view_d_syn:(g_id + 1) * group_per_gpu * self.n_view_d_syn]
                        tf_weight_map = tf.clip_by_value(tf_weight_map, 0.0, 1.0)
                        tf_smooth_gather_map_synbatch = batch_smooth_gather_map_synbatch[g_id * group_per_gpu * self.n_view_d_syn:(g_id + 1) * group_per_gpu * self.n_view_d_syn]
                        tf_smooth_mask_synbatch = batch_smooth_mask_synbatch[g_id * group_per_gpu * self.n_view_d_syn:(g_id + 1) * group_per_gpu * self.n_view_d_syn]
                        tf_real_synbatch_all_gpu.append(tf_real_synbatch)
                        tf_attr_synbatch_chart_all_gpu.append(tf_attr_pyramid_synbatch)
                        tf_gather_map_synbatch_all_gpu.append(tf_gather_map_synbatch)
                        tf_weight_map_synbatch_all_gpu.append(tf_weight_map)
                        tf_smooth_gather_map_synbatch_all_gpu.append(tf_smooth_gather_map_synbatch)
                        tf_smooth_mask_synbatch_all_gpu.append(tf_smooth_mask_synbatch)

                self.public_ops['input_real_img_realbatch'] = tf_real_realbatch_all_gpu
                self.public_ops['input_chart_realbatch'] = tf_attr_realbatch_chart_all_gpu
                self.public_ops['input_gather_map_realbatch'] = tf_gather_map_realbatch_all_gpu

                self.public_ops['input_real_img_synbatch'] = tf_real_synbatch_all_gpu
                self.public_ops['input_chart_synbatch'] = tf_attr_synbatch_chart_all_gpu
                self.public_ops['input_gather_map_synbatch'] = tf_gather_map_synbatch_all_gpu
                if Config.loss.smooth_loss > 0.0:
                    self.public_ops['input_smooth_gather_map_synbatch'] = tf_smooth_gather_map_synbatch_all_gpu
                    self.public_ops['input_smooth_mask_synbatch'] = tf_smooth_mask_synbatch_all_gpu
                if Config.gather_cfg.normal_weight > 0.0:
                    self.public_ops['input_weight_map_synbatch'] = tf_weight_map_synbatch_all_gpu

                # gather_map
                vis_input_gather = tileImage(tf.cast(tf.concat(tf_gather_map_synbatch_all_gpu, axis=0),
                                                   tf.uint8), nCol=Config.global_cfg.exp_cfg.n_view_d_synbatch)
                vis_input_gather_x = tf.tile(vis_input_gather[::, ::, ::, 0:1], [1, 1, 1, 3]) * 2
                self.add_summary(vis_input_gather_x, 'train', 'image', 'vis_input_gather_u')
                vis_input_gather_y = tf.tile(vis_input_gather[::, ::, ::, 1:2], [1, 1, 1, 3]) / 3
                self.add_summary(vis_input_gather_y, 'train', 'image', 'vis_input_gather_v')

                # weight_map
                if Config.gather_cfg.normal_weight > 0.0:
                    vis_input_weight_map = tileImage(tf.cast(tf.concat(tf_weight_map_synbatch_all_gpu, axis=0) * 255,
                                                         tf.uint8), nCol=Config.global_cfg.exp_cfg.n_view_d_synbatch)
                    self.add_summary(vis_input_weight_map, 'train', 'image', 'vis_input_weight')

                # attr_chart
                vis_input_attr = tileImage(batch_attr_chart_synbatch, nCol=Config.global_cfg.exp_cfg.n_view_g)
                vis_input_mask = tf.tile(vis_input_attr, [1, 1, 1, 3])
                self.add_summary(vis_input_mask, 'train', 'image', 'vis_input_mask')

                # real albedo
                vis_input_albedo = tileImage(denormalizeInput(tf.concat(tf_real_synbatch_all_gpu, axis=0)[::, ::, ::, 0:3]), nCol=Config.global_cfg.exp_cfg.n_view_d_synbatch)
                self.add_summary(vis_input_albedo, 'train', 'image', 'input_real_img')
        elif phase == 'test':
            with tf.name_scope('dataset_{}'.format(phase)):
                tf_attr_chart_synbatch_all_gpu = []
                tf_gather_map_synbatch_all_gpu = []
                with tf.device('/cpu:0'):
                    tf_attr_chart_synbatch = tf.placeholder(
                        shape=[self.n_view_g, Config.global_cfg.exp_cfg.chart_res, Config.global_cfg.exp_cfg.chart_res, 1], dtype=tf.float32)
                    tf_gather_map_synbatch = tf.placeholder(
                        shape=[None, Config.global_cfg.exp_cfg.gather_map_res, Config.global_cfg.exp_cfg.gather_map_res, 2],
                        dtype=tf.float32)
                self.public_ops['attr_chart_place_holder_synbatch'] = tf_attr_chart_synbatch
                self.public_ops['gather_map_place_holder_synbatch'] = tf_gather_map_synbatch

                with tf.device('/gpu:{}'.format(0)), tf.name_scope('gpu{}'.format(0)):
                    tf_attr_chart_pyramid_synbatch = self.get_pyramid(tf_attr_chart_synbatch, res_log2_list[-1], res_log2_list[0])
                    tf_attr_chart_synbatch_all_gpu.append(tf_attr_chart_pyramid_synbatch)
                    tf_gather_map_synbatch_all_gpu.append(tf_gather_map_synbatch)
            self.public_ops['input_chart_synbatch'] = tf_attr_chart_synbatch_all_gpu
            self.public_ops['input_gather_map_synbatch'] = tf_gather_map_synbatch_all_gpu
        else:
            raise NotImplementedError


    def preprocess(self, x):
        with tf.name_scope('ProcessReals'):
            assert x.shape[-1] == 4 or x.shape[-1] == 3
            if x.shape[-1] == 4:
                mask = x[::, ::, ::, 3:]
                if Config.global_cfg.exp_cfg.random_background:
                    bg_color = tf.random_uniform(shape=[1, 1, 1, 1], minval=-0.9, maxval=-0.1, name='random_background_realimg') * (1 - mask)
                    x = tf.concat([normalizeInput(x[::, ::, ::, 0:3]) * mask + bg_color, mask], axis=3)
                else:
                    bg_color = tf.constant(shape=[1, 1, 1, 1], value=-1, dtype=tf.float32)
                    x = tf.concat([normalizeInput(x[::, ::, ::, 0:3]), x[::, ::, ::, 3:]], axis=3)
            elif x.shape[-1] == 3:
                x = normalizeInput(x)
                bg_color = tf.constant(shape=[1, 1, 1, 1], value=-1)
                if Config.global_cfg.exp_cfg.random_background:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            if Config.global_cfg.exp_cfg.random_shift:
                res = Config.global_cfg.exp_cfg.img_res
                scale = tf.random_uniform(shape=[2], minval=0.7, maxval=0.9, name='rd_scale_real')
                scale_size_1 = tf.floor(res * scale[0])
                scale_size_2 = tf.floor(res * scale[1])
                x = tf.image.resize(x, size=[scale_size_1, scale_size_2])
                # padding = tf.Variable(initial_value=0, shape=[4, 2])
                shift_1 = tf.floor(tf.random_uniform(shape=[1], minval=0, maxval=(res - scale_size_1)))
                shift_2 = tf.floor(tf.random_uniform(shape=[1], minval=0, maxval=(res - scale_size_2)))
                padding = [[0, 0], [shift_1[0], res - shift_1[0] - scale_size_1], [shift_2[0], res - shift_2[0] - scale_size_2], [0, 0]]
                x = tf.pad(x, paddings=padding, constant_values=bg_color[0, 0, 0, 0])
                x.set_shape([x.shape[0], res, res, x.shape[3]])
            return x


    def get_pyramid(self, batch_input, bot_res_log2, top_res_log2, interpolation='BILINEAR'):
        pyramid = []
        assert len(batch_input.get_shape().as_list()) == 4 # NHWC
        if interpolation == 'BILINEAR':
            pyramid.append(batch_input)
            for res_log2 in range(bot_res_log2 - 1, top_res_log2 - 1, -1):
                batch_input = tf.image.resize(batch_input, size=(2 ** res_log2, 2 ** res_log2))
                pyramid.append(batch_input)
        else:
            raise NotImplementedError
        pyramid.reverse()
        return pyramid


class data_loader_stylegan_npy(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        if phase == 'train':
            batch_size = Config.dataloader_cfg.batch_size
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_size_per_gpu = batch_size // len(gpu_list)
            assert batch_size_per_gpu * len(gpu_list) == batch_size
            with tf.name_scope('dataset_{}'.format(phase)):
                batch_realimg = tf.placeholder(shape=[batch_size, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, Config.tex_gen_cfg.struct.output_channel],
                                               dtype=tf.float32, name='batch_realimg')
                self.public_ops['realimg_ph'] = batch_realimg

                tf_real_all_gpu = []
                for g_id in range(0, len(gpu_list)):
                    with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                        # real batch
                        tf_real = batch_realimg[g_id * batch_size_per_gpu:(g_id + 1) * batch_size_per_gpu]
                        with tf.name_scope('normalize'):
                            tf_real = normalizeInput(tf_real)
                        tf_real_all_gpu.append(tf_real)

                self.public_ops['input_real_img'] = tf_real_all_gpu
                # real albedo
                tf_real_all = tf.concat(tf_real_all_gpu, axis=0)
                img_per_group = Config.tex_gen_cfg.struct.output_channel // 3
                group_num = tf_real_all.shape[0]
                real_img_all_list = []
                for group_id in range(group_num):
                    for img_id in range(img_per_group):
                        real_img_all_list.append(
                            tf_real_all[group_id:(group_id + 1), ::, ::, img_id * 3:(img_id + 1) * 3])
                real_img_sum = tf.concat(real_img_all_list, axis=0)
                vis_input_albedo = tileImage(denormalizeInput(real_img_sum), nCol=min(2, batch_size_per_gpu) * 6)
                self.add_summary(vis_input_albedo, 'train', 'image', 'input_real_img')


class data_loader_ae_npy(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        global_data_dict = kwargs['global_data_dict']
        phase = kwargs['phase']
        if phase == 'train':
            batch_size = Config.dataloader_cfg.batch_size
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_size_per_gpu = batch_size // len(gpu_list)
            assert batch_size_per_gpu * len(gpu_list) == batch_size
            with tf.name_scope('dataset_{}'.format(phase)):
                batch_input_mask = tf.placeholder(shape=[batch_size, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 1], dtype=tf.float32, name='input_mask')
                self.public_ops['input_mask_ph'] = batch_input_mask

                tf_input_mask_all_gpu = []

                for g_id in range(0, len(gpu_list)):
                    with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                        # real batch
                        tf_input_mask = batch_input_mask[g_id * batch_size_per_gpu:(g_id + 1) * batch_size_per_gpu]
                        tf_input_mask_all_gpu.append(tf_input_mask)
                self.public_ops['input_mask'] = tf_input_mask_all_gpu
                vis_input_mask= tileImage(tf.cast(batch_input_mask * 255, tf.uint8), nCol=8)
                self.add_summary(vis_input_mask, 'train', 'image', 'input_mask')
        elif phase == 'test':
            with tf.name_scope('dataset_{}'.format(phase)):
                tf_input_mask_all_gpu = []
                with tf.device('/cpu:0'):
                    tf_input_mask = tf.placeholder(shape=[None, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 1], dtype=tf.float32)
                self.public_ops['input_mask_place_holder'] = tf_input_mask
                tf_input_mask_all_gpu.append(tf_input_mask)
                self.public_ops['input_mask'] = tf_input_mask_all_gpu
        else:
            raise NotImplementedError


class data_loader_vp_npy(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        global_data_dict = kwargs['global_data_dict']
        phase = kwargs['phase']
        if phase == 'train':
            batch_size = Config.dataloader_cfg.batch_size
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_size_per_gpu = batch_size // len(gpu_list)
            assert batch_size_per_gpu * len(gpu_list) == batch_size
            with tf.name_scope('dataset_{}'.format(phase)):
                batch_input_mask = tf.placeholder(shape=[batch_size, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 1], dtype=tf.float32, name='input_mask')
                self.public_ops['input_mask_ph'] = batch_input_mask
                batch_input_view_label = tf.placeholder(
                    shape=[batch_size, Config.vp_cfg.struct.view_bin_num],
                    dtype=tf.float32, name='view_label')
                self.public_ops['input_view_label_ph'] = batch_input_view_label

                tf_input_mask_all_gpu = []
                tf_view_label_all_gpu = []

                for g_id in range(0, len(gpu_list)):
                    with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                        tf_input_mask = batch_input_mask[g_id * batch_size_per_gpu:(g_id + 1) * batch_size_per_gpu]
                        tf_input_mask_all_gpu.append(tf_input_mask)
                        tf_input_view_label = batch_input_view_label[g_id * batch_size_per_gpu:(g_id + 1) * batch_size_per_gpu]
                        tf_view_label_all_gpu.append(tf_input_view_label)
                self.public_ops['input_mask'] = tf_input_mask_all_gpu
                vis_input_mask= tileImage(tf.cast(batch_input_mask * 255, tf.uint8), nCol=8)
                self.add_summary(vis_input_mask, 'train', 'image', 'input_mask')

                self.public_ops['input_view_label'] = tf_view_label_all_gpu
        elif phase == 'test':
            with tf.name_scope('dataset_{}'.format(phase)):
                tf_input_mask_all_gpu = []
                with tf.device('/cpu:0'):
                    tf_input_mask = tf.placeholder(shape=[None, Config.global_cfg.exp_cfg.img_res, Config.global_cfg.exp_cfg.img_res, 1], dtype=tf.float32)
                self.public_ops['input_mask_place_holder'] = tf_input_mask
                tf_input_mask_all_gpu.append(tf_input_mask)
                self.public_ops['input_mask'] = tf_input_mask_all_gpu
        else:
            raise NotImplementedError
