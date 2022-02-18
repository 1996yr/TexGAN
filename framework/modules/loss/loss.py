import tensorflow as tf
from framework.utils.log import log_message
from framework.modules.codebase.loss_func import *
from framework.modules.module_base import module_base
from exp_config.config import Config


class loss_chart_gan_mix(module_base):
    def __init__(self):
        super().__init__()

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        tex_generator = kwargs['tex_generator']
        tex_discriminator = kwargs['tex_discriminator']
        gather_module = kwargs['gather_module']
        global_data_dict = kwargs['global_data_dict']
        if Config.global_cfg.exp_cfg.real_batch > 0:
            use_real_batch = True
        else:
            use_real_batch = False

        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        batch_group_per_gpu = Config.dataloader_cfg.batch_group // len(gpu_list)
        assert batch_group_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_group

        tf_tex_d_loss_realbatch_all_gpu = []
        tf_tex_g_loss_realbatch_all_gpu = []
        tf_tex_gp_loss_realbatch_all_gpu = []

        tf_tex_d_loss_synbatch_all_gpu = []
        tf_tex_g_loss_synbatch_all_gpu = []
        tf_tex_gp_loss_synbatch_all_gpu = []

        tf_smooth_loss_synbatch_all_gpu = []

        with tf.name_scope('tex_loss'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    if Config.loss.smooth_loss > 0:
                        with tf.name_scope('smooth_loss'):
                            _loss_smooth_synbatch = Config.loss.smooth_loss * smooth_loss(
                                smooth_gather_img=gather_module.public_ops['smooth_gather_img_recons_synbatch'][g_id],
                                fake_recons_img=gather_module.public_ops['fake_img_recons_synbatch'][g_id],
                                smooth_mask=data_loader.public_ops['input_smooth_mask_synbatch'][g_id])
                            tf_smooth_loss_synbatch_all_gpu.append(_loss_smooth_synbatch)
                    with tf.name_scope('g_loss'):
                        if Config.loss.gan_loss == 'style_base':
                            if use_real_batch:
                                _loss_tex_g_realbatch = loss_logistic_nonsaturating_g(tex_discriminator.public_ops['fake_d_realbatch'][g_id])
                            _loss_tex_g_synbatch = loss_logistic_nonsaturating_g(tex_discriminator.public_ops['fake_d_synbatch'][g_id])
                        else:
                            raise NotImplementedError
                        if use_real_batch:
                            tf_tex_g_loss_realbatch_all_gpu.append(_loss_tex_g_realbatch)
                        tf_tex_g_loss_synbatch_all_gpu.append(_loss_tex_g_synbatch)
                    with tf.name_scope('d_loss'):
                        if Config.loss.gan_loss == 'style_base':
                            if use_real_batch:
                                _loss_tex_d_realbatch = loss_logistic_simplegp_d(
                                    tex_discriminator.public_ops['real_d_realbatch'][g_id],
                                    tex_discriminator.public_ops['fake_d_realbatch'][g_id])
                            _loss_tex_d_synbatch = loss_logistic_simplegp_d(
                                tex_discriminator.public_ops['real_d_synbatch'][g_id],
                                tex_discriminator.public_ops['fake_d_synbatch'][g_id])
                        else:
                            raise NotImplementedError
                        if use_real_batch:
                            tf_tex_d_loss_realbatch_all_gpu.append(_loss_tex_d_realbatch)
                        tf_tex_d_loss_synbatch_all_gpu.append(_loss_tex_d_synbatch)
                    if Config.loss.simplegp > 0:
                        with tf.name_scope('tex_gp_loss'):
                            if use_real_batch:
                                _loss_gp_realbatch = Config.loss.simplegp * loss_simple_gp(
                                    data_loader.public_ops['input_real_img_realbatch'][g_id],
                                    tex_discriminator.public_ops['real_d_realbatch'][g_id]) * 0.5
                                tf_tex_gp_loss_realbatch_all_gpu.append(_loss_gp_realbatch)

                            _loss_gp_synbatch = Config.loss.simplegp * loss_simple_gp(
                                data_loader.public_ops['input_real_img_synbatch'][g_id],
                                tex_discriminator.public_ops['real_d_synbatch'][g_id]) * 0.5
                            tf_tex_gp_loss_synbatch_all_gpu.append(_loss_gp_synbatch)

        if use_real_batch:
            self.public_ops['loss_tex_g_realbatch'] = tf_tex_g_loss_realbatch_all_gpu
            if Config.global_cfg.exp_cfg.real_batch > 0:
                self.add_summary(tf.reduce_mean(tf_tex_g_loss_realbatch_all_gpu), 'train', 'scalar', 'loss_tex_g_realbatch')
        self.public_ops['loss_tex_g_synbatch'] = tf_tex_g_loss_synbatch_all_gpu
        self.add_summary(tf.reduce_mean(tf_tex_g_loss_synbatch_all_gpu), 'train', 'scalar', 'loss_tex_g_synbatch')

        if use_real_batch:
            self.public_ops['loss_tex_d_realbatch'] = tf_tex_d_loss_realbatch_all_gpu
            if Config.global_cfg.exp_cfg.real_batch > 0:
                self.add_summary(tf.reduce_mean(tf_tex_d_loss_realbatch_all_gpu), 'train', 'scalar', 'loss_tex_d_realbatch')
        self.public_ops['loss_tex_d_synbatch'] = tf_tex_d_loss_synbatch_all_gpu
        self.add_summary(tf.reduce_mean(tf_tex_d_loss_synbatch_all_gpu), 'train', 'scalar', 'loss_tex_d_synbatch')

        if Config.loss.simplegp > 0:
            if use_real_batch:
                self.public_ops['loss_tex_gp_realbatch'] = tf_tex_gp_loss_realbatch_all_gpu
                self.add_summary(tf.reduce_mean(tf_tex_gp_loss_realbatch_all_gpu), 'train', 'scalar',
                                 'loss_tex_gp_realbatch')
            self.public_ops['loss_tex_gp_synbatch'] = tf_tex_gp_loss_synbatch_all_gpu
            self.add_summary(tf.reduce_mean(tf_tex_gp_loss_synbatch_all_gpu), 'train', 'scalar',
                             'loss_tex_gp_synbatch')

        if Config.loss.smooth_loss > 0:
            self.public_ops['loss_smooth_synbatch'] = tf_smooth_loss_synbatch_all_gpu
            smooth_loss_mean_synbatch = tf.reduce_mean(tf_smooth_loss_synbatch_all_gpu)
            self.public_ops['loss_smooth_mean_synbatch'] = smooth_loss_mean_synbatch
            self.add_summary(smooth_loss_mean_synbatch, 'train', 'scalar', 'loss_smooth_synbatch')


class style_gan_loss(module_base):
    def __init__(self):
        super().__init__()

    def build_graph(self, **kwargs):
        tex_discriminator = kwargs['tex_discriminator']
        data_loader = kwargs['data_loader']
        global_data_dict = kwargs['global_data_dict']

        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        batch_size_per_gpu = Config.dataloader_cfg.batch_size // len(gpu_list)
        assert batch_size_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_size

        tf_tex_d_loss_all_gpu = []
        tf_tex_g_loss_all_gpu = []
        tf_tex_gp_loss_all_gpu = []

        with tf.name_scope('tex_loss'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    with tf.name_scope('g_loss'):
                        if Config.loss.gan_loss == 'style_base':
                            _loss_tex_g = loss_logistic_nonsaturating_g(tex_discriminator.public_ops['fake_d'][g_id])
                        else:
                            raise NotImplementedError
                        tf_tex_g_loss_all_gpu.append(_loss_tex_g)

                    with tf.name_scope('d_loss'):
                        if Config.loss.gan_loss == 'style_base':
                            _loss_tex_d = loss_logistic_simplegp_d(
                                tex_discriminator.public_ops['real_d'][g_id],
                                tex_discriminator.public_ops['fake_d'][g_id])
                        else:
                            raise NotImplementedError
                        tf_tex_d_loss_all_gpu.append(_loss_tex_d)

                    if Config.loss.simplegp > 0:
                        with tf.name_scope('tex_gp_loss'):
                            _loss_gp = Config.loss.simplegp * loss_simple_gp(
                                data_loader.public_ops['input_real_img'][g_id],
                                tex_discriminator.public_ops['real_d'][g_id]) * 0.5
                            tf_tex_gp_loss_all_gpu.append(_loss_gp)

        self.public_ops['loss_tex_g'] = tf_tex_g_loss_all_gpu
        self.add_summary(tf.reduce_mean(tf_tex_g_loss_all_gpu), 'train', 'scalar', 'loss_tex_g')

        self.public_ops['loss_tex_d'] = tf_tex_d_loss_all_gpu
        self.add_summary(tf.reduce_mean(tf_tex_d_loss_all_gpu), 'train', 'scalar', 'loss_tex_d')

        if Config.loss.simplegp > 0:
            self.public_ops['loss_tex_gp'] = tf_tex_gp_loss_all_gpu
            self.add_summary(tf.reduce_mean(tf_tex_gp_loss_all_gpu), 'train', 'scalar', 'loss_tex_gp')


class loss_ae(module_base):
    def __init__(self):
        super().__init__()

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        ae_module = kwargs['ae_module']
        global_data_dict = kwargs['global_data_dict']

        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        batch_size_per_gpu = Config.dataloader_cfg.batch_size // len(gpu_list)
        assert batch_size_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_size


        source_img_all_gpu = data_loader.public_ops['input_mask']
        recons_img_all_gpu = ae_module.public_ops['recons_mask']

        tf_recons_loss_all_gpu = []
        with tf.name_scope('auto_enconder_loss'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    source_img = source_img_all_gpu[g_id]
                    recons_img = recons_img_all_gpu[g_id]
                    with tf.name_scope('recons_l2'):
                        loss = l2_img_loss(source_img, recons_img)
                        tf_recons_loss_all_gpu.append(loss)

        self.public_ops['loss_recons_l2'] = tf_recons_loss_all_gpu
        self.add_summary(tf.reduce_mean(tf_recons_loss_all_gpu), 'train', 'scalar', 'loss_recons')


class loss_vp(module_base):
    def __init__(self):
        super().__init__()

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        vp_module = kwargs['vp_module']
        global_data_dict = kwargs['global_data_dict']

        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        batch_size_per_gpu = Config.dataloader_cfg.batch_size // len(gpu_list)
        assert batch_size_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_size


        view_label_all_gpu = data_loader.public_ops['input_view_label']
        view_est_all_gpu = vp_module.public_ops['view_est']

        view_label_loss_all_gpu = []
        with tf.name_scope('vp_loss'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    label = view_label_all_gpu[g_id]
                    est = view_est_all_gpu[g_id]
                    with tf.name_scope('view_loss_ce'):
                        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=est))
                        view_label_loss_all_gpu.append(loss)

        self.public_ops['loss_view_pred_ce'] = view_label_loss_all_gpu
        self.add_summary(tf.reduce_mean(view_label_loss_all_gpu), 'train', 'scalar', 'loss_view_pred_ce')

