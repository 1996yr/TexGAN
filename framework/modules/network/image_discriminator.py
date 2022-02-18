import tensorflow as tf
from exp_config.config import Config
from framework.modules.module_base import module_base
from framework.utils.log import log_message
from framework.modules.codebase import net_structures


class tex_discriminator_spade_chart_MD_Mix(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        gather_module = kwargs['gather_module']
        global_data_dict = kwargs['global_data_dict']
        if 'phase' in kwargs.keys():
            phase = kwargs['phase']
        else:
            phase = 'train'

        log_message(self.__class__.__name__, '---Building subgraph...---')

        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        d_num = Config.tex_dis_cfg.d_num
        d_order_syn = Config.tex_dis_cfg.d_order_syn
        d_weight_syn = Config.tex_dis_cfg.d_weight_syn
        d_order_real = Config.tex_dis_cfg.d_order_real
        d_weight_real = Config.tex_dis_cfg.d_weight_real
        # d_order = [0]
        assert len(d_order_syn) == Config.global_cfg.exp_cfg.n_view_d_synbatch
        assert len(d_order_real) == Config.global_cfg.exp_cfg.n_view_d_realbatch
        d_list = []
        for d_id in range(d_num):
            net_discriminator = getattr(net_structures, Config.tex_dis_cfg.struct.struct_name)(d_id)
            d_list.append(net_discriminator)
        if phase == 'train':
            tf_real_d_realbatch_all_gpu = []
            tf_fake_d_realbatch_all_gpu = []
            tf_real_score_realbatch_all_d = {'real_realbatch_d{}'.format(i) : [] for i in range(Config.tex_dis_cfg.d_num)}
            tf_fake_score_realbatch_all_d = {'fake_realbatch_d{}'.format(i) : [] for i in range(Config.tex_dis_cfg.d_num)}

            tf_real_d_synbatch_all_gpu = []
            tf_fake_d_synbatch_all_gpu = []
            tf_real_score_synbatch_all_d = {'real_synbatch_d{}'.format(i): [] for i in range(Config.tex_dis_cfg.d_num)}
            tf_fake_score_synbatch_all_d = {'fake_synbatch_d{}'.format(i): [] for i in range(Config.tex_dis_cfg.d_num)}

            scope_list = ['_dscope{}'.format(str(d_id)) for d_id in range(d_num)]
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    lod = global_data_dict['lod_all_gpu'][g_id]
                    with tf.name_scope('real'):
                        tex_d_input_realbatch = data_loader.public_ops['input_real_img_realbatch'][g_id][::,::,::,0:3]
                        tex_d_input_synbatch = data_loader.public_ops['input_real_img_synbatch'][g_id][::,::,::,0:3]
                        real_d_gpu_realbatch = []
                        real_d_gpu_synbatch = []
                        for img_id in range(Config.global_cfg.exp_cfg.n_view_d_realbatch):
                            d_id = d_order_real[img_id]
                            scope = scope_list[d_id]
                            net_d = d_list[d_id]
                            input_img_realbatch = tex_d_input_realbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_realbatch, ::, ::, ::]
                            _real_realbatch_d = net_d.forward(input_img_realbatch, name_scope='tex_dis'+scope, lod_in=lod)
                            tf_real_score_realbatch_all_d['real_realbatch_d{}'.format(d_id)].append(_real_realbatch_d)
                            real_d_gpu_realbatch.append(d_weight_real[img_id] * _real_realbatch_d)
                        real_d_realbatch = tf.concat(real_d_gpu_realbatch, axis=0)

                        for img_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                            d_id = d_order_syn[img_id]
                            scope = scope_list[d_id]
                            net_d = d_list[d_id]
                            input_img_synbatch = tex_d_input_synbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_synbatch, ::, ::, ::]
                            _real_synbatch_d = net_d.forward(input_img_synbatch, name_scope='tex_dis' + scope, lod_in=lod)
                            tf_real_score_synbatch_all_d['real_synbatch_d{}'.format(d_id)].append(_real_synbatch_d)
                            # if d_weight[img_id] > 0.0:
                            real_d_gpu_synbatch.append(d_weight_syn[img_id] * _real_synbatch_d)
                        real_d_synbatch = tf.concat(real_d_gpu_synbatch, axis=0)
                    with tf.name_scope('fake'):
                        tex_d_input_realbatch = gather_module.public_ops['fake_img_recons_realbatch'][g_id]
                        tex_d_input_synbatch = gather_module.public_ops['fake_img_recons_synbatch'][g_id]
                        fake_d_gpu_realbatch = []
                        fake_d_gpu_synbatch = []

                        for img_id in range(Config.global_cfg.exp_cfg.n_view_d_realbatch):
                            d_id = d_order_real[img_id]
                            scope = scope_list[d_id]
                            net_d = d_list[d_id]
                            input_img = tex_d_input_realbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_realbatch, ::, ::, ::]
                            _fake_realbatch_d = net_d.forward(input_img, name_scope='tex_dis' + scope, lod_in=lod)
                            tf_fake_score_realbatch_all_d['fake_realbatch_d{}'.format(d_id)].append(_fake_realbatch_d)

                            fake_d_gpu_realbatch.append(d_weight_real[img_id]*_fake_realbatch_d)
                        fake_d_realbatch = tf.concat(fake_d_gpu_realbatch, axis=0)

                        for img_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                            d_id = d_order_syn[img_id]
                            scope = scope_list[d_id]
                            net_d = d_list[d_id]
                            input_img = tex_d_input_synbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_synbatch, ::, ::, ::]
                            _fake_synbatch_d = net_d.forward(input_img, name_scope='tex_dis' + scope, lod_in=lod)
                            tf_fake_score_synbatch_all_d['fake_synbatch_d{}'.format(d_id)].append(_fake_synbatch_d)
                            # if d_weight[img_id] > 0.0:
                            fake_d_gpu_synbatch.append(d_weight_syn[img_id] * _fake_synbatch_d)
                        fake_d_synbatch = tf.concat(fake_d_gpu_synbatch, axis=0)

                    tf_real_d_realbatch_all_gpu.append(real_d_realbatch)
                    tf_fake_d_realbatch_all_gpu.append(fake_d_realbatch)
                    tf_real_d_synbatch_all_gpu.append(real_d_synbatch)
                    tf_fake_d_synbatch_all_gpu.append(fake_d_synbatch)
            self.public_ops['real_d_realbatch'] = tf_real_d_realbatch_all_gpu
            self.public_ops['fake_d_realbatch'] = tf_fake_d_realbatch_all_gpu
            self.public_ops['real_score_realbatch'] = tf_real_score_realbatch_all_d
            self.public_ops['fake_score_realbatch'] = tf_fake_score_realbatch_all_d

            self.public_ops['real_d_synbatch'] = tf_real_d_synbatch_all_gpu
            self.public_ops['fake_d_synbatch'] = tf_fake_d_synbatch_all_gpu
            self.public_ops['real_score_synbatch'] = tf_real_score_synbatch_all_d
            self.public_ops['fake_score_synbatch'] = tf_fake_score_synbatch_all_d

            for d_id in list(set(d_order_real)):
                real_score_realbatch = tf.reduce_mean(tf_real_score_realbatch_all_d['real_realbatch_d{}'.format(d_id)])
                fake_score_realbatch = tf.reduce_mean(tf_fake_score_realbatch_all_d['fake_realbatch_d{}'.format(d_id)])
                if Config.global_cfg.exp_cfg.real_batch > 0:
                    self.add_summary(real_score_realbatch, 'train', 'scalar', 'real_realbatch_d{}'.format(d_id))
                    self.add_summary(fake_score_realbatch, 'train', 'scalar', 'fake_realbatch_d{}'.format(d_id))

            for d_id in list(set(d_order_syn)):
                real_score_synbatch_ = tf.reduce_mean(tf_real_score_synbatch_all_d['real_synbatch_d{}'.format(d_id)])
                fake_score_synbatch_ = tf.reduce_mean(tf_fake_score_synbatch_all_d['fake_synbatch_d{}'.format(d_id)])
                self.add_summary(real_score_synbatch_, 'train', 'scalar', 'real_synbatch_d{}'.format(d_id))
                self.add_summary(fake_score_synbatch_, 'train', 'scalar', 'fake_synbatch_d{}'.format(d_id))
            if Config.global_cfg.exp_cfg.real_batch > 0:
                self.add_summary(tf.reduce_mean(tf_real_d_realbatch_all_gpu), 'train', 'scalar', 'real_img_score_realbatch')
                self.add_summary(tf.reduce_mean(tf_fake_d_realbatch_all_gpu), 'train', 'scalar', 'fake_img_score_realbatch')
            self.add_summary(tf.reduce_mean(tf_real_d_synbatch_all_gpu), 'train', 'scalar', 'real_img_score_synbatch')
            self.add_summary(tf.reduce_mean(tf_fake_d_synbatch_all_gpu), 'train', 'scalar', 'fake_img_score_synbatch')

        else:
            try: # debug d score
                pass
                # gpu_list = [0]
                # tf_fake_score_synbatch_all_d = {'fake_synbatch_d{}'.format(i): [] for i in range(Config.tex_dis_cfg.d_num)}
                # scope_list = ['_front_view', '_back_view', '_side_view', '_left_side_view', '_right_side_view']
                # for g_id in range(0, len(gpu_list)):
                #     with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                #         lod = global_data_dict['lod_all_gpu'][g_id]
                #         with tf.name_scope('fake'):
                #             tex_d_input_synbatch = gather_module.public_ops['fake_img_recons_synbatch'][g_id]
                #             for img_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                #                 d_id = d_order[img_id]
                #                 scope = scope_list[d_id]
                #                 net_d = d_list[d_id]
                #                 input_img = tex_d_input_synbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_synbatch, ::,
                #                             ::, ::]
                #                 _fake_synbatch_d = net_d.forward(input_img, name_scope='tex_dis' + scope, lod_in=lod)
                #                 tf_fake_score_synbatch_all_d['fake_synbatch_d{}'.format(d_id)].append(_fake_synbatch_d)
                # self.public_ops['fake_score_synbatch'] = tf_fake_score_synbatch_all_d
            except:
                pass

class tex_discriminator_spade_chart_ImgConcat_ShareD_Mix(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        gather_module = kwargs['gather_module']
        global_data_dict = kwargs['global_data_dict']

        log_message(self.__class__.__name__, '---Building subgraph...---')

        gpu_list = Config.global_cfg.meta_cfg.gpu_list

        net_discriminator_synbatch = getattr(net_structures, Config.tex_dis_cfg.struct.struct_name)(0, channel_times=2.5)

        tf_real_d_synbatch_all_gpu = []
        tf_fake_d_synbatch_all_gpu = []

        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                lod = global_data_dict['lod_all_gpu'][g_id]
                with tf.name_scope('real'):
                    tex_d_input_synbatch = data_loader.public_ops['input_real_img_synbatch'][g_id][::,::,::,0:3]
                    all_img = []
                    scope = "_synbatch_{}view_d".format(Config.global_cfg.exp_cfg.n_view_d_synbatch)
                    net_d = net_discriminator_synbatch
                    for img_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                        input_img_synbatch = tex_d_input_synbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_synbatch, ::, ::, ::]
                        all_img.append(input_img_synbatch)
                    input_concat_img_synbatch = tf.concat(all_img, axis=3)
                    real_d_synbatch = net_d.forward(input_concat_img_synbatch, name_scope='tex_dis' + scope, lod_in=lod)

                with tf.name_scope('fake'):
                    tex_d_input_synbatch = gather_module.public_ops['fake_img_recons_synbatch'][g_id]
                    all_img = []
                    scope = "_synbatch_{}view_d".format(Config.global_cfg.exp_cfg.n_view_d_synbatch)
                    net_d = net_discriminator_synbatch
                    for img_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                        input_img = tex_d_input_synbatch[img_id::Config.global_cfg.exp_cfg.n_view_d_synbatch, ::, ::, ::]
                        all_img.append(input_img)
                    input_concat_img_synbatch = tf.concat(all_img, axis=3)
                    fake_d_synbatch = net_d.forward(input_concat_img_synbatch, name_scope='tex_dis' + scope, lod_in=lod)

                tf_real_d_synbatch_all_gpu.append(real_d_synbatch)
                tf_fake_d_synbatch_all_gpu.append(fake_d_synbatch)

        self.public_ops['real_d_synbatch'] = tf_real_d_synbatch_all_gpu
        self.public_ops['fake_d_synbatch'] = tf_fake_d_synbatch_all_gpu
        self.add_summary(tf.reduce_mean(tf_real_d_synbatch_all_gpu), 'train', 'scalar', 'real_img_score_synbatch')
        self.add_summary(tf.reduce_mean(tf_fake_d_synbatch_all_gpu), 'train', 'scalar', 'fake_img_score_synbatch')


class style_dis(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        data_loader = kwargs['data_loader']
        global_data_dict = kwargs['global_data_dict']
        tex_generator_module = kwargs['tex_generator']
        log_message(self.__class__.__name__, '---Building subgraph...---')
        gpu_list = Config.global_cfg.meta_cfg.gpu_list

        net_discriminator = getattr(net_structures, Config.tex_dis_cfg.struct.struct_name)(0)

        tf_real_d_all_gpu = []
        tf_fake_d_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                lod = global_data_dict['lod_all_gpu'][g_id]
                with tf.name_scope('real'):
                    tex_d_input = data_loader.public_ops['input_real_img'][g_id][::,::,::, 0:Config.tex_gen_cfg.struct.output_channel]
                    _real_d = net_discriminator.forward(tex_d_input, name_scope='tex_dis', lod_in=lod)
                    tf_real_d_all_gpu.append(_real_d)
                with tf.name_scope('fake'):
                    tex_d_input = tex_generator_module.public_ops['fake_img'][g_id]
                    _fake_d = net_discriminator.forward(tex_d_input, name_scope='tex_dis', lod_in=lod)
                    tf_fake_d_all_gpu.append(_fake_d)
        self.public_ops['real_d'] = tf_real_d_all_gpu
        self.public_ops['fake_d'] = tf_fake_d_all_gpu
        self.add_summary(tf.reduce_mean(tf_real_d_all_gpu), 'train', 'scalar', 'real_d_score')
        self.add_summary(tf.reduce_mean(tf_fake_d_all_gpu), 'train', 'scalar', 'fake_d_score')
