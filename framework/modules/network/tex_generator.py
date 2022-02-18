import tensorflow as tf
from exp_config.config import Config
from framework.modules.module_base import module_base
from framework.modules.codebase import net_structures
from framework.modules.codebase.visualization import tileImage
from framework.modules.codebase.layers import denormalizeInput

class tex_generator_spade_6chart_mix(module_base):
    def __init__(self):
        module_base.__init__(self)
        self.net_z_map_name = Config.z_mapping_cfg.struct_name

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        global_data_dict = kwargs['global_data_dict']
        data_loader = kwargs['data_loader']

        if phase == 'train':
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_group_per_gpu = Config.dataloader_cfg.batch_group // len(gpu_list)
            assert batch_group_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_group
        elif phase == 'test':
            gpu_list = [0]
            batch_group_per_gpu = 1
        else:
            raise NotImplementedError

        if 'net_tex_g' not in global_data_dict:
            net_generator = getattr(net_structures, Config.tex_gen_cfg.struct.struct_name)(0)
        else:
            net_generator = global_data_dict['net_tex_g']

        if 'net_z_map' not in global_data_dict:
            net_z_map = getattr(net_structures, self.net_z_map_name)(0)
        else:
            net_z_map = global_data_dict['net_z_map']

        # random_z
        trunc_place_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        if phase == 'train':
            tf_zt_all_gpu = []
            tf_zt2_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.random_normal(shape=[batch_group_per_gpu, Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    _z = tf.expand_dims(_z, axis=1)
                    _z = tf.tile(_z, [1, Config.global_cfg.exp_cfg.n_view_g, 1])
                    _z = tf.reshape(_z, [batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g,
                                         Config.tex_gen_cfg.struct.z_dim])
                    tf_zt_all_gpu.append(_z)
                    if Config.tex_gen_cfg.struct.style_mix:
                        _z2 = tf.random_normal(shape=[batch_group_per_gpu, Config.tex_gen_cfg.struct.z_dim],
                                              dtype=tf.float32, name='z_code2')
                        _z2 = tf.expand_dims(_z2, axis=1)
                        _z2 = tf.tile(_z2, [1, Config.global_cfg.exp_cfg.n_view_g, 1])
                        _z2 = tf.reshape(_z2, [batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g,
                                             Config.tex_gen_cfg.struct.z_dim])
                        tf_zt2_all_gpu.append(_z2)
        elif phase == 'test':
            tf_zt_all_gpu = []
            tf_zt2_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.placeholder(shape=[batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g,
                                               Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    tf_zt_all_gpu.append(_z)
        else:
            raise NotImplementedError
        dlatents_all_gpu = []
        dlatents2_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            lod = global_data_dict['lod_all_gpu'][g_id]
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                _dlatents = net_z_map.forward(tf_zt_all_gpu[g_id], phase, trunc_place_holder=trunc_place_holder,variable_scope_prefix='tex_gen')
                dlatents_all_gpu.append(_dlatents)
                if len(tf_zt2_all_gpu) > 0:
                    _dlatents2 = net_z_map.forward(tf_zt2_all_gpu[g_id], phase, trunc_place_holder=trunc_place_holder,variable_scope_prefix='tex_gen')
                    dlatents2_all_gpu.append(_dlatents2)


        # tex generation
        output_chart_synbatch_all_gpu = []
        output_chart_realbatch_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            input_chart_pyramid_synbatch = data_loader.public_ops['input_chart_synbatch'][g_id]
            if phase == 'test' or phase == 'train':
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    lod = global_data_dict['lod_all_gpu'][g_id]
                    dlatents2 = None
                    _output_albedo_synbatch = net_generator.forward(input_chart_pyramid_synbatch,
                                                                    dlatents_all_gpu[g_id], dlatents2, lod_in=lod,
                                                                    name_scope='tex_gen')
                    output_chart_synbatch_all_gpu.append(_output_albedo_synbatch)

            if phase == 'train':
                input_chart_pyramid_realbatch = data_loader.public_ops['input_chart_realbatch'][g_id]
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    if len(dlatents2_all_gpu) > 0:
                        dlatents2 = dlatents2_all_gpu[g_id]
                    else:
                        dlatents2 = None
                    _output_albedo_realbatch = net_generator.forward(input_chart_pyramid_realbatch,
                                                                     dlatents_all_gpu[g_id], dlatents2, lod_in=lod,
                                                                     name_scope='tex_gen', global_data_dict=global_data_dict)
                    output_chart_realbatch_all_gpu.append(_output_albedo_realbatch)

        self.public_ops['trunc_place_holder'] = trunc_place_holder
        self.public_ops['fake_chart6view_synbatch'] = output_chart_synbatch_all_gpu
        self.public_ops['random_z'] = tf_zt_all_gpu
        self.public_ops['dlatents'] = dlatents_all_gpu
        if phase == 'train':
            self.public_ops['fake_chart6view_realbatch'] = output_chart_realbatch_all_gpu
        if 'net_tex_g' not in global_data_dict:
            global_data_dict['net_tex_g'] = net_generator
        if 'net_z_map' not in global_data_dict:
            global_data_dict['net_z_map'] = net_z_map
        if phase == 'train':
            fake_chart = []
            for g_id in range(0, len(gpu_list)):
                # for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g):
                clip_albedo = tf.clip_by_value(denormalizeInput(output_chart_synbatch_all_gpu[g_id]), 0.0, 1.0)
                fake_chart.append(clip_albedo)
            vis_fake_albedo = tileImage(tf.concat(fake_chart, axis=0), nCol=Config.global_cfg.exp_cfg.n_view_g)
            self.add_summary(vis_fake_albedo, 'train', 'image', 'fake_chart6view')


class tex_generator_spade_6chart_mix_split_coursew(module_base):
    def __init__(self):
        module_base.__init__(self)
        self.net_z_map_name = Config.z_mapping_cfg.struct_name

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        global_data_dict = kwargs['global_data_dict']
        data_loader = kwargs['data_loader']

        if phase == 'train':
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_group_per_gpu = Config.dataloader_cfg.batch_group // len(gpu_list)
            assert batch_group_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_group
        elif phase == 'test':
            gpu_list = [0]
            batch_group_per_gpu = 1
        else:
            raise NotImplementedError

        if 'net_tex_g' not in global_data_dict:
            net_generator = getattr(net_structures, Config.tex_gen_cfg.struct.struct_name)(0)
        else:
            net_generator = global_data_dict['net_tex_g']

        if 'net_z_map_share' not in global_data_dict:
            net_z_map_share = getattr(net_structures, self.net_z_map_name)(0)
        else:
            net_z_map_share = global_data_dict['net_z_map_share']

        net_z_map_split_list = []
        for view_id in range(1, 7):
            if 'net_z_map_split_{}'.format(view_id) not in global_data_dict:
                net_z_map_split = getattr(net_structures, self.net_z_map_name)(view_id)
            else:
                net_z_map_split = global_data_dict['net_z_map_split_{}'.format(view_id)]
            net_z_map_split_list.append(net_z_map_split)

        # random_z
        trunc_place_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        if phase == 'train':
            tf_zt_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.random_normal(shape=[batch_group_per_gpu, Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    _z = tf.expand_dims(_z, axis=1)
                    _z = tf.tile(_z, [1, Config.global_cfg.exp_cfg.n_view_g, 1])
                    _z = tf.reshape(_z, [batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g,
                                         Config.tex_gen_cfg.struct.z_dim])
                    tf_zt_all_gpu.append(_z)
        elif phase == 'test':
            tf_zt_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.placeholder(shape=[batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g,
                                               Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    tf_zt_all_gpu.append(_z)
        else:
            raise NotImplementedError
        dlatents_share_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                _dlatens_share = net_z_map_share.forward(tf_zt_all_gpu[g_id], phase, trunc_place_holder=trunc_place_holder,variable_scope_prefix='tex_gen')
                dlatents_share_all_gpu.append(_dlatens_share)

        dlatens_split_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                dlatens_split_per_gpu_list = []
                for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g):
                    net_id = i // Config.global_cfg.exp_cfg.n_view_g
                    dlatens_split = net_z_map_split_list[net_id].forward(tf_zt_all_gpu[g_id][i:(i+1)], phase, trunc_place_holder=trunc_place_holder,variable_scope_prefix='tex_gen', use_suffix=True)
                    dlatens_split_per_gpu_list.append(dlatens_split)
                dlatens_split_per_gpu = tf.concat(dlatens_split_per_gpu_list, axis=0)
            dlatens_split_all_gpu.append(dlatens_split_per_gpu)


        # tex generation
        output_chart_synbatch_all_gpu = []
        output_chart_realbatch_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            input_chart_pyramid_synbatch = data_loader.public_ops['input_chart_synbatch'][g_id]

            if phase == 'test' or phase == 'train':
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    lod = global_data_dict['lod_all_gpu'][g_id]
                    _output_albedo_synbatch = net_generator.forward(input_chart_pyramid_synbatch,
                                                                    dlatents_share_all_gpu[g_id], dlatens_split_all_gpu[g_id], lod_in=lod,
                                                                    name_scope='tex_gen')
                    output_chart_synbatch_all_gpu.append(_output_albedo_synbatch)

            if phase == 'train':
                input_chart_pyramid_realbatch = data_loader.public_ops['input_chart_realbatch'][g_id]
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _output_albedo_realbatch = net_generator.forward(input_chart_pyramid_realbatch,
                                                                     dlatents_share_all_gpu[g_id], dlatens_split_all_gpu[g_id], lod_in=lod,
                                                                     name_scope='tex_gen', global_data_dict=global_data_dict)
                    output_chart_realbatch_all_gpu.append(_output_albedo_realbatch)

        self.public_ops['trunc_place_holder'] = trunc_place_holder
        self.public_ops['fake_chart6view_synbatch'] = output_chart_synbatch_all_gpu
        self.public_ops['random_z'] = tf_zt_all_gpu
        # self.public_ops['dlatents'] = dlatents_all_gpu
        if phase == 'train':
            self.public_ops['fake_chart6view_realbatch'] = output_chart_realbatch_all_gpu
        if 'net_tex_g' not in global_data_dict:
            global_data_dict['net_tex_g'] = net_generator
        if 'net_z_map_share' not in global_data_dict:
            global_data_dict['net_z_map_share'] = net_z_map_share
        for view_id in range(1, 7):
            if 'net_z_map_split_{}'.format(view_id) not in global_data_dict:
                global_data_dict['net_z_map_split_{}'.format(view_id)] = net_z_map_split_list[view_id - 1]
        if phase == 'train':
            fake_chart = []
            for g_id in range(0, len(gpu_list)):
                # for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g):
                clip_albedo = tf.clip_by_value(denormalizeInput(output_chart_synbatch_all_gpu[g_id]), 0.0, 1.0)
                fake_chart.append(clip_albedo)
            vis_fake_albedo = tileImage(tf.concat(fake_chart, axis=0), nCol=6)
            self.add_summary(vis_fake_albedo, 'train', 'image', 'fake_chart6view')


class style_gen(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        global_data_dict = kwargs['global_data_dict']

        if phase == 'train':
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_size_per_gpu = Config.dataloader_cfg.batch_size // len(gpu_list)
            assert batch_size_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_size
        elif phase == 'test':
            gpu_list = [0]
            batch_size_per_gpu = 1
        else:
            raise NotImplementedError

        if 'net_tex_g' not in global_data_dict:
            net_generator = getattr(net_structures, Config.tex_gen_cfg.struct.struct_name)(0)
        else:
            net_generator = global_data_dict['net_tex_g']

        if 'net_z_map' not in global_data_dict:
            net_z_map = getattr(net_structures, 'z_mapping_to_w')(0)
        else:
            net_z_map = global_data_dict['net_z_map']

        # random_z
        trunc_place_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        if phase == 'train':
            tf_zt_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.random_normal(shape=[batch_size_per_gpu, Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    tf_zt_all_gpu.append(_z)
        elif phase == 'test':
            tf_zt_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    _z = tf.placeholder(shape=[batch_size_per_gpu, Config.tex_gen_cfg.struct.z_dim], dtype=tf.float32, name='z_code')
                    tf_zt_all_gpu.append(_z)
        else:
            raise NotImplementedError
        dlatents_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                _dlatens = net_z_map.forward(tf_zt_all_gpu[g_id], phase, trunc_place_holder=trunc_place_holder,variable_scope_prefix='tex_gen')
                dlatents_all_gpu.append(_dlatens)

        # tex generation
        fake_img_all_gpu = []
        fake_img_const1_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                lod = global_data_dict['lod_all_gpu'][g_id]
                if phase == 'train':
                    _fake_img = net_generator.forward(dlatents_all_gpu[g_id], lod_in=lod, name_scope='tex_gen', const_train=True, global_data_dict=global_data_dict)
                elif phase == 'test':
                    _fake_img = net_generator.forward(dlatents_all_gpu[g_id], lod_in=lod, name_scope='tex_gen', const_train=True)
                    _fake_img_const1 = net_generator.forward(dlatents_all_gpu[g_id], lod_in=lod, name_scope='tex_gen', const_train=False)
                    fake_img_const1_all_gpu.append(_fake_img_const1)
                fake_img_all_gpu.append(_fake_img)

        self.public_ops['trunc_place_holder'] = trunc_place_holder
        self.public_ops['fake_img'] = fake_img_all_gpu
        self.public_ops['fake_img_const1'] = fake_img_const1_all_gpu
        self.public_ops['random_z'] = tf_zt_all_gpu
        self.public_ops['dlatents'] = dlatents_all_gpu
        if 'net_tex_g' not in global_data_dict:
            global_data_dict['net_tex_g'] = net_generator
        if 'net_z_map' not in global_data_dict:
            global_data_dict['net_z_map'] = net_z_map
        if phase == 'train':
            # denormalize_fake = denormalizeInput(tf.concat(fake_img_all_gpu, axis=0)[::, ::, ::, 0:3])
            # denormalize_fake_tile = tileImage(tf.clip_by_value(denormalize_fake, 0, 1), nCol=min(8, batch_size_per_gpu))
            # self.add_summary(denormalize_fake_tile, 'train', 'image', 'fake_img')
            fake_img_all = tf.concat(fake_img_all_gpu, axis=0) # N * res * res * c
            fake_img_all_list = []
            img_per_group = Config.tex_gen_cfg.struct.output_channel // 3
            group_num = fake_img_all.shape[0]
            for group_id in range(group_num):
                for img_id in range(img_per_group):
                    fake_img_all_list.append(fake_img_all[group_id:(group_id + 1), ::, ::, img_id*3:(img_id + 1) * 3])
            fake_img_sum = tf.concat(fake_img_all_list, axis=0)
            denormalize_fake = tf.clip_by_value(denormalizeInput(fake_img_sum), 0.0, 1.0)
            denormalize_fake_tile = tileImage(denormalize_fake, nCol=min(2, batch_size_per_gpu) * img_per_group)
            self.add_summary(denormalize_fake_tile, 'train', 'image', 'fake_img')

