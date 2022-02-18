import os, cv2, json, shutil, time, random
import tensorflow as tf
import numpy as np

from exp_config.config import Config
from framework.utils.log import log_message
from framework.utils.io import zip_folder, save_pfm
from tensorflow.python.client import timeline

from framework.graph_runner.graph_runner_base import graph_runner
from framework.modules.data_loader import data_loader_ph
from framework.modules.network import tex_generator as network_tex_g
from framework.modules.gather import gather as gather_module
from framework.modules.network import image_discriminator as network_image_d
from framework.modules.loss import loss
from framework.modules.solver import solver
from framework.modules.codebase.layers import denormalizeInput

class npy_data_iterator:
    def __iter__(self):
        self.data_id_input_synbatch = 0
        self.data_id_realimg_synbatch = 0
        self.data_id_input_realbatch = 0
        self.data_id_realimg_realbatch = 0

        self.group_per_iter = Config.dataloader_cfg.batch_group
        npy_root = os.path.join(Config.dataloader_cfg.root_folder, 'PackToNpy')

        if Config.global_cfg.exp_cfg.syn_batch > 0:
            self.syn_batch_conditional_input_attr = np.load(
                os.path.join(npy_root, 'SynBatchConditionalInput', 'attr_input.npy'))  # N*n_view_g*mask

            self.syn_batch_conditional_input_gather = np.load(
                os.path.join(npy_root, 'SynBatchConditionalInput', 'gather_input.npy'))  # N*n_view_d_syn*gather_map
            self.syn_batch_realimg = np.load(
                os.path.join(npy_root, 'RealImageSynBatch', 'realimg.npy'))  # M*n_view_d_syn*realimg
            assert self.syn_batch_conditional_input_attr.shape[0] == self.syn_batch_conditional_input_gather.shape[0]
        if Config.loss.smooth_loss > 0.0:
            self.syn_batch_conditional_input_smooth_gather = np.load(
                os.path.join(npy_root, 'SynBatchConditionalInput', 'smooth_gather_input.npy'))  # N*n_view_d_syn*gather_map_boundary
            self.syn_batch_conditional_input_smooth_mask = np.load(
                os.path.join(npy_root, 'SynBatchConditionalInput', 'smooth_mask_input.npy'))  # N*n_view_d_syn*mask_boundary
        if Config.gather_cfg.normal_weight > 0.0:
            self.syn_batch_conditional_input_normal_weight = np.load(
                os.path.join(npy_root, 'SynBatchConditionalInput', 'weight_map_input.npy')
            )

        if Config.global_cfg.exp_cfg.real_batch > 0:
            self.real_batch_conditional_input_attr = np.load(
                os.path.join(npy_root, 'RealBatchConditionalInput', 'attr_input.npy'))  # N*n_view_g*mask
            self.real_batch_conditional_input_gather = np.load(
                os.path.join(npy_root, 'RealBatchConditionalInput', 'gather_input.npy'))  # N*n_view_d_real*gather_map
            self.real_batch_realimg = np.load(
                os.path.join(npy_root, 'RealImageRealBatch', 'realimg.npy'))  # M*n_view_d_real*realimg
            assert self.real_batch_conditional_input_attr.shape[0] == self.real_batch_conditional_input_gather.shape[0]
        return self

    def __next__(self):
        result_dict = {}
        if Config.global_cfg.exp_cfg.real_batch > 0:
            id_list = self.get_data_id_list(current_id=self.data_id_realimg_realbatch,
                                            total_length=self.real_batch_realimg.shape[0],
                                            output_length=self.group_per_iter)
            result_dict['realimg_realbatch_ph'] = np.concatenate([self.real_batch_realimg[i] for i in id_list], axis=0)
            self.data_id_realimg_realbatch = (self.data_id_realimg_realbatch + self.group_per_iter) % \
                                             self.real_batch_realimg.shape[0]

            id_list = self.get_data_id_list(current_id=self.data_id_input_realbatch,
                                            total_length=self.real_batch_conditional_input_attr.shape[0],
                                            output_length=self.group_per_iter)
            result_dict['attrchart_realbatch_ph'] = np.concatenate(
                [self.real_batch_conditional_input_attr[i] for i in id_list], axis=0)
            result_dict['gathermap_realbatch_ph'] = np.concatenate(
                [self.real_batch_conditional_input_gather[i] for i in id_list], axis=0)
            self.data_id_input_realbatch = (self.data_id_input_realbatch + self.group_per_iter) % \
                                           self.real_batch_conditional_input_attr.shape[0]
        if Config.global_cfg.exp_cfg.syn_batch > 0:
            id_list = self.get_data_id_list(current_id=self.data_id_realimg_synbatch,
                                            total_length=self.syn_batch_realimg.shape[0], output_length=self.group_per_iter)
            result_dict['realimg_synbatch_ph'] = np.concatenate([self.syn_batch_realimg[i] for i in id_list], axis=0)
            self.data_id_realimg_synbatch = (self.data_id_realimg_synbatch + self.group_per_iter) % \
                                            self.syn_batch_realimg.shape[0]

            id_list = self.get_data_id_list(current_id=self.data_id_input_synbatch,
                                            total_length=self.syn_batch_conditional_input_attr.shape[0],
                                            output_length=self.group_per_iter)
            result_dict['attrchart_synbatch_ph'] = np.concatenate(
                [self.syn_batch_conditional_input_attr[i] for i in id_list], axis=0)

            gather_batch_list = []
            for i in id_list:
                if Config.global_cfg.exp_cfg.type == 'car' or Config.global_cfg.exp_cfg.type == 'shoe':
                    bin_count = (self.syn_batch_conditional_input_gather.shape[1] - 4) // 4
                    random_index_fsr = random.randint(a=4, b=int(3 + bin_count))
                    random_index_fsl = random.randint(a=int(4 + bin_count), b=int(3 + 2 * bin_count))
                    random_index_bsl = random.randint(a=int(4 + 2 * bin_count), b=int(3 + 3 * bin_count))
                    random_index_bsr = random.randint(a=int(4 + 3 * bin_count), b=int(3 + 4 * bin_count))
                    batch_index = [0, 1, 2, 3, random_index_fsr, random_index_fsl, random_index_bsl, random_index_bsr]
                elif Config.global_cfg.exp_cfg.type == 'ffhq':
                    random_index_0 = random.randint(a=0, b=3)
                    random_index_1 = random.randint(a=4, b=7)
                    random_index_2 = random.randint(a=8, b=11)
                    batch_index = [random_index_0, random_index_1, random_index_2]
                elif Config.global_cfg.exp_cfg.type == '2d':
                    batch_index = [0]
                else:
                    raise NotImplementedError
                gather_batch_list.append(self.syn_batch_conditional_input_gather[i][batch_index])
            result_dict['gathermap_synbatch_ph'] = np.concatenate(gather_batch_list, axis=0)
            if Config.loss.smooth_loss > 0.0:
                result_dict['smooth_gather_synbatch_ph'] = np.concatenate(
                    [self.syn_batch_conditional_input_smooth_gather[i] for i in id_list], axis=0)
                result_dict['smooth_mask_synbatch_ph'] = np.concatenate(
                    [self.syn_batch_conditional_input_smooth_mask[i] for i in id_list], axis=0)
            if Config.gather_cfg.normal_weight > 0.0:
                result_dict['weight_map_synbatch_ph'] = np.concatenate(
                    [self.syn_batch_conditional_input_normal_weight[i] for i in id_list], axis=0)
            self.data_id_input_synbatch = (self.data_id_input_synbatch + self.group_per_iter) % \
                                          self.syn_batch_conditional_input_attr.shape[0]
        return result_dict

    def get_data_id_list(self, current_id, total_length, output_length):
        output = [i % total_length for i in range(current_id, current_id + output_length)]
        return output

class spade_chart6view_mix_train(graph_runner):
    def __init__(self, init_for_train=True):
        super().__init__() # make folder, init log file
        # load validation input
        try:
            validation_input_folder = os.path.join(Config.dataloader_cfg.root_folder,
                                                   Config.dataloader_cfg.validation_input_folder)

            self.attr_chart = np.load(os.path.join(validation_input_folder, 'attr_chart.npy')).astype(np.float32)[
                              0:Config.global_cfg.meta_cfg.validation_output_num]
            self.fix_view_gather_map = np.load(os.path.join(validation_input_folder, 'fix_view_gather_map.npy')).astype(
                np.float32)[0:Config.global_cfg.meta_cfg.validation_output_num]
            self.fix_8view_gather_map = np.load(os.path.join(validation_input_folder, 'fix_8view_gather_map.npy')).astype(
                np.float32)[0:Config.global_cfg.meta_cfg.validation_output_num]
            self.round_view_gather_map = np.load(os.path.join(validation_input_folder, 'round_view_gather_map.npy')).astype(
                np.float32)[0:Config.global_cfg.meta_cfg.validation_output_num]
            grid_img_num = 64
            self.grid_attr_chart = np.load(os.path.join(validation_input_folder, 'grid_attr_chart.npy')).astype(np.float32)[
                                   0:grid_img_num]
            self.grid_gather_map = np.load(os.path.join(validation_input_folder, 'grid_fix_view_gather_map.npy')).astype(
                np.float32)[0:grid_img_num]
            self.grid_z = []
            np.random.seed(0)
            for i in range(grid_img_num):
                latents_tex = np.random.standard_normal(size=Config.tex_gen_cfg.struct.z_dim)
                latents_tex = latents_tex.astype(np.float32)
                self.grid_z.append(latents_tex)
        except:
            print('warning: validation data not found')
        if init_for_train:  # need validation data
            # init data reader
            npy_data_iterator_class = npy_data_iterator()
            self.npy_data_iterator = iter(npy_data_iterator_class)
            try:
                self.real_front_attr_chart = np.load(
                    os.path.join(validation_input_folder, 'real_front_attr_chart.npy')).astype(np.float32)
                self.real_front_gather_map = np.load(
                    os.path.join(validation_input_folder, 'real_front_gather_map.npy')).astype(np.float32)

                self.dense_attr_chart_dict = np.load(
                    os.path.join(validation_input_folder, 'dense_name_attr_chart_dict.npy'), allow_pickle=True).item()
                self.dense_gather_map_dict = np.load(
                    os.path.join(validation_input_folder, 'dense_name_gather_map_dict.npy'), allow_pickle=True).item()
            except:
                log_message('validation', 'not use real front validation/dense syn validation')

        self.n_view_g = Config.global_cfg.exp_cfg.n_view_g
        self.n_view_d_syn = Config.global_cfg.exp_cfg.n_view_d_synbatch
        self.n_view_d_real = Config.global_cfg.exp_cfg.n_view_d_realbatch
        self.chart_res = Config.global_cfg.exp_cfg.chart_res

    def build_graph(self, is_train=True, add_inference_part=True):
        assert os.path.exists(Config.global_cfg.folder_cfg.output_dir)
        assert os.path.exists(Config.global_cfg.folder_cfg.validation_dir)
        assert os.path.exists(Config.global_cfg.folder_cfg.model_dir)
        assert os.path.exists(Config.global_cfg.folder_cfg.log_dir)
        assert os.path.exists(Config.global_cfg.folder_cfg.dense_gen_dir)
        assert os.path.exists(Config.global_cfg.folder_cfg.test_result_dir)
        super().build_graph()
        log_message(self.__class__.__name__, '-----Building Graph...-----')

        self.global_data_dict = {}
        if not is_train:
            add_inference_part = True
        with self.graph.as_default():
            with tf.name_scope('Inputs'), tf.device('/cpu:0'):
                self.lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])  # enable progressive
                self.g_lr_in = tf.placeholder(tf.float32, name='glr', shape=[])  # enable progressive
                self.d_lr_in = tf.placeholder(tf.float32, name='dlr', shape=[])  # enable progressive

            lod_all_gpu = []
            lod_assign_ops = []
            for g_id in range(0, len(Config.global_cfg.meta_cfg.gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    with tf.variable_scope('gpu{}'.format(g_id)):
                        lod = tf.get_variable(name='lod', trainable=False, initializer=np.float32(0.0),
                                                  dtype='float32')
                        lod_all_gpu.append(lod)
                    assign_ops = tf.assign(lod, self.lod_in, name='assign_lod')
                    lod_assign_ops.append(assign_ops)

            self.global_data_dict['lod_all_gpu'] = lod_all_gpu
            self.global_data_dict['lod_assign_ops'] = lod_assign_ops

            if is_train:
                log_message(self.__class__.__name__, '-----Building data loader, using {}...-----'.format(Config.dataloader_cfg.module_name))
                data_module_train = getattr(data_loader_ph, Config.dataloader_cfg.module_name)()
                data_module_train.build_graph(
                    phase='train',
                    global_data_dict=self.global_data_dict,
                )
                self.submodule_dict['data_loader_train'] = data_module_train

                log_message(self.__class__.__name__, '-----Building texture generator, using {}.{}...-----'.format(
                    Config.tex_gen_cfg.module_name, Config.tex_gen_cfg.struct.struct_name))
                # generator submodule
                tex_generator_module = getattr(network_tex_g, Config.tex_gen_cfg.module_name)()
                tex_generator_module.build_graph(
                    phase='train',
                    global_data_dict=self.global_data_dict,
                    data_loader = data_module_train
                )
                self.submodule_dict['tex_generator'] = tex_generator_module

                log_message(self.__class__.__name__, '-----Building gather, using {}...-----'.format(Config.gather_cfg.module_name))
                # discriminator submodule
                gather = getattr(gather_module, Config.gather_cfg.module_name)()
                gather.build_graph(
                    phase='train',
                    data_loader=data_module_train,
                    tex_generator=tex_generator_module,
                    global_data_dict=self.global_data_dict
                )
                self.submodule_dict['gather_module'] = gather

                log_message(self.__class__.__name__, '-----Building texture discriminator, using {}...-----'.format(Config.tex_dis_cfg.module_name))
                # discriminator submodule
                tex_discriminator_module = getattr(network_image_d, Config.tex_dis_cfg.module_name)()
                tex_discriminator_module.build_graph(
                    data_loader=data_module_train,
                    gather_module=gather,
                    global_data_dict=self.global_data_dict,
                    generator_module = tex_generator_module
                )
                self.submodule_dict['tex_discriminator'] = tex_discriminator_module

                log_message(self.__class__.__name__, '-----Building tex loss, using {}...-----'.format(Config.loss.tex_loss_module_name))
                tex_loss_module = getattr(loss, Config.loss.tex_loss_module_name)()
                tex_loss_module.build_graph(
                    data_loader=data_module_train,
                    tex_generator=tex_generator_module,
                    gather_module=gather,
                    tex_discriminator=tex_discriminator_module,
                    global_data_dict=self.global_data_dict
                )
                self.submodule_dict['tex_loss'] = tex_loss_module

                log_message(self.__class__.__name__, '-----Building solver tex G for synthetic batch...-----')
                # G solver submodule
                var_prefix = ['tex_gen']
                loss_list_g = [self.submodule_dict['tex_loss'].public_ops['loss_tex_g_synbatch']]
                if Config.loss.smooth_loss > 0:
                    loss_list_g.append(self.submodule_dict['tex_loss'].public_ops['loss_smooth_synbatch'])
                solver_module_tex_G = getattr(solver, Config.solver_tex_G.module_name)(var_scope='tex_g_solver')
                solver_module_tex_G.build_graph(
                    loss_list=loss_list_g,
                    var_prefix=var_prefix,
                    params=Config.solver_tex_G.params,
                    lr_in=self.g_lr_in,
                    reuse = False
                )
                self.submodule_dict['solver_tex_g_synbatch'] = solver_module_tex_G

                if Config.global_cfg.exp_cfg.real_batch > 0:
                    log_message(self.__class__.__name__, '-----Building solver tex G for real batch...-----')
                    # G solver submodule
                    var_prefix = ['tex_gen']
                    loss_list_g = [self.submodule_dict['tex_loss'].public_ops['loss_tex_g_realbatch']]
                    solver_module_tex_G = getattr(solver, Config.solver_tex_G.module_name)(var_scope='tex_g_solver')
                    solver_module_tex_G.build_graph(
                        loss_list=loss_list_g,
                        var_prefix=var_prefix,
                        params=Config.solver_tex_G.params,
                        lr_in=self.g_lr_in,
                        reuse=True
                    )
                    self.submodule_dict['solver_tex_g_realbatch'] = solver_module_tex_G

                log_message(self.__class__.__name__, '-----Building solver tex D for synthetic batch...-----')
                # D solver submodule
                loss_list_d = [self.submodule_dict['tex_loss'].public_ops['loss_tex_d_synbatch']]
                if Config.loss.simplegp > 0:
                    loss_list_d.append(self.submodule_dict['tex_loss'].public_ops['loss_tex_gp_synbatch'])
                solver_module_tex_D = getattr(solver, Config.solver_tex_D.module_name)(var_scope='tex_d_solver')
                solver_module_tex_D.build_graph(
                    loss_list=loss_list_d,
                    var_prefix=['tex_dis'],
                    params=Config.solver_tex_D.params,
                    lr_in=self.d_lr_in,
                    reuse=False
                )
                self.submodule_dict['solver_tex_d_synbatch'] = solver_module_tex_D

                if Config.global_cfg.exp_cfg.real_batch > 0:
                    log_message(self.__class__.__name__, '-----Building solver tex D for real batch...-----')
                    # D solver submodule
                    loss_list_d = [self.submodule_dict['tex_loss'].public_ops['loss_tex_d_realbatch']]
                    if Config.loss.simplegp > 0:
                        loss_list_d.append(self.submodule_dict['tex_loss'].public_ops['loss_tex_gp_realbatch'])
                    solver_module_tex_D = getattr(solver, Config.solver_tex_D.module_name)(var_scope='tex_d_solver')
                    d_order_real = list(set(Config.tex_dis_cfg.d_order_real))
                    suffix_list = ['_dscope{}'.format(i) for i in d_order_real]
                    solver_module_tex_D.build_graph(
                        loss_list=loss_list_d,
                        var_prefix=['tex_dis'+suffix for suffix in suffix_list],
                        params=Config.solver_tex_D.params,
                        lr_in=self.d_lr_in,
                        reuse=True
                    )
                    self.submodule_dict['solver_tex_d_realbatch'] = solver_module_tex_D

                tensorboard_train = []
                for _name in self.submodule_dict:
                    tensorboard_train += self.submodule_dict[_name].tensorboard_ops['train']
                self.summary_train_op = tf.summary.merge(tensorboard_train)
            if add_inference_part:
                data_module_test = getattr(data_loader_ph, Config.dataloader_cfg.module_name)()
                data_module_test.build_graph(
                    phase='test',
                    global_data_dict=self.global_data_dict
                )
                self.submodule_dict['data_loader_test'] = data_module_test

                # generator submodule
                tex_generator_module_inference = getattr(network_tex_g, Config.tex_gen_cfg.module_name)()
                tex_generator_module_inference.build_graph(
                    phase='test',
                    global_data_dict=self.global_data_dict,
                    data_loader=data_module_test
                )
                self.submodule_dict['tex_generator_inference'] = tex_generator_module_inference

                gather_inference = getattr(gather_module, Config.gather_cfg.module_name)()
                gather_inference.build_graph(
                    phase='test',
                    data_loader=data_module_test,
                    tex_generator=tex_generator_module_inference,
                    global_data_dict=self.global_data_dict
                )
                self.submodule_dict['gather_module_inference'] = gather_inference

                # discriminator submodule
                tex_discriminator_inference = getattr(network_image_d, Config.tex_dis_cfg.module_name)()
                tex_discriminator_inference.build_graph(
                    data_loader=data_module_test,
                    gather_module=gather_inference,
                    global_data_dict=self.global_data_dict,
                    generator_module=tex_generator_module_inference,
                    phase='test'
                )
                self.submodule_dict['tex_discriminator_inference'] = tex_discriminator_inference

            self.global_data_dict = self.global_data_dict

    def get_feed_dict_for_train(self, batch_type, input_data_dict):
        if batch_type == 'syn':
            feed_dict = {self.lod_in: 0.0,
                         self.submodule_dict['data_loader_train'].public_ops['realimg_synbatch_ph']: input_data_dict['realimg_synbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['attrchart_synbatch_ph']: input_data_dict['attrchart_synbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['gathermap_synbatch_ph']: input_data_dict['gathermap_synbatch_ph']
                         }
            if Config.loss.smooth_loss > 0.0:
                feed_dict[self.submodule_dict['data_loader_train'].public_ops['smooth_gather_synbatch_ph']] = input_data_dict['smooth_gather_synbatch_ph']
                feed_dict[self.submodule_dict['data_loader_train'].public_ops['smooth_mask_synbatch_ph']] = input_data_dict['smooth_mask_synbatch_ph']
            if Config.gather_cfg.normal_weight > 0.0:
                feed_dict[self.submodule_dict['data_loader_train'].public_ops['weight_map_synbatch_ph']] = input_data_dict['weight_map_synbatch_ph']
        elif batch_type == 'real':
            feed_dict = {self.lod_in: 0.0,
                         self.submodule_dict['data_loader_train'].public_ops['realimg_realbatch_ph']:input_data_dict['realimg_realbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['attrchart_realbatch_ph']: input_data_dict['attrchart_realbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['gathermap_realbatch_ph']: input_data_dict['gathermap_realbatch_ph'],
                         }
        else:
            feed_dict = {self.lod_in: 0.0,
                         self.submodule_dict['data_loader_train'].public_ops['realimg_synbatch_ph']:
                             input_data_dict['realimg_synbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['attrchart_synbatch_ph']:
                             input_data_dict['attrchart_synbatch_ph'],
                         self.submodule_dict['data_loader_train'].public_ops['gathermap_synbatch_ph']:
                             input_data_dict['gathermap_synbatch_ph']
                         }
            real_dict = {}
            smooth_dict = {}
            if Config.global_cfg.exp_cfg.real_batch > 0:
                real_dict = {self.submodule_dict['data_loader_train'].public_ops['realimg_realbatch_ph']:
                                input_data_dict['realimg_realbatch_ph'],
                             self.submodule_dict['data_loader_train'].public_ops['attrchart_realbatch_ph']:
                                input_data_dict['attrchart_realbatch_ph'],
                             self.submodule_dict['data_loader_train'].public_ops['gathermap_realbatch_ph']:
                                input_data_dict['gathermap_realbatch_ph']}

            if Config.loss.smooth_loss > 0.0:
                smooth_dict = {
                    self.submodule_dict['data_loader_train'].public_ops['smooth_gather_synbatch_ph']: input_data_dict['smooth_gather_synbatch_ph'],
                    self.submodule_dict['data_loader_train'].public_ops['smooth_mask_synbatch_ph']: input_data_dict['smooth_mask_synbatch_ph']}
            if Config.gather_cfg.normal_weight > 0.0:
                feed_dict[self.submodule_dict['data_loader_train'].public_ops['weight_map_synbatch_ph']] = input_data_dict['weight_map_synbatch_ph']
            feed_dict.update(real_dict)
            feed_dict.update(smooth_dict)
        return feed_dict

    def run_training(self):
        log_message(self.__class__.__name__, '-----Begin Training...-----')
        cur_nimg = self.skip_iter
        prev_lod = -1

        np.random.seed(0)
        profiled = False
        skipped_first_profile = False

        while cur_nimg < Config.global_cfg.meta_cfg.max_img:
            lod, g_lr, d_lr = self.calculate_sched_param(cur_nimg)
            self.sess.run(self.global_data_dict['lod_assign_ops'], feed_dict={self.lod_in: lod})

            img_per_log = Config.global_cfg.meta_cfg.batch_repeat * Config.dataloader_cfg.batch_group * \
                          Config.global_cfg.exp_cfg.n_view_g * \
                          (Config.global_cfg.exp_cfg.syn_batch + Config.global_cfg.exp_cfg.real_batch) * \
                          Config.global_cfg.exp_cfg.grad_repeat

            if cur_nimg % Config.global_cfg.meta_cfg.sum_step < img_per_log and Config.global_cfg.meta_cfg.enable_summary:
                log_message('run training', 'run summary')
                data_dict = next(self.npy_data_iterator)
                feed_dict = self.get_feed_dict_for_train(batch_type='all', input_data_dict=data_dict)
                summary_train = self.sess.run(self.summary_train_op, feed_dict)
                log_message('run training', 'write summary')
                self.tensorboard_writer.add_summary(summary_train, cur_nimg)
                log_message('run training', 'write summary completed')

            if cur_nimg % Config.global_cfg.meta_cfg.run_validation_step < img_per_log:

                assert 'tex_generator_inference' in self.submodule_dict

                if Config.global_cfg.folder_cfg.temp_output_dir == '':
                    validation_dir = os.path.join(Config.global_cfg.folder_cfg.validation_dir, '{}'.format(str(cur_nimg).zfill(10)))
                    os.makedirs(validation_dir, exist_ok=True)
                else:
                    validation_dir = os.path.join(Config.global_cfg.folder_cfg.temp_validation_dir, '{}'.format(str(cur_nimg).zfill(10)))
                    os.makedirs(validation_dir, exist_ok=True)
                if Config.global_cfg.meta_cfg.enable_validation:
                    log_message('run training', 'validation')
                    self.run_discriminator_input(folder_root=validation_dir, folder_name='discriminator_input')
                    self.run_validation(folder_root=validation_dir, folder_name='wo_trunc_training_sample', modellist=list(range(0, 32)), trunc=1.0)
                    self.run_validation(folder_root=validation_dir, folder_name='trunc_training_sample', modellist=list(range(0, 32)), trunc=0.7)
                    self.run_validation(folder_root=validation_dir, folder_name='wo_trunc_test_sample', modellist=list(range(32, 64)), trunc=1.0)
                    self.run_validation(folder_root=validation_dir, folder_name='trunc_test_sample', modellist=list(range(32, 64)), trunc=0.7)


                    if Config.global_cfg.folder_cfg.temp_output_dir != '':
                        temp_zip_fn = os.path.join(Config.global_cfg.folder_cfg.temp_output_dir, 'validation_{}.zip'.format(str(cur_nimg).zfill(10)))
                        zip_folder(validation_dir, temp_zip_fn)
                        dst_fn = os.path.join(Config.global_cfg.folder_cfg.validation_dir, '{}.zip'.format(str(cur_nimg).zfill(10)))
                        log_message('validation', 'cp {} {}'.format(temp_zip_fn, dst_fn))
                        shutil.copyfile(temp_zip_fn, dst_fn)
                        os.remove(temp_zip_fn)


            if cur_nimg % Config.global_cfg.meta_cfg.checkpoint_step < img_per_log:
                self.saver.save(self.sess, os.path.join(Config.global_cfg.folder_cfg.model_dir, 'model'), global_step=cur_nimg)

            for batch_repeat in range(Config.global_cfg.meta_cfg.batch_repeat):
                if Config.global_cfg.meta_cfg.profile == 1 and not profiled and lod == 0.0:
                    if skipped_first_profile:
                        # profile grad computation of generator
                        log_message(self.__class__.__name__, 'profile grad computation of generator')
                        run_metadata_gen = tf.RunMetadata()
                        data_dict = next(self.npy_data_iterator)
                        feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                        feed_dict[self.g_lr_in] = g_lr
                        tf_g_v_gen = self.submodule_dict['solver_tex_g_synbatch'].public_ops['grads_and_vars']
                        g_v_gen = self.sess.run(tf_g_v_gen, feed_dict,
                                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                      run_metadata=run_metadata_gen
                                      )
                        trace = timeline.Timeline(step_stats=run_metadata_gen.step_stats)
                        trace_file_gen = open(os.path.join(Config.global_cfg.folder_cfg.log_dir, 'timeline_gen_grad_and_var.ctf.json'), 'w')
                        trace_file_gen.write(trace.generate_chrome_trace_format())
                        trace_file_gen.close()
                        log_message(self.__class__.__name__, 'profile grad computation of generator completed')

                        # compute avg of grad in cpu
                        s_t = time.time()
                        g_v_list_gen = [g_v_gen] * Config.global_cfg.exp_cfg.grad_repeat
                        g_dict_gen = {tf_g_v_gen[j][0]: sum([g_v_list_gen[i][j][0] for i in range(len(g_v_list_gen))]) / len(g_v_list_gen) for j in range(len(tf_g_v_gen))}
                        g_dict_gen[self.g_lr_in] = g_lr
                        e_t = time.time()
                        log_message(self.__class__.__name__, 'avg grad in cpu {} seconds'.format(e_t - s_t))

                        # profile variable update of generator
                        log_message(self.__class__.__name__, 'profile train_op of generator')
                        run_metadata_gen_train_op = tf.RunMetadata()
                        self.sess.run(self.submodule_dict['solver_tex_g_synbatch'].public_ops['train_op'],
                                      feed_dict=g_dict_gen,
                                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                      run_metadata=run_metadata_gen_train_op)
                        trace = timeline.Timeline(step_stats=run_metadata_gen_train_op.step_stats)
                        trace_file_gen_train_op = open(
                            os.path.join(Config.global_cfg.folder_cfg.log_dir, 'timeline_gen_train_op.ctf.json'), 'w')
                        trace_file_gen_train_op.write(trace.generate_chrome_trace_format())
                        trace_file_gen_train_op.close()
                        log_message(self.__class__.__name__, 'profile train_op of generator completed')

                        # profile grad computation of discriminator
                        log_message(self.__class__.__name__, 'profile grad computation of discriminator')
                        run_metadata_dis = tf.RunMetadata()
                        data_dict = next(self.npy_data_iterator)
                        feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                        feed_dict[self.d_lr_in] = d_lr
                        tf_g_v_dis = self.submodule_dict['solver_tex_d_synbatch'].public_ops['grads_and_vars']
                        g_v_dis = self.sess.run(tf_g_v_dis, feed_dict,
                                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                      run_metadata=run_metadata_dis
                                      )
                        trace = timeline.Timeline(step_stats=run_metadata_dis.step_stats)
                        trace_file_dis = open(os.path.join(Config.global_cfg.folder_cfg.log_dir, 'timeline_dis_grad_and_var.ctf.json'), 'w')
                        trace_file_dis.write(trace.generate_chrome_trace_format())
                        trace_file_dis.close()
                        log_message(self.__class__.__name__, 'profile grad computation of discriminator completed')

                        # compute avg of grad in cpu
                        s_t = time.time()
                        g_v_list_dis = [g_v_dis] * Config.global_cfg.exp_cfg.grad_repeat
                        g_dict_dis = {
                            tf_g_v_dis[j][0]: sum([g_v_list_dis[i][j][0] for i in range(len(g_v_list_dis))]) / len(
                                g_v_list_dis) for j in range(len(tf_g_v_dis))}
                        g_dict_dis[self.d_lr_in] = d_lr
                        e_t = time.time()
                        log_message(self.__class__.__name__, 'avg grad in cpu {} seconds'.format(e_t - s_t))

                        # profile variable update of discriminator
                        log_message(self.__class__.__name__, 'profile train_op of discriminator')
                        run_metadata_dis_train_op = tf.RunMetadata()
                        self.sess.run(self.submodule_dict['solver_tex_d_synbatch'].public_ops['train_op'],
                                      feed_dict=g_dict_dis,
                                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                      run_metadata=run_metadata_dis_train_op)
                        trace = timeline.Timeline(step_stats=run_metadata_dis_train_op.step_stats)
                        trace_file_dis_train_op = open(
                            os.path.join(Config.global_cfg.folder_cfg.log_dir, 'timeline_dis_train_op.ctf.json'), 'w')
                        trace_file_dis_train_op.write(trace.generate_chrome_trace_format())
                        trace_file_dis_train_op.close()
                        log_message(self.__class__.__name__, 'profile train_op of discriminator completed')

                        profiled = True
                    else:
                        skipped_first_profile = True

                        data_dict = next(self.npy_data_iterator)
                        feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                        feed_dict[self.g_lr_in] = g_lr
                        self.sess.run(self.submodule_dict['solver_tex_g_synbatch'].public_ops['train_op'],
                                      feed_dict)

                        data_dict = next(self.npy_data_iterator)
                        feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                        feed_dict[self.d_lr_in] = d_lr
                        self.sess.run(self.submodule_dict['solver_tex_d_synbatch'].public_ops['train_op'],
                                      feed_dict)

                else:
                    for syn_batch in range(Config.global_cfg.exp_cfg.syn_batch):
                        g_v_list_gen = []
                        g_v_list_dis = []
                        tf_g_v_gen = self.submodule_dict['solver_tex_g_synbatch'].public_ops['grads_and_vars']
                        tf_g_v_dis = self.submodule_dict['solver_tex_d_synbatch'].public_ops['grads_and_vars']
                        for grad_repeat in range(Config.global_cfg.exp_cfg.grad_repeat):
                            data_dict = next(self.npy_data_iterator)
                            feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                            g_v_gen, g_v_dis = self.sess.run([tf_g_v_gen, tf_g_v_dis], feed_dict)
                            g_v_list_gen.append(g_v_gen)
                            g_v_list_dis.append(g_v_dis)

                        g_dict_gen = {tf_g_v_gen[j][0]: sum([g_v_list_gen[i][j][0] for i in range(len(g_v_list_gen))]) / len(g_v_list_gen) for j in range(len(tf_g_v_gen))}
                        g_dict_gen[self.g_lr_in] = g_lr
                        self.sess.run(self.submodule_dict['solver_tex_g_synbatch'].public_ops['train_op'], feed_dict=g_dict_gen)

                        g_dict_dis = {tf_g_v_dis[j][0]: sum([g_v_list_dis[i][j][0] for i in range(len(g_v_list_dis))]) / len(g_v_list_dis) for j in range(len(tf_g_v_dis))}
                        g_dict_dis[self.d_lr_in] = d_lr
                        self.sess.run(self.submodule_dict['solver_tex_d_synbatch'].public_ops['train_op'], g_dict_dis)
                        if batch_repeat == 0:
                            log_string = 'nimg ' + str(cur_nimg) + ' lod ' + str(lod) + ' minibatch ' + str(
                                Config.dataloader_cfg.batch_group * Config.global_cfg.exp_cfg.n_view_g * Config.global_cfg.exp_cfg.grad_repeat)
                            log_message('training process', log_string)

                    for real_batch in range(Config.global_cfg.exp_cfg.real_batch):
                        g_v_list_gen = []
                        g_v_list_dis = []
                        tf_g_v_gen = self.submodule_dict['solver_tex_g_realbatch'].public_ops['grads_and_vars']
                        tf_g_v_dis = self.submodule_dict['solver_tex_d_realbatch'].public_ops['grads_and_vars']
                        for grad_repeat in range(Config.global_cfg.exp_cfg.grad_repeat):
                            data_dict = next(self.npy_data_iterator)
                            feed_dict = self.get_feed_dict_for_train(batch_type='real', input_data_dict=data_dict)


                            g_v_gen, g_v_dis = self.sess.run([tf_g_v_gen, tf_g_v_dis], feed_dict)
                            g_v_list_gen.append(g_v_gen)
                            g_v_list_dis.append(g_v_dis)

                        g_dict_gen = {tf_g_v_gen[j][0]: sum([g_v_list_gen[i][j][0] for i in range(len(g_v_list_gen))]) / len(g_v_list_gen) for j in range(len(tf_g_v_gen))}
                        g_dict_gen[self.g_lr_in] = g_lr
                        self.sess.run(self.submodule_dict['solver_tex_g_realbatch'].public_ops['train_op'], feed_dict=g_dict_gen)

                        g_dict_dis = {tf_g_v_dis[j][0]: sum([g_v_list_dis[i][j][0] for i in range(len(g_v_list_dis))]) / len(g_v_list_dis) for j in range(len(tf_g_v_dis))}
                        g_dict_dis[self.d_lr_in] = d_lr
                        self.sess.run(self.submodule_dict['solver_tex_d_realbatch'].public_ops['train_op'], g_dict_dis)
                        if batch_repeat == 0:
                            log_string = 'nimg ' + str(cur_nimg) + ' lod ' + str(lod) + ' minibatch ' + str(
                                Config.dataloader_cfg.batch_group * Config.global_cfg.exp_cfg.n_view_g * Config.global_cfg.exp_cfg.grad_repeat)
                            log_message('training process', log_string)
                cur_nimg += Config.dataloader_cfg.batch_group * Config.global_cfg.exp_cfg.n_view_g * \
                            (Config.global_cfg.exp_cfg.syn_batch + Config.global_cfg.exp_cfg.real_batch) * Config.global_cfg.exp_cfg.grad_repeat

        log_message(self.__class__.__name__, '-----Finished Training...-----')
        self.saver.save(self.sess, os.path.join(Config.global_cfg.folder_cfg.model_dir, 'model_latest'), global_step=cur_nimg)

    def calculate_sched_param(self, cur_nimg):
        phase_img = Config.tex_gen_cfg.struct.phase_kimg * 1000
        full_res_log2 = np.log2(Config.global_cfg.exp_cfg.chart_res)
        if phase_img == 0.0:
            lod = 0.0
        else:
            lod = full_res_log2 - 3
            phase_id = cur_nimg // (2 * phase_img)
            lod = lod - phase_id
            phase_nimg = cur_nimg - (2 * phase_img) * phase_id
            if phase_nimg > phase_img:
                lod = lod - float(phase_nimg - phase_img) / float(phase_img)
            if lod <= 0:
                lod = 0
        cur_res = 2 ** (full_res_log2 - int(np.floor(lod)))
        batch_size = max(Config.global_cfg.meta_cfg.batch_dict[cur_res], Config.dataloader_cfg.batch_group)
        Config.global_cfg.meta_cfg.batch_dict[cur_res] = batch_size
        Config.global_cfg.exp_cfg.grad_repeat = batch_size // Config.dataloader_cfg.batch_group
        if cur_res < 128:
            g_lr = 0.001
            d_lr = 0.001
        else:
            g_lr = Config.solver_tex_G.lr[cur_res]
            d_lr = Config.solver_tex_D.lr[cur_res]
        return lod, g_lr, d_lr

    def run_generation(self, lod, val_latents_tex, modellist=[], trunc=1.0, keep_gen_fix8_view_pfm=False):
        z_num = len(val_latents_tex)

        _gen_fix_view = np.zeros(shape=[len(modellist), z_num, 6, Config.global_cfg.exp_cfg.gather_map_res,
                                        Config.global_cfg.exp_cfg.gather_map_res, 3])
        _gen_fix8view = np.zeros(shape=[len(modellist), z_num, 8, Config.global_cfg.exp_cfg.gather_map_res,
                                        Config.global_cfg.exp_cfg.gather_map_res, 3])
        _gen_chart = np.zeros(shape=[len(modellist), z_num, Config.global_cfg.exp_cfg.n_view_g, Config.global_cfg.exp_cfg.chart_res,
                                        Config.global_cfg.exp_cfg.chart_res, 3])
        _gen_chart_masked = np.zeros(shape=[len(modellist), z_num, Config.global_cfg.exp_cfg.n_view_g, Config.global_cfg.exp_cfg.chart_res,
                                        Config.global_cfg.exp_cfg.chart_res, 3])

        _gen_fix_z = np.zeros(shape=[len(modellist), 36, Config.global_cfg.exp_cfg.gather_map_res,
                                        Config.global_cfg.exp_cfg.gather_map_res, 3])
        _gen_grid_img = np.zeros(shape=[len(self.grid_z), 128, 128, 3])
        if len(modellist) == 0:
            modellist = list(range(Config.global_cfg.meta_cfg.validation_output_num))
        min_model_id = min(modellist)
        for i in modellist:
            for z_id in range(z_num):
                if Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen' or Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen_mix':
                    latent = val_latents_tex[z_id:z_id + 1]
                else:
                    latent = np.tile(val_latents_tex[z_id:z_id + 1], (self.n_view_g, 1))
                feed_dict = {self.submodule_dict['tex_generator_inference'].public_ops['random_z'][0]: latent,
                             self.submodule_dict['data_loader_test'].public_ops['attr_chart_place_holder_synbatch']:
                                 self.attr_chart[i],
                             self.submodule_dict['data_loader_test'].public_ops['gather_map_place_holder_synbatch']:
                                 self.fix_view_gather_map[i],
                             self.submodule_dict['tex_generator_inference'].public_ops['trunc_place_holder']:
                                 trunc * np.ones(shape=[1], dtype=np.float32)}
                validation_ops = [self.submodule_dict['tex_generator_inference'].public_ops['fake_chart6view_synbatch'][0],
                                  self.submodule_dict['gather_module_inference'].public_ops['fake_img_recons_synbatch'][0]]
                chart, fake_recons = self.sess.run(validation_ops, feed_dict=feed_dict)
                mask = self.attr_chart[i][::, ::, ::, 0:1]
                fake_recons = np.clip(fake_recons, a_min=-1, a_max=1)
                chart = np.clip(chart, a_min=-1, a_max=1)

                _gen_fix_view[i - min_model_id, z_id] = fake_recons
                _gen_chart[i - min_model_id, z_id] = chart
                _gen_chart_masked[i - min_model_id, z_id] = chart * mask

                feed_dict = {self.submodule_dict['tex_generator_inference'].public_ops['random_z'][0]: latent,
                             self.submodule_dict['data_loader_test'].public_ops['attr_chart_place_holder_synbatch']:
                                 self.attr_chart[i],
                             self.submodule_dict['data_loader_test'].public_ops['gather_map_place_holder_synbatch']:
                                 self.fix_8view_gather_map[i],
                             self.submodule_dict['tex_generator_inference'].public_ops['trunc_place_holder']:
                                 trunc * np.ones(shape=[1], dtype=np.float32)}
                validation_ops = [self.submodule_dict['gather_module_inference'].public_ops['fake_img_recons_synbatch'][0]]
                fake_recons = self.sess.run(validation_ops, feed_dict=feed_dict)
                if keep_gen_fix8_view_pfm:
                    fake_recons = np.array(fake_recons)
                else:
                    fake_recons = np.clip(fake_recons, a_min=-1, a_max=1)
                _gen_fix8view[i - min_model_id, z_id] = fake_recons
            for view_group_id in range(6):
                if Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen' or Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen_mix':
                    latent = val_latents_tex[0:1]
                else:
                    latent = np.tile(val_latents_tex[0:1], (self.n_view_g, 1))
                feed_dict = {self.submodule_dict['tex_generator_inference'].public_ops['random_z'][0]: latent,
                             self.submodule_dict['data_loader_test'].public_ops['attr_chart_place_holder_synbatch']:
                                 self.attr_chart[i],
                             self.submodule_dict['data_loader_test'].public_ops['gather_map_place_holder_synbatch']:
                                 self.round_view_gather_map[i, view_group_id * Config.global_cfg.exp_cfg.n_view_g:(
                                 view_group_id + 1) * Config.global_cfg.exp_cfg.n_view_g],
                             self.submodule_dict['tex_generator_inference'].public_ops['trunc_place_holder']: trunc * np.ones(shape=[1], dtype=np.float32)}
                validation_ops = self.submodule_dict['gather_module_inference'].public_ops['fake_img_recons_synbatch'][0]
                fake_recons = self.sess.run(validation_ops, feed_dict=feed_dict)
                fake_recons = np.clip(fake_recons, a_min=-1, a_max=1)
                _gen_fix_z[i - min_model_id, view_group_id * Config.global_cfg.exp_cfg.n_view_g: (view_group_id+1)*Config.global_cfg.exp_cfg.n_view_g] = fake_recons

        for z_id in range(len(self.grid_z)):

                if Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen' or Config.tex_gen_cfg.module_name == 'tex_generator_spade_6gen_mix':
                    latent = self.grid_z[z_id:z_id + 1]
                else:
                    latent = np.tile(self.grid_z[z_id:z_id + 1], (self.n_view_g, 1))
                feed_dict = {self.submodule_dict['tex_generator_inference'].public_ops['random_z'][0]: latent,
                             self.submodule_dict['data_loader_test'].public_ops['attr_chart_place_holder_synbatch']:
                                 self.grid_attr_chart[z_id],
                             self.submodule_dict['data_loader_test'].public_ops['gather_map_place_holder_synbatch']:
                                 self.grid_gather_map[z_id:(z_id+1)],
                             self.submodule_dict['tex_generator_inference'].public_ops['trunc_place_holder']: trunc * np.ones(shape=[1], dtype=np.float32)}
                validation_ops = self.submodule_dict['gather_module_inference'].public_ops['fake_img_recons_synbatch'][0]
                fake_recons = self.sess.run(validation_ops, feed_dict=feed_dict)
                fake_recons = np.clip(fake_recons, a_min=-1, a_max=1)
                _gen_grid_img[z_id] = fake_recons[0]
        return _gen_fix_view, _gen_fix_z, _gen_chart, _gen_chart_masked, _gen_grid_img, _gen_fix8view

    def run_validation(self, folder_root, folder_name, trunc, modellist, keep_gen_fix8_view_pfm=False):
        np.random.seed(0)
        val_latents_tex = []
        z_num = 32
        for val_id in range(z_num):
            latents_tex = np.random.standard_normal(size=Config.tex_gen_cfg.struct.z_dim)
            latents_tex = latents_tex.astype('float32')
            val_latents_tex.append(latents_tex)
        _gen_fix_view, _gen_fix_z, _gen_chart, _gen_chart_masked, _gen_grid, _gen_fix8view = self.run_generation(lod=0.0,
                                                                                                  val_latents_tex=val_latents_tex,
                                                                                                  modellist=modellist,
                                                                                                  trunc=trunc, keep_gen_fix8_view_pfm=keep_gen_fix8_view_pfm)
        log_message('run training', 'validation completed')
        img_res = Config.global_cfg.exp_cfg.img_res
        gen_fix_view = np.zeros(shape=[len(modellist) * img_res,
                                       len(val_latents_tex) * img_res, 3], dtype=np.float32)
        gen_fix_8view = np.zeros(shape=[len(modellist) * img_res,
                                       len(val_latents_tex) * 8 * img_res, 3], dtype=np.float32)
        gen_fix_z = np.zeros(shape=[len(modellist) * img_res,
                                    36 * img_res, 3])
        gen_chart = np.zeros(shape=[len(modellist) * img_res,
                                    len(val_latents_tex) * Config.global_cfg.exp_cfg.n_view_g * img_res, 3])
        gen_chart_masked = np.zeros(shape=[len(modellist) * img_res,
                                           len(val_latents_tex) * Config.global_cfg.exp_cfg.n_view_g * img_res, 3])
        gen_chart_front = np.zeros(shape=[len(modellist) * img_res,
                                           len(val_latents_tex) * img_res, 3])

        gen_grid = np.zeros(shape=[8 * 128, 8 * 128, 3])
        validation_dir = os.path.join(folder_root, folder_name)
        os.makedirs(validation_dir, exist_ok=True)
        # write grid img
        for i in range(8):
            for j in range(8):
                img = denormalizeInput(_gen_grid[i * 8 + j])[::, ::, ::-1]
                img = np.array(img * 255, dtype=np.uint8)
                gen_grid[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = img
        cv2.imwrite(os.path.join(validation_dir, 'grid_fake.png'), gen_grid)

        # write fix view img
        for v_id in range(Config.global_cfg.exp_cfg.n_view_g):
            for model_id in range(len(modellist)):
                for z_id in range(z_num):
                    output = denormalizeInput(_gen_fix_view[model_id, z_id, v_id][::, ::, ::-1])
                    gen_fix_view[model_id * img_res:(model_id + 1) * img_res,
                    z_id * img_res:(z_id + 1) * img_res] = np.array(output)
            cv2.imwrite(os.path.join(validation_dir, 'recons_fix_view{}.png'.format(v_id)),
                        np.array(gen_fix_view * 255, dtype=np.uint8))

        for model_id in range(len(modellist)):
            for z_id in range(z_num):
                for v_id in range(Config.global_cfg.exp_cfg.n_view_g):
                    output_chart = np.array(denormalizeInput(_gen_chart[model_id, z_id, v_id][::, ::, ::-1]))
                    gen_chart[model_id * img_res:(model_id + 1) * img_res,
                        (z_id * Config.global_cfg.exp_cfg.n_view_g + v_id) * img_res:
                        (z_id * Config.global_cfg.exp_cfg.n_view_g + v_id + 1) * img_res] = output_chart

                    output_chart_masked = np.array(denormalizeInput(_gen_chart_masked[model_id, z_id, v_id][::, ::, ::-1]))
                    gen_chart_masked[model_id * img_res:(model_id + 1) * img_res,
                        (z_id * Config.global_cfg.exp_cfg.n_view_g + v_id) * img_res:
                        (z_id * Config.global_cfg.exp_cfg.n_view_g + v_id + 1) * img_res] = output_chart_masked
                gen_chart_front[model_id * img_res:(model_id + 1) * img_res, z_id * img_res: (z_id + 1) * img_res] = \
                    gen_chart_masked[model_id * img_res:(model_id + 1) * img_res,
                        (z_id * Config.global_cfg.exp_cfg.n_view_g) * img_res:
                        (z_id * Config.global_cfg.exp_cfg.n_view_g + 1) * img_res]
                for v_id in range(8):
                    output_img = np.array(denormalizeInput(_gen_fix8view[model_id, z_id, v_id][::, ::, ::-1]))
                    gen_fix_8view[model_id * img_res:(model_id + 1) * img_res, (z_id * 8 + v_id) * img_res:
                        (z_id * 8 + v_id + 1) * img_res] = output_img

            for v_id in range(36):
                output = np.array(denormalizeInput(_gen_fix_z[model_id, v_id][::, ::, ::-1]))
                gen_fix_z[model_id * img_res:(model_id + 1) * img_res, v_id * img_res:(v_id + 1) * img_res] = output
        cv2.imwrite(os.path.join(validation_dir, 'chart.png'), np.array(gen_chart * 255, dtype=np.uint8))
        cv2.imwrite(os.path.join(validation_dir, 'chart_masked.png'), np.array(gen_chart_masked * 255, dtype=np.uint8))
        cv2.imwrite(os.path.join(validation_dir, 'fix_z_round_view.png'), np.array(gen_fix_z * 255, dtype=np.uint8))
        if keep_gen_fix8_view_pfm:
            cv2.imwrite(os.path.join(validation_dir, 'fake_fix8view.png'),
                        np.array(np.clip(gen_fix_8view, a_min=0, a_max=1) * 255, dtype=np.uint8))
            save_pfm(os.path.join(validation_dir, 'fake_fix8view.pfm'), np.array(gen_fix_8view, dtype=np.float32))
        else:
            cv2.imwrite(os.path.join(validation_dir, 'fake_fix8view.png'),
                        np.array(gen_fix_8view * 255, dtype=np.uint8))
        cv2.imwrite(os.path.join(validation_dir, 'crop_front_chart.png'), np.array(gen_chart_front * 255, dtype=np.uint8))

    def run_discriminator_input(self, folder_root, folder_name):
        output_folder = os.path.join(folder_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        gpu_list = Config.global_cfg.meta_cfg.gpu_list
        real_list = []
        fake_list = []
        gather_map_list = []

        img_num_row = 12
        img_num_col = self.n_view_d_syn * 12
        # img_num = self.n_view_d * img_num_row * 12
        syn_batch = Config.global_cfg.exp_cfg.syn_batch
        real_batch = Config.global_cfg.exp_cfg.real_batch
        for repeat in range(img_num_row * img_num_col // (Config.dataloader_cfg.batch_group * self.n_view_d_syn * (syn_batch + real_batch)) + 1):
            for suffix in ['_synbatch'] * syn_batch + ['_realbatch'] * real_batch:
                data_dict = next(self.npy_data_iterator)
                if suffix == '_synbatch':
                    feed_dict = self.get_feed_dict_for_train(batch_type='syn', input_data_dict=data_dict)
                else:
                    feed_dict = self.get_feed_dict_for_train(batch_type='real', input_data_dict=data_dict)

                for gid in range(len(gpu_list)):
                    run_ops_real = self.submodule_dict['data_loader_train'].public_ops['input_real_img'+suffix][gid]
                    run_ops_fake = self.submodule_dict['gather_module'].public_ops['fake_img_recons'+suffix][gid]
                    run_ops_gather_map = self.submodule_dict['data_loader_train'].public_ops['input_gather_map'+suffix][gid]
                    real, fake, gather_map = self.sess.run(
                        [run_ops_real, run_ops_fake, run_ops_gather_map],
                        feed_dict=feed_dict)  # bs_gpu * 128 * 128 * 3
                    real = np.clip(real, -1, 1)
                    fake = np.clip(fake, -1, 1)
                    mask = np.where(gather_map[::, ::, ::, 0:1] >=0, 1, 0)
                    if suffix == '_synbatch':
                        # real_fill = real[::, ::, ::, 0:3] * real[::, ::, ::, 3:4]
                        # fake_fill = fake * mask
                        real_fill = real[::, ::, ::, 0:3]
                        fake_fill = fake
                        gather_map_fill = gather_map
                    elif suffix == '_realbatch':
                        batch_group_per_gpu = Config.dataloader_cfg.batch_group // len(gpu_list)
                        real_fill = np.zeros(shape=[Config.global_cfg.exp_cfg.n_view_d_synbatch * batch_group_per_gpu, 128, 128, 3])
                        fake_fill = np.zeros(shape=[Config.global_cfg.exp_cfg.n_view_d_synbatch * batch_group_per_gpu, 128, 128, 3])
                        gather_map_fill = np.zeros(shape=[Config.global_cfg.exp_cfg.n_view_d_synbatch * batch_group_per_gpu, 128, 128, 2])
                        for group_id in range(batch_group_per_gpu):
                            offset = (Config.global_cfg.exp_cfg.n_view_d_synbatch - Config.global_cfg.exp_cfg.n_view_d_realbatch) // 2 + group_id * Config.global_cfg.exp_cfg.n_view_d_synbatch
                            real_fill[offset:Config.global_cfg.exp_cfg.n_view_d_realbatch + offset] = \
                                real[group_id * Config.global_cfg.exp_cfg.n_view_d_realbatch: (group_id + 1) * Config.global_cfg.exp_cfg.n_view_d_realbatch, ::, ::, 0:3]
                            fake_fill[offset:Config.global_cfg.exp_cfg.n_view_d_realbatch + offset] = \
                                fake[group_id * Config.global_cfg.exp_cfg.n_view_d_realbatch:(group_id + 1) * Config.global_cfg.exp_cfg.n_view_d_realbatch]
                            gather_map_fill[offset:Config.global_cfg.exp_cfg.n_view_d_realbatch + offset] = \
                                gather_map[group_id * Config.global_cfg.exp_cfg.n_view_d_realbatch:(group_id + 1) * Config.global_cfg.exp_cfg.n_view_d_realbatch]

                    else:
                        raise NotImplementedError

                    gather_map_list.append(gather_map_fill)
                    real_list.append(real_fill)
                    fake_list.append(fake_fill)
        real_np = np.array(denormalizeInput(np.concatenate(real_list, axis=0)) * 255)
        fake_np = np.array(denormalizeInput(np.concatenate(fake_list, axis=0)) * 255)
        gather_np = np.array(np.concatenate(gather_map_list, axis=0), dtype=np.float32)

        real_out = np.zeros(shape=[img_num_row * 128, img_num_col * 128, 3])
        fake_out = np.zeros(shape=[img_num_row * 128, img_num_col * 128, 3])
        gather_out = np.ones(shape=[img_num_row * 128, img_num_col * 128, 3])
        for i in range(img_num_row):
            for j in range(img_num_col):
                real_out[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, ::] = real_np[i * img_num_col + j, ::, ::, ::]
                fake_out[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, ::] = fake_np[i * img_num_col + j, ::, ::, ::]
                gather_out[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, 1:3] = gather_np[i * img_num_col + j, ::, ::,
                                                                                ::] / 128.0
        cv2.imwrite(os.path.join(output_folder, 'real.png'), real_out[::, ::, ::-1])
        cv2.imwrite(os.path.join(output_folder, 'fake.png'), fake_out[::, ::, ::-1])
        save_pfm(os.path.join(output_folder, 'gather_map.pfm'), gather_out)

    def run_chart(self, model_name_list, test_name):
        self.sess.run(self.global_data_dict['lod_assign_ops'], feed_dict={self.lod_in: 0.0})
        model_num = len(model_name_list)
        output_root = os.path.join(Config.global_cfg.folder_cfg.test_result_dir, 'output_demo', test_name)
        os.makedirs(output_root, exist_ok=True)
        input_condition = np.load(os.path.join(Config.dataloader_cfg.root_folder, 'PackToNpy', 'SynBatchConditionalInput', 'attr_input.npy'))
        print('input_condition.shape: ', input_condition.shape)
        data_length = input_condition.shape[0]
        input_gather = np.load(
            os.path.join(Config.dataloader_cfg.root_folder, 'PackToNpy', 'SynBatchConditionalInput', 'gather_input.npy'))
        print('data length: ', data_length, 'model num: ', model_num)
        np.random.seed(0)
        trunc = 1.0

        for data_id in range(model_num):
            print('{}/{}'.format(data_id, model_num))
            output_chart_folder = os.path.join(output_root, model_name_list[data_id])
            os.makedirs(output_chart_folder, exist_ok=True)
            for chart_id in range(Config.global_cfg.exp_cfg.n_view_g):
                spade_in = np.array(input_condition[data_id] * 255, dtype=np.uint8)
                cv2.imwrite(os.path.join(output_chart_folder, 'spade_mask{}.png'.format(chart_id)), spade_in[chart_id])
            for gather_map_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                gather_map = input_gather[data_id][gather_map_id]
                save_pfm(os.path.join(output_chart_folder, 'gather_map{}.pfm'.format(gather_map_id)),
                         np.concatenate([gather_map, np.ones(shape=[gather_map.shape[0], gather_map.shape[1], 1])], axis=2))
            latents_tex = np.random.standard_normal(size=Config.tex_gen_cfg.struct.z_dim)
            latents_tex = np.reshape(latents_tex.astype('float32'), newshape=[1, Config.tex_gen_cfg.struct.z_dim])
            latent_tile = np.tile(latents_tex, [Config.global_cfg.exp_cfg.n_view_g, 1])

            feed_dict = {self.submodule_dict['tex_generator_inference'].public_ops['random_z'][0]: latent_tile,
                         self.submodule_dict['data_loader_test'].public_ops['attr_chart_place_holder_synbatch']: input_condition[data_id],
                         self.submodule_dict['tex_generator_inference'].public_ops['trunc_place_holder']: trunc * np.ones(shape=[1], dtype=np.float32),
                         self.submodule_dict['data_loader_test'].public_ops['gather_map_place_holder_synbatch']: input_gather[data_id]}
            validation_ops = [self.submodule_dict['tex_generator_inference'].public_ops['fake_chart6view_synbatch'][0],
                              self.submodule_dict['gather_module_inference'].public_ops['fake_img_recons_synbatch'][0]]
            fake_chart, fake_projection = self.sess.run(validation_ops, feed_dict=feed_dict)
            fake_chart = fake_chart[::, ::, ::, ::-1]
            output_chart = denormalizeInput(np.clip(fake_chart, -1, 1)) * 255
            for chart_id in range(Config.global_cfg.exp_cfg.n_view_g):
                mask_per_chart = input_condition[data_id][chart_id]
                output_img = np.array(output_chart[chart_id] * mask_per_chart, dtype=np.uint8) + np.array((1 - mask_per_chart), dtype=np.uint8) * 128
                cv2.imwrite(os.path.join(output_chart_folder, 'chart{}.png'.format(chart_id)), output_img)

            fake_projection = fake_projection[::, ::, ::, ::-1]
            output_projected_img = denormalizeInput(np.clip(fake_projection, -1, 1)) * 255
            for gather_map_id in range(Config.global_cfg.exp_cfg.n_view_d_synbatch):
                output_img = np.array(output_projected_img[gather_map_id], dtype=np.uint8)
                cv2.imwrite(os.path.join(output_chart_folder, 'projected_img_{}.png'.format(gather_map_id)), output_img)
