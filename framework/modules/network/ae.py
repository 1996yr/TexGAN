import tensorflow as tf
from exp_config.config import Config
from framework.modules.module_base import module_base
from framework.modules.codebase import net_structures
from framework.modules.codebase.visualization import tileImage
# from framework.modules.codebase.layers import denormalizeInput

class auto_encoder(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        phase = kwargs['phase']
        global_data_dict = kwargs['global_data_dict']
        data_loader = kwargs['data_loader']

        if phase == 'train':
            gpu_list = Config.global_cfg.meta_cfg.gpu_list
            batch_size_per_gpu = Config.dataloader_cfg.batch_size // len(gpu_list)
            assert batch_size_per_gpu * len(gpu_list) == Config.dataloader_cfg.batch_size
        elif phase == 'test':
            gpu_list = [0]
            batch_size_per_gpu = 1
        else:
            raise NotImplementedError

        if 'net_ae' not in global_data_dict:
            net_ae = getattr(net_structures, Config.ae_cfg.struct.struct_name)(0)
        else:
            net_ae = global_data_dict['net_ae']


        recons_mask_all_gpu = []
        latents_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            input_mask = data_loader.public_ops['input_mask'][g_id]
            if phase == 'test' or phase == 'train':
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    recons_mask, latents = net_ae.forward(input_mask, name_scope='auto_encoder')
                    recons_mask_all_gpu.append(recons_mask)
                    latents_all_gpu.append(latents)
        self.public_ops['latents'] = latents_all_gpu
        self.public_ops['recons_mask'] = recons_mask_all_gpu
        if 'net_ae' not in global_data_dict:
            global_data_dict['net_ae'] = net_ae
        if phase == 'train':
            recons_mask_list = []
            for g_id in range(0, len(gpu_list)):
                # for i in range(batch_group_per_gpu * Config.global_cfg.exp_cfg.n_view_g):
                clip_output = tf.clip_by_value(recons_mask_all_gpu[g_id], 0.0, 1.0)
                recons_mask_list.append(clip_output)
            vis_fake_albedo = tileImage(tf.concat(recons_mask_list, axis=0), nCol=8)
            self.add_summary(vis_fake_albedo, 'train', 'image', 'recons_mask')


