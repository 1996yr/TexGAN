import tensorflow as tf
from exp_config.config import Config
from framework.modules.module_base import module_base
from framework.modules.codebase import net_structures
from framework.modules.codebase.visualization import tileImage
# from framework.modules.codebase.layers import denormalizeInput

class view_predicion(module_base):
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

        if 'net_vp' not in global_data_dict:
            net_vp = getattr(net_structures, Config.vp_cfg.struct.struct_name)(0)
        else:
            net_vp = global_data_dict['net_vp']


        view_label_est_all_gpu = []
        for g_id in range(0, len(gpu_list)):
            input_mask = data_loader.public_ops['input_mask'][g_id]
            if phase == 'test' or phase == 'train':
                with tf.device('/gpu:{}'.format(g_id)), tf.name_scope('gpu{}'.format(g_id)):
                    view_label_est = net_vp.forward(input_mask, name_scope='view_prediction')
                    view_label_est_all_gpu.append(view_label_est)
        self.public_ops['view_est'] = view_label_est_all_gpu
        if 'net_vp' not in global_data_dict:
            global_data_dict['net_vp'] = net_vp
