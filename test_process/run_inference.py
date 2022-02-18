import os, sys, shutil, random
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(project_root)
import shutil, cv2
import numpy as np
import importlib
from exp_config.config import Config
from framework.parser.parser_base import parser
from tqdm import tqdm
from framework.utils.io import save_pfm, load_pfm
from framework.utils.io import get_fn_from_txt
_runner_gan = importlib.import_module('framework.graph_runner.' + Config.global_cfg.exp_cfg.graph_type)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = 'PCI_BUS_ID'
    args_dict = vars(parser.parse_args())

    # output folder
    assert args_dict['output_dir'] != ''
    Config.global_cfg.folder_cfg.output_dir = args_dict['output_dir']
    Config.global_cfg.folder_cfg.log_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'log')
    Config.global_cfg.folder_cfg.model_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'model')
    Config.global_cfg.folder_cfg.copy_script_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'script')
    Config.global_cfg.folder_cfg.validation_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'validation')
    Config.global_cfg.folder_cfg.dense_gen_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'dense_generation')
    Config.global_cfg.folder_cfg.test_result_dir = os.path.join(Config.global_cfg.folder_cfg.output_dir, 'test_result')
    assert os.path.exists(args_dict['data_root'])
    Config.dataloader_cfg.root_folder = args_dict['data_root']

    # environment
    # assert args_dict['gpu_id'] != ''
    Config.global_cfg.meta_cfg.gpu_list = [0]

    runner_gan = getattr(_runner_gan, Config.global_cfg.exp_cfg.gan_type)(init_for_train=False)
    # build graph
    runner_gan.build_graph(is_train=False)
    # init tf session
    runner_gan.init_session()
    runner_gan.load_previous_model(model=args_dict['model_path'])

    res = Config.global_cfg.exp_cfg.img_res
    view_num = Config.global_cfg.exp_cfg.n_view_d_synbatch

    model_name_list = []
    if str(args_dict['dataset']).lower() == 'car':
        name_txt = r'car_name_vae_sampled_shuffled.txt'
        test_name = 'CarVAESampledForDemo'
        model_name_list = get_fn_from_txt(r'./framework/utils/{}'.format(name_txt))[0:100]
    elif str(args_dict['dataset']).lower() == 'ffhq':
        name_txt = r'ffhq.txt'
        test_name = 'FFHQDemo'
        model_name_list = get_fn_from_txt(r'./framework/utils/{}'.format(name_txt))[0:100]
    elif str(args_dict['dataset']).lower() == 'shoe':
        test_name = 'ShoeDemo'
        model_name_list = [str(i) for i in range(100)]
    else:
        raise NotImplementedError

    runner_gan.run_chart(model_name_list, test_name)
    runner_gan.close_session()
