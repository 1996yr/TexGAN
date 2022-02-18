import os, sys, shutil
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(project_root)
import shutil
import importlib
from exp_config.config import Config
from framework.parser.parser_base import parser

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
    Config.tex_gen_cfg.struct.phase_kimg = args_dict['phase_kimg'] * Config.global_cfg.exp_cfg.n_view_g
    Config.global_cfg.exp_cfg.grad_repeat = args_dict['grad_repeat']
    # environment
    assert args_dict['gpu_id'] != ''
    Config.global_cfg.meta_cfg.gpu_list = list(map(int, args_dict['gpu_id'].strip().split(',')))

    # for philly temp storage
    if args_dict['temp_output_dir'] != '':
        assert args_dict['temp_output_dir'] != '/home/v-ruiyu'
        if os.path.exists(args_dict['temp_output_dir']):
            shutil.rmtree(args_dict['temp_output_dir'])
        Config.global_cfg.folder_cfg.temp_output_dir = args_dict['temp_output_dir']
        Config.global_cfg.folder_cfg.temp_validation_dir = os.path.join(Config.global_cfg.folder_cfg.temp_output_dir, 'validation')
        Config.global_cfg.folder_cfg.temp_dense_gen_dir = os.path.join(Config.global_cfg.folder_cfg.temp_output_dir, 'dense_generation')

    # analysis
    Config.global_cfg.meta_cfg.profile = args_dict['profile']
    Config.global_cfg.meta_cfg.enable_summary = args_dict['enable_summary']
    Config.global_cfg.meta_cfg.enable_validation = args_dict['enable_validation']

    # exp config
    Config.dataloader_cfg.batch_group = args_dict['batch_group']
    Config.global_cfg.exp_cfg.syn_batch = args_dict['syn_batch']
    Config.global_cfg.exp_cfg.real_batch = args_dict['real_batch']
    Config.dataloader_cfg.shuffle = args_dict['data_shuffle']
    Config.loss.smooth_loss = args_dict['smooth_loss']
    Config.gather_cfg.normal_weight = args_dict['grad_weight']
    Config.global_cfg.exp_cfg.random_background = args_dict['random_bg']
    Config.global_cfg.exp_cfg.random_shift = args_dict['random_shift']

    model_path = args_dict['model_path']
    Config.global_cfg.exp_cfg.type = args_dict['exp_type']


    # copy script and config to output dir
    try:
        script_cp_dst = Config.global_cfg.folder_cfg.copy_script_dir
        if os.path.exists(script_cp_dst):
            shutil.rmtree(script_cp_dst)
        shutil.copytree(project_root, script_cp_dst, ignore=shutil.ignore_patterns(".git"))
    except:
        pass



    runner_gan = getattr(_runner_gan, Config.global_cfg.exp_cfg.gan_type)()
    # build graph
    runner_gan.build_graph(add_inference_part=True)
    # init tf session
    runner_gan.init_session()
    if model_path != '':
        runner_gan.load_previous_model(model_path)
    else:
        runner_gan.load_previous_model()

    # run training
    runner_gan.run_training()
    runner_gan.close_session()

