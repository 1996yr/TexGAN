import os

zt_dim = 512
dlatents_size = 512
fmbase = 4096
progressive_type = 'recursive'
N_G_out = 6
N_D_out_SynBatch = 8
N_D_out_RealBatch = 4

class Config(object):
    class global_cfg():
        class folder_cfg():
            output_dir = ''
            log_dir = os.path.join(output_dir, 'log')
            model_dir = os.path.join(output_dir, 'model')
            copy_script_dir = os.path.join(output_dir, 'script')
            validation_dir = os.path.join(output_dir, 'validation')
            dense_gen_dir = os.path.join(output_dir, 'dense_generation')
            test_result_dir = os.path.join(output_dir, 'test_result')

            temp_output_dir = ''  # for philly data io
            temp_validation_dir = ''
            temp_dense_gen_dir = ''
            previous_model = ''

        class meta_cfg():
            gpu_list = [0]
            profile = 0
            auto_restart = 1
            batch_repeat = 4
            batch_dict = {8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8} # bs = max(bathsize, data_cfg.bathc_group)
            max_img = 25000 * 1000 * N_G_out
            enable_summary = 0
            sum_step = 8000
            max_num_checkpoint = int(600)
            checkpoint_step = 200 * 1000 * N_G_out
            enable_validation = 0
            validation_output_num = 64
            run_validation_step = 200 * 1000 * N_G_out

        class exp_cfg():
            graph_type = 'tex_gan_runner'
            gan_type = 'spade_chart6view_mix_train'

            syn_batch = 0
            real_batch = 0
            grad_repeat = 0

            n_view_g = N_G_out
            n_view_d_synbatch = N_D_out_SynBatch
            n_view_d_realbatch = N_D_out_RealBatch
            chart_res = 128
            img_res = 128
            gather_map_res = 128
            random_background = 0
            random_shift = 0
            fused = 'auto'
            type = 'shoe'

    class dataloader_cfg():
        module_name = r'data_loader_npy'
        root_folder = r''
        validation_input_folder = 'ValidationInput'
        batch_group = 0
        shuffle = 1


    class z_mapping_cfg():
        struct_name = 'z_mapping_to_w'
        FC_layers = 8
        dlatent_size_zt = dlatents_size
        use_trunc = True
        trunc_psi = 1.0

    class tex_gen_cfg():
        module_name = 'tex_generator_spade_6chart_mix'
        class struct():
            spade_type = 'mul2cov'
            struct_name = 'tex_generator_spade_6chart_split_const'
            z_dim = zt_dim
            const_res = 4
            fm_base = fmbase
            fmap_max = 512
            phase_kimg = 600 * N_G_out
            progressive_type = progressive_type
            style_mix = True

    class gather_cfg():
        module_name = 'gather6chart_mix'
        normal_weight = 0.0

    class tex_dis_cfg():
        module_name = r'tex_discriminator_spade_chart_MD_Mix'

        d_num = 5
        d_order_syn = [0, 1, 2, 2, 3, 3, 4, 4]
        d_weight_syn = [1, 1, 1, 1, 1, 1, 1, 1]
        d_order_real = [0, 1, 2, 2]
        d_weight_real = [1, 1, 1, 1]

        class struct():
            struct_name = 'multi_view_image_discriminator_tex_spade'
            fmap_base = fmbase
            fmap_max = 512
            progressive_type = progressive_type
            use_minibatch_stddev_layer = False

    class loss():
        smooth_loss = 0.0
        tex_loss_module_name = 'loss_chart_gan_mix'
        gan_loss = "style_base"
        simplegp = 10.0

    class solver_tex_G():
        module_name = 'solver_base'
        params = 'adam-0.001-0.0-0.99-sum'
        lr = {128: 0.0015, 256: 0.002}

    class solver_tex_D():
        module_name = 'solver_base'
        params = 'adam-0.001-0.0-0.99-sum'
        lr = {128: 0.0015, 256: 0.002}

    def __init__(self):
        super().__init__()

