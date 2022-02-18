from framework.modules.codebase.layers import *
from exp_config.config import Config


class z_mapping_to_w:
    def __init__(self, id=0):
        self.map_id = id
        self.inited = False

        self.FC_layers = Config.z_mapping_cfg.FC_layers
        self.dlatent_size = Config.z_mapping_cfg.dlatent_size_zt
        self.use_trunc = Config.z_mapping_cfg.use_trunc
        self.trunc_psi = Config.z_mapping_cfg.trunc_psi
        self.public_ops = {}

        self.gain = np.sqrt(2)
        self.use_wscale = True
        self.mapping_lrmul = 0.01
    def forward(self, latents_in, phase, trunc_place_holder, variable_scope_prefix, use_suffix=False):
        # with tf.variable_scope('{}_zmap_{}'.format(variable_scope_prefix, self.map_id)):
        if not use_suffix:
            suffix = ''
        else:
            suffix = '_{}'.format(self.map_id)
        with tf.variable_scope('{}_zmap{}'.format(variable_scope_prefix, suffix), reuse=tf.AUTO_REUSE):
            dlatents = pixel_norm(latents_in)
            # mapping layers
            for layer_id in range(self.FC_layers):
                with tf.variable_scope('dense_{}'.format(layer_id)):
                    fmaps = self.dlatent_size
                    dlatents = dense(dlatents, fmaps=fmaps, gain=self.gain, use_wscale=self.use_wscale, lrmul=self.mapping_lrmul)
                    dlatents = apply_bias(dlatents, lrmul=self.mapping_lrmul)
                    dlatents = tf.nn.leaky_relu(dlatents, alpha=0.2)

            dlatent_avg = tf.get_variable('dlatent_avg', shape=[1, self.dlatent_size],
                                          initializer=tf.initializers.zeros(),
                                          trainable=False)
            if phase == 'train' and not self.inited:
                with tf.name_scope('DlatentAvgUpdate_{}'.format(phase)):
                    batch_avg = tf.reduce_mean(dlatents, axis=0, keepdims=True)
                    update_op = tf.assign(dlatent_avg, lerp(batch_avg, dlatent_avg, 0.995))
                with tf.control_dependencies([update_op]):
                    dlatents = tf.identity(dlatents)
            if phase == 'test' and self.use_trunc:
                with tf.name_scope('Truncation_{}'.format(phase)):
                    dlatents = lerp(dlatent_avg, dlatents, trunc_place_holder)
                    self.public_ops['inference_dlatent'] = dlatents
            self.inited = True
            return dlatents


class tex_generator_spade_6chart_split_const:
    def __init__(self, id=0):  # nc_f = 512, nd_f = 4, out_dim = 32, sn = 0, norm = 'bn', act = 'relu',  voxel_order = 'NDHWC', mirror = False):
        self.inited = False
        struct = Config.tex_gen_cfg.struct
        self.chart_res = Config.global_cfg.exp_cfg.chart_res
        self.const_res = struct.const_res
        self.fm_base = struct.fm_base
        self.fmap_max = struct.fmap_max
        self.z_map_layers = int(np.log2(self.chart_res)) * 2 - 2
        self.progressive_type = struct.progressive_type

        self.gain = np.sqrt(2)
        self.use_wscale = True
        # self.mapping_lrmul = 0.01
        self.blur_filter = [1, 2, 1]
        self.public_ops = {}
        if Config.global_cfg.exp_cfg.fused == 't':
            self.fused = True
        elif Config.global_cfg.exp_cfg.fused == 'f':
            self.fused = False
        else:
            self.fused = 'auto'

    def forward(self, input_chart_pyramid, dlatents_in, dlatents2_in, lod_in, name_scope, global_data_dict={}):
        pyramid_layers = int(np.log2(self.chart_res) - np.log2(self.const_res)) + 1
        assert len(input_chart_pyramid) == pyramid_layers
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            dlatents = tf.tile(dlatents_in[:, np.newaxis], [1, self.z_map_layers, 1])
            if dlatents2_in is not None:
                dlatents2 = tf.tile(dlatents2_in[:, np.newaxis], [1, self.z_map_layers, 1])
                layer_idx = np.arange(self.z_map_layers)[np.newaxis, :, np.newaxis]
                cur_layers = self.z_map_layers - tf.cast(lod_in, tf.int32) * 2
                mixing_cutoff = tf.cond(
                    tf.random_uniform([], 0.0, 1.0) < 0.9,
                    lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                    lambda: cur_layers)
                dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
            def nf(stage): return min(int(self.fm_base / (2.0 ** (stage))), self.fmap_max)
            def blur(x): return blur2d(x, self.blur_filter) if self.blur_filter else x

            noise_inputs = []
            for layer_id in range(self.z_map_layers):
                res = layer_id // 2 + 2
                shape = [1, 2**res, 2**res, 1]
                noise_inputs.append(tf.get_variable('nosie{}'.format(layer_id), shape=shape, initializer=tf.initializers.random_normal(), trainable=False))
            def layer_epilogue(x, layer_idx, zero_init=True):
                # x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=True, zero_init=zero_init)
                x = apply_bias(x)
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = instance_norm(x)
                if Config.tex_gen_cfg.struct.spade_type == 'mul':
                    x = spade_mod_attr_chart_1cov_mask_mul(x, dlatents[:, layer_idx],
                                          input_chart_pyramid[layer_idx // 2],
                                          use_wscale=True)
                elif Config.tex_gen_cfg.struct.spade_type == 'mul2cov':
                    x = spade_mod_attr_chart_2cov_mask_mul(x, dlatents[:, layer_idx],
                                          input_chart_pyramid[layer_idx // 2],
                                          use_wscale=True)
                elif Config.tex_gen_cfg.struct.spade_type == 'concat':
                    x = spade_mod_attr_chart_2cov_mask_concat(x, dlatents[:, layer_idx],
                                                                  input_chart_pyramid[layer_idx // 2],
                                                                  use_wscale=True)
                elif Config.tex_gen_cfg.struct.spade_type == 'mul_concat':
                    x = spade_mod_attr_chart_2cov_mask_mul_concat(x, dlatents[:, layer_idx],
                                                              input_chart_pyramid[layer_idx // 2],
                                                              use_wscale=True)
                elif Config.tex_gen_cfg.struct.spade_type == 'none':
                    x = spade_mod_none_mask(x, dlatents[:, layer_idx],
                                                                  input_chart_pyramid[layer_idx // 2],
                                                                  use_wscale=True)
                else:
                    raise NotImplementedError
                return x
            # final
            with tf.variable_scope('4x4'):
                with tf.variable_scope('Const'):
                    x = tf.get_variable('const', shape=[Config.global_cfg.exp_cfg.n_view_g, 4, 4, nf(1)],
                                        initializer=tf.initializers.ones(), dtype='float32')
                    x = layer_epilogue(tf.tile(x, [tf.shape(dlatents_in)[0]//Config.global_cfg.exp_cfg.n_view_g, 1, 1, 1]), 0, zero_init=True)
                with tf.variable_scope('Conv'):
                    x = layer_epilogue(conv2d(x, fmaps=nf(1), kernel=3, gain=self.gain, use_wscale=True), 1)

            def block(res_log2, x):
                with tf.variable_scope('{}x{}'.format(2 ** res_log2, 2 ** res_log2)):
                # with tf.variable_scope('{}x{}'.format(2 ** res_log2, 2 ** res_log2), reuse=self.inited):
                    with tf.variable_scope('Conv0_up'):
                        x = upscale2d_conv2d(x, fmaps=nf(res_log2 - 1), kernel=3, gain=self.gain, use_wscale=True,
                                             fused_scale=self.fused)
                        x = layer_epilogue(blur(x), (res_log2 * 2 - 4))
                    with tf.variable_scope('Conv1'):
                        x = layer_epilogue(conv2d(x, fmaps=nf(res_log2 - 1), kernel=3, gain=self.gain, use_wscale=True),
                                           (res_log2 * 2 - 3))
                    return x
            def torgb(res_log2, x):
                lod = int(np.log2(self.chart_res) - res_log2)
                with tf.variable_scope('ToRGB_lod{}'.format(lod)):
                    return apply_bias(conv2d(x, fmaps=3, kernel=1, gain=1, use_wscale=True))

            if self.progressive_type == 'recursive':
                def cset(cur_lambda, new_cond, new_lambda):
                    return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
                def grow(x, res, lod):
                    y = block(res, x)
                    img = lambda: upscale2d(torgb(res, y), 2 ** lod)
                    img = cset(img, tf.greater(lod_in, lod),
                               lambda: upscale2d(lerp(torgb(res, y), upscale2d(torgb(res - 1, x)), lod_in - lod),
                                                 2 ** lod))
                    if lod > 0: img = cset(img, tf.less(lod_in, lod), lambda: grow(y, res + 1, lod - 1))
                    return img()
                with tf.variable_scope('style_gen'):
                    images_out = grow(x, 3, int(np.log2(self.chart_res) - 3))

            elif self.progressive_type == 'linear':
                with tf.variable_scope('style_gen'):
                    images_out = torgb(2, x)
                    for res in range(3, int(np.log2(self.chart_res)) + 1):
                        lod = int(np.log2(self.chart_res)) - res
                        x = block(res, x)
                        img = torgb(res, x)
                        images_out = upscale2d(images_out)
                        with tf.variable_scope('Grow_lod%d' % lod):
                            images_out = lerp_clip(img, images_out, lod_in - lod)
        assert images_out.shape[1:] == [self.chart_res, self.chart_res, 3]
        self.inited = True
        return images_out


class multi_view_image_discriminator_tex_spade:
    def __init__(self, id):  # nc_f = 32, nd_l = 4, n_shared = 3, n_head = 8, sn = 1, norm = 'None', act = 'relu':
        self.d_id = id
        self.inited = False
        struct = Config.tex_dis_cfg.struct
        self.img_res = Config.global_cfg.exp_cfg.img_res
        self.fmap_base = struct.fmap_base
        self.fmap_max = struct.fmap_max
        self.progressive_type = struct.progressive_type

        self.img_res_log2 = int(np.log2(self.img_res))
        self.use_wscale = True
        self.gain = np.sqrt(2)
        self.use_minibatch_stddev_layer = struct.use_minibatch_stddev_layer
        self.blur_filter = [1, 2, 1]
        if Config.global_cfg.exp_cfg.fused == 't':
            self.fused = True
        elif Config.global_cfg.exp_cfg.fused == 'f':
            self.fused = False
        else:
            self.fused = 'auto'

    def forward(self, _image, name_scope, lod_in=None):
        with tf.variable_scope('{}_{}'.format(name_scope, self.d_id), reuse=tf.AUTO_REUSE):
            def nf(stage): return int(min(int(self.fmap_base / (2.0 ** stage)), self.fmap_max))
            def fromrgb(x, res):
                with tf.variable_scope('FromRGB_lod%d' % (self.img_res_log2 - res)):
                    return tf.nn.leaky_relu(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=1, gain=self.gain, use_wscale=self.use_wscale)), alpha=0.2)

            def blur(x): return blur2d(x, self.blur_filter) if self.blur_filter else x
            def block(x, res_log2):
                with tf.variable_scope('{}x{}'.format(2**res_log2, 2**res_log2)):
                    if res_log2 >= 3:
                        with tf.variable_scope('Conv0'):
                            x = tf.nn.leaky_relu(apply_bias(conv2d(x, fmaps=nf(res_log2 - 1), kernel=3, gain = self.gain, use_wscale=True)), alpha=0.2)
                        with tf.variable_scope('Conv1_down'):
                            x = tf.nn.leaky_relu(apply_bias(
                                conv2d_downscale2d(blur(x), fmaps=nf(res_log2 - 2), kernel=3, gain=self.gain, use_wscale=True,
                                                   fused_scale=self.fused)), alpha=0.2)
                    else:
                        if self.use_minibatch_stddev_layer:
                            mbstd_group_size = 4
                            mbstd_num_features = 1
                            x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                        with tf.variable_scope('Conv'):
                            x = tf.nn.leaky_relu(apply_bias(conv2d(x, fmaps=nf(res_log2 - 1), kernel=3, gain=self.gain, use_wscale=True)), alpha=0.2)
                        with tf.variable_scope('Dense0'):
                            x = tf.nn.leaky_relu(apply_bias(dense(x, fmaps=nf(res_log2 - 2), gain=self.gain, use_wscale=True)), alpha=0.2)
                        with tf.variable_scope('Dense1'):
                            x = apply_bias(dense(x, fmaps=1, gain=1, use_wscale=True))
                    return x

            if self.progressive_type == 'recursive':
                with tf.variable_scope('style_dis'):
                    assert lod_in is not None
                    assert all(dim.value is not None for dim in _image.shape[1:])
                    def cset(cur_lambda, new_cond, new_lambda):
                        return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

                    def grow(res, lod):
                        x = lambda: fromrgb(downscale2d(_image, 2 ** lod), res)
                        if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
                        x = block(x(), res); y = lambda: x
                        if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(_image, 2 ** (lod + 1)), res - 1), lod_in - lod))
                        return y()
                    scores_out = grow(2, self.img_res_log2 - 2)
            elif self.progressive_type == 'linear':
                with tf.variable_scope('style_dis'):
                    img = _image
                    x = fromrgb(img, self.img_res_log2)
                    for res in range(self.img_res_log2, 2, -1):
                        lod = self.img_res_log2 - res
                        x = block(x, res)
                        img = downscale2d(img)
                        y = fromrgb(img, res - 1)
                        with tf.variable_scope('Grow_lod%d' % lod):
                            x = lerp_clip(x, y, lod_in - lod)
                    scores_out = block(x, 2)
            elif self.progressive_type == 'linear_lod0':
                with tf.variable_scope('style_dis'):
                    x = fromrgb(_image, self.img_res_log2)
                    for res in range(self.img_res_log2, 2, -1):
                        x = block(x, res)
                    scores_out = block(x, 2)
            # scores_out = tf.identity(scores_out, name='scores_out')
        self.inited = True
        return scores_out
