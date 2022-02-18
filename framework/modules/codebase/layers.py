import tensorflow as tf
import numpy as np

def normalizeInput(input_data):
    return input_data * 2.0 - 1.0


def denormalizeInput(input_data):
    return 0.5 * (input_data + 1.0)



# from styleGAN
def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, tf.float32)
    return tf.matmul(x, w)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # init = tf.initializers.random_normal(0, init_std, seed=np.random.randint(0, 1000))
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init, dtype=tf.float32) * runtime_coef


def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        output = x + b
    elif len(x.shape) == 4:
        output = x + tf.reshape(b, [1, 1, 1, -1])
    elif len(x.shape) == 5:
        output = x + tf.reshape(b, [1, 1, 1, 1, -1])
    else:
        raise NotImplementedError
    return output


def lerp(a, b, t):
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    """Linear interpolation."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)


def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    # f = np.tile(f, [1, 1, int(x.shape[1]), 1])
    f = np.tile(f, [1, 1, int(x.shape[-1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    # strides = [1, 1, stride, stride]
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
    x = tf.cast(x, orig_dtype)
    return x


def apply_noise(x, noise_var=None, randomize_noise=True, zero_init=True):
    with tf.variable_scope('Noise'):
        if zero_init:
            weight = tf.get_variable('weight', shape=[x.shape[-1].value], initializer=tf.initializers.zeros())
        else:
            weight = tf.get_variable('weight', shape=[x.shape[-1].value], initializer=tf.initializers.ones()) * 0.001
        if len(x.shape) == 5:
            if randomize_noise or noise_var is None:
                noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3], 1], dtype=x.dtype)
            else:
                noise = tf.cast(noise_var, x.dtype)
            return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, 1, 1, 1, -1])
        elif len(x.shape) == 4:
            if randomize_noise or noise_var is None:
                noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1], dtype=x.dtype)
            else:
                noise = tf.cast(noise_var, x.dtype)
            return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, 1, 1, -1])
        else:
            raise NotImplementedError


def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 4:
            axis_list = [1, 2]
        else:
            raise NotImplementedError
        x -= tf.reduce_mean(x, axis=axis_list, keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_list, keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

def spade_mod_attr_chart_2cov_mask_mul(x, dlatent, attr_chart, **kwargs):
    try:
        assert x.shape[0:3] == attr_chart.shape[0:3] # (BS, h, w)
    except:
        print(x.shape, attr_chart.shape)
        exit()
    with tf.variable_scope('StpadeModAttrChart'):
        style = tf.reshape(dlatent, [dlatent.shape[0], 1, 1, dlatent.shape[1]]) # N, 1, 1, 512
        style = tf.tile(style, [1, x.shape[1], x.shape[2], 1]) # N, res, res, 512
        style = style * attr_chart[::, ::, ::, -1:]    # mul mask
        with tf.variable_scope('cov1'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))
        with tf.variable_scope('cov2'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))

        style_shape = style.shape
        style = tf.reshape(style, [-1, style_shape[1], style_shape[2], style_shape[-1] // 2, 2])
        scale = style[...,0] + 1
        bias = style[...,1]
        assert scale.shape == x.shape
        assert bias.shape == x.shape
        return x * scale + bias



def spade_mod_attr_chart_1cov_mask_mul(x, dlatent, attr_chart, **kwargs):
    try:
        assert x.shape[0:3] == attr_chart.shape[0:3] # (BS, h, w)
    except:
        print(x.shape, attr_chart.shape)
        exit()
    with tf.variable_scope('StpadeModAttrChart'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[-1] * 2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 1, 1, x.shape[-1] * 2])
        style = tf.tile(style, [1, x.shape[1], x.shape[2], 1])
        style = style * attr_chart[::, ::, ::, -1:]
        return x * (style[::, ::, ::, 0:x.shape[-1]] + 1) + style[::, ::, ::, x.shape[-1]:]

def spade_mod_attr_chart_2cov_mask_concat(x, dlatent, attr_chart, **kwargs):
    assert x.shape[0:3] == attr_chart.shape[0:3] # (BS, h, w)
    with tf.variable_scope('StpadeModAttrChart'):
        style = tf.reshape(dlatent, [dlatent.shape[0], 1, 1, dlatent.shape[1]]) # N, 1, 1, 512
        style = tf.tile(style, [1, x.shape[1], x.shape[2], 1])
        # style = style * attr_chart[::, ::, ::, -1:]
        style = tf.concat([style, attr_chart[::, ::, ::, -1:]], axis=3)
        with tf.variable_scope('cov1'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))
        with tf.variable_scope('cov2'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))

        style_shape = style.shape
        style = tf.reshape(style, [-1, style_shape[1], style_shape[2], style_shape[-1] // 2, 2])
        scale = style[...,0] + 1
        bias = style[...,1]
        assert scale.shape == x.shape
        assert bias.shape == x.shape
        return x * scale + bias


def spade_mod_none_mask(x, dlatent, attr_chart, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[-1] * 2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
        return x * (style[:, 0] + 1) + style[:, 1]

def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[-1] * 2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
        return x * (style[:, 0] + 1) + style[:, 1]


def spade_mod_attr_chart_2cov_mask_mul_concat(x, dlatent, attr_chart, **kwargs):
    assert x.shape[0:3] == attr_chart.shape[0:3] # (BS, h, w)
    with tf.variable_scope('StpadeModAttrChart'):
        style = tf.reshape(dlatent, [dlatent.shape[0], 1, 1, dlatent.shape[1]]) # N, 1, 1, 512
        style = tf.tile(style, [1, x.shape[1], x.shape[2], 1])
        style = style * attr_chart[::, ::, ::, -1:]
        style = tf.concat([style, attr_chart[::, ::, ::, 0:6]], axis=3)
        with tf.variable_scope('cov1'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))
        with tf.variable_scope('cov2'):
            style = apply_bias(conv2d(style, fmaps=x.shape[-1] * 2, kernel=1, **kwargs))

        style_shape = style.shape
        style = tf.reshape(style, [-1, style_shape[1], style_shape[2], style_shape[-1] // 2, 2])
        scale = style[...,0] + 1
        bias = style[...,1]
        assert scale.shape == x.shape
        assert bias.shape == x.shape
        return x * scale + bias

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
        output = tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.
        output = tf.transpose(output, perm=[0, 2, 3, 1])
        return output

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format="NHWC")

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128
    else:
        raise NotImplementedError

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    # w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], **kwargs)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,2,2,1], padding='SAME', data_format='NHWC')


def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[-1]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[-1]])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4
    assert all(dim.value is not None for dim in x.shape[1:])
    # assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NHWC')
def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

def conv2d_downscale2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    # w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], **kwargs)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,2,2,1], padding='SAME', data_format='NHWC')

