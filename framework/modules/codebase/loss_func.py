import tensorflow as tf

def smooth_loss(smooth_gather_img, fake_recons_img, smooth_mask):
    try:
        assert smooth_gather_img.shape[0:2] == fake_recons_img.shape[0:2] == smooth_mask.shape[0:2]
    except:
        print(smooth_gather_img.shape, fake_recons_img.shape, smooth_mask.shape)
        raise Exception
    delta = (smooth_gather_img - fake_recons_img) * smooth_mask
    loss = tf.reduce_sum(delta * delta) / tf.reduce_sum(smooth_mask)
    return loss

def loss_logistic_nonsaturating_g(fake_d):
    # return tf.nn.softplus(tf.reduce_mean(-fake_d))
    return tf.reduce_mean(tf.nn.softplus(-fake_d))

def loss_logistic_simplegp_d(real_d, fake_d):
    loss_d = tf.nn.softplus(fake_d)  # -log(1 - logistic(fake_scores_out))
    loss_d += tf.nn.softplus(
        -real_d)
    return tf.reduce_mean(loss_d)
    # return tf.nn.softplus(tf.reduce_mean(fake_d - real_d))

def l2_img_loss(img1, img2):
    err = img1 - img2
    loss = tf.reduce_mean(err ** 2)
    return tf.reduce_mean(loss)

def loss_simple_gp(real_data, d_real):
    real_loss = tf.reduce_sum(d_real)
    real_grads = tf.gradients(real_loss, [real_data])[0]
    gp_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
    return tf.reduce_mean(gp_penalty)