import tensorflow as tf



def perturb_gaussian(image, std=0.1):
    noise = tf.random.normal(shape=tf.shape(image), stddev=std)
    return tf.clip_by_value(image + noise, -1, 1)


def perturb_blur(image, kernel_size=5):
    image = tf.expand_dims(image, 0)
    blurred = tf.nn.avg_pool2d(image, ksize=kernel_size, strides=1, padding='SAME')
    return tf.squeeze(blurred, 0)


def perturb_salt_pepper(image, amount=0.05):
    mask = tf.random.uniform(tf.shape(image))
    image = tf.where(mask < amount / 2, -1.0, image)
    image = tf.where(mask > 1 - (amount / 2), 1.0, image)
    return image


def perturb_contrast(image, factor=1.0):
    return tf.clip_by_value(image * factor, -1.0, 1.0)
