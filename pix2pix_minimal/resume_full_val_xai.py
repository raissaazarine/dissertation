"""
Standalone resume script for the notebook's "FULL VALIDATION SET perturbation-based XAI
evaluation" cell (pix2pix.ipynb, last cell). The notebook kernel died partway through
(processed up to image_idx 3178 of 7373), so this reproduces the exact same setup and
re-runs the exact same resumable loop as a background process, appending to the same CSV.
"""
import tensorflow as tf
import tensorflow_io as tfio
import os
import pathlib
import csv
import numpy as np
from contextlib import contextmanager

dataset_name = "polyps_v7"
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3
PATH = pathlib.Path(r'./datasets') / dataset_name
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tfio.experimental.image.decode_tiff(image)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :3]
    real_image = image[:, :w, :3]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    input_image = input_image[:, :, :INPUT_CHANNELS]
    return input_image, real_image


val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.tif'), shuffle=True, seed=42)
val_dataset = val_dataset.map(load_image_test)
val_dataset = val_dataset.batch(BATCH_SIZE)



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                                 kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[None, None, INPUT_CHANNELS])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                            kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                   use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


generator = Generator()
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
status.expect_partial()
print(f'restored checkpoint: {tf.train.latest_checkpoint(checkpoint_dir)}')



def saliency_map(model, input_image):
    input_tensor = tf.cast(input_image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = model(input_tensor, training=True)
        loss = tf.reduce_mean(pred)
    grads = tape.gradient(loss, input_tensor)
    sal = tf.abs(grads[0, :, :, 0]).numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal, pred[0].numpy()


def grad_times_input(model, input_image):
    input_tensor = tf.cast(input_image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = model(input_tensor, training=True)
        loss = tf.reduce_mean(pred)
    grads = tape.gradient(loss, input_tensor)
    gxi = tf.abs(grads * input_tensor)[0, :, :, 0].numpy()
    gxi = (gxi - gxi.min()) / (gxi.max() - gxi.min() + 1e-8)
    return gxi, pred[0].numpy()


def smoothgrad(model, input_image, n_samples=20, noise_std=0.1):
    input_tensor = tf.cast(input_image, tf.float32)
    grads_sum = tf.zeros_like(input_tensor)
    for _ in range(n_samples):
        noise = tf.random.normal(tf.shape(input_tensor), stddev=noise_std)
        noisy = input_tensor + noise
        with tf.GradientTape() as tape:
            tape.watch(noisy)
            pred = model(noisy, training=True)
            loss = tf.reduce_mean(pred)
        grads_sum = grads_sum + tf.abs(tape.gradient(loss, noisy))
    avg_grads = (grads_sum / n_samples)[0, :, :, 0].numpy()
    avg_grads = (avg_grads - avg_grads.min()) / (avg_grads.max() - avg_grads.min() + 1e-8)
    return avg_grads, pred[0].numpy()



out_root_full = pathlib.Path('./xai_evaluation/full_val')
out_root_full.mkdir(parents=True, exist_ok=True)
csv_path = out_root_full / 'perturbation_metrics_per_image.csv'


@contextmanager
def dropout_disabled(model):
    dropout_layers = [l for l in model.submodules if isinstance(l, tf.keras.layers.Dropout)]
    original_rates = [l.rate for l in dropout_layers]
    for l in dropout_layers:
        l.rate = 0.0
    try:
        yield
    finally:
        for l, r in zip(dropout_layers, original_rates):
            l.rate = r


N_STEPS_FULL = 10
N_NOISE_FULL = 5
SMOOTHGRAD_N_SAMPLES_FULL = 5
NOISE_STD = 0.03
FILL_VALUE = -1.0


def smoothgrad_fast(model, input_image):
    return smoothgrad(model, input_image, n_samples=SMOOTHGRAD_N_SAMPLES_FULL, noise_std=0.1)


XAI_METHODS_FULL = {
    'Saliency': saliency_map,
    'Grad x Input': grad_times_input,
    'SmoothGrad': smoothgrad_fast,
}


def rank_positions(attribution):
    h, w = attribution.shape
    order = np.argsort(-attribution.reshape(-1))
    return np.unravel_index(order, (h, w))


def mask_delete(image, ys, xs, n_masked, fill_value=FILL_VALUE):
    image_np = image.numpy().copy()
    image_np[ys[:n_masked], xs[:n_masked], :] = fill_value
    return tf.convert_to_tensor(image_np, dtype=tf.float32)


def mask_insert(image, ys, xs, n_inserted, fill_value=FILL_VALUE):
    image_np = image.numpy()
    base = np.full_like(image_np, fill_value)
    base[ys[:n_inserted], xs[:n_inserted], :] = image_np[ys[:n_inserted], xs[:n_inserted], :]
    return tf.convert_to_tensor(base, dtype=tf.float32)


def deletion_insertion_auc(model, test_input, attribution, clean_01, steps=N_STEPS_FULL):
    inp = test_input[0]
    h, w = attribution.shape
    n_pix = h * w
    ys, xs = rank_positions(attribution)
    fractions = np.linspace(0, 1, steps + 1)
    del_ssim = np.empty(steps + 1, dtype=np.float32)
    ins_ssim = np.empty(steps + 1, dtype=np.float32)
    for i, frac in enumerate(fractions):
        n = int(round(frac * n_pix))
        del_pred01 = model(mask_delete(inp, ys, xs, n)[tf.newaxis], training=True)[0] * 0.5 + 0.5
        del_ssim[i] = float(tf.image.ssim(del_pred01, clean_01, max_val=1))
        ins_pred01 = model(mask_insert(inp, ys, xs, n)[tf.newaxis], training=True)[0] * 0.5 + 0.5
        ins_ssim[i] = float(tf.image.ssim(ins_pred01, clean_01, max_val=1))
    return float(np.trapz(del_ssim, fractions)), float(np.trapz(ins_ssim, fractions))


def explanation_sensitivity(attribution_fn, model, test_input, base_map,
                             n_noise=N_NOISE_FULL, noise_std=NOISE_STD):
    base_flat = base_map.reshape(-1)
    corrs = []
    for _ in range(n_noise):
        noisy_input = tf.clip_by_value(
            test_input + tf.random.normal(tf.shape(test_input), stddev=noise_std), -1.0, 1.0)
        noisy_map, _ = attribution_fn(model, noisy_input)
        corrs.append(np.corrcoef(base_flat, noisy_map.reshape(-1))[0, 1])
    return float(np.mean(corrs))


def explanation_entropy(attribution):
    p = attribution.reshape(-1).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log2(p)).sum())


done_pairs = set()
if csv_path.exists():
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            done_pairs.add((int(row['image_idx']), row['method']))
    print(f'resuming: {len(done_pairs)} (image, method) results already saved in {csv_path}', flush=True)

fieldnames = ['image_idx', 'method', 'deletion_auc', 'insertion_auc', 'sensitivity', 'entropy']
write_header = not csv_path.exists()

with open(csv_path, 'a', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fh.flush()

    with dropout_disabled(generator):
        for img_idx, (test_input, ground_truth) in enumerate(val_dataset):
            needed = [name for name in XAI_METHODS_FULL if (img_idx, name) not in done_pairs]
            if not needed:
                continue

            clean_pred = generator(test_input, training=True)[0]
            clean_01 = clean_pred * 0.5 + 0.5

            for name in needed:
                fn = XAI_METHODS_FULL[name]
                attribution, _ = fn(generator, test_input)

                deletion_auc, insertion_auc = deletion_insertion_auc(
                    generator, test_input, attribution, clean_01)
                sensitivity = explanation_sensitivity(fn, generator, test_input, attribution)
                entropy = explanation_entropy(attribution)

                writer.writerow({
                    'image_idx': img_idx, 'method': name,
                    'deletion_auc': deletion_auc, 'insertion_auc': insertion_auc,
                    'sensitivity': sensitivity, 'entropy': entropy,
                })
                fh.flush()

            if (img_idx + 1) % 50 == 0:
                print(f'processed {img_idx + 1} images...', flush=True)

print(f'done -> results saved to {csv_path}', flush=True)
