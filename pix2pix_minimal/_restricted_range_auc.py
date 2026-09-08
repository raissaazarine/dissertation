"""Recomputes deletion/insertion AUC restricted to the 0-20% masking-fraction
range, alongside the existing full 0-100% range, for the 3 fixed samples used
in xai_evaluation.py's curve visualisation (pix2pix.ipynb). Tests whether the
0.008-0.016 between-method gap in the full-range AUC is mostly determined by
the 20-100% regime (where deletion SSIM was observed to recover, U-shaped,
rather than continue declining) instead of the meaningful 0-20% decline.
"""
import tensorflow as tf
import tensorflow_io as tfio
import pathlib
import numpy as np
from contextlib import contextmanager

from pix2pix import Generator, INPUT_CHANNELS

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
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
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    return (input_image / 127.5) - 1, (real_image / 127.5) - 1


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    input_image = input_image[:, :, :INPUT_CHANNELS]
    return input_image, real_image


val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.tif'), shuffle=True, seed=42)
val_dataset = val_dataset.map(load_image_test).batch(1)

generator = Generator()
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
status.expect_partial()
print(f'restored checkpoint: {tf.train.latest_checkpoint(checkpoint_dir)}', flush=True)

N_SAMPLES = 3
fixed_samples = [next(iter(val_dataset)) for _ in range(N_SAMPLES)]


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


def smoothgrad(model, input_image, n_samples=5, noise_std=0.1):
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


XAI_METHODS_FULL = {
    'Saliency': saliency_map,
    'Grad x Input': grad_times_input,
    'SmoothGrad': smoothgrad,
}

N_STEPS_FULL = 10
FILL_VALUE = -1.0


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


def deletion_insertion_curve(model, test_input, attribution, clean_01, steps=N_STEPS_FULL):
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
    return fractions, del_ssim, ins_ssim


import csv
out_path = pathlib.Path('./xai_evaluation/restricted_range_auc_3sample.csv')
fieldnames = ['sample', 'method', 'del_auc_full', 'del_auc_20', 'ins_auc_full', 'ins_auc_20']
fh = open(out_path, 'w', newline='')
writer = csv.DictWriter(fh, fieldnames=fieldnames)
writer.writeheader()
fh.flush()

rows = []
with dropout_disabled(generator):
    for idx, (test_input, ground_truth) in enumerate(fixed_samples):
        clean_pred = generator(test_input, training=True)[0]
        clean_01 = clean_pred * 0.5 + 0.5

        for name, fn in XAI_METHODS_FULL.items():
            attribution, _ = fn(generator, test_input)
            fractions, del_ssim, ins_ssim = deletion_insertion_curve(generator, test_input, attribution, clean_01)

            del_auc_full = float(np.trapz(del_ssim, fractions))
            ins_auc_full = float(np.trapz(ins_ssim, fractions))
            del_auc_20 = float(np.trapz(del_ssim[0:3], fractions[0:3])) / 0.2
            ins_auc_20 = float(np.trapz(ins_ssim[0:3], fractions[0:3])) / 0.2

            print(f'sample {idx + 1} | {name:13s} | '
                  f'del_full={del_auc_full:.4f} del_0-20={del_auc_20:.4f} | '
                  f'ins_full={ins_auc_full:.4f} ins_0-20={ins_auc_20:.4f} | '
                  f'del_ssim@0,10,20,50,100%={del_ssim[0]:.3f},{del_ssim[1]:.3f},{del_ssim[2]:.3f},{del_ssim[5]:.3f},{del_ssim[10]:.3f}',
                  flush=True)
            row = dict(sample=idx + 1, method=name,
                       del_auc_full=del_auc_full, del_auc_20=del_auc_20,
                       ins_auc_full=ins_auc_full, ins_auc_20=ins_auc_20)
            rows.append(row)
            writer.writerow(row)
            fh.flush()

fh.close()
print('saved ->', out_path, flush=True)

print()
print('=== per-method mean across 3 samples ===')
for name in XAI_METHODS_FULL:
    sub = [r for r in rows if r['method'] == name]
    del_full = np.mean([r['del_auc_full'] for r in sub])
    del_20 = np.mean([r['del_auc_20'] for r in sub])
    ins_full = np.mean([r['ins_auc_full'] for r in sub])
    ins_20 = np.mean([r['ins_auc_20'] for r in sub])
    print(f'{name:13s} | del_full={del_full:.4f} del_0-20(normalized)={del_20:.4f} | '
          f'ins_full={ins_full:.4f} ins_0-20(normalized)={ins_20:.4f}')
