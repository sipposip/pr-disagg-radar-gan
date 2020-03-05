"""
minimal functionality necessary for using the trained generator.


"""

import numpy as np
import tensorflow as tf
from pylab import plt
from matplotlib.colors import LogNorm
from tensorflow.keras import backend as K

norm_scale = 127.4
generator_file = f'trained_models/gen_20090101-20161231-tp_thresh_daily5_n_thresh20_ndomain16_stride16_0020.h5'


# we need the custom layer PixelNormalization to load the generator
class PixelNormalization(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = K.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = K.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


# load the trained generator network
gen = tf.keras.models.load_model(generator_file, compile=False,
                                 custom_objects={'PixelNormalization': PixelNormalization,
                                                 'tf':tf})

latent_dim = gen.inputs[0].shape[1]
# in order to use the model, we need to compile it, and specify a loss functio (which wont be used)
gen.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))


def generate_scenarios(cond, n_scenarios):
    # the generator takes normalized daily sums, so we have to divide by norm_scale
    cond = cond / norm_scale
    # for each cond, make several predictions with different latent noise
    latent = np.random.normal(size=(n_scenarios, latent_dim))
    # for efficiency reason, we dont make a single forecast with the network, but
    # we batch all n_fake_per_real together
    cond_batch = np.repeat(cond[np.newaxis], repeats=n_scenarios, axis=0)
    generated = gen.predict([latent, cond_batch])
    # remove empty channel dimension
    generated = generated.squeeze()
    # this now contains daily fractions. convert to mm/h
    generated_precip = generated * cond.squeeze() * norm_scale
    return generated_precip


def plot_scenarios(scenarios):
    nrows = len(scenarios)
    fig = plt.figure(figsize=(24, nrows))
    n_plot = nrows
    plt.axis('off')
    # plot fake samples
    for iplot in range(nrows):
        for jplot in range(24):
            ax = plt.subplot(n_plot, 24, iplot * 24 + jplot + 1)
            if iplot == 0:
                ax.annotate(f'{jplot:02d}'':00', xy=(0.5, 1), xytext=(0, 5),
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')
            im = plt.imshow(scenarios[iplot, jplot - 1, :, :], cmap=plt.cm.gist_earth_r,
                            norm=LogNorm(vmin=0.01, vmax=50))
            plt.axis('off')
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.007, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('fraction of daily precipitation', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    return fig



