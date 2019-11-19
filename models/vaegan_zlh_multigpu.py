import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate, Dropout
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
K.set_image_dim_ordering('tf')

from .cond_base import CondBaseModel
from .layers import *
from .utils import *
from .nn_blocks import *
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def sample_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(keras.metrics.categorical_crossentropy(c_true, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake_f = keras.metrics.binary_crossentropy(y_neg, y_fake_f)
        return K.mean(loss_real + loss_fake_f)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        loss = self.lossfun(y_real, y_fake_f)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_r, x_f):
        loss_x = 0.5 * K.mean(K.square(x_r - x_f))

        return loss_x

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        loss = self.lossfun(x_r, x_f)
        self.add_loss(loss, inputs=inputs)

        return x_r

class FeatureMatchingLayer(Layer):
    __name__ = 'feature_matching_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(FeatureMatchingLayer, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        loss = 0.5 * K.mean(K.square(f1 - f2))
        return loss

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(loss, inputs=inputs)

        return f1

class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(KLLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = -0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var))
        return kl_loss

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return z_avg


class VAEGAN_zlh_multigpu(CondBaseModel):
    def __init__(self,
        input_shape=(128, 128, 3),
        num_id=15,
        z_dims=4096,
        gpus=8,
        name='vaegan',
        **kwargs
    ):
        super(VAEGAN_zlh_multigpu, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.input_shape = input_shape
        self.num_id = num_id
        self.z_dims = z_dims
        self.i_dims = z_dims
        self.gpus = gpus

        self.f_enc = None
        self.f_dec = None
        self.f_dis = None
        self.f_cls = None
        self.enc_trainer = None
        self.dec_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch, lr):
        x_r, c = x_batch
        batchsize = (len(x_r)//2)
        # print(c.shape)

        x_s = x_r[0:batchsize,:,:,:]
        c_s = c[0:batchsize]

        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        x_dummy = np.zeros(x_s.shape, dtype='float32')
        c_dummy = np.zeros(c_s.shape, dtype='float32')
        z_dummy = np.zeros(z_p.shape, dtype='float32')
        y_dummy = np.zeros((batchsize, 1), dtype='float32')
        f_dummy = np.zeros((batchsize, 4096), dtype='float32')

        # Train I
        K.set_value(self.cls_trainer.optimizer.lr, lr)
        i_loss = self.cls_trainer.train_on_batch([x_s, c_s], c_dummy)

        # Train autoencoder
        K.set_value(self.enc_trainer.optimizer.lr, lr)
        #print(self.enc_trainer.metrics_names)
        e_loss, _, kl_loss = self.enc_trainer.train_on_batch([x_s, x_s], [x_dummy, z_dummy])

        # Train generator
        K.set_value(self.dec_trainer.optimizer.lr, lr)
        #print(self.dec_trainer.metrics_names)
        g_loss, gr_loss, gd_loss, gc_loss = self.dec_trainer.train_on_batch([x_s, x_s], [x_dummy, f_dummy, f_dummy])

        # Train classifier
        K.set_value(self.cls_trainer.optimizer.lr, lr)
        c_loss = self.cls_trainer.train_on_batch([x_s, c_s], c_dummy)

        # Train discriminator
        K.set_value(self.dis_trainer.optimizer.lr, lr)
        d_loss = self.dis_trainer.train_on_batch([x_s, x_s], y_dummy)


        x_a = x_r[batchsize:,:,:,:]
        # Train I
        K.set_value(self.cls_trainer.optimizer.lr, lr)
        i_loss += self.cls_trainer.train_on_batch([x_s, c_s], c_dummy)

        # Train autoencoder
        K.set_value(self.enc_trainer.optimizer.lr, 0.1*lr)
        e1_loss, _, kl1_loss = self.enc_trainer.train_on_batch([x_s, x_a], [x_dummy, z_dummy])

        # Train generator
        K.set_value(self.dec_trainer.optimizer.lr, 0.7*lr)
        g1_loss, gr1_loss, gd1_loss, gc1_loss = self.dec_trainer.train_on_batch([x_s, x_a], [x_dummy, f_dummy, f_dummy])

        # Train classifier
        K.set_value(self.cls_trainer.optimizer.lr, lr)
        c_loss += self.cls_trainer.train_on_batch([x_s, c_s], c_dummy)

        # Train discriminator
        K.set_value(self.dis_trainer.optimizer.lr, lr)
        d_loss += self.dis_trainer.train_on_batch([x_s, x_a], y_dummy)

        loss = {
            'i_loss': i_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'c_loss': c_loss,
            'gr_loss': gr_loss,
            'gd_loss': gd_loss,
            'gc_loss': gc_loss,
            'e_loss': e_loss,
            'g1_loss': g1_loss,
            'gr1_loss': gr1_loss,
            'gd1_loss': gd1_loss,
            'gc1_loss': gc1_loss,
            'e1_loss': e1_loss,
        }
        return loss

    def test(self, datasets, batchsize, epoch, output_path):
        num_data = len(datasets)
        perm = np.random.permutation(num_data)
        index = perm[0:batchsize*2]

        x_test = self.make_batch(datasets, index)
        x, c = x_test
        x_s = x[0:batchsize,:,:,:]
        x_a = x[batchsize:,:,:,:]

        z_params = self.f_enc.predict(x_a)

        z_avg = z_params[:, :self.z_dims]
        z_log_var = z_params[:, self.z_dims:]
        z = z = sample_normal([z_avg, z_log_var])
        z = K.eval(z)

        c_i, x_i = self.f_cls.predict(x_s)
        x_f = self.f_dec.predict([z, x_i])

        fig = plt.figure(figsize=(1, batchsize))
        grid = gridspec.GridSpec(batchsize, 1, wspace=0.1, hspace=0.1)
        imgs = x_s * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        for i in range(batchsize):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(output_path + f"/epoch{epoch}_s.jpg", dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(1, batchsize))
        grid = gridspec.GridSpec(batchsize, 1, wspace=0.1, hspace=0.1)
        imgs = x_a * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        for i in range(batchsize):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(output_path + f"/epoch{epoch}_a.jpg", dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(1, batchsize))
        grid = gridspec.GridSpec(batchsize, 1, wspace=0.1, hspace=0.1)
        imgs = x_f * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        for i in range(batchsize):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(output_path + f"/epoch{epoch}_f.jpg", dpi=200)
        plt.close(fig)

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
        self.f_enc.summary()
        self.f_dec = self.build_decoder()
        self.f_dec.summary()
        self.f_dis = self.build_discriminator()
        self.f_dis.summary()
        self.f_cls = self.build_classifier()
        self.f_cls.summary()

        # Algorithm
        x_s = Input(shape=self.input_shape)
        x_a = Input(shape=self.input_shape)
        c = Input(shape=(self.num_id,))

        z_params = self.f_enc(x_a)

        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)
        z = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])

        kl_loss = KLLossLayer()([z_avg, z_log_var])

        c_i, x_i = self.f_cls(x_s)
        #i_loss = ClassifierLossLayer()([c, c_i])

        x_f = self.f_dec([z, x_i])

        y_r, f_D_x_r = self.f_dis(x_a)
        y_f, f_D_x_f = self.f_dis(x_f)

        d_loss = DiscriminatorLossLayer()([y_r, y_f])

        c_r, f_C_x_r = self.f_cls(x_s)
        c_f, f_C_x_f = self.f_cls(x_f)

        g_loss = GeneratorLossLayer()([x_a, x_f])

        gd_loss = FeatureMatchingLayer()([f_D_x_r, f_D_x_f])
        gc_loss = FeatureMatchingLayer()([f_C_x_r, f_C_x_f])

        c_loss = ClassifierLossLayer()([c, c_r])

        # Build classifier trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, True)

        self.cls_trainer = Model(inputs=[x_s, c],
                                 outputs=[c_loss])
        self.cls_trainer = multi_gpu_model(self.cls_trainer, gpus=self.gpus)
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))
        #self.cls_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, True)
        set_trainable(self.f_cls, False)

        self.dis_trainer = Model(inputs=[x_s, x_a],
                                 outputs=[d_loss])
        self.dis_trainer = multi_gpu_model(self.dis_trainer, gpus=self.gpus)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))
        #self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, True)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.dec_trainer = Model(inputs=[x_s, x_a],
                                 outputs=[g_loss, gd_loss, gc_loss])

        self.dec_trainer = multi_gpu_model(self.dec_trainer, gpus=self.gpus)
        self.dec_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))

         # Build autoencoder
        set_trainable(self.f_enc, True)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.enc_trainer = Model(inputs=[x_s, x_a],
                                outputs=[g_loss, kl_loss])
        self.enc_trainer = multi_gpu_model(self.enc_trainer, gpus=self.gpus)
        self.enc_trainer.compile(loss=[zero_loss, zero_loss],
                                optimizer=Adam(lr=1.0e-4, beta_1=0.5))
        #self.enc_trainer.summary()
        # Store trainers
        self.store_to_save('cls_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('dec_trainer')
        self.store_to_save('enc_trainer')

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)

        use_norm = True
        x = conv_block(x_inputs, 64, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 64, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 128, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 128, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)

        x = Dense(output_dims)(x)
        x = Activation('linear')(x)

        return Model(x_inputs, x)

    def build_decoder(self):
        z_inputs = Input(shape=(self.z_dims,))
        x_inputs = Input(shape=(self.i_dims,))
        z = Concatenate()([x_inputs, z_inputs])

        w = self.input_shape[0] // (2 ** 5)
        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        upscale_block = upscale_nn
        use_norm = True

        x = upscale_block(x, 512, use_norm, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')

        x = upscale_block(x, 512, use_norm, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')

        x = upscale_block(x, 256, use_norm, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')

        x = upscale_block(x, 128, use_norm, norm='batchnorm')
        x = conv_block(x, 128, use_norm, strides=1, norm='batchnorm')

        x = upscale_block(x, 64, use_norm, norm='batchnorm')
        x = conv_block(x, 64, use_norm, strides=1, norm='batchnorm')

        x = Conv2D(3, kernel_size=3, padding='same', activation="tanh")(x)

        return Model([z_inputs, x_inputs], x)

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2), bnorm=True)(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, strides=(2, 2), bnorm=True)(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        f = x

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_classifier(self):
        inputs = Input(shape=self.input_shape)

        use_norm = True
        x = conv_block(inputs, 64, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 64, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 128, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 128, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 256, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = conv_block(x, 512, use_norm, strides=1, norm='batchnorm')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)

        f = x

        x = Dense(self.num_id)(x)
        x = Activation('softmax')(x)

        return Model(inputs, [x, f])
