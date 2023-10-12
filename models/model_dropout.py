#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class test_DropoutLayer(Layer):
    """
        The :class:`DropoutLayer` class is a dropout layer which randomly set some
        values to zero by a given keeping probability.
        
        Parameters
        ----------
        layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
        keep : float
        The keeping probability, the lower more values will be set to zero.
        is_fix : boolean
        Default False, if True, the keeping probability is fixed and cannot be changed via feed_dict.
        is_train : boolean
        If False, skip this layer, default is True.
        seed : int or None
        An integer or None to create random seed.
        name : a string or None
        An optional name to attach to this layer.
        
        Examples
        --------
        - Define network
        >>> network = tl.layers.InputLayer(x, name='input_layer')
        >>> network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
        >>> network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
        >>> ...
        
        - For training, enable dropout as follow.
        >>> feed_dict = {x: X_train_a, y_: y_train_a}
        >>> feed_dict.update( network.all_drop )     # enable dropout layers
        >>> sess.run(train_op, feed_dict=feed_dict)
        >>> ...
        
        - For testing, disable dropout as follow.
        >>> dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable dropout layers
        >>> feed_dict = {x: X_val_a, y_: y_val_a}
        >>> feed_dict.update(dp_dict)
        >>> err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        >>> ...
        
        Notes
        -------
        - A frequent question regarding :class:`DropoutLayer` is that why it donot have `is_train` like :class:`BatchNormLayer`.
        In many simple cases, user may find it is better to use one inference instead of two inferences for training and testing seperately, :class:`DropoutLayer`
        allows you to control the dropout rate via `feed_dict`. However, you can fix the keeping probability by setting `is_fix` to True.
        """
    def __init__(
                 self,
                 layer = None,
                 keep = 0.9,
                 is_fix = True,
                 is_train = True,
                 seed = None,
                 name = 'dropout_layer',
                 ):
        Layer.__init__(self, name=name)
        if is_train is False:
            print("  [TL] skip DropoutLayer")
            self.outputs = layer.outputs
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
        else:
            self.inputs = layer.outputs
            print("[TL] DropoutLayer %s: keep:%f is_fix:%s" % (self.name, keep, is_fix))
            
            # The name of placeholder for keep_prob is the same with the name
            # of the Layer.
            if is_fix:
                self.outputs = tf.nn.dropout(self.inputs, keep, seed=seed, name=name)
                print('keep:',keep)
            
            else:
                set_keep[name] = tf.placeholder(tf.float32)
                self.outputs = tf.nn.dropout(self.inputs, set_keep[name], seed=seed, name=name) # 1.2
            
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
            if is_fix is False:
                self.all_drop.update({set_keep[name]: keep})
                print(set_keep[name])
            self.all_layers.extend([self.outputs])

def SRGAN_g(t_image,keep_prob,variance,std,is_variance=False, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        # during inference
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        # B residual blacks end
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        n = test_DropoutLayer(n, keep = keep_prob)
        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        
        return n

def SRGAN_g0(t_image,keep_prob,variance,std,is_variance=False, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
        feature maps (n) and stride (s) feature maps (n) and stride (s)
        """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        
        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn
    
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end
        
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        n = test_DropoutLayer(n, keep = keep_prob)
        n = DropoutLayer(n, keep=0.9, name='vgg_out/drop1')
        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def SRGAN_g2(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)

    96x96 --> 384x384

    Use Resize Conv
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    size = t_image.get_shape().as_list()

    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        n = UpSampling2dLayer(n, size=[size[1]*2, size[2]*2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
        n = Conv2d(n, 64, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')   # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu,
                is_train=is_train, gamma_init=g_init, name='up1/batch_norm')

        n = UpSampling2dLayer(n, size=[size[1]*4, size[2]*4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
        n = Conv2d(n, 32, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')     # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu,
                is_train=is_train, gamma_init=g_init, name='up2/batch_norm')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d2(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits
def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
                padding='SAME', W_init=w_init, name='h0/c')
            
        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train,
                gamma_init=gamma_init, name='h7/bn')
                        
        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train,
                gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer(layer=[net_h7, net],
                combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
                        
        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity,
                W_init = w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv