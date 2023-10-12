#! /usr/bin/python
# -*- coding: utf8 -*-
import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
#from model_noise_last2 import *
from utils import *
from config import config, log_config
import argparse
import pdb

#parser = argparse.ArgumentParser()


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')
    noise_sdv=args.noise
    #net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_g0= SRGAN_g0(t_image,[],noise_sdv,is_variance=False,is_train=True, reuse=False)
    #pdb.set_trace()
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g0.outputs, is_train=True, reuse=True)

    net_g0.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g0.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)
    # print(vgg_predict_emb.outputs)
    # # exit()
    
    ## test inference
    
    net_g_test0 = SRGAN_g0(t_image,[],noise_sdv,is_variance=False,is_train=False, reuse=True)
    net_g_test = SRGAN_g(t_image,[],noise_sdv,is_variance=False,is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')                 # paper 1e-3
    mse_loss = tl.cost.mean_squared_error(net_g0.outputs , t_target_image, is_mean=True)                                 # paper
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)    # simiao
    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g0)
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')
    '''
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
    '''
    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0):
            out0 = sess.run(net_g_test0.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            print("[*] save images")
            tl.vis.save_images(out0, [ni, ni], save_dir_gan+'/train_0_%d_dropout_%s.png' % (epoch,0.1))
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d_dropout_%s.png' % (epoch,0.1))

        ## save model
        if (epoch != 0):
            
            tl.files.save_npz(net_g0.all_params, name=checkpoint_dir+'/g_dropout_%s_srgan.npz' %(0.1), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_noise_%s_srgan.npz' %(0.1), sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = 'samples_layer_%d_dropout_%0.2f_no_train/set_14_evaluate' %(args.layer,1-args.drop_out_keep)
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    #valid_variance_list = sorted(tl.files.load_file_list(path=config.VALID.variance_path, regx='.*.jpg', printable=False))
    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs_o = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs_o = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    #valid_variance_imgs = read_all_imgs(valid_variance_list, path=config.VALID.variance_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()
    
    i=0
    noise_sdv=args.noise
    for imid_i in range(1):
        imid=args.image_id
    #imid=list_imid[imid_i]
    #for imid in range(13,14):
        ###========================== DEFINE MODEL ============================###
        #imid = 10 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        
        for rotate_time in range(1):
            i+=1
            print (rotate_time)
            valid_lr_img = np.rot90(valid_lr_imgs_o[imid],rotate_time)
            valid_hr_img = np.rot90(valid_hr_imgs_o[imid],rotate_time)
            #valid_variance_img = np.rot90(valid_variance_imgs[imid],rotate_time)
            # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
            valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
            # print(valid_lr_img.min(), valid_lr_img.max())
            size = valid_lr_img.shape
            t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
            keep_prob = args.drop_out_keep
            #pdb.set_trace
            if i>1:

                net_g = SRGAN_g(t_image,keep_prob,[],noise_sdv,is_variance=False,is_train=False, reuse=True)
                '''
                _,g_out2 = SRGAN_g(tf.image.rot90(t_image,1),[],is_variance=False,is_train=False, reuse=True)
                _,g_out3 = SRGAN_g(tf.image.rot90(t_image,2),[],is_variance=False,is_train=False, reuse=True)
                _,g_out4 = SRGAN_g(tf.image.rot90(t_image,3),[],is_variance=False,is_train=False, reuse=True)
    
                cat_g_output=tf.stack([g_out1[:,:,:,2],tf.image.rot90(g_out2,3)[:,:,:,2],tf.image.rot90(g_out3,2)[:,:,:,2],tf.image.rot90(g_out4,1)[:,:,:,2]])
                mean, var = tf.nn.moments(cat_g_output, axes=[0])
                var_unsqueeze=tf.expand_dims(var, -1)
                '''
            else:
                
                net_g0 = SRGAN_g0(t_image,[],noise_sdv,is_variance=False,is_train=False, reuse=False)
                net_g = SRGAN_g(t_image,keep_prob,[],noise_sdv,is_variance=False,is_train=False, reuse=True)
                '''
                net_g, g_out1 = SRGAN_g(t_image,[],is_variance=False,is_train=False, reuse=True)
                _,g_out2 = SRGAN_g(tf.image.rot90(t_image,1),[],is_variance=False,is_train=False, reuse=True)
                _,g_out3 = SRGAN_g(tf.image.rot90(t_image,2),[],is_variance=False,is_train=False, reuse=True)
                _,g_out4 = SRGAN_g(tf.image.rot90(t_image,3),[],is_variance=False,is_train=False, reuse=True)
                
                cat_g_output=tf.stack([g_out1[:,:,:,2],tf.image.rot90(g_out2,3)[:,:,:,2],tf.image.rot90(g_out3,2)[:,:,:,2],tf.image.rot90(g_out4,1)[:,:,:,2]])
                mean, var = tf.nn.moments(cat_g_output, axes=[0])
                var_unsqueeze=tf.expand_dims(var, -1)
                '''
            ###========================== RESTORE G =============================###
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            
            tl.layers.initialize_global_variables(sess)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g0)
            ###======================= EVALUATION =============================###
            
            start_time = time.time()
            #pdb.set_trace()
            #drop out sampling
            for si in range(args.sample):
                
                feed_dict={t_image: [valid_lr_img]}
                out = sess.run(net_g.outputs,feed_dict=feed_dict)
                #pdb.set_trace()
                tl.vis.save_image(np.rot90(out[0],4-rotate_time), save_dir+'/valid_gen_id_%02d_r%d_dropout_%0.2f_r%02d.png' % (imid,rotate_time,1-keep_prob,si))
            
            out0 = sess.run(net_g0.outputs, {t_image: [valid_lr_img]})
            tl.vis.save_image(np.rot90(out0[0],4-rotate_time), save_dir+'/valid_gen_id_%02d_r%d_r0.png' % (imid,rotate_time))
            #out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
            print("took: %4.4fs" % (time.time() - start_time))

            #print("LR size: %s /  generated HR size: %s" % (size, out.shape))
            # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
            #print("[*] save images")
            #tl.vis.save_image(np.rot90(out[0],4-rotate_time), save_dir+'/valid_gen_id_%02d_r%d.png' % (imid,rotate_time))
            #tl.vis.save_image(np.rot90(valid_lr_img,4-rotate_time), save_dir+'/valid_lr_id_%02d_r%d.png' % (imid,rotate_time))
            #tl.vis.save_image(np.rot90(valid_hr_img,4-rotate_time), save_dir+'/valid_hr_id_%02d_r%d.png' % (imid,rotate_time))

            #out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
            #tl.vis.save_image(np.rot90(out_bicu,4-rotate_time), save_dir+'/valid_bicubic_id_%02d_r%d.png' % (imid,rotate_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
    parser.add_argument('--noise', type=float, default=0, help='noise')
    parser.add_argument('--layer', type=int, default=0, help='dropout')
    parser.add_argument('--dataset', type=str, default='DIVI2K', help='validation dataset')
    parser.add_argument('--image_id', type=int, default=0, help='image id')
    parser.add_argument('--sample', type=int, default=32, help='sample numbers')
    parser.add_argument('--drop_out_keep', type=float, default=0.5, help='sample numbers')
    args = parser.parse_args()
    
    if args.layer ==4:
        from model_dropout_last4 import *
    elif args.layer ==2:
        from model_dropout_last2 import *
    elif args.layer ==8:
        from model_dropout_last8 import *
    elif args.layer ==1:
        from model_dropout_last1 import *
    elif args.layer ==9:
        from model_dropout_last9 import *
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
