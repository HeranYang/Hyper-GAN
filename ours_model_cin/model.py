from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from collections import namedtuple
import numpy as np

from module import *
from utils import *
from PIL import Image
from skimage import transform
import imageio

import random



class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        
        self.batch_size = args.batch_size
        self.image_size0 = args.fine_size0
        self.image_size1 = args.fine_size1
        self.code_size0 = 1
        self.code_size1 = 1
        self.code_size2 = args.n_domains
        
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        
        self.n_domains = args.n_domains
        
        self.L1_lambda = args.L1_lambda
        self.L2_lambda = args.L2_lambda
        self.L3_lambda = args.L3_lambda
        self.L4_lambda = args.L4_lambda
        
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.define_D = define_D
        self.encoder_resnet = encoder_resnet
        self.decoder_resnet = decoder_resnet
        self.latentEncodeNet = latentEncodeNet
        self.latentDecodeNet = latentDecodeNet
        self.classifer = classifer
        
        
        self.max_update_num = args.max_update_num
        
        
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size0 image_size1 n_domains \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size0, args.fine_size1, args.n_domains,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.pool = ImagePool(args.max_size)



    def _build_model(self):
        
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size0, self.image_size1,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.input_code = tf.placeholder(tf.float32,
                                        [self.batch_size, self.code_size0, self.code_size1,
                                         self.code_size2 + self.code_size2],
                                        name='A_and_B_code')
        self.domain_code = tf.placeholder(tf.int32,
                                        [self.batch_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='A_and_B_domain_code')
        
        # input image in two domains.
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        # one hot code for distinguishing input domain, e.g., [1,0,0], [0,1,0], etc.
        self.real_codeA = self.input_code[:, :, :, :self.code_size2]
        self.real_codeB = self.input_code[:, :, :, self.code_size2:self.code_size2 + self.code_size2]
        # a number for denoting input domain, e.g., 1, 2, etc.
        self.DA = self.domain_code[:, :self.input_c_dim]
        self.DB = self.domain_code[:, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        
        # a fully connected layer for modifying the filters in generators based on input code.
        self.latEnScale_A = self.latentEncodeNet(self.real_codeA, self.options, False, name="generator_latEnScaleNet")
        self.latEnOffset_A = self.latentEncodeNet(self.real_codeA, self.options, False, name="generator_latEnOffsetNet")
        
        self.latEnScale_B = self.latentEncodeNet(self.real_codeB, self.options, True, name="generator_latEnScaleNet")
        self.latEnOffset_B = self.latentEncodeNet(self.real_codeB, self.options, True, name="generator_latEnOffsetNet")
        
        self.latDeScale_A = self.latentDecodeNet(self.real_codeA, self.options, False, name="generator_latDeScaleNet")
        self.latDeOffset_A = self.latentDecodeNet(self.real_codeA, self.options, False, name="generator_latDeOffsetNet")
        
        self.latDeScale_B = self.latentDecodeNet(self.real_codeB, self.options, True, name="generator_latDeScaleNet")
        self.latDeOffset_B = self.latentDecodeNet(self.real_codeB, self.options, True, name="generator_latDeOffsetNet")
        
        # cycle of "real_A to fake_B to recon_A"
        self.encode_A = self.encoder_resnet(self.real_A, self.latEnScale_A, self.latEnOffset_A, self.options, reuse=False, name="generator_encoder")
        self.fake_B = self.decoder_resnet(self.encode_A, self.latDeScale_B, self.latDeOffset_B, self.options, reuse=False, name="generator_decoder")
        self.encode_fakeB = self.encoder_resnet(self.fake_B, self.latEnScale_B, self.latEnOffset_B, self.options, reuse=True, name="generator_encoder")
        self.fake_A_ = self.decoder_resnet(self.encode_fakeB, self.latDeScale_A, self.latDeOffset_A, self.options, reuse=True, name="generator_decoder")
        
        # cycle of "real_B to fake_A to recon_B"
        self.encode_B = self.encoder_resnet(self.real_B, self.latEnScale_B, self.latEnOffset_B, self.options, reuse=True, name="generator_encoder")
        self.fake_A = self.decoder_resnet(self.encode_B, self.latDeScale_A, self.latDeOffset_A, self.options, reuse=True, name="generator_decoder")
        self.encode_fakeA = self.encoder_resnet(self.fake_A, self.latEnScale_A, self.latEnOffset_A, self.options, reuse=True, name="generator_encoder")
        self.fake_B_ = self.decoder_resnet(self.encode_fakeA, self.latDeScale_B, self.latDeOffset_B, self.options, reuse=True, name="generator_decoder")
        
        # discriminator.
        self.DB_fake = self.define_D(self.fake_B, self.DB, self.options, reuse=False, name="discriminator")
        self.DA_fake = self.define_D(self.fake_A, self.DA, self.options, reuse=True, name="discriminator")
        
        # classifier.
        self.LabA = self.classifer(self.encode_A, self.options, reuse=False, name="generator_classifier")
        self.LabB = self.classifer(self.encode_B, self.options, reuse=True, name="generator_classifier")
        
        # discriminator for encode.
        self.LabfakeB = self.classifer(self.encode_fakeB, self.options, reuse=True, name="generator_classifier")
        self.LabfakeA = self.classifer(self.encode_fakeA, self.options, reuse=True, name="generator_classifier")
        
	    # self reconstruction.
        self.recon_A = self.decoder_resnet(self.encode_A, self.latDeScale_A, self.latDeOffset_A, self.options, reuse=True, name="generator_decoder")
        self.recon_B = self.decoder_resnet(self.encode_B, self.latDeScale_B, self.latDeOffset_B, self.options, reuse=True, name="generator_decoder")
        

        # define generator loss.
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_ads = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))
        self.g_loss_cycle = abs_criterion(self.real_A, self.fake_A_) \
            + abs_criterion(self.real_B, self.fake_B_)

        self.g_loss_class = sce_criterion(self.LabA, tf.multiply(tf.ones_like(self.LabA), self.real_codeA)) \
            + sce_criterion(self.LabB, tf.multiply(tf.ones_like(self.LabB), self.real_codeB))
        self.g_loss_class_fake = sce_criterion(self.LabfakeA, tf.multiply(tf.ones_like(self.LabfakeA), self.real_codeA)) \
            + sce_criterion(self.LabfakeB, tf.multiply(tf.ones_like(self.LabfakeB), self.real_codeB))

        self.g_loss_common = abs_criterion(self.encode_A, self.encode_fakeB) \
            + abs_criterion(self.encode_B, self.encode_fakeA)

        self.g_loss_recon = abs_criterion(self.real_A, self.recon_A) \
            + abs_criterion(self.real_B, self.recon_B)

        self.g_loss = self.g_loss_ads \
                        + self.L1_lambda * self.g_loss_cycle \
                        + self.L2_lambda * self.g_loss_class \
                        + self.L2_lambda * self.g_loss_class_fake \
                        + self.L3_lambda * self.g_loss_common \
                        + self.L4_lambda * self.g_loss_recon


        # define parameter for discriminator.
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size0, self.image_size1,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size0, self.image_size1,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.define_D(self.real_B, self.DB, self.options, reuse=True, name="discriminator")
        self.DA_real = self.define_D(self.real_A, self.DA, self.options, reuse=True, name="discriminator")
        self.DB_fake_sample = self.define_D(self.fake_B_sample, self.DB, self.options, reuse=True, name="discriminator")
        self.DA_fake_sample = self.define_D(self.fake_A_sample, self.DA, self.options, reuse=True, name="discriminator")

        # define discriminator loss.
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss


        # define loss for tensorboard.
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_ads_sum = tf.summary.scalar("g_loss_ads", self.g_loss_ads)
        self.g_loss_cycle_sum = tf.summary.scalar("g_loss_cycle", self.g_loss_cycle)
        self.g_loss_class_sum = tf.summary.scalar("g_loss_class", self.g_loss_class)
        self.g_loss_class_fake_sum = tf.summary.scalar("g_loss_class_fake", self.g_loss_class_fake)
        self.g_loss_common_sum = tf.summary.scalar("g_loss_common", self.g_loss_common)
        self.g_loss_recon_sum = tf.summary.scalar("g_loss_recon", self.g_loss_recon)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge(
            [self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_ads_sum, self.g_loss_cycle_sum,
             self.g_loss_class_sum, self.g_loss_class_fake_sum, self.g_loss_common_sum, 
             self.g_loss_recon_sum, self.g_loss_sum])
        
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )


        # test code.
        self.test_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size0, self.image_size1,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size0, self.image_size1,
                                      self.output_c_dim], name='test_B')
        self.test_codeA = tf.placeholder(tf.float32,
                                        [self.batch_size, self.code_size0, self.code_size1,
                                         self.code_size2], name='test_A_code')
        self.test_codeB = tf.placeholder(tf.float32,
                                        [self.batch_size, self.code_size0, self.code_size1,
                                         self.code_size2], name='test_B_code')
        
        self.latEnScale_test_A = self.latentEncodeNet(self.test_codeA, self.options, True, name="generator_latEnScaleNet")
        self.latEnOffset_test_A = self.latentEncodeNet(self.test_codeA, self.options, True, name="generator_latEnOffsetNet")
        
        self.latEnScale_test_B = self.latentEncodeNet(self.test_codeB, self.options, True, name="generator_latEnScaleNet")
        self.latEnOffset_test_B = self.latentEncodeNet(self.test_codeB, self.options, True, name="generator_latEnOffsetNet")
        
        self.latDeScale_test_A = self.latentDecodeNet(self.test_codeA, self.options, True, name="generator_latDeScaleNet")
        self.latDeOffset_test_A = self.latentDecodeNet(self.test_codeA, self.options, True, name="generator_latDeOffsetNet")
        
        self.latDeScale_test_B = self.latentDecodeNet(self.test_codeB, self.options, True, name="generator_latDeScaleNet")
        self.latDeOffset_test_B = self.latentDecodeNet(self.test_codeB, self.options, True, name="generator_latDeOffsetNet")
        
        
        self.encode_test_A = self.encoder_resnet(self.test_A, 
                                                 self.latEnScale_test_A, self.latEnOffset_test_A, 
                                                 self.options, reuse=True, 
                                                 name="generator_encoder")
        self.fake_test_B = self.decoder_resnet(self.encode_test_A, 
                                               self.latDeScale_test_B, self.latDeOffset_test_B, 
                                               self.options, reuse=True, 
                                               name="generator_decoder")
        self.encode_test_B = self.encoder_resnet(self.test_B, 
                                                 self.latEnScale_test_B, self.latEnOffset_test_B, 
                                                 self.options, reuse=True, 
                                                 name="generator_encoder")
        self.fake_test_A = self.decoder_resnet(self.encode_test_B,  
                                               self.latDeScale_test_A, self.latDeOffset_test_A, 
                                               self.options, reuse=True, 
                                               name="generator_decoder")


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)


    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        
        
        # # ==============================================================
        # # ONLY used for continue training.
        # if args.continue_train:
        #     self.load_valid(r'./r1/checkpoint/', 80)
        #     print(" [*] Load SUCCESS")
        # # ==============================================================
        

        saver_all = tf.train.Saver(max_to_keep = None)

        for epoch in range(args.epoch):
            
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)
            
            data_domA = glob('../data/BraTS-Dataset/SlicedData/Train/TrainA/*.npy')
            data_domB = glob('../data/BraTS-Dataset/SlicedData/Train/TrainB/*.npy')
            data_domC = glob('../data/BraTS-Dataset/SlicedData/Train/TrainC/*.npy')
            data_domD = glob('../data/BraTS-Dataset/SlicedData/Train/TrainD/*.npy')
            
            np.random.shuffle(data_domA)
            np.random.shuffle(data_domB)
            np.random.shuffle(data_domC)
            np.random.shuffle(data_domD)
            
            dataList = [data_domA, data_domB, data_domC, data_domD]
            
            dataNumList = [len(data_domA), len(data_domB), len(data_domC), len(data_domD)]

            batch_idxs = self.max_update_num // self.batch_size

            for idx in range(0, batch_idxs):

                # domain_code
                DA, DB = random.sample(range(self.n_domains), 2)
                domain_code_tmp = np.array(np.dstack((DA, DB))).astype(np.float32)
                domain_code = domain_code_tmp.reshape([self.batch_size, self.input_c_dim + self.output_c_dim])
                
                # input_code
                code_domA = prod_input_code(self.n_domains, DA)
                code_domB = prod_input_code(self.n_domains, DB)
                input_code_tmp = np.array(np.hstack((code_domA, code_domB))).astype(np.float32)
                input_code = input_code_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                                     self.code_size2 + self.code_size2])

                dataA_idxs = dataList[DA][idx*self.batch_size : (idx+1)*self.batch_size]
                dataB_idxs = dataList[DB][idx*self.batch_size : (idx+1)*self.batch_size]
                
                batch_files = list(zip(dataA_idxs, dataB_idxs))

                # real_image.
                batch_images = [load_train_data(batch_file, args.load_size0, args.load_size1, 
                                                args.fine_size0, args.fine_size1) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                
                
                
                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, 
                               self.input_code: input_code, 
                               self.domain_code: domain_code, 
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.domain_code: domain_code,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, DA, DB)

            save_epoch_file = os.path.join(args.checkpoint_dir, 'MRCT_epoch')
            if not os.path.exists(save_epoch_file):
                os.makedirs(save_epoch_file)
            if np.mod(epoch, args.save_freq) == 0:
                saver_all.save(self.sess, os.path.join(save_epoch_file, 'cyclegans.epoch'), global_step=epoch)




    def load_valid(self, checkpoint_dir, epoch):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_epoch" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt_name = os.path.basename('cyclegans.epoch-{}'.format(epoch))
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))



    def sample_model(self, sample_dir, epoch,  idx, DA, DB):

        data_domA = glob('../data/BraTS-Dataset/SlicedData/Train/TrainA/*.npy')
        data_domB = glob('../data/BraTS-Dataset/SlicedData/Train/TrainB/*.npy')
        data_domC = glob('../data/BraTS-Dataset/SlicedData/Train/TrainC/*.npy')
        data_domD = glob('../data/BraTS-Dataset/SlicedData/Train/TrainD/*.npy')
        
        np.random.shuffle(data_domA)
        np.random.shuffle(data_domB)
        np.random.shuffle(data_domC)
        np.random.shuffle(data_domD)
        
        dataList = [data_domA, data_domB, data_domC, data_domD]
        
        # domain_code
        domain_code_tmp = np.array(np.dstack((DA, DB))).astype(np.float32)
        domain_code = domain_code_tmp.reshape([self.batch_size, self.input_c_dim + self.output_c_dim])
        
        # input_code
        code_domA = prod_input_code(self.n_domains, DA)
        code_domB = prod_input_code(self.n_domains, DB)
        input_code_tmp = np.array(np.hstack((code_domA, code_domB))).astype(np.float32)
        input_code = input_code_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                             self.code_size2 + self.code_size2])


        dataA_idxs = dataList[DA][0 : self.batch_size]
        dataB_idxs = dataList[DB][0 : self.batch_size]
        
        batch_files = list(zip(dataA_idxs, dataB_idxs))
        sample_image = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_image).astype(np.float32)
        
        
        fake_A, fake_B, fake_A_, fake_B_ = self.sess.run(
            [self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
            feed_dict={self.real_data: sample_images, 
                       self.input_code: input_code, 
                       self.domain_code: domain_code}
        )

        inputB = np.load(dataB_idxs[0])
        syn_A = np.array(fake_A).reshape([self.image_size0, self.image_size1])
        recon_B = np.array(fake_B_).reshape([self.image_size0, self.image_size1])

        inputA = np.load(dataA_idxs[0])
        syn_B = np.array(fake_B).reshape([self.image_size0, self.image_size1])
        recon_A = np.array(fake_A_).reshape([self.image_size0, self.image_size1])
        
        images_B2A = np.concatenate([inputB, syn_A, recon_B], axis = 1)
        images_A2B = np.concatenate([inputA, syn_B, recon_A], axis = 1)
        images_show = np.concatenate([images_B2A, images_A2B], axis = 0)
        
        imageio.imwrite('./{}/Epoch{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx), (images_show + 1.) * 127.5)


    def valid(self, args):
        """valid cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        
        image_path = r'../data/BraTS-Dataset/VolumeData/Valid/Valid{}/Dom{}{:03d}.nii.gz'
        
        valNum = 5
        valVec = np.arange(valNum) + 1
        
        epochNum = 61
        epochVec = np.arange(epochNum) * 2 + 0

        dataInfoFile = open(r'../data/BraTS-dataInfo.txt', 'r')
        sourceInLines = dataInfoFile.readlines()
        dataInfoFile.close()
        dataInfo = []
        for line in sourceInLines:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')
            dataInfo.append(temp2)


        dom_name = ['A', 'B', 'C', 'D']
        inputDomVec = np.arange(len(dom_name))
        
        
        for epoch in epochVec:
        
            valid_dir = './valid'
            if not os.path.exists(valid_dir):
                os.makedirs(valid_dir)

            valid_checkpoint = './checkpoint'

            self.load_valid(valid_checkpoint, epoch)

            
            for input_domain_id in inputDomVec:
                
                input_domain_name = dom_name[input_domain_id]
                output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)
                
                for output_domain_id in output_domain_list:
                    
                    output_domain_name = dom_name[output_domain_id]
                    
                    if output_domain_id==0:
                        output_range = 3000
                    elif output_domain_id==1:
                        output_range = 5000
                    elif output_domain_id==2:
                        output_range = 6000
                    else:
                        output_range = 7000
                    
                    # input_code
                    code_domA = prod_input_code(self.n_domains, input_domain_id)
                    code_domB = prod_input_code(self.n_domains, output_domain_id)
                    input_code_domA_tmp = np.array(code_domA).astype(np.float32)
                    input_code_domA = input_code_domA_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                                          self.code_size2])
                    input_code_domB_tmp = np.array(code_domB).astype(np.float32)
                    input_code_domB = input_code_domB_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                                          self.code_size2])
                    
                    namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)
                    
                    for val_id in valVec:
                        
                        input_image_path = image_path.format(input_domain_name, input_domain_name, val_id)
                        input_imageAll = nib.load(input_image_path)
                        gt_image_path = image_path.format(output_domain_name, output_domain_name, val_id)
                        gt_imageAll = nib.load(gt_image_path)
        
                        sliceNum = int(dataInfo[val_id - 1 + 100][3])
                        sliceVec = np.arange(sliceNum)
                        
                        teResults = np.zeros(input_imageAll.shape, dtype=np.int16)
                        inputImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                        gtImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                        
                        sample_vol = [load_test_data(input_image_path, input_domain_id)]
                        sample_vol = np.array(sample_vol).astype(np.float32)
                        
                        for iSlicet in sliceVec:
        
                            iSlice = iSlicet + int(dataInfo[val_id - 1 + 100][4]) - 1
                            print('Processing image: id ' + str(val_id) + 'slice' + str(iSlicet))
                            
                            sample_image = sample_vol[:, int(iSlice), :, :]
                            sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])
        
                            if epoch == epochVec[0]:
                                
                                gt_image = gt_imageAll.get_data()[int(iSlice), :, :].astype('int16')
                                gtImage[int(iSlice), :, :] = np.array(gt_image).astype('int16')
        
                                input_image = input_imageAll.get_data()[int(iSlice), :, :].astype('int16')
                                inputImage[int(iSlice), :, :] = np.array(input_image).astype('int16')
        
                            fake_img = self.sess.run(self.fake_test_B, 
                                                     feed_dict={self.test_A: sample_image, 
                                                                self.test_codeA: input_code_domA, 
                                                                self.test_codeB: input_code_domB})
                            
                            temp = (fake_img + 1.) / 2. * output_range
                            teResults[int(iSlice), :, :] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
        
                        head_output = input_imageAll.get_header()
                        affine_output = input_imageAll.affine
                        
                        saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
                        nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, val_id, epoch))
        
                        if epoch == epochVec[0]:
                            
                            gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
                            gt_path = os.path.join(valid_dir, '{}'.format(os.path.basename(gt_image_path)))
                            nib.save(gtResults, gt_path)
        
                            inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
                            input_path = os.path.join(valid_dir, '{}'.format(os.path.basename(input_image_path)))
                            nib.save(inputResults, input_path)


    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        
        image_path = r'../data/BraTS-Dataset/VolumeData/Test/Test{}/Dom{}{:03d}.nii.gz'

        epoch = 94
        test_checkpoint = './checkpoint'
        self.load_valid(test_checkpoint, epoch)

        test_base = './test'
        if not os.path.exists(test_base):
            os.makedirs(test_base)

        test_dir = '{}/epoch_{}'.format(test_base, epoch)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        

        tedataSize = 45
        teIdVec = np.arange(tedataSize) + 1

        dataInfoFile = open(r'../data/BraTS-dataInfo.txt', 'r')
        sourceInLines = dataInfoFile.readlines()
        dataInfoFile.close()
        dataInfo = []
        for line in sourceInLines:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')
            dataInfo.append(temp2)
        
        
        dom_name = ['A', 'B', 'C', 'D']
        inputDomVec = np.arange(len(dom_name))


        for teId in teIdVec:
            
            sliceNum = int(dataInfo[teId - 1 + 105][3])
            sliceVec = np.arange(sliceNum)
            
            
            for input_domain_id in inputDomVec:
                
                input_domain_name = dom_name[input_domain_id]
                output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)
                
                
                for output_domain_id in output_domain_list:
                    
                    output_domain_name = dom_name[output_domain_id]
                    
                    if output_domain_id==0:
                        output_range = 3000
                    elif output_domain_id==1:
                        output_range = 5000
                    elif output_domain_id==2:
                        output_range = 6000
                    else:
                        output_range = 7000
                    
                    # input_code
                    code_domA = prod_input_code(self.n_domains, input_domain_id)
                    code_domB = prod_input_code(self.n_domains, output_domain_id)
                    input_code_domA_tmp = np.array(code_domA).astype(np.float32)
                    input_code_domA = input_code_domA_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                                          self.code_size2])
                    input_code_domB_tmp = np.array(code_domB).astype(np.float32)
                    input_code_domB = input_code_domB_tmp.reshape([self.batch_size, self.code_size0, self.code_size1, 
                                                          self.code_size2])
                    
                    namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)
                    
                    
                    gt_image_path = image_path.format(output_domain_name, output_domain_name, teId)
                    gt_imageAll = nib.load(gt_image_path)

                    input_image_path = image_path.format(input_domain_name, input_domain_name, teId)
                    input_imageAll = nib.load(input_image_path)
                    
                    sample_vol = [load_test_data(input_image_path, input_domain_id)]
                    sample_vol = np.array(sample_vol).astype(np.float32)
                    
                    teResults = np.zeros(input_imageAll.shape, dtype=np.int16)
                    inputImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                    gtImage = np.zeros(input_imageAll.shape, dtype=np.int16)
        
                    for iSlicet in sliceVec:
        
                        iSlice = iSlicet + int(dataInfo[teId - 1 + 105][4]) - 1
                        print('Processing image: id ' + str(teId) + 'slice' + str(iSlicet))
                        
                        sample_image = sample_vol[:, int(iSlice), :, :]
                        sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])
                        
                        gt_image = gt_imageAll.get_data()[int(iSlice), :, :].astype('int16')
                        gtImage[int(iSlice), :, :] = np.array(gt_image).astype('int16')
        
                        input_image = input_imageAll.get_data()[int(iSlice), :, :].astype('int16')
                        inputImage[int(iSlice), :, :] = np.array(input_image).astype('int16')
        
        
                        fake_img = self.sess.run(self.fake_test_B, 
                                                 feed_dict={self.test_A: sample_image, 
                                                            self.test_codeA: input_code_domA, 
                                                            self.test_codeB: input_code_domB})
                            
                        temp = (fake_img + 1.) / 2. * output_range
                        teResults[int(iSlice), :, :] = np.array(temp).astype('int16').reshape([args.fine_size0,args.fine_size1])

        
                    head_output = input_imageAll.get_header()
                    affine_output = input_imageAll.affine
                    saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
                    nib.save(saveResults, '{}/{}_{:0>2d}.nii.gz'.format(test_dir, namehd, teId))
        
                    gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
                    gt_path = os.path.join(test_dir, '{}'.format(os.path.basename(gt_image_path)))
                    nib.save(gtResults, gt_path)
        
                    inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
                    input_path = os.path.join(test_dir, '{}'.format(os.path.basename(input_image_path)))
                    nib.save(inputResults, input_path)
