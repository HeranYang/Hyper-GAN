from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def latentEncodeNet(input_code, options, reuse=False, name="latentEncodeNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        hc1 = dense1d(input_code, options.gf_dim, name="ls_h1_dense1d")
        # h1 is self.df_dim, for c1 in encoder.
        hc2 = dense1d(hc1, options.gf_dim*2, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for c2 in encoder.
        hc3 = dense1d(hc2, options.gf_dim*4, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for c3 in encoder.
        hr11 = dense1d(hc3, options.gf_dim*4, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for r11 in encoder.
        hr12 = dense1d(hr11, options.gf_dim*4, name="ls_h5_dense1d")
        # h5 is self.df_dim * 4, for r12 in encoder.
        hr21 = dense1d(hr12, options.gf_dim*4, name="ls_h6_dense1d")
        # h6 is self.df_dim * 4, for r21 in encoder.
        hr22 = dense1d(hr21, options.gf_dim*4, name="ls_h7_dense1d")
        # h7 is self.df_dim * 4, for r22 in encoder.
        hr31 = dense1d(hr22, options.gf_dim*4, name="ls_h8_dense1d")
        # h8 is self.df_dim * 4, for r31 in encoder.
        hr32 = dense1d(hr31, options.gf_dim*4, name="ls_h9_dense1d")
        # h9 is self.df_dim * 4, for r32 in encoder.
        hr41 = dense1d(hr32, options.gf_dim*4, name="ls_h10_dense1d")
        # h10 is self.df_dim * 4, for r41 in encoder.
        hr42 = dense1d(hr41, options.gf_dim*4, name="ls_h11_dense1d")
        # h11 is self.df_dim * 4, for r41 in encoder.
        hr51 = dense1d(hr42, options.gf_dim*4, name="ls_h12_dense1d")
        # h12 is self.df_dim * 4, for r41 in encoder.
        hr52 = dense1d(hr51, options.gf_dim*4, name="ls_h13_dense1d")
        # h13 is self.df_dim * 4, for r41 in encoder.
        return {'hc1':hc1, 'hc2':hc2, 'hc3':hc3,
                'hr11':hr11, 'hr12':hr12, 
                'hr21':hr21, 'hr22':hr22,
                'hr31':hr31, 'hr32':hr32, 
                'hr41':hr41, 'hr42':hr42,
                'hr51':hr51, 'hr52':hr52}
                

def latentDecodeNet(input_code, options, reuse=False, name="latentDecodeNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        hd2 = dense1d(input_code, options.gf_dim, name="ls_h1_dense1d")
        # h1 is self.df_dim, for d2 in encoder.
        hd1 = dense1d(hd2, options.gf_dim*2, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for d1 in encoder.
        hr92 = dense1d(hd1, options.gf_dim*4, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for r92 in encoder.
        hr91 = dense1d(hr92, options.gf_dim*4, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for r91 in encoder.
        hr82 = dense1d(hr91, options.gf_dim*4, name="ls_h5_dense1d")
        # h5 is self.df_dim * 4, for r82 in encoder.
        hr81 = dense1d(hr82, options.gf_dim*4, name="ls_h6_dense1d")
        # h6 is self.df_dim * 4, for r81 in encoder.
        hr72 = dense1d(hr81, options.gf_dim*4, name="ls_h7_dense1d")
        # h7 is self.df_dim * 4, for r72 in encoder.
        hr71 = dense1d(hr72, options.gf_dim*4, name="ls_h8_dense1d")
        # h8 is self.df_dim * 4, for r71 in encoder.
        hr62 = dense1d(hr71, options.gf_dim*4, name="ls_h9_dense1d")
        # h9 is self.df_dim * 4, for r62 in encoder.
        hr61 = dense1d(hr62, options.gf_dim*4, name="ls_h10_dense1d")
        # h10 is self.df_dim * 4, for r61 in encoder.
        return {'hd2':hd2, 'hd1':hd1,
                'hr91':hr91, 'hr92':hr92, 
                'hr81':hr81, 'hr82':hr82,
                'hr71':hr71, 'hr72':hr72, 
                'hr61':hr61, 'hr62':hr62}
        

def define_D(image, domain_code, options, reuse=False, name="discriminator"):
    
    plex_netD = [discriminator(image, options, reuse, name+str(domain_id)) for domain_id in range(options.n_domains)]
    
    dA = tf.constant(0)
    dB = tf.constant(1)
    dC = tf.constant(2)
    dD = tf.constant(3)
    
    fA = lambda: plex_netD[0]
    fB = lambda: plex_netD[1]
    fC = lambda: plex_netD[2]
    fD = lambda: plex_netD[3]
    
    netD = tf.case({tf.equal(domain_code[0,0], dA): fA, 
                    tf.equal(domain_code[0,0], dB): fB, 
                    tf.equal(domain_code[0,0], dC): fC}, 
                   default = fD, exclusive=True)
    
    return netD


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def encoder_resnet(image, scale, offset, options, reuse=False, name="encoder"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, scale1, offset1, scale2, offset2, dim, ks=3, s=1, name='encoder_res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = condition_instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), scale1, offset1, name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = condition_instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), scale2, offset2, name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(condition_instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='encoder_e1_c'), scale['hc1'], offset['hc1'], 'encoder_e1_bn'))
        c2 = tf.nn.relu(condition_instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='encoder_e2_c'), scale['hc2'], offset['hc2'], 'encoder_e2_bn'))
        c3 = tf.nn.relu(condition_instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='encoder_e3_c'), scale['hc3'], offset['hc3'], 'encoder_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, scale['hr11'], offset['hr11'], scale['hr12'], offset['hr12'], options.gf_dim*4, name='encoder_r1')
        r2 = residule_block(r1, scale['hr21'], offset['hr21'], scale['hr22'], offset['hr22'], options.gf_dim*4, name='encoder_r2')
        r3 = residule_block(r2, scale['hr31'], offset['hr31'], scale['hr32'], offset['hr32'], options.gf_dim*4, name='encoder_r3')
        r4 = residule_block(r3, scale['hr41'], offset['hr41'], scale['hr42'], offset['hr42'], options.gf_dim*4, name='encoder_r4')
        r5 = residule_block(r4, scale['hr51'], offset['hr51'], scale['hr52'], offset['hr52'], options.gf_dim*4, name='encoder_r5')
        
        return r5


def decoder_resnet(r5, scale, offset, options, reuse=False, name="decoder"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, scale1, offset1, scale2, offset2, dim, ks=3, s=1, name='decoder_res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = condition_instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), scale1, offset1, name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = condition_instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), scale2, offset2, name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        r6 = residule_block(r5, scale['hr61'], offset['hr61'], scale['hr62'], offset['hr62'], options.gf_dim*4, name='decoder_r5')
        r7 = residule_block(r6, scale['hr71'], offset['hr71'], scale['hr72'], offset['hr72'], options.gf_dim*4, name='decoder_r4')
        r8 = residule_block(r7, scale['hr81'], offset['hr81'], scale['hr82'], offset['hr82'], options.gf_dim*4, name='decoder_r3')
        r9 = residule_block(r8, scale['hr91'], offset['hr91'], scale['hr92'], offset['hr92'], options.gf_dim*4, name='decoder_r2')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='decoder_d1_dc')
        d1 = tf.nn.relu(condition_instance_norm(d1, scale['hd1'], offset['hd1'], 'decoder_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='decoder_d2_dc')
        d2 = tf.nn.relu(condition_instance_norm(d2, scale['hd2'], offset['hd2'], 'decoder_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='decoder_pred_c'))

        return pred



def classifer(r5, options, reuse=False, name="classifer"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        
        h0 = flip_gradient(r5, l=1.0, name='c_h0_fgrad')
        # h0 is (64 x 48 x self.df_dim*4)
        h1 = lrelu(conv2d(h0, options.df_dim*2, ks=1, s=1, name='c_h1_conv'))
        # h1 is (64 x 48 x self.df_dim*2)
        h2 = lrelu(conv2d(h1, options.df_dim, ks=1, s=1, name='c_h2_conv'))
        # h2 is (64 x 48 x self.df_dim)
        h3 = lrelu(conv2d(h2, options.df_dim/4, ks=1, s=1, name='c_h3_conv'))
        # h3 is (64 x 48 x self.df_dim/4)
        h4 = lrelu(conv2d(h3, options.n_domains, ks=1, s=1, name='c_h4_conv'))
        # h4 is (64 x 48 x 3)
        return h4
        

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def prod_input_code(n_domains, DA):
    input_code = np.zeros(n_domains)
    input_code[DA] = 1.
    return input_code
