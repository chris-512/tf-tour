#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import numpy as np
from numpy import prod

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './mnist_data',
                           """Directory where mnist data is stored.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './mnist_ckpts',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean(
    'use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size.""")
tf.app.flags.DEFINE_integer('num_epochs', 20, """Number of epochs.""")
tf.app.flags.DEFINE_integer('max_steps', 10000, """Number of max steps.""")
tf.app.flags.DEFINE_integer('decay_steps', 100000,
                            """Number of decay steps.""")
tf.app.flags.DEFINE_float(
    'decay_rate', 0.96, """The learning rate decay rate.""")
tf.app.flags.DEFINE_float('initial_lr', 0.001, """The initial learning rate.""")


def inference(images, keep_prob=1.0):
    """Build the MNIST model.
    Args:

    """
    x_input = tf.reshape(images, [-1, 28, 28, 1])

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(
                            0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.005)):
        net = slim.conv2d(x_input, 20, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 50, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = tf.reshape(net, [-1, prod(net.get_shape()[1:])])
        net = slim.fully_connected(net, 500, scope='fc1')
        net = slim.fully_connected(
            net, FLAGS.num_classes, activation_fn=None, scope='fc2')

    return net


def main(_):

    with tf.Graph().as_default() as g:

        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

        print('mnist.train.images shape: ', mnist.train.images.shape)
        print('mnist.train.labels shape: ', mnist.train.labels.shape)

        print('mnist.test.images shape: ', mnist.test.images.shape)
        print('mnist.test.labels shape: ', mnist.test.labels.shape)

        inputs = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.int64, [None])

        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("simple_cnn", reuse=tf.AUTO_REUSE):
            logits = inference(inputs)

        # loss
        # labels = tf.cast(labels, dtype=tf.int64)
        labels = tf.to_int64(labels)
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy')
        cls_loss = tf.reduce_mean(cls_loss, name='xentropy_mean')

        correct_prediction = tf.equal(tf.argmax(logits, axis=1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32)) # out of 100, how many you got correct?

        # learning rate
        lr = tf.train.exponential_decay(
            FLAGS.initial_lr, global_step, FLAGS.decay_steps, FLAGS.decay_rate)

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999)

        # minimize the cross_entropy
        train_op = opt.minimize(cls_loss, global_step=global_step)

        sess = tf.Session()

        tf.global_variables_initializer().run(session=sess)

        for epoch in range(FLAGS.num_epochs):

            print('train epoch-{}...'.format(epoch))

            # Iterating over all training images
            for step in range(FLAGS.max_steps):

                batch_xs, batch_ys = mnist.train.next_batch(batch_size=FLAGS.batch_size)

                start_time = time.time()
                _, loss_value, accuracy_ = sess.run(
                    [train_op, cls_loss, accuracy], feed_dict={inputs: batch_xs, labels: batch_ys})
                duration = time.time() - start_time

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                if step % 50 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration

                    format_str = (
                        '%s: (train) step %d, loss = %.4f, acc = %.4f (%.1f examples/sec; %.3f sec/batch)'
                    )
                    print(format_str % (datetime.now(), step,
                                        loss_value, accuracy_, examples_per_sec, sec_per_batch))

                # Evaluate
                if step != 0 and step % 300 == 0:
                    test_batch_xs, test_batch_ys = mnist.test.next_batch(batch_size=1000)
                    test_loss_value, test_accuracy_ = sess.run(
                    [cls_loss, accuracy], feed_dict={inputs: test_batch_xs, labels: test_batch_ys})

                    format_str = (
                        '%s: (test) step %d, loss = %.4f, acc = %.4f'
                    )
                    print(format_str % (datetime.now(), step,
                                        test_loss_value, test_accuracy_))


if __name__ == '__main__':
    tf.app.run()
