# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def main(_):
    csv_file_name = './data/hour-post-single-3.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=32, window_size=32)
    with tf.Session() as sess:
        data = reader.read_full()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data = sess.run(data)
        coord.request_stop()

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=12, input_window_size=16, output_window_size=16,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)#,
        #optimizer=tf.train.AdamOptimizer(0.01))
        

    ar.train(input_fn=train_input_fn, steps=40000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=24*4)))
    np.savetxt('predict.txt',predictions['mean'])
    #np.savetxt('evaluate.txt',evaluation['mean'])
    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result.jpg')
    
    plt.figure(figsize=(15,5))
    plt.plot(predictions['times'].reshape(-1),predictions['mean'].reshape(-1),label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result-1.jpg')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
