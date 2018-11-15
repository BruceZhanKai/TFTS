# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def main(_):
    csvname='hour-post-single-5'
    csv_file_name = './data/'+csvname+'.csv'
    modelname=   "ar"
    batchsize=    32
    windowsize=   36
    preiodicity=  12
    inwindowsize= 18
    outwindowsize=18
    modelallname="./model/"+modelname+"-"+str(batchsize)+"-"+str(windowsize)+"-"+str(preiodicity)+"-"+str(inwindowsize)+"-"+str(outwindowsize)+"-"+csvname
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=batchsize,
                                                               window_size=windowsize)
    with tf.Session() as sess:
        data = reader.read_full()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data = sess.run(data)
        coord.request_stop()

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=preiodicity, input_window_size=inwindowsize, output_window_size=outwindowsize,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,
        #optimizer=tf.train.AdamOptimizer(0.1),
        model_dir=modelallname)
        

    ar.train(input_fn=train_input_fn, steps=6000)

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
    plt.title("%s-%s-%s-%s-%s-%s-%s"%(modelname,batchsize,windowsize,preiodicity,inwindowsize,
                                      outwindowsize,csvname))
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
