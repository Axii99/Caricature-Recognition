import io
import os
import yaml
import pickle
import argparse
import numpy as np
import tensorflow as tf
import cv2

from scipy import misc

from model import get_embd
from eval.utils import calculate_roc, calculate_tar, distance, calculate_accuracy, get_best_threshold, calculate_tar_far


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='/data/hhd/InsightFace-tensorflow/output/20190116-130753/checkpoints/ckpt-m-116000', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')

    return parser.parse_args()


def load_bin(path, image_size):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        img = misc.imread(io.BytesIO(bin))
        img = misc.imresize(img, [image_size, image_size])
        # cv2.imshow('a',img)
        # cv2.waitKey(0)
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images[cnt] = img
        images_f[cnt] = img_f
        cnt += 1
    print('done!')
    return (images, images_f, issame_list)

def find_wrong(path, image_size):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        img = misc.imread(io.BytesIO(bin))
        img = misc.imresize(img, [image_size, image_size])
        # cv2.imshow('a',img)
        # cv2.waitKey(0)
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        # img = img/127.5-1.0
        # img_f = img_f/127.5-1.0
        images[cnt] = img
        images_f[cnt] = img_f
        cnt += 1
    print('done!')
    list = [True, True, True, True, False, False, True, True, True, True, True, True,
            True, False, False, False, False, False, False, False, False, True, False, False,
            True, False, True, False, False, False, True, True, False, False, True, False,
            False, False, False, False, False, True, False, False, True, True, False, False,
            False, True, False, False, False, True, False, False, False, True, False, False,
            False, False, False, False, False, True, False, True, True, True, False, False,
            False, False, False, True, False, False, False, True, True, False, False, False,
            False, False, False, False, True, True, True, False, False, True, False, True,
            False, False, True, True, True, True, False, False, True, False, True, False,
            False, False, True, False, True, True, False, False, True, True, False, False,
            False, True, True, True, True, True, False, True, False, False, False, True,
            False, False, True, True, True, False, True]
    for i in range(len(list)):
        if list[i] == True:
            cv2.imwrite("D:\WorkSpace\WebCaricature\wrong/" +str(i) +"a.jpg",images[2*i])
            cv2.imwrite("D:\WorkSpace\WebCaricature\wrong/" + str(i) + "b.jpg", images[2*i+1])
    return (images, images_f, issame_list)

def evaluate(embeddings, actual_issame, far_target=1e-3, distance_metric=0, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    if distance_metric == 1:
        thresholds = np.arange(0, 1, 0.0025)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_threshold = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), distance_metric=distance_metric, nrof_folds=nrof_folds)
    tar, tar_std, far = calculate_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), far_target=far_target, distance_metric=distance_metric, nrof_folds=nrof_folds)
    dist_array = distance(embeddings1, embeddings2, distance_metric)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return tpr, fpr, acc_mean, acc_std, tar, tar_std, far,best_threshold,dist_array

def run_embds(sess, images, batch_size, image_size, train_mode, embds_ph, image_ph, train_ph_dropout, train_ph_bn):
    if train_mode >= 1:
        train = True
    else:
        train = False
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)
        print('%d/%d' % (i, batch_num), end='\r')
    if left > 0:
        image_batch = np.zeros([batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = images[-left:]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)[:left]
    print()
    print('done!')
    return np.array(embds)


if __name__ == '__main__':
    args = get_args()
    adapted_sum = 0
    ens_dist = {}
    ef_dist = {} #eyes + faces
    mf_dist = {}  # mouths + faces
    if args.mode == 'build':
        print('building...')
        config = yaml.load(open(args.config_path))
        images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
        train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
        train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
        embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
        print('done!')
        print('============================================')
        print(config['eye_model'])
        print(config['mouth_model'])
        print(config['face_model'])

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        ens_threshold = []

        with tf.Session(config=tf_config) as sess:
            if not config['eye_model'] == '':
                tf.global_variables_initializer().run()
                print('loading...')
                saver = tf.train.Saver()
                saver.restore(sess, config['eye_model'])
                print(config['eye_model'])
                print('done!')

                batch_size = config['batch_size']
                # batch_size = 32
                print('evaluating...')
                val_data = {}
                if args.val_data == '':
                    val_data = config['val_data1']
                else:
                    val_data[os.path.basename(args.val_data)] = args.val_data
                for k, v in val_data.items():
                    imgs, imgs_f, issame = load_bin(v, config['image_size'])
                    print('forward running...')
                    embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds, images,
                                          train_phase_dropout, train_phase_bn)
                    embds_f_arr = run_embds(sess, imgs_f, batch_size, config['image_size'], args.train_mode, embds, images,
                                            train_phase_dropout, train_phase_bn)
                    embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
                        embds_f_arr, axis=1, keepdims=True)
                    print('done!')
                    tpr, fpr, acc_mean, acc_std, tar, tar_std, far, bthd, dist_array = evaluate(embds_arr, issame, far_target=args.target_far,
                                                                              distance_metric=1)
                    print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (k, acc_mean, acc_std, tar, tar_std, far))
                    ens_threshold.append(bthd)
                    ens_dist = dist_array
                    ens_dist_a = dist_array
                    ens_dist_b = dist_array
                    adapted_sum +=acc_mean
                    ef_dist = dist_array
                    print(ens_dist)
                    print(ens_threshold)
                    # find_wrong(v, config['image_size'])
                print('done!')
                ################################################################
                if not config['mouth_model'] == '':
                    tf.global_variables_initializer().run()
                    print('loading...')
                    saver = tf.train.Saver()
                    saver.restore(sess, config['mouth_model'])
                    print(config['mouth_model'])
                    print('done!')

                    batch_size = config['batch_size']
                    # batch_size = 32
                    print('evaluating...')
                    val_data = {}
                    if args.val_data == '':
                        val_data = config['val_data2']
                    else:
                        val_data[os.path.basename(args.val_data)] = args.val_data
                    for k, v in val_data.items():
                        imgs, imgs_f, issame = load_bin(v, config['image_size'])
                        print('forward running...')
                        embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds, images,
                                              train_phase_dropout, train_phase_bn)
                        embds_f_arr = run_embds(sess, imgs_f, batch_size, config['image_size'], args.train_mode, embds, images,
                                                train_phase_dropout, train_phase_bn)
                        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
                            embds_f_arr, axis=1, keepdims=True)
                        print('done!')
                        tpr, fpr, acc_mean, acc_std, tar, tar_std, far, bthd, dist_array = evaluate(embds_arr, issame,
                                                                                                    far_target=args.target_far,
                                                                                                    distance_metric=1)
                        print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (
                        k, acc_mean, acc_std, tar, tar_std, far))
                        ens_threshold.append(bthd)
                        ens_dist = ens_dist*0.5 + dist_array*0.25
                        ens_dist_a = ens_dist_a*0.25 + dist_array*0.5
                        ens_dist_b = ens_dist_a*0.25 + dist_array*0.25
                        adapted_sum += acc_mean
                        mf_dist = dist_array
                        print("ens_dist: ")
                        print(ens_dist)
                        bestth = np.mean(ens_threshold)
                        print(bestth)
                    print('done!')
                    ####################################################################################
                    if not config['face_model'] == '':
                        tf.global_variables_initializer().run()
                        print('loading...')
                        saver = tf.train.Saver()
                        saver.restore(sess, config['face_model'])
                        print(config['face_model'])
                        print('done!')

                        batch_size = config['batch_size']
                        # batch_size = 32
                        print('evaluating...')
                        val_data = {}
                        if args.val_data == '':
                            val_data = config['val_data3']
                        else:
                            val_data[os.path.basename(args.val_data)] = args.val_data
                        for k, v in val_data.items():
                            imgs, imgs_f, issame = load_bin(v, config['image_size'])
                            print('forward running...')
                            embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds,
                                                  images,
                                                  train_phase_dropout, train_phase_bn)
                            embds_f_arr = run_embds(sess, imgs_f, batch_size, config['image_size'], args.train_mode,
                                                    embds, images,
                                                    train_phase_dropout, train_phase_bn)
                            embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1,
                                                                   keepdims=True) + embds_f_arr / np.linalg.norm(
                                embds_f_arr, axis=1, keepdims=True)
                            print('done!')
                            tpr, fpr, acc_mean, acc_std, tar, tar_std, far, bthd, dist_array = evaluate(embds_arr,
                                                                                                        issame,
                                                                                                        far_target=args.target_far,
                                                                                                        distance_metric=1)
                            print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (
                                k, acc_mean, acc_std, tar, tar_std, far))
                            ens_threshold.append(bthd)
                            ens_dist = ens_dist + dist_array*0.25
                            #adapted
                            ens_dist_a = ens_dist_a + dist_array * 0.25
                            ens_dist_b = ens_dist_a + dist_array * 0.5
                            adapted_sum += acc_mean
                            ens_dist_a = ens_dist_a / adapted_sum
                            ef_dist = ef_dist*0.5 + dist_array * 0.5
                            mf_dist = mf_dist * 0.5 + dist_array * 0.5
                            print("ens_dist: ")
                            print(ens_dist)
                        print('done!')
           # _, _, ens_acc = calculate_accuracy(bestth, ens_dist/3, issame)
            print("ensemble result:")
            thresholds = np.arange(0, 1, 0.0025)
            tpr,fpr, accuracy, best_threshold = get_best_threshold(thresholds, ens_dist,np.asarray(issame),distance_metric=1)
            tpr_a, fpr_a, accuracy_a, best_threshold_a = get_best_threshold(thresholds, ens_dist_a, np.asarray(issame),distance_metric=1)
            tpr_b, fpr_b, accuracy_b, best_threshold_b = get_best_threshold(thresholds, ens_dist_b, np.asarray(issame),
                                                                            distance_metric=1)
            etpr, efpr, ef_accuracy, ef_best_threshold = get_best_threshold(thresholds, ef_dist, np.asarray(issame),
                                                                distance_metric=1)
            mtpr, mfpr, mf_accuracy, mf_best_threshold = get_best_threshold(thresholds, mf_dist, np.asarray(issame),
                                                                      distance_metric=1)

            tar, far = calculate_tar_far(best_threshold, ens_dist,np.asarray(issame))
            tar_a, far_a = calculate_tar_far(best_threshold_a, ens_dist_a, np.asarray(issame))
            tar_b, far_b = calculate_tar_far(best_threshold_b, ens_dist_b, np.asarray(issame))
            acc_mean = np.mean(accuracy)
            acc_std = np.std(accuracy)
            acc_mean_a = np.mean(accuracy_a)
            acc_std_a = np.std(accuracy_a)
            acc_mean_b = np.mean(accuracy_b)
            acc_std_b = np.std(accuracy_b)
            print(best_threshold)
            print('All(2:1:1): acc--%1.5f+-%1.5f tpr---%1.5f, fpr---%1.5f; tar---%1.5f, far---%1.5f'% (acc_mean, acc_std, tpr, fpr, tar, far))
            print(best_threshold_a)
            print('All(1:2:1): acc--%1.5f+-%1.5f tpr---%1.5f, fpr---%1.5f; tar---%1.5f, far---%1.5f' % (acc_mean_a, acc_std_a, tpr_a, fpr_a, tar_a, far_a))
            print(best_threshold_b)
            print('All(1:1:2): acc--%1.5f+-%1.5f tpr---%1.5f, fpr---%1.5f; tar---%1.5f, far---%1.5f' % (acc_mean_b, acc_std_b, tpr_b, fpr_b, tar_b, far_b))
            print(ef_best_threshold)
            print('Eyes+Faces(5:5): acc--%1.5f+-%1.5f tpr---%1.5f, fpr---%1.5f' % (np.mean(ef_accuracy), np.std(ef_accuracy), etpr, efpr))
            print(mf_best_threshold)
            print('Mouths+Faces(5:5): acc--%1.5f+-%1.5f tpr---%1.5f, fpr---%1.5f' % (np.mean(mf_accuracy), np.std(mf_accuracy), mtpr, mfpr))
    else:
        raise ValueError("Invalid value for --mode.")

