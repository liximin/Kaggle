import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 224
train_batch_size = 1

def convert_to_tfrecord(rootpath, target_record_dir, target_record_filename):
    target_tfrecord_file = target_record_dir+"/"+target_record_filename+".tfrecords"
    os.makedirs(target_record_dir, mode=0o775, exist_ok=True)
    if os.path.isfile(target_tfrecord_file) is True:
        print("the "+target_tfrecord_file+" exist, no need to run the  convert_to_tfrecord")
        return
    writer = tf.python_io.TFRecordWriter(target_tfrecord_file)
    filenames = os.listdir(rootpath)
    for name in filenames:
        print("Processing image:" + name)
        img = Image.open(rootpath+"/"+name)
        label = name.rstrip(".jpg")
        
        if img.mode == "RGB":
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=
                                              tf.train.Int64List(value=train_labels[train_labels["id"]==label]['breed'].ravel().reshape(-1)[0])), 
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
            writer.write(example.SerializeToString())
    writer.close()

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding="SAME")
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        # add L2 loss
        weight_loss = tf.multiply(tf.nn.l2_loss(kernel), 0.001, name="weight_loss")
        weight_loss = tf.reduce_mean(weight_loss)
        tf.add_to_collection('losses', weight_loss)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

def fc_op(input_op, name, n_out, p, wl2=0.004, relu_tag=True):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        if relu_tag is True:
            biases = tf.Variable(tf.constant(0.001, shape=[n_out], dtype=tf.float32), name="b")
            activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        else:
            biases = tf.Variable(tf.constant(0.001, shape=[n_out], dtype=tf.float32), name="b")
            activation = tf.matmul(input_op, kernel) + biases
        # add L2 loss
        weight_loss = tf.multiply(tf.nn.l2_loss(kernel), wl2, name="weight_loss")
        weight_loss = tf.reduce_mean(weight_loss)
        tf.add_to_collection('losses', weight_loss)
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding="SAME",
                          name=name)

def inference_op(input_op, keep_prob):
    p = []
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2_dropout = tf.nn.dropout(conv1_2, 1.0, name="conv1_2_dropout")
    pool1 = mpool_op(conv1_2_dropout, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2_dropout = tf.nn.dropout(conv2_2, 1.0, name="conv2_2_dropout")
    pool2 = mpool_op(conv2_2_dropout, name="pool2", kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3_dropout = tf.nn.dropout(conv3_3, 1.0, name="conv3_3_dropout")
    pool3 = mpool_op(conv3_3_dropout, name="pool3", kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3_dropout = tf.nn.dropout(conv4_3, 1.0, name="conv4_3_dropout")
    pool4 = mpool_op(conv4_3_dropout, name="pool4", kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3_dropout = tf.nn.dropout(conv5_3, 1.0, name="conv5_3_dropout")
    pool5 = mpool_op(conv5_3_dropout, name="pool5", kh=2, kw=2, dw=2, dh=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p, wl2=0.004)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p, wl2=0.004)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")


    fc8 = fc_op(fc7_drop, name="fc8", n_out=120, p=p)
    softmax = tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax, 1)
    return prediction, softmax, fc8, p


def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def read_tfrecord(filenames, batch_num):
    read_tfrecord_dataset = tf.contrib.data.TFRecordDataset(filenames)
    read_tfrecord_dataset = read_tfrecord_dataset.map(_parse_function)
    read_tfrecord_dataset = read_tfrecord_dataset.repeat(10)
    read_tfrecord_dataset = read_tfrecord_dataset.batch(batch_num)
    return read_tfrecord_dataset

def save_model(sess, variable_list, checkpointfile):
    saverdict = {}
    for item in variable_list:
        saverdict.update({item.name[:-2]: item})
    saver = tf.train.Saver(saverdict)
    save_path = saver.save(sess, checkpointfile)
    print("model saved in the "+save_path)
def _parse_function(filenames):
    features = {"img_raw": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature(1, tf.int64)}
    parsed_features = tf.parse_single_example(filenames, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    return image, parsed_features["label"]

if __name__ == '__main__':
    train_labels = pd.read_csv("./data/labels.csv")
    labels_str   = pd.read_csv("./data/sample_submission.csv").drop("id", axis = 1).columns.tolist()
    labels_str_lenth=len(labels_str)
    labels_map={}
    for index,labels_index in enumerate(labels_str):
        labels_map[labels_index]=[index]
    train_labels['breed']=train_labels['breed'].map(labels_map)

    print( train_labels[train_labels["id"]=='fa2a33c1dc8b39ad51738408b289a0de']['breed'].ravel().reshape(-1)[0] )
    print( train_labels[train_labels["id"]=='fa2a33c1dc8b39ad51738408b289a0de']['breed'].ravel().reshape(-1)[0][0])
    convert_to_tfrecord("./data/train", "./process_workspace", "TrainSet")
    dataset = read_tfrecord(["./process_workspace/TrainSet.tfrecords"], train_batch_size)
    keep_prod = tf.placeholder(tf.float32)
    image_holder = tf.placeholder(tf.float32, [train_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_holder = tf.placeholder(tf.int64, [train_batch_size])
    testimage_holder = tf.placeholder(tf.float32, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    # softmax output use
    prediction, softmax, fc8, p = inference_op(image_holder, keep_prod)
    loss = loss(fc8, label_holder)
    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
    with tf.Session() as sess:
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        saverdict = {}
        for item in p:
            saverdict.update({item.name[:-2]: item})
        saver = tf.train.Saver(saverdict)
        if os.path.isfile("./process_workspace/DogBreedIdentificationVgg16_model.ckpt.meta") is True:
            print("Found checkpoint files , start to training from restored data")
            saver.restore(sess, "./process_workspace/DogBreedIdentificationVgg16_model.ckpt")
            print(p[0], sess.run(p[-1]))
            print("Model restored.\n")
        Step_Continue = 0
        step = -1
                                                                                                                                                                                                                                                                                                        # end in 800
        for step in range(10):
            if step < Step_Continue:
                sess.run(next_element)
                continue
            print("########## train  %d step(s) start: #######" % step, end="\n")
            try:
                preprocess_batch_img = sess.run(next_element)
                train_softmax, _,  result_loss = sess.run([softmax,train_op, loss], feed_dict={
                    image_holder: preprocess_batch_img[0].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)/255 - 0.5,
                    label_holder: preprocess_batch_img[1].reshape(-1),
                    keep_prod: 0.5
                })
                print("result_loss:")
                print(result_loss)
                print("train_softmax:")
                print(train_softmax)
            except tf.errors.OutOfRangeError:
                save_model(sess, p, "./process_workspace/DogBreedIdentificationVgg16_model.ckpt")
                break
            if step % 5 == 0:
                save_model(sess, p, "./process_workspace/DogBreedIdentificationVgg16_model.ckpt")
        save_model(sess, p, "./process_workspace/CatsDogsVgg16_model.ckpt")
        print("########## train end at  %d  range: #######" % (step+1), end="\n")
        print(sess.run(p[-1]))

 
