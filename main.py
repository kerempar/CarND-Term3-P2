import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from   moviepy.editor import VideoFileClip
import scipy
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    
    print("load_vgg...")
    
    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # Apply skip-layers and upsampling
    
    # The pretrained VGG-16 model is already fully convolutional,
    # containing all the convolutions that replace the fully connected layers and retain spatial information.
    # We must add 1x1 convolutions on top of the VGG to reduce the number of filters
    # from 4096 to the number of classes for our specific model
    
    # we have a binary classification, road or not road, so number of classes is 2.
    # kernel size is 1
    # we use a regularizer to prevent weights from becoming too large
    # and it will be error prone to overfitting and produce garbage
    
    # Adding skip connections to the model.
    # We will combine the output of two layers.
    # The first output is the output of the current layer.
    # The second output is the output of a layer further back in the network
    
    print("layers...")
    
    # 1x1 convolution of vgg layer 7
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                       padding= 'same',
                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # upsample (de-convolution, transpose convolution)
    layer7_upsampled = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4,
                                                  strides= (2, 2),
                                                  padding= 'same',
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    tf.Print(layer7_upsampled, [tf.shape(layer7_upsampled)[1:3]])
    # 1x1 convolution of vgg layer 4
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                       padding= 'same',
                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # skip connection (element-wise addition)
    layer4_output = tf.add(layer7_upsampled, layer4_conv_1x1)

    # upsample (de-convolution, transpose convolution)
    layer4_upsampled = tf.layers.conv2d_transpose(layer4_output, num_classes, 4,
                                                  strides= (2, 2),
                                                  padding= 'same',
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 3
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                       padding= 'same',
                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # skip connection (element-wise addition)
    layer3_output = tf.add(layer4_upsampled, layer3_conv_1x1)

    # upsample (to original image size)
    nn_last_layer = tf.layers.conv2d_transpose(layer3_output, num_classes, 16,
                                            strides= (8, 8),
                                            padding= 'same',
                                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    print("optimize...")
    
    # logits is a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    # loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # training operation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())
    
    print("training...")
    print()
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            print("Loss: {:.4f}".format(loss))
        print()
tests.test_train_nn(train_nn)

def process_image(img):
    #print("process image...")
    img = scipy.misc.imresize(img, image_shape)
    
    #inference
    im_softmax = session.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [img]})
    
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(img)
    street_im.paste(mask, box=None, mask=mask)
    
    return np.array(street_im)

session = None
logits = None
image_shape = (160, 576)
keep_prob = None
image_pl = None

def run():
    num_classes = 2
    data_dir = './data'
    runs_dir = './runs'
    
    global session
    global logits
    global image_shape
    global keep_prob
    global image_pl
    
    print("Run starting...")
    
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        num_epochs = 70
        batch_size = 5
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
    
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        print("Applying trained model to video...")
        # update global variables before calling video functions
        session = sess
        logits = logits
        keep_prob = keep_prob
        image_pl = input_image
        
        video_input1 = VideoFileClip('solidWhiteRight.mp4')
        video_output1 = 'solidWhiteRight_video_output.mp4'
        processed_video = video_input1.fl_image(process_image)
        processed_video.write_videofile(video_output1, audio=False)
    
        video_input1 = VideoFileClip('advanced_lane_lines_video.mp4')
        video_output1 = 'advanced_lane_lines_video_output.mp4'
        processed_video = video_input1.fl_image(process_image)
        processed_video.write_videofile(video_output1, audio=False)

    print("Run ending...")

if __name__ == '__main__':
    run()
