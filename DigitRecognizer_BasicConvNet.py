import numpy as np
import tensorflow as tf
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa 

REMOTEDEBUG = True # set to True if debugging remotely
REMOTEDEBUG = False
if (REMOTEDEBUG == True):
   # ssh -R 5678:localhost:5678 kbhit@192.168.1.149 -p 49
   import pydevd
   from pydevd_file_utils import setup_client_server_paths
   MY_PATHS_FROM_ECLIPSE_TO_PYTHON = [
       ('/Users/pal004/Documents/LiClipse Workspace/RemoteSystemsTempFiles/192.168.1.149/home/kbhit/git/DigitRecognizer', '/home/kbhit/git/DigitRecognizer'),
       ]
   setup_client_server_paths(MY_PATHS_FROM_ECLIPSE_TO_PYTHON)
   pydevd.settrace("localhost", port=5678, stdoutToServer=True, stderrToServer=True)
   
PERCENT_TRAIN = 0.8 # the other part will be validation
train_fn = 'data/train.csv'
test_fn  = 'data/test.csv'

pixel_depth = 255.0  # Number of levels per pixel.
image_size = 28  #224 before
num_labels = 10
num_channels = 1 # greyscale

# some knobs to turn
LEARNING_RATE = 0.0001 
num_steps = 1050*30 # lets run for 30 epochs.   n=1050 steps per epoch at 33600 images with n=32 batchsize
batch_size = 32
num_hidden = 64 # number of neurons in FC layer (right before softmax layer)
data_augmentation_prob = 0.05 # the probability of each data augmention filter

def normalize_image (image_data): # convert values between -.5 and .5
   for i in range (image_data.shape[0]):
      image_data[i,] = (image_data[i,].astype(np.float32) - (pixel_depth / 2)) / pixel_depth
      
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

# read in training and validation set
train_matrix = pd.read_csv(train_fn, delimiter=",")
indexes = np.arange (train_matrix.shape[0])
np.random.shuffle (indexes)
num_validation = int (train_matrix.shape[0]-(train_matrix.shape[0]*PERCENT_TRAIN))

train_set = train_matrix.iloc [indexes[:-num_validation]]
valid_set = train_matrix.iloc [indexes[-num_validation:]]

train_labels  = train_set['label'].as_matrix ().astype (np.int32)
valid_labels  = valid_set['label'].as_matrix ().astype (np.int32)
train_dataset = train_set[np.arange(1,train_set.shape[1])].as_matrix ().astype(np.float32).reshape (-1, image_size, image_size, num_channels)
valid_dataset = valid_set[np.arange(1,valid_set.shape[1])].as_matrix ().astype(np.float32).reshape (-1, image_size, image_size, num_channels)

# read in test set
test_matrix = pd.read_csv(test_fn, delimiter=",")
test_dataset = test_matrix.as_matrix ().astype(np.float32).reshape (-1, image_size, image_size, num_channels) # no labels and no need to randomize


normalize_image (valid_dataset);
normalize_image (test_dataset);

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, ))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
 
  # variables 
  kernel_conv1 = tf.Variable(tf.truncated_normal([3, 3, num_channels, 32], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv1')
  biases_conv1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                        trainable=True, name='biases_conv1')
  kernel_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv2')
  biases_conv2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                        trainable=True, name='biases_conv2')
  kernel_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv3')
  biases_conv3 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                        trainable=True, name='biases_conv3')
  fc1w = tf.Variable(tf.truncated_normal([1024, num_hidden], 
                                                dtype=tf.float32,
                                                stddev=1e-1), name='weights') # 1024 from pool3.get_shape () of 4*4*64
  fc1b = tf.Variable(tf.constant(1.0, shape=[num_hidden], dtype=tf.float32),
                        trainable=True, name='biases')
  fc2w = tf.Variable(tf.truncated_normal([num_hidden, num_labels],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='weights')
  fc2b = tf.Variable(tf.constant(1.0, shape=[num_labels], dtype=tf.float32),
                        trainable=True, name='biases')
 
  
  def model(data):
     parameters = []
     with tf.name_scope('conv1_1') as scope:
         conv = tf.nn.conv2d(data, kernel_conv1, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv1)
         conv1_1 = tf.nn.relu(out, name=scope)
         parameters += [kernel_conv1, biases_conv1]
         
     # pool1
     pool1 = tf.nn.max_pool(conv1_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool1')
     
     with tf.name_scope('conv2_1') as scope:
         conv = tf.nn.conv2d(pool1, kernel_conv2, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv2)
         conv2_1 = tf.nn.relu(out, name=scope)
         parameters += [kernel_conv2, biases_conv2]
         
     # pool2
     pool2 = tf.nn.max_pool(conv2_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')
     
     with tf.name_scope('conv3_1') as scope:
         conv = tf.nn.conv2d(pool2, kernel_conv3, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv3)
         conv3_1 = tf.nn.relu(out, name=scope)
         parameters += [kernel_conv3, biases_conv3]
         
     # pool3
     pool3 = tf.nn.max_pool(conv3_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')
         
     # fc1
     with tf.name_scope('fc1') as scope:
         shape = int(np.prod(pool3.get_shape()[1:])) # except for batch size (the first one), multiple the dimensions
         pool3_flat = tf.reshape(pool3, [-1, shape])
         fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
         fc1 = tf.nn.relu(fc1l)
         parameters += [fc1w, fc1b]

     # fc2
     with tf.name_scope('fcnum_hidden2') as scope:
         fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
         parameters += [fc2w, fc2b]
     return fc2l;
     
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
  
  
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(data_augmentation_prob, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
#        iaa.Fliplr(0.5), # horizontally flip 50% of all images
#        iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.03))), # crop images by 0-3% of their height/width
        st(iaa.GaussianBlur((0, 0.5))), # blur images with a sigma between 0 and 0.5
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
        st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-8, 8), "y": (-8, 8)}, # translate by -8 to +8 pixels (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-8, 8), # shear by -8 to +8 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            cval=(0, 1.0), # if mode is constant, use a cval between 0 and 1.0
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.20)) # apply elastic transformations with random strengths
    ],
    random_order=True # do all of the above in random order
)

with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run ()
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = np.copy (train_dataset[offset:(offset + batch_size), :, :, :]) # copying so a reference won't be created or else we'll be normalizing normalized stuff which would yield odd results
      batch_data = seq.augment_images(batch_data.astype (np.uint8)).astype (np.float32); # add data augmentation
      normalize_image (batch_data);
    
      batch_labels = train_labels[offset:(offset + batch_size)]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run(
                                      [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 200 == 0):
         print ("offset is %d" % (offset))
         print ("Minibatch loss at step", step, ":", l)
         print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
         print ("Validation accuracy: %.1f%%" % accuracy(
           valid_prediction.eval(), valid_labels))
#  print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
