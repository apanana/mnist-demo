# import tensorflow as tf

# # from tensorflow.examples.tutorials.mnist import input_data
# # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are oh-encoded
# mnist = tf.keras.datasets.mnist

# # n_train = mnist.train.num_examples  # 55,000
# # n_validation = mnist.validation.num_examples  # 5000
# # n_test = mnist.test.num_examples  # 10,000
# n_train = 55000
# n_validation = 5000
# n_test = 10000

# n_input = 784  # input layer (28x28 pixels)
# n_hidden1 = 512  # 1st hidden layer
# n_hidden2 = 256  # 2nd hidden layer
# n_hidden3 = 128  # 3rd hidden layer
# n_output = 10  # output layer (0-9 digits)

# learning_rate = 1e-4
# n_iterations = 1000
# batch_size = 128
# dropout = 0.5

# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_output])
# keep_prob = tf.placeholder(tf.float32)

# weights = {
#     'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
#     'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
#     'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
#     'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
# }

# biases = {
#     'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
#     'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
#     'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
#     'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
# }

# layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
# layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
# layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
# layer_drop = tf.nn.dropout(layer_3, keep_prob)
# output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(
#         labels=Y, logits=output_layer
#         ))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# # train on mini batches
# for i in range(n_iterations):
#     batch_x, batch_y = mnist.train.next_batch(batch_size)
#     sess.run(train_step, feed_dict={
#         X: batch_x, Y: batch_y, keep_prob: dropout
#         })

#     # print loss and accuracy (per minibatch)
#     if i % 100 == 0:
#         minibatch_loss, minibatch_accuracy = sess.run(
#             [cross_entropy, accuracy],
#             feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
#             )
#         print(
#             "Iteration",
#             str(i),
#             "\t| Loss =",
#             str(minibatch_loss),
#             "\t| Accuracy =",
#             str(minibatch_accuracy)
#             )

# test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
# print("\nAccuracy on test set:", test_accuracy)


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_labels))
# print(train_labels[0:10])


# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# print(train_images[0])

train_images = train_images / 255.0

test_images = test_images / 255.0


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# print(train_images[0])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


predictions[0]

np.argmax(predictions[0])

test_labels[0]


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()