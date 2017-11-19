import numpy as np
import tensorflow as tf
import random

class RegressionModel:
    def __init__(self, num_params):
        self.num_weights = num_params
        self.x_place = tf.placeholder(tf.float32, shape=(None, num_params), name="x")
        self.labels_place = tf.placeholder(tf.int64, shape=(None,), name="labels")
        self.setup_model()

    def setup_model(self):
        self.preds = self.predict(self.x_place)
        self.loss_val = self.loss(self.preds)
        self.train_op = self.train_step(self.loss_val)

    def predict(self, x):
        # w = tf.Variable((self.num_weights, 2), dtype=tf.float32, initializer=tf.zeros())
        # b = tf.Variable((1,), dtype=tf.float32, initializer=tf.zeros_initializer())
        # preds = tf.matmul(x, w) + b
        # ^ Shorthand for the above ^
        prediction = tf.layers.dense(inputs=x, units=2, use_bias=True)
        return prediction # batch_size x 2

    def loss(self, preds):
        one_hot = tf.one_hot(self.labels_place, 2)
        sofmax_ce_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot,
                logits=preds,
            )
        loss_val = tf.reduce_mean(sofmax_ce_loss)
        return loss_val

    def train_step(self, loss_val):
        learning_rate = 1e-4
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_val)
        return train_op

    def train(self, session, x_batch, labels_batch):
        input_feed = {
            self.x_place: x_batch,
            self.labels_place: labels_batch
        }
        output_feed = [self.loss_val, self.train_op]
        curr_loss, _ = session.run(output_feed, feed_dict=input_feed)
        print(loss)

class NeuralNetModel:
def __init__(self, num_params):
        self.num_weights = num_params
        self.x_place = tf.placeholder(tf.float32, shape=(None, num_params), name="x")
        self.labels_place = tf.placeholder(tf.int64, shape=(None,), name="labels")
        self.setup_model()

    def setup_model(self):
        self.preds = self.predict(self.x_place)
        self.loss_val = self.loss(self.preds)
        self.train_op = self.train_step(self.loss_val)

    def predict(self, x):
        # w = tf.Variable((self.num_weights, 2), dtype=tf.float32, initializer=tf.zeros())
        # b = tf.Variable((1,), dtype=tf.float32, initializer=tf.zeros_initializer())
        # preds = tf.matmul(x, w) + b
        # ^ Shorthand for the above ^
        prediction = tf.layers.dense(inputs=x, units=2, use_bias=True)
        return prediction # batch_size x 2

    def loss(self, preds):
        one_hot = tf.one_hot(self.labels_place, 2)
        sofmax_ce_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot,
                logits=preds,
            )
        loss_val = tf.reduce_mean(sofmax_ce_loss)
        return loss_val

    def train_step(self, loss_val):
        learning_rate = 1e-4
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_val)
        return train_op

    def train(self, session, x_batch, labels_batch):
        input_feed = {
            self.x_place: x_batch,
            self.labels_place: labels_batch
        }
        output_feed = [self.loss_val, self.train_op]
        curr_loss, _ = session.run(output_feed, feed_dict=input_feed)
        print(loss)



def separate_data(data, labels):
    pos_examples = data[labels == 1]
    neg_examples = data[labels == -1]
    return pos_examples, neg_examples

def split_data(pos_data, neg_data, SPLIT=0.8):
    indices = list(range(len(pos_data)))
    random.shuffle(indices)
    pos_train = indices[:int(len(pos_data) * SPLIT), :]
    pos_valid = indices[int(len(pos_data) * SPLIT):, :]

    indices = list(range(len(neg_data)))
    random.shuffle(indices)
    neg_train = indices[:int(len(neg_data) * SPLIT), :]
    neg_valid = indices[int(len(neg_data) * SPLIT):, :]

    return pos_train, neg_train, pos_valid, neg_valid

model = RegressionModel(10)
sess = tf.Session()


model_nn = NeuralNetModel(10)
sess_nn = tf.Session()
