import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

VISTA_labels = "./Enhancer_Prediction/Tables/enhancers.xlsx"

def get_labels():
    df = pd.read_excel(VISTA_labels)
    activity = df["Limb-activity"]
    f = lambda label: -1 if label == "negative" else 1
    labels = np.array([f(label) for label in reversed(activity)])
    return labels.transpose()

class NeuralNetModel:
    def __init__(self, num_params):
        self.num_weights = num_params
        self.x_place = tf.placeholder(tf.float32, shape=(None, num_params), name="x")
        self.labels_place = tf.placeholder(tf.int64, shape=(None,), name="labels")
        self.setup_model()
        self.summary = tf.summary.merge_all()

    def setup_model(self):
        self.preds = self.predict(self.x_place)
        self.loss_val = self.loss(self.preds)
        self.train_op = self.train_step(self.loss_val)

    def predict(self, x):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.regularizer = regularizer
        l1 = tf.layers.dense(inputs=x, units=self.num_weights, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=regularizer)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(inputs=l1, units=self.num_weights, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=regularizer)
        self.featurization = l2
        l2 = tf.nn.relu(l2)
        prediction = tf.layers.dense(inputs=l2, units=2, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=regularizer)
        return prediction # batch_size x 2

    # def build_embeddings(self, pos_data, neg_data):

    def loss(self, preds):
        predicted_labels = tf.argmax(preds, axis=-1)
        correct_prediction = tf.equal(predicted_labels, self.labels_place)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        one_hot = tf.one_hot(self.labels_place, 2)
        self.debug_labels = one_hot
        sofmax_ce_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot,
                logits=preds,
            )
        loss_val = tf.reduce_mean(sofmax_ce_loss)
        self.auc = tf.metrics.auc(self.labels_place, predicted_labels)
        self.precision = tf.metrics.precision(self.labels_place, predicted_labels)
        tf.summary.scalar('loss', loss_val)
        return loss_val

    def train_step(self, loss_val):
        learning_rate = 1e-3
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_val)
        return train_op

    def train(self, session, x_batch, labels_batch):
        input_feed = {
            self.x_place: x_batch,
            self.labels_place: labels_batch
        }
        output_feed = [self.loss_val, self.train_op, self.accuracy, self.summary]
        curr_loss, _, accuracy, summary = session.run(output_feed, feed_dict=input_feed)
        print("accuracy: " + str(accuracy))
        return curr_loss, summary

    def validate(self, session, x_batch, labels_batch):
        input_feed = {
            self.x_place: x_batch,
            self.labels_place: labels_batch
        }
        output_feed = [self.loss_val, self.accuracy, self.summary, self.auc, self.precision]
        curr_loss, accuracy, summary, auc, precision = session.run(output_feed, feed_dict=input_feed)
        print("accuracy: " + str(accuracy))
        print("auc: " + str(auc))
        return curr_loss, summary, auc, precision

    def get_featurization(self, session, x_batch):
        input_feed = {
            self.x_place: x_batch
        }
        output_feed = self.featurization
        features = session.run(output_feed, feed_dict=input_feed)
        return features


def separate_data(data, labels):
    pos_examples = data[labels == 1, :]
    neg_examples = data[labels == -1, :]
    return pos_examples, neg_examples

def split_data(pos_data, neg_data, SPLIT=0.8):
    indices = list(range(len(pos_data)))
    random.shuffle(indices)
    POS_SPLIT = int(pos_data.shape[0] * SPLIT)
    pos_train = pos_data[indices[:int(pos_data.shape[0] * SPLIT)], :]
    pos_valid = pos_data[indices[int(pos_data.shape[0] * SPLIT):], :]

    indices = list(range(len(neg_data)))
    random.shuffle(indices)
    neg_train = neg_data[indices[:int(neg_data.shape[0] * SPLIT)], :]
    neg_valid = neg_data[indices[int(neg_data.shape[0] * SPLIT):], :]

    return pos_train, neg_train, pos_valid, neg_valid

def sample_batch(pos_train, neg_train, num_samples):
    pos_indices = list(range(pos_train.shape[0]))
    neg_indices = list(range(neg_train.shape[0]))

    random.shuffle(pos_indices)
    random.shuffle(neg_indices)

    pos_indices = pos_indices[:num_samples]
    neg_indices = neg_indices[:num_samples]

    return pos_train[pos_indices, :], neg_train[neg_indices, :]

def batch_validate(pos_valid, neg_valid, model, session):
    num_samples = int(pos_valid.shape[0])
    pos_sample, neg_sample = sample_batch(pos_valid, neg_valid, num_samples)
    samples = np.concatenate((pos_sample, neg_sample))
    labels = np.array([1] * num_samples + [0] * num_samples)
    shuf_indices = list(range(num_samples * 2))
    random.shuffle(shuf_indices)
    labels = labels[shuf_indices]
    samples = samples[shuf_indices]
    loss, summary = model.validate(session, samples, labels)
    return loss, summary

def whole_validate(pos_valid, neg_valid, model, session):
    pos_sample, neg_sample = pos_train, neg_train
    samples = np.concatenate((pos_sample, neg_sample))
    labels = np.array([1] * len(pos_sample) + [0] * len(neg_sample))
    shuf_indices = list(range(len(pos_sample) + len(neg_sample)))
    random.shuffle(shuf_indices)
    labels = labels[shuf_indices]
    samples = samples[shuf_indices]
    loss, summary, auc, precision = model.validate(session, samples, labels)
    return loss, summary, auc, precision

auc_values = []
prec_values = []

def batch_train(train_data, validation_data, model, session, iters = 10):
    pos_train, neg_train = train_data
    pos_valid, neg_valid = validation_data
    num_samples = int(pos_train.shape[0])
    train_writer = tf.summary.FileWriter('summaries/train', sess.graph)
    test_writer = tf.summary.FileWriter('summaries/validation')
    # num_samples = 200
    for i in range(iters):
        pos_sample, neg_sample = sample_batch(pos_train, neg_train, num_samples)
        samples = np.concatenate((pos_sample, neg_sample))
        labels = np.array([1] * num_samples + [0] * num_samples)
        shuf_indices = list(range(num_samples * 2))
        random.shuffle(shuf_indices)
        labels = labels[shuf_indices]
        samples = samples[shuf_indices]
        loss, summary = model.train(session, samples, labels)
        train_writer.add_summary(summary, i)
        print(loss)
        if i % 10 == 0:
            print("Validating...")
            # validation_loss, summary = batch_validate(pos_valid, neg_valid, model, session)
            validation_loss, summary, auc, precision = whole_validate(pos_valid, neg_valid, model, session)
            auc_values.append(auc)
            prec_values.append(auc)
            test_writer.add_summary(summary, i)

def generate_pca(data, model, session):
    pos_sample, neg_sample = data
    samples = np.concatenate((pos_sample, neg_sample))
    labels = np.array([1] * len(pos_sample) + [0] * len(neg_sample))
    featurization = model.get_featurization(session, samples)
    pca = PCA(n_components=2) #2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(featurization))
    plt.scatter(transformed[labels==1][0], transformed[labels==1][1], label='Positive', c='red')
    plt.scatter(transformed[labels==0][0], transformed[labels==0][1], label='Negative', c='blue')
    plt.legend()
    plt.show()

def generate_auc_plot():
    auc_arr = np.array(auc_values)
    x = auc_arr[:, 0]
    y = auc_arr[:, 1]
    plt.plot(x, y)
    plt.title('AUC plot for 3-layer neural net')
    plt.xlabel('Negative Labels')
    plt.ylabel('Positive Labels')
    plt.show()

# load and separate the data
# data_frame = pd.read_csv('train_data_fully_processed.csv')
data_frame = pd.read_csv('new_feature_extraction.csv')
# data_frame = data_frame.merge(data_frame_2)
data = data_frame.as_matrix().astype(np.float32)[:, 1:]

# normalize the data
data_mean = np.mean(data, axis=0)
data_range = np.max(data, axis=0) - np.min(data, axis=0)
data_dev = np.std(data, axis=0)
data = (data - data_mean) / data_dev
print(data)

labels = get_labels()
labels = np.array(labels.astype(float))
pos_data, neg_data = separate_data(data, labels)
pos_train, neg_train, pos_valid, neg_valid = split_data(pos_data, neg_data)

# define the model
model = NeuralNetModel(pos_data.shape[1])
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
# train the model
batch_train((pos_train, neg_train), (pos_valid, neg_valid), model, sess, 1000)
auc_arr = np.array(auc_values)
print(auc_arr.shape)
generate_auc_plot()
# np.save("auc", np.array(auc_values))
# np.save("precision", np.array(prec_values))
# generate_pca((pos_train, neg_train), model, sess)
