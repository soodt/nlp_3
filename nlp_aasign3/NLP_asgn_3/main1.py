import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from collections import Counter
import tensorflow_hub as hub

# Load dataset
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

# Build vocabulary
vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

# Set vocabulary size and initialize the lookup table
vocab_size = 10000
truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

#Model 1
embed_size = 16
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, input_shape=[None]),
    keras.layers.SimpleRNN(64),
    keras.layers.Dense(1, activation="sigmoid")
])

#model2
K = keras.backend
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model2 = keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])

history = model.fit(train_set, epochs=20)

# Load the test dataset
test_set = datasets["test"].batch(32).map(preprocess)
test_set = test_set.map(encode_words).prefetch(1)

# Evaluate the model
train_loss, train_acc = model.evaluate(train_set)
test_loss, test_acc = model.evaluate(test_set)

print(f"Training Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")

#1.2

embed_size = 16
model_lstm = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, input_shape=[None]),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation="sigmoid")
])

model_lstm.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])

history_lstm = model_lstm.fit(train_set, epochs=20)

# Evaluate the LSTM model
train_loss_lstm, train_acc_lstm = model_lstm.evaluate(train_set)
test_loss_lstm, test_acc_lstm = model_lstm.evaluate(test_set)

print(f"LSTM Training Accuracy: {train_acc_lstm}")
print(f"LSTM Test Accuracy: {test_acc_lstm}")


