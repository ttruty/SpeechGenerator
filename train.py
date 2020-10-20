import numpy as np # linear algebra
import tensorflow as tf
import nltk
import spacy
from nltk import word_tokenize
from nltk.util import ngrams
import re
import os
from datetime import date

today = date.today()
input_path = "training_data"

text = ""

for dirname, _, filenames in os.walk(input_path):
    for filename in filenames:
        text_file = os.path.join(dirname, filename)
        temp = open(text_file, 'r', encoding="utf-8")
        text += temp.read()
        temp.close()

        print ('Length of {}: {} characters'.format(filename, len(text)))
print("Sample:")
print(text[:500])

print ('Length of text: {} characters'.format(len(text)))
print(text[:500])

# adding stopwords
nlp = spacy.load("en")
nlp.Defaults.stop_words |= {"oh", "uh", "na", "okay", "li", "y1"}
stopwords = nlp.Defaults.stop_words

all_words = nltk.tokenize.word_tokenize(text.lower())
all_words_no_stop = nltk.FreqDist(w.lower() for w in all_words if w not in stopwords)

vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


def text_to_int(text):
    return np.array([char2idx[c] for c in text])


text_as_int = text_to_int(text)


# extra step, better to do it
def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])


print(int_to_text(text_as_int[:13]))

seq_length = 1000  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello


dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()


for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(data, epochs=100, callbacks=[checkpoint_callback])

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
date = today.strftime("%b-%d-%Y")
model.save(f"./models/{date}-speech.model")
