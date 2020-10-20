import numpy as np # linear algebra
import os
import tensorflow as tf

def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    text = ""

    for dirname, _, filenames in os.walk('training_data'):
        for filename in filenames:
            text_file = os.path.join(dirname, filename)
            temp = open(text_file, 'r', encoding="utf-8")
            text += temp.read()
            temp.close()


    vocab = sorted(set(text))
    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Number of characters to generate
    num_generate = 1500

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.5

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))
new_model = tf.keras.models.load_model('trump.model')
new_model.summary()
print(generate_text(new_model, "Hello Americans"))