import numpy as np
import tensorflow as tf
from preprocess import clean_text, preprocess_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

# Load trained model
model = load_model("saved_model/recipe_generator.keras")

# Load tokenizers and sequence lengths
_, _, ingredient_tokenizer, instruction_tokenizer, max_ing_len, max_instr_len = preprocess_data()

def generate_recipe(ingredient_text):
    # Preprocess input
    ingredient_seq = ingredient_tokenizer.texts_to_sequences([clean_text(ingredient_text)])
    ingredient_seq = pad_sequences(ingredient_seq, maxlen=max_ing_len, padding='post')

    # Initialize decoder input
    decoder_seq = np.zeros((1, max_instr_len))
    decoder_seq[0, 0] = 1  # Assume '1' is the start token

    generated_text = []
    for i in range(max_instr_len - 1):
        predictions = model.predict([ingredient_seq, decoder_seq])
        next_word_index = np.argmax(predictions[0, i])

        if next_word_index == 0:
            break

        generated_text.append(instruction_tokenizer.index_word.get(next_word_index, ""))
        decoder_seq[0, i+1] = next_word_index

    return " ".join(generated_text)

# Example usage
if __name__ == "__main__":
    print(generate_recipe("chicken, garlic, onions, tomatoes"))