import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import DATA_PATH

nltk.download('punkt')

# Load dataset
def load_data():
    df = pd.read_csv(DATA_PATH).dropna()
    df = df[['ingredients', 'cooking_method']]
    return df

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)
    return text

def preprocess_data():
    df = load_data()
    df['ingredients'] = df['ingredients'].apply(clean_text)
    df['cooking_method'] = df['cooking_method'].apply(clean_text)

    # Tokenization
    ingredient_tokenizer = Tokenizer()
    ingredient_tokenizer.fit_on_texts(df['ingredients'])

    instruction_tokenizer = Tokenizer()
    instruction_tokenizer.fit_on_texts(df['cooking_method'])

    # Convert text to sequences
    ingredient_sequences = ingredient_tokenizer.texts_to_sequences(df['ingredients'])
    instruction_sequences = instruction_tokenizer.texts_to_sequences(df['cooking_method'])

    max_ing_len = max(len(seq) for seq in ingredient_sequences)
    max_instr_len = max(len(seq) for seq in instruction_sequences)

    ingredient_sequences = pad_sequences(ingredient_sequences, maxlen=max_ing_len, padding='post')
    instruction_sequences = pad_sequences(instruction_sequences, maxlen=max_instr_len, padding='post')

    return ingredient_sequences, instruction_sequences, ingredient_tokenizer, instruction_tokenizer, max_ing_len, max_instr_len

if __name__ == "__main__":
    preprocess_data()