from preprocess import preprocess_data
from model import build_model
from config import BATCH_SIZE, EPOCHS
import numpy as np

# Load and preprocess data
X, y, ingredient_tokenizer, instruction_tokenizer, max_ing_len, max_instr_len = preprocess_data()

# Vocabulary sizes
vocab_size_ing = len(ingredient_tokenizer.word_index) + 1
vocab_size_instr = len(instruction_tokenizer.word_index) + 1

# Build model
model = build_model(vocab_size_ing, vocab_size_instr, max_ing_len, max_instr_len)

# Prepare decoder inputs
y_input = np.hstack((np.ones((y.shape[0], 1)), y[:, :-1]))  # Add start token

# Train model
model.fit([X, y_input], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# Save model
model.save("saved_model/recipe_generator82k.keras")