import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input, Attention, Concatenate, AdditiveAttention
from tensorflow.keras.models import Model
from config import EMBEDDING_DIM, LSTM_UNITS

def build_model(vocab_size_ing, vocab_size_instr, max_ing_len, max_instr_len):
    # Encoder
    encoder_inputs = Input(shape=(max_ing_len,))
    encoder_embedding = Embedding(vocab_size_ing, EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_instr_len,))
    decoder_embedding = Embedding(vocab_size_instr, EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    # Attention
    attention = AdditiveAttention()
    context_vector = attention([decoder_outputs, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, context_vector])

    # Output
    decoder_dense = Dense(vocab_size_instr, activation='softmax')
    decoder_outputs = decoder_dense(decoder_combined_context)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model