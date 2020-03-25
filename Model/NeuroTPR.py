from keras.models import Model
from keras.layers import TimeDistributed, Embedding, Input, LSTM, Bidirectional, concatenate, Dense, average
from keras.initializers import RandomUniform
from ELMo import ElmoEmbeddingLayer
from keras_contrib.layers import CRF


def NeuroTPR_initiate(wordEmbeddings, posEmbeddings, params):
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                      trainable=False)(words_input)

    pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
    pos = Embedding(input_dim=posEmbeddings.shape[0], output_dim=posEmbeddings.shape[1], weights=[posEmbeddings],
                    trainable=False)(pos_input)

    character_input1 = Input(shape=(None, 52), name='char_input_normal')
    embed_char_out1 = TimeDistributed(Embedding(input_dim=params["char_length"], output_dim=50, embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding1')(character_input1)
    char_lstm1 = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False, recurrent_dropout=0.25)),
                                name='char_LSTM1')(embed_char_out1)

    character_input2 = Input(shape=(None, 52), name='char_input_caseless')
    embed_char_out2 = TimeDistributed(Embedding(input_dim=params["char_length_caseless"], output_dim=50, embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding2')(character_input2)
    char_lstm2 = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False, recurrent_dropout=0.25)),
                                name='char_LSTM2')(embed_char_out2)

    char_lstm3 = average([char_lstm1, char_lstm2])

    words_elmo_input = Input(shape=(1,), dtype='string', name='words_elmo_input')
    words_elmo = ElmoEmbeddingLayer(trainable=False)(words_elmo_input)

    output = concatenate([words, pos, char_lstm3, words_elmo], axis=-1)

    word_lstm = Bidirectional(LSTM(60, return_sequences=True, dropout=0.50, recurrent_dropout=0.25), name='word_LSTM')(output)

    fc1 = TimeDistributed(Dense(50),name='fully_connected1')(word_lstm)

    my_crf = CRF(params["label_length"], sparse_target=True, name='CRF_layer')(fc1)

    model = Model(inputs=[words_input, character_input1, character_input2, pos_input, words_elmo_input], outputs=[my_crf]) # 

    return model