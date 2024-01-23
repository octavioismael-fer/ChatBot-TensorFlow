# Impotamos Librearias
import numpy as np 
import json
import re
import tensorflow as tf
import warnings
import random
import spacy


nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings('ignore')

# Leyendo el Dataset
with open('Intent.json', 'rb') as file:
    data = json.load(file)

# Generamos el pre procesamiento
def pre_procesamiento(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line

# Dividimos nuestros datos en "inputs" y "targets"
inputs, targets = [], []
cls = []
intent_doc = {}

for i in data['intents']:
    if i['intent'] not in cls:
        cls.append(i['intent'])

    if i['intent'] not in intent_doc:
        intent_doc[i['intent']] = []

    for text in i['text']:
        inputs.append(pre_procesamiento(text))
        targets.append(i['intent'])

    for response in i['responses']:
        intent_doc[i['intent']].append(response)

# Tokenizamo datos
def token_data(inp_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    tokenizer.fit_on_texts(inp_list)

    inp_seq = tokenizer.texts_to_sequences(inp_list)

    # Añadimos padding
    inp_seq = tf.keras.preprocessing.sequence.pad_sequences(inp_seq, padding='pre')

    return tokenizer, inp_seq

# Pre procesamos los datos de input
tokenizer, inp_tensor = token_data(inputs)

def cr_cat_target(targets):
    word = {}
    cat_t = []
    counter = 0

    for trg in targets:
        if trg not in word:
            word[trg]=counter
            counter+=1
        cat_t.append(word[trg])

    cat_tensor = tf.keras.utils.to_categorical(cat_t, num_classes=len(word), dtype='int32')
    return cat_tensor, dict((v,k) for k, v in word.items())

# Pre procesamos los datos de Output
target_tensor, target_idx_word = cr_cat_target(targets)

'input shape: {} and output shape: {}'.format(inp_tensor.shape, target_tensor.shape)

## Construccion de modelo
# Hiperparametros
epochs= 60
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 512
units= 128
target_len = target_tensor.shape[1]

# Modelo
model = tf.keras.models.Sequential([
    # Capa de Embedding
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
    # Capa de Hidden
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Capa de Clasificaicon
    tf.keras.layers.Dense(target_len, activation='softmax')
])

# Compilacion del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

# Vemo como se veria el modelo
model.summary()

# EarlyStopping -> Sirve para parar las iteraciones o "epochs" de aprendizaje
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# Entrenamiento
model.fit(inp_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])

# Respuesta
def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))

    # Se Divide las oraciones de entrada en palabras
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # Manejamos el error de palabras desconocidas
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])
    
    sent_seq = tf.expand_dims(sent_seq, 0)
    # Predecir la categoría de las oraciones de entrada
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)

    # Elegir una respuesta aleatoria para las oraciones predichas
    return random.choice(intent_doc[target_idx_word[pred_class[0]]]), target_idx_word[pred_class[0]]

# Chat con el bot
print("Note: Enter 'quit' to break the loop.")
while True:
    input_ = input('You: ')
    if input_.lower() == 'quit':
        break
    res, typ = response(input_)
    print('Bot: {} -- TYPE: {}'.format(res, typ))
    print()

