# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lg
   Description :
   Author :       lizhenhui
   date：          2024/3/25
-------------------------------------------------
   Change Activity:
                   2024/3/25:
-------------------------------------------------
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM

# 定义文本数据
text = """这是你的训练数据。你可以在这里放入你的文本数据，确保它包含足够的文本量以便模型学习。"""

# 创建字符索引映射
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# 准备训练数据
maxlen = 10
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# 创建输入和标签
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)

y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 构建模型
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')


# 定义生成文本函数
def generate_text(model, seed_text, num_chars=50, temperature=1.0):
    generated_text = seed_text
    for i in range(num_chars):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            sampled[0, t, char_indices[char]] = 1.

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text


# 辅助函数：使用 softmax 温度对预测进行采样
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 训练模型
model.fit(x, y, batch_size=128, epochs=10)

# 生成文本示例
generated_text = generate_text(model, seed_text="这是", num_chars=200, temperature=0.5)
print(generated_text)
