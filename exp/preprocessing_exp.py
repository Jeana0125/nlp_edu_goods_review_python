from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

#print(word_tokenize("Don‘t be afraid.Jone’s mother is coming!"))

from nltk.tokenize import WordPunctTokenizer
#print(WordPunctTokenizer().tokenize("Don‘t be afraid. Jone’s mother is coming!"))

import re
word = re.compile(r'\W*\b\w{1,2}\b')
#print(word.sub('','I am a student.I live Korea'))

from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don‘t be afraid. Jone’s mother is coming!"))

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [['barber', 'person'], ['barber', 'good', 'person'],
             ['barber', 'huge', 'person'], ['knew', 'secret'],
             ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
             ['barber', 'kept', 'word'], ['barber', 'kept', 'word'],
             ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
             ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.
tokenizer.fit_on_texts(sentences)
#텍스트 시퀀스의 모든 단어들을 각 정수로 맵핑
encoded = tokenizer.texts_to_sequences(sentences)
#print(encoded)

#패딩 하기 위하여 길이가 제일 긴 문장의 길이를 계산
max_len = max(len(item) for item in encoded)
print(max_len)

for item in encoded: # 각 문장에 대해서
    while len(item) < max_len:   # max_len보다 작으면
        item.append(0)

padded_np = np.array(encoded)
print(padded_np)

