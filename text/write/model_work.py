import random
import os
import numpy as np

import keras
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam

from text.write.data_util import *
import connect_db.connect_hdfs as cdch
from text.write.config import Config



"""
자동 리뷰 쓰기
"""

class ReviewModel(object):

    ###########1.prepare############
    def __init__(self,config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config

        #파일 얻어오기
        self.word2numF, self.num2word, self.words, self.files_content = prepare_file()

        #model 파일이 존재하면 training 시작
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            # parameter 컴퓨팅 과정 출력(keras)
            self.model.summary()
        else:
            self.train()

        self.do_train = False
        self.loaded_model = True


    ###########2.build model############
    def build_model(self):

        #shape=(self.config.max_len,) ----> [1,2,3,4,5,6,...]
        #tensor object 출력
        input_tensor = Input(shape=(self.config.max_len,))
        # dense vector 생성 ( [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] )
        embedd = Embedding(len(self.num2word), 300, input_length=self.config.max_len)(input_tensor)
        # Bidirectional LSTM
        # RNN : GRU , LSTM
        # 상세 내용 : https://keras.io/ko/layers/recurrent/ 참조
        #lstm = Bidirectional(GRU(128, return_sequences=True))(embedd)
        lstm = Bidirectional(LSTM(128, return_sequences=True))(embedd)
        # overfitting 제어
        dropout = Dropout(0.3)(lstm)
        #lstm = LSTM(256)(dropout)
        # dropout = Dropout(0.6)(lstm)
        flatten = Flatten()(dropout)
        dense = Dense(len(self.words), activation='softmax')(flatten)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    def sample(self,preds,temperature=1.0):
        """
        temperature = 1.0, normal model
        temperature = 0.5, open model
        temperature = 1.5, conservative model
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1,preds,1)
        return np.argmax(probas)


    def generate_sample_result(self,epoch,logs):
        """
        매번 epoch의 결과 출력
        """
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.5, 1.0, 1.5]:
            print("------------Diversity {}--------------".format(diversity))
            file_list = self.files_content.split(" ")
            start_index = random.randint(0,len(file_list) - self.config.max_len - 1)
            generated = ''
            pre_sentence = file_list[start_index: start_index + self.config.max_len]
            print(pre_sentence)
            sentence = " ".join(pre_sentence)
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-self.config.max_len:]):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]

                generated += next_char
                sentence = sentence + " " + next_char
            print(sentence)



    ###########3.predict y############
    def predict(self,text):
        """
        입력한 단어로 리뷰 작성
        """
        if not self.loaded_model:
            return
        status = text.split(" ")[0]
        product = text.split(" ")[1]

        content = cdch.open_file_list("C:\\Users\\user\\Desktop\\edu\\train_sub.csv")
        file_temp = []
        for con in content:
            if con[0] == status:
                temp = con[1]+" "+con[2]
                if product in temp:
                    file_temp.append(temp)
        random_line = random.choice(file_temp)
        print("random_line============" + random_line)

        res = ''
        random_words = random_line.split(" ")

        x_pred = np.zeros((1, self.config.max_len))
        for t in range(self.config.max_len):
            word = random_words[t]
            x_pred[0, t] = self.word2numF(word)
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, 1.0)
            next_char = self.num2word[next_index]
            print(next_char)
            res += " "+ next_char
        return res


    def data_generator(self):
        i = 0
        while 1:
            _x = self.words[i:i+self.config.max_len]
            x = " ".join(_x)
            y = self.words[i + self.config.max_len]
            #print(x, y)

            puncs = [']', '[', '（', '）', '{', '}', '：', '....', '!', ':']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue

            y_vec = np.zeros(shape=(1, len(self.words)),dtype=np.bool)
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(shape=(1, self.config.max_len),dtype=np.int32)

            for t in range(len(x.split(" "))):
                x_vec[0, t] = self.word2numF(x.split(" ")[t])

            yield x_vec, y_vec
            i += 1


    ###########4.train model############
    def train(self):
        #number_of_epoch = len(self.files_content) // self.config.batch_size
        number_of_epoch = 1
        if not self.model:
            self.build_model()

        self.model.summary()

        self.model.fit_generator(
            generator=self.data_generator(),
            #verbos=0,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                # save_weights_only=False 완전한 model 저장
                keras.callbacks.ModelCheckpoint(self.config.weight_file,save_weights_only=False),
                #epoch가 끝날때 generate_sample_result 함수 호출
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


if __name__ == '__main__':

    model = ReviewModel(Config)
    while 1:
        text = input("text:")
        sentence = model.predict(text)
        print(sentence)



