import connect_db.connect_hdfs as cdch
import nltk
from nltk.corpus import stopwords as pw
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from random import shuffle
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#hdfs에서 파일 읽어오기
#pdd = tt.Process_Data_HDFS()
#content = pdd.connect_hdfs_read("/data/input/test.csv")
content = cdch.open_file_list("C:\\Users\\user\\Desktop\\edu\\train_sub.csv")
#print(content)

"""
알고리즘 테스트하여 최상의 세개 알고리즘 추출
"""

#레이블 값
y_train = []
#X값
X_pre_train = []
X_train = []
#정서 값
emotion_dict = {'positive':2,'negative':1}

###############1.레이블과 X값 얻기#######################
#stop words
nltk.download('stopwords')
cacheStopWords = pw.words("english")

def read_file():

    str = []
    for con in content:
        # title & content
        sentence = con[1]+' '+con[2]
        line = []
        line.append(con[0])
        line.append([word for word in sentence.split(' ') if word.lower() not in cacheStopWords])
        #print(line)
        str.append(line)

    return str

###############2.doc2vec##################

def bag_of_words(words):
    return dict([(word, True) for word in words])


def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  #리뷰를 단어 두개로 묶는다
    bigrams = bigram_finder.nbest(score_fn, n) #카이제곱 사용
    newBigrams = [u + v for (u, v) in bigrams]
    return bag_of_words(newBigrams)


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    newBigrams = [u + v for (u, v) in bigrams]
    a = bag_of_words(words)
    b = bag_of_words(newBigrams)
    a.update(b) #dict a를 b에 병합
    return a  #모든 단어 + 두개 단어 묶음

#['2',['Amazing!','pleased',.....]]
file_words = read_file()

# number의 많은 정보를 가진 특징을 얻기
def get_feature(number):
    posWords=[]
    negWords=[]

    for line in file_words:
        if line[0] == '2':
            for word in line[1]:
                #print("pos    "+word)
                posWords.append(word)
        else:
            for word in line[1]:
                #print("neg    "+word)
                negWords.append(word)

    # 모든 단어의 나타난 횟수를 통계
    word_fd = FreqDist()
    # positive & negative words 통계
    con_word_fd = ConditionalFreqDist()

    for word in posWords:
            word_fd[word]+=1
            con_word_fd['pos'][word]+=1

    for word in negWords:
        word_fd[word]+=1
        con_word_fd['neg'][word]+=1

    pos_word_count=con_word_fd['pos'].N()    # positive word count
    neg_word_count=con_word_fd['neg'].N()    # negative word count
    total_word_count=pos_word_count+neg_word_count

    word_scores={}
    for word,freq in word_fd.items():
        print(word,freq)
        pos_score=BigramAssocMeasures.chi_sq(con_word_fd['pos'][word],(freq,pos_word_count),total_word_count)
        neg_score=BigramAssocMeasures.chi_sq(con_word_fd['neg'][word],(freq,neg_word_count),total_word_count)
        word_scores[word]=pos_score+neg_score
        best_vals=sorted(word_scores.items(),key=lambda item: item[1],reverse=True)[:number]
        best_words=set([w for w,s in best_vals])

    return dict([(word,True) for word in best_words])


# 트레이닝 데이터를 구축
def build_features():
    #feature = bag_of_words(text())
    #feature = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=900)
    # feature =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=900)
    feature = get_feature(300)  # 단어 얻기

    posFeatures = []
    negFeatures = []
    for items in file_words:
        a = {}
        if items[0] == '2':
            for item in items[1]:
                if item in feature.keys():
                    a[item] = 'True'
            posWords = [a, 'pos']  # 좋은 리뷰 "pos"
            posFeatures.append(posWords)
        else:
            for item in items[1]:
                if item in feature.keys():
                    a[item] = 'True'
            negWords = [a, 'neg']  # 나쁜 리뷰 "neg"
            negFeatures.append(negWords)

    return posFeatures, negFeatures

#traing data
posFeatures,negFeatures =  build_features()

###############3.알고리즘##################

# 텍스트 랜덤
shuffle(posFeatures)
shuffle(negFeatures)

train = posFeatures[200:]+negFeatures[200:]
test = posFeatures[:200]+negFeatures[:200]
data,tag = zip(*test)

def score(classifier):

    classifier = SklearnClassifier(classifier)
    classifier.train(train)
    pred = classifier.classify_many(data)
    n = 0
    s = len(pred)
    for i in range(0, s):
        if pred[i] == tag[i]:
            n = n + 1

    return n / s  # 레이블 값과 비교


print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))
print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression()))
print('SVC`s accuracy is %f' % score(SVC()))
print('LinearSVC`s accuracy is %f' % score(LinearSVC()))
print('NuSVC`s accuracy is %f' % score(NuSVC()))





