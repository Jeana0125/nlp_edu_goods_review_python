from random import randint
import re

from text.write.data_util import *
import connect_db.connect_hdfs as cdch

def wordListSum(wordList):
    sum = 0
    for word, value in wordList.items():
        sum = sum + value
    return sum


def retrieveRandomWord(wordList):

    randomIndex = randint(1, wordListSum(wordList))
    for word, value in wordList.items():
        randomIndex -= value
        if randomIndex <= 0:
            return word



def buildWordDict(text):

    status = text.split(' ')[0]
    product = text.split(' ')[1]
    file_list = cdch.open_file_list("C:\\Users\\user\\Desktop\\edu\\train_sub.csv")
    file_temp = []
    for line in file_list:
        if line[0] == status:
            if product in line[1] + " " + line[2]:
                file_temp.append(line[1] + " " + line[2])

    words = [word for word in file_temp if word != ""]
    wordDict = {}
    for i in range(1, len(words)):
        if words[i-1] not in wordDict:
            wordDict[words[i-1]] = {}
        if words[i] not in wordDict[words[i-1]]:
            wordDict[words[i-1]][words[i]] = 0
        wordDict[words[i-1]][words[i]] = wordDict[words[i-1]][words[i]] + 1

    return wordDict


def randomFirstWord(wordDict):
    randomIndex = randint(0, len(wordDict))
    return list(wordDict.keys())[randomIndex]


if __name__ == '__main__':

    text = input("text:")
    wordDict = buildWordDict(text)
    length = 20
    chain = ""
    currentWord = randomFirstWord(wordDict)
    for i in range(0, length):
        chain += currentWord + " "
        currentWord = retrieveRandomWord(wordDict[currentWord])
    print(chain)