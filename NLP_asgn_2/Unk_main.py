import math
from collections import defaultdict
import random

def tokenize(sentence):
    sentence = sentence.strip('\n')
    tokens = sentence.split(" ")
    tokens.append("<STOP>")
    return tokens

def countTokenOccurences(sentences):
    tokenOccurences = {}

    for sentence in sentences:
        for token in sentence:
            if token in tokenOccurences:
                tokenOccurences[token] += 1
            else:
                tokenOccurences[token] = 1

    return tokenOccurences

def parseTrainingOOV(sentences, margin):
    
    new_sentences = []
    vocabulary = set()
    tokenOccurences = countTokenOccurences(sentences)

    for key, value in tokenOccurences.items():
        if (value > margin):
            vocabulary.add(key)
    vocabulary.add("<UNK>")  

    for sentence in sentences:
        new_sentence2 = []
        for token in sentence:
            if (token in vocabulary):
                new_sentence2.append(token)
            else:
                new_sentence2.append("<UNK>")
        new_sentences.append(new_sentence2)

    return new_sentences, vocabulary

def parseTrainingOOV_with_fraction(sentences, fraction):
    new_sentences = []
    vocabulary = set()
    tokenOccurences = countTokenOccurences(sentences)

    for key, value in tokenOccurences.items():
        if (value > 1):  
            vocabulary.add(key)
        if (value == 1 and random.random() >= fraction):
            vocabulary.add(key)
    vocabulary.add("<UNK>") 

    for sentence in sentences:
        new_sentence2 = []

        for token in sentence:
            if (token in vocabulary):
                new_sentence2.append(token)
            else:
                new_sentence2.append("<UNK>")
        new_sentences.append(new_sentence2)

    return new_sentences, vocabulary

def parseOOV(sentences, vocabulary):

    new_sentences = []
    
    for sentence in sentences:
        new_sentence2 = []
        for token in sentence:
            if (token not in vocabulary):
                new_sentence2.append("<UNK>")
            else:
                new_sentence2.append(token)
        new_sentences.append(new_sentence2)

    return new_sentences

def ngram(sentence, n):
    ngramList = []

    for i in range(len(sentence)-n+1):
        ngramList.append(sentence[i:i+n])

    return ngramList

def unigramModelTrain(sentences):
    model = defaultdict(int)
    N = 0

    for sentence in sentences:
        N += len(sentence)
        for token in sentence:
            model[token] += 1

    modelProbabilities = {token: count / N for token, count in model.items()}
    return modelProbabilities

def ngramModelTrain(sentences, n):
    model = defaultdict(lambda: defaultdict(int))
    modelProbabilities = {}

    for sentence in sentences:
        for ng in ngram(sentence, n):
            context = tuple(ng[0:-1])
            word = ng[-1]
            model[context][word] += 1

    for context, nextWords in model.items():
        total_count = sum(nextWords.values())
        discounted_total = total_count - 0.75 * len(nextWords)
        modelProbabilities[context] = {}
        for word, count in nextWords.items():
            modelProbabilities[context][word] = max(count - 0.75, 0) / total_count
        modelProbabilities[context]["<UNK>"] = 0.75 * len(nextWords) / total_count

    return modelProbabilities

def unigramModelPerplexity(sentences, model):
    m = 0
    l = 0

    for sentence in sentences:
        for word in sentence:
            m += 1
            p = model.get(word, model.get("<UNK>", 1e-10))
            l += math.log(p, 2)

    perplexity = pow(2, -l / m)
    return perplexity

def ngramModelPerplexity(sentences, model):
    m = 0
    l = 0

    n = len(next(iter(model))) + 1

    for sentence in sentences:
        ng = ngram(sentence, n)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            p = model.get(context, {}).get(word, model.get(context, {}).get("<UNK>", 1e-10))
            l += math.log(p, 2)

    perplexity = pow(2, -l / m)
    return perplexity

def additiveSmoothing(sentences, n, delta):
    model = defaultdict(lambda: defaultdict(int))
    modelProbabilities = {}

    for sentence in sentences:
        for ng in ngram(sentence, n):
            context = tuple(ng[0:-1])
            word = ng[-1]
            model[context][word] += 1

    for context, nextWords in model.items():
        total_count = sum(nextWords.values())
        modelProbabilities[context] = {}
        for word, count in nextWords.items():
            modelProbabilities[context][word] = (count + delta) / (total_count + delta * len(nextWords))
        modelProbabilities[context]["<UNK>"] = delta / (total_count + delta * len(nextWords))

    return modelProbabilities

def smoothedModelPerplexity(sentences, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3):
    m = 0
    l = 0

    for sentence in sentences:
        ng = ngram(sentence, 3)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            p = 0
            p += trigramModel.get(context, {}).get(word, trigramModel.get(context, {}).get("<UNK>", 0)) * lambda_3
            p += bigramModel.get(context[1:], {}).get(word, bigramModel.get(context[1:], {}).get("<UNK>", 0)) * lambda_2
            p += unigramModel.get(word, unigramModel.get("<UNK>", 1e-10)) * lambda_1

            l += math.log(p, 2)

    perplexity = pow(2, -l / m)
    return perplexity

def main():
    sentenceList_train = []
    with open("1b_benchmark.train.tokens") as training_file:
        for line in training_file:
            sentenceList_train.append(tokenize(line))

    sentenceList_test = []
    with open("1b_benchmark.test.tokens") as testing_file:
        for line in testing_file:
            sentenceList_test.append(tokenize(line))

    sentenceList_dev = []
    with open("1b_benchmark.dev.tokens") as dev_file:
        for line in dev_file:
            sentenceList_dev.append(tokenize(line))

    # Training and evaluating with all tokens appearing less than 5 times converted to <UNK>
    print("Training models with all tokens appearing less than 5 times converted to <UNK>...")
    sentenceList_train_5, vocab_5 = parseTrainingOOV(sentenceList_train, 4)
    unigramModel_5 = unigramModelTrain(sentenceList_train_5)
    bigramModel_5 = ngramModelTrain(sentenceList_train_5, 2)
    trigramModel_5 = ngramModelTrain(sentenceList_train_5, 3)
    print("Training complete.")

    sentenceList_dev_5 = parseOOV(sentenceList_dev, vocab_5)
    sentenceList_test_5 = parseOOV(sentenceList_test, vocab_5)

    print("Calculating perplexity with tokens < 5 times as <UNK>...")
    unigramPerplexity_dev_5 = unigramModelPerplexity(sentenceList_dev_5, unigramModel_5)
    bigramPerplexity_dev_5 = ngramModelPerplexity(sentenceList_dev_5, bigramModel_5)
    trigramPerplexity_dev_5 = ngramModelPerplexity(sentenceList_dev_5, trigramModel_5)
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev_5))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev_5))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev_5))

    print("")
    print("====================")
    print("")

    # Training and evaluating with a fraction of words appearing just once converted to <UNK>
    fraction = 0.5
    print("Training models with fraction of tokens appearing just once converted to <UNK>...")
    sentenceList_train_fraction, vocab_fraction = parseTrainingOOV_with_fraction(sentenceList_train, fraction)
    unigramModel_fraction = unigramModelTrain(sentenceList_train_fraction)
    bigramModel_fraction = ngramModelTrain(sentenceList_train_fraction, 2)
    trigramModel_fraction = ngramModelTrain(sentenceList_train_fraction, 3)
    print("Training complete.")

    sentenceList_dev_fraction = parseOOV(sentenceList_dev, vocab_fraction)
    sentenceList_test_fraction = parseOOV(sentenceList_test, vocab_fraction)

    print("Calculating perplexity with fraction of tokens appearing just once as <UNK>...")
    unigramPerplexity_dev_fraction = unigramModelPerplexity(sentenceList_dev_fraction, unigramModel_fraction)
    bigramPerplexity_dev_fraction = ngramModelPerplexity(sentenceList_dev_fraction, bigramModel_fraction)
    trigramPerplexity_dev_fraction = ngramModelPerplexity(sentenceList_dev_fraction, trigramModel_fraction)
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev_fraction))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev_fraction))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev_fraction))

    print("")
    print("====================")
    print("")

    # Print the comparison of the two approaches
    print("Comparison of Perplexities:")
    print("Approach 1: Convert all tokens appearing less than 5 times to <UNK>")
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev_5))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev_5))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev_5))

    print("Approach 2: Convert a fraction of tokens appearing just once to <UNK>")
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev_fraction))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev_fraction))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev_fraction))

if __name__ == '__main__':
    main()
