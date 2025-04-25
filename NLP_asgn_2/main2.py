import math

def tokenize(sentence):
    sentence = sentence.strip('\n')
    tokens = sentence.split(" ")
    tokens.append("<STOP>")
    return tokens

def countTokenOccurences(sentenceList):
    tokenOccurences = {}

    for sentence in sentenceList:
        for token in sentence:
            if token in tokenOccurences:
                tokenOccurences[token] += 1
            else:
                tokenOccurences[token] = 1

    return tokenOccurences

def parseTrainingOOV(sentenceList, margin):
    newSentenceList = []
    vocab = set()
    tokenOccurences = countTokenOccurences(sentenceList)

    # Remove tokens with more than margin occurrences from tokenOccurences
    for key, value in tokenOccurences.copy().items():
        if value > margin:
            tokenOccurences.pop(key)
            vocab.add(key) # adds keys to vocab set
    vocab.add("<UNK>")  # add <UNK> to vocab set

    # Replace tokens with less than margin occurrences with UNK
    for sentence in sentenceList:
        newSentence = []
        for token in sentence:
            if token not in tokenOccurences:
                newSentence.append(token)
            else:
                newSentence.append("<UNK>")
        newSentenceList.append(newSentence)

    return newSentenceList, vocab

def parseOOV(sentenceList, vocab):
    newSentenceList = []
    for sentence in sentenceList:
        newSentence = []
        for token in sentence:
            if token not in vocab:
                newSentence.append("<UNK>")
            else:
                newSentence.append(token)
        newSentenceList.append(newSentence)

    return newSentenceList

def ngram(sentence, n):
    ngramList = []

    for i in range(len(sentence)-n+1):
        ngramList.append(sentence[i:i+n])

    return ngramList

def ngramProbability(languageModel, word, context):
    numerator = 0
    denominator = 1

    if context in languageModel:
        denominator = sum(languageModel[context].values())

        if word in languageModel[context]:
            numerator = languageModel[context][word]

    return float(numerator) / float(denominator)

def unigramModelTrain(sentenceList):
    model = {}
    modelProbabilities = {}
    N = 0

    for sentence in sentenceList:
        N += len(sentence)
        for token in sentence:
            if token not in model:
                model[token] = 1
            else:
                model[token] += 1

    for key in model:
        modelProbabilities[key] = model.get(key) / N

    return modelProbabilities

def ngramModelTrain(sentenceList, n):
    model = {}
    modelProbabilities = {}

    # Construct the model
    for sentence in sentenceList:
        for ng in ngram(sentence, n):
            context = tuple(ng[0:-1])
            word = ng[-1]

            if context not in model:
                model[context] = {}

            if word not in model[context]:
                model[context][word] = 1
            else:
                model[context][word] += 1

    # Apply smoothing to handle unseen contexts and words
    for context, nextWords in model.items():
        total_count = sum(nextWords.values())
        modelProbabilities[context] = {}
        for word, count in nextWords.items():
            modelProbabilities[context][word] = (count + 1) / (total_count + len(nextWords))

    return modelProbabilities

def unigramModelPerplexity(sentenceList, model):
    m = 0
    l = 0

    for sentence in sentenceList:
        for word in sentence:
            m += 1
            if word not in model:
                p = 0
            else:
                p = model[word]

            if p > 0:
                l += math.log((p), 2)
            else:
                return math.inf

    perplexity = pow(2, -l / m)

    return perplexity

def ngramModelPerplexity(sentenceList, model):
    m = 0
    l = 0

    n = len(next(iter(model))) + 1

    for sentence in sentenceList:
        ng = ngram(sentence, n)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            if context not in model or word not in model[context]:
                p = 1e-10  # Assign a small probability for unseen n-grams
            else:
                p = model[context][word]

            l += math.log((p), 2)

    perplexity = pow(2, -l / m)

    return perplexity

def additiveSmoothing(sentenceList, n, delta):
    model = {}
    modelProbabilities = {}

    # Construct the model
    for sentence in sentenceList:
        for ng in ngram(sentence, n):
            context = tuple(ng[0:-1])
            word = ng[-1]

            if context not in model:
                model[context] = {}

            if word not in model[context]:
                model[context][word] = 1
            else:
                model[context][word] += 1

    # Apply additive smoothing
    for context, nextWords in model.items():
        total_count = sum(nextWords.values())
        modelProbabilities[context] = {}
        for word, count in nextWords.items():
            modelProbabilities[context][word] = (count + delta) / (total_count + delta * len(nextWords))

    return modelProbabilities

def smoothedModelPerplexity(sentenceList, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3):
    m = 0
    l = 0

    for sentence in sentenceList:
        ng = ngram(sentence, 3)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            p = 0

            if context in trigramModel and word in trigramModel[context]:
                p += trigramModel[context][word] * lambda_3
            if context[1:] in bigramModel and word in bigramModel[context[1:]]:
                p += bigramModel[context[1:]][word] * lambda_2
            if word in unigramModel:
                p += unigramModel[word] * lambda_1

            l += math.log((p), 2)

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

    # training
    print("Training models...")
    sentenceList_train_, vocab = parseTrainingOOV(sentenceList_train, 2)
    unigramModel = unigramModelTrain(sentenceList_train_)
    bigramModel = ngramModelTrain(sentenceList_train_, 2)
    trigramModel = ngramModelTrain(sentenceList_train_, 3)
    print("Training complete.")

    print("")
    print("====================")
    print("")

    # get perplexity on training data
    print("Calculating perplexity on training set...")
    unigramPerplexity_train = unigramModelPerplexity(sentenceList_train_, unigramModel)
    bigramPerplexity_train = ngramModelPerplexity(sentenceList_train_, bigramModel)
    trigramPerplexity_train = ngramModelPerplexity(sentenceList_train_, trigramModel)
    print("unigram Perplexity on training set: {}".format(unigramPerplexity_train))
    print("bigram Perplexity on training set: {}".format(bigramPerplexity_train))
    print("trigram Perplexity on training set: {}".format(trigramPerplexity_train))

    print("")
    print("====================")
    print("")

    # get perplexity on dev data
    print("Calculating perplexity on dev set...")
    sentenceList_dev_ = parseOOV(sentenceList_dev, vocab)
    unigramPerplexity_dev = unigramModelPerplexity(sentenceList_dev_, unigramModel)
    bigramPerplexity_dev = ngramModelPerplexity(sentenceList_dev_, bigramModel)
    trigramPerplexity_dev = ngramModelPerplexity(sentenceList_dev_, trigramModel)
    print("unigram Perplexity on dev set: {}".format(unigramPerplexity_dev))
    print("bigram Perplexity on dev set: {}".format(bigramPerplexity_dev))
    print("trigram Perplexity on dev set: {}".format(trigramPerplexity_dev))

    print("")
    print("====================")
    print("")

    # get perplexity on test data
    print("Calculating perplexity on test set...")
    sentenceList_test_ = parseOOV(sentenceList_test, vocab)
    unigramPerplexity_test = unigramModelPerplexity(sentenceList_test_, unigramModel)
    bigramPerplexity_test = ngramModelPerplexity(sentenceList_test_, bigramModel)
    trigramPerplexity_test = ngramModelPerplexity(sentenceList_test_, trigramModel)
    print("unigram Perplexity on test set: {}".format(unigramPerplexity_test))
    print("bigram Perplexity on test set: {}".format(bigramPerplexity_test))
    print("trigram Perplexity on test set: {}".format(trigramPerplexity_test))

    print("")
    print("====================")
    print("")
    deltas = [1, 0.5, 0.1]  # You can choose other values as well
    for delta in deltas:
        print("====================")
        print("Calculating additive smoothing perplexities with delta={}".format(delta))
        smoothedUnigramModel = additiveSmoothing(sentenceList_train_, 1, delta)
        smoothedBigramModel = additiveSmoothing(sentenceList_train_, 2, delta)
        smoothedTrigramModel = additiveSmoothing(sentenceList_train_, 3, delta)

        unigramPerplexity_train_smoothed = ngramModelPerplexity(sentenceList_train_, smoothedUnigramModel)
        bigramPerplexity_train_smoothed = ngramModelPerplexity(sentenceList_train_, smoothedBigramModel)
        trigramPerplexity_train_smoothed = ngramModelPerplexity(sentenceList_train_, smoothedTrigramModel)
        print("Additive smoothing perplexity on training set with delta={}: ".format(delta))
        print("Unigram: ", unigramPerplexity_train_smoothed)
        print("Bigram: ", bigramPerplexity_train_smoothed)
        print("Trigram: ", trigramPerplexity_train_smoothed)

        smoothedUnigramDev = additiveSmoothing(sentenceList_dev_, 1, delta)
        smoothedBigramDev = additiveSmoothing(sentenceList_dev_, 2, delta)
        smoothedTrigramDev = additiveSmoothing(sentenceList_dev_, 3, delta)

        unigramPerplexity_dev_smoothed = ngramModelPerplexity(sentenceList_dev_, smoothedUnigramDev)
        bigramPerplexity_dev_smoothed = ngramModelPerplexity(sentenceList_dev_, smoothedBigramDev)
        trigramPerplexity_dev_smoothed = ngramModelPerplexity(sentenceList_dev_, smoothedTrigramDev)
        print("Additive smoothing perplexity on dev set with delta={}: ".format(delta))
        print("Unigram: ", unigramPerplexity_dev_smoothed)
        print("Bigram: ", bigramPerplexity_dev_smoothed)
        print("Trigram: ", trigramPerplexity_dev_smoothed)

    print("====================")
    print("Aplha of 0.5 seems to have the best results on the dev set.")
    smoothedUnigramTest = additiveSmoothing(sentenceList_test_, 1, delta)
    smoothedBigramTest = additiveSmoothing(sentenceList_test_, 2, delta)
    smoothedTrigramTest = additiveSmoothing(sentenceList_test_, 3, delta)

    unigramPerplexity_test_smoothed = ngramModelPerplexity(sentenceList_test_, smoothedUnigramTest)
    bigramPerplexity_test_smoothed = ngramModelPerplexity(sentenceList_test_, smoothedBigramTest)
    trigramPerplexity_test_smoothed = ngramModelPerplexity(sentenceList_test_, smoothedTrigramTest)
    # print("Additive smoothing perplexity on dev set with delta={}: ".format(delta))
    print("Unigram: ", unigramPerplexity_test_smoothed)
    print("Bigram: ", bigramPerplexity_test_smoothed)
    print("Trigram: ", trigramPerplexity_test_smoothed)

    print("Training smoothed models...")
    lambdas = [(0.1, 0.3, 0.6), (0.7, 0.15, 0.15), (0.15, 0.7, 0.15), (0.15, 0.15, 0.7), (0.33, 0.33, 0.33)]
    for (lambda_1, lambda_2, lambda_3) in lambdas:
        print("====================")
        print("Calculating smoothed model perplexities with hyperparameters lambda_1={}, lambda_2={}, lambda_3={}".format(lambda_1, lambda_2, lambda_3))
        smoothedPerplexity_training = smoothedModelPerplexity(sentenceList_train_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        smoothedPerplexity_dev = smoothedModelPerplexity(sentenceList_dev_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        smoothedPerplexity_test = smoothedModelPerplexity(sentenceList_test_, unigramModel, bigramModel, trigramModel, lambda_1, lambda_2, lambda_3)
        print("smoothed model perplexity on training set: {}".format(smoothedPerplexity_training))
        print("smoothed model perplexity on dev set: {}".format(smoothedPerplexity_dev))
        print("smoothed model perplexity on test set: {}".format(smoothedPerplexity_test))

if __name__ == '__main__':
    main()
