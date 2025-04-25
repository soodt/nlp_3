import math
import random
from collections import defaultdict

def read_file(file):
    data = []
    with open(file) as f:
        for tokens in f:
            data.append(get_tokens(tokens)) 
    return data

# countTokenOccurences
def count_tokens(tokenArr):
    dict = defaultdict(int)
    for i in range(len(tokenArr)):
        s = tokenArr[i]
        for token in s:
            dict[token] += 1
    return dict

# tokenize
def get_tokens(tokens):
    line = tokens.strip()
    sol = line.split()
    sol.append("<STOP>")
    return sol

def parse_tokens(tokenArr, vocab):
    arr = []
    for i in range(len(tokenArr)):
        s = tokenArr[i]
        sol = []
        for token in s:
            if token not in vocab:
                sol.append("<UNK>")
            else:
                sol.append(token)
        arr.append(sol)

    return arr

def unk_tokens(tokenArr):
    arr = []
    v = set()
    numTokens = count_tokens(tokenArr)

    for key, value in numTokens.copy().items():
        if value >= 3:
            numTokens.pop(key)
            v.add(key)
    v.add("<UNK>")

    for i in range(len(tokenArr)):
        s = tokenArr[i]
        sol = []
        for token in s:
            if token not in numTokens:
                sol.append(token)
            else:
                sol.append("<UNK>")
        arr.append(sol)
    return arr, v

def get_unigram(tokenArr):
    dict = defaultdict(int)
    n = 0

    for i in  range(len(tokenArr)):
        s = tokenArr[i]
        n += len(s)
        for token in s:
            dict[token] += 1

    sol = {}
    for key, value in sol.items():
        sol[key] = value / n

    return sol

def get_ngram(tokens, n):
    ngramList = []
    for i in range((len(tokens) + 1) - n ):
        ngramList.append(tokens[i:i+n])
    return ngramList

def trainUnigram(sentences, n = 0):
    model = defaultdict(int)
    for i in range(len(sentences)):
        s = sentences[i]
        n += len(s)
        for word in s:
            model[word] += 1
    
    p = {}
    for key, value in model.items():
        temp = value / n
        p[key] = temp
        
    return p

def trainModel(tokenArr, n):
    model = defaultdict(lambda: defaultdict(int))
    prob = {}

    for sentence in tokenArr:
        for ng in get_ngram(sentence, n):
            key = tuple(ng[0:-1])
            latest = ng[-1]
            model[key][latest] += 1

    for key, value in model.items():
        total = sum(value.values())
        prob[key] = {}
        n = len(value)
        for key2, value2 in value.items():
            prob[key][key2] = max(value2 - 0.75, 0) / total
            prob[key]["<UNK>"] = 0.75 * n / total
    return prob

def unigram_perplex(tokenArr, model):
    d, n = 0, 0

    for i in range(len(tokenArr)):
        s = tokenArr[i]
        for word in s:
            d += 1
            p = model.get(word, model.get("<UNK>", 1e-7))
            n += math.log(p, 2)

    sol = pow(2, -n / d)

    return sol

def ng_Perplex(tokenArr, model):
    m, l = 0, 0

    n = len(next(iter(model))) + 1

    for i in range(len(tokenArr)):
        s = tokenArr[i]
        ng = get_ngram(s, n)
        for ntuple in ng:
            context = tuple(ntuple[0:-1])
            word = ntuple[-1]

            m += 1
            p = model.get(context, {}).get(word, model.get(context, {}).get("<UNK>", 1e-10))
            l += math.log(p, 2)
    perplexity = pow(2, -l / m)
    return perplexity

trainData = read_file("1b_benchmark.train.tokens")
devData = read_file("1b_benchmark.test.tokens")
testData = read_file("1b_benchmark.dev.tokens")
trainData_, vocab = unk_tokens(trainData)
unigram = trainUnigram(trainData_)
bigram = trainModel(trainData_, 2)
trigram = trainModel(trainData_, 3)
devData_ = parse_tokens(devData, vocab)
testData_ = parse_tokens(testData, vocab)
#testdata = "HDTV ."
spacer = ("***********************\n***********************\n***********************")
print(f"unigram Perplexity on training set: {unigram_perplex(trainData_, unigram)}")
print(f"bigram Perplexity on training set: {ng_Perplex(trainData_, bigram)}")
print(f"trigram Perplexity on training set: {ng_Perplex(trainData_, trigram)}")
print(spacer)
print(f"unigram Perplexity on dev set: {unigram_perplex(devData_, unigram)}")
print(f"bigram Perplexity on dev set: {ng_Perplex(devData_, bigram)}")
print(f"trigram Perplexity on dev set: {ng_Perplex(devData_, trigram)}")
print(spacer)
print(f"unigram Perplexity on test set: {unigram_perplex(testData_, unigram)}")
print(f"bigram Perplexity on test set: {ng_Perplex(testData_, bigram)}")
print(f"trigram Perplexity on test set: {ng_Perplex(testData_, trigram)}")
print(spacer)


#Needs work
def addSmoothing(tokenArr, n, d):
    model = defaultdict(lambda: defaultdict(int))
    p = {}

    for i in range(len(tokenArr)):
        s = tokenArr[i]
        for ng in get_ngram(s, n):
            key = tuple(ng[0:-1])
            value = ng[-1]
            model[key][value] += 1

    for key, value in model.items():
        total = sum(value.values())
        p[key] = {}
        for word, count in value.items():
            p[key][word] = (count + d) / (total + d * len(value))
            p[key]["<UNK>"] = d / (total + d * len(value))

    return p

#Needs work
def smooth_Perplex(tokenArr, unigram, bigram, trigram, lambdaa):
    
    lam1, lam2, lam3 = lambdaa
    m, l = 0, 0

    for i in range(len(tokenArr)):
        s = tokenArr[i]
        model = get_ngram(s, 3)
        for t in model:
            context = tuple(t[0:-1])
            word = t[-1]

            m += 1
            p = 0

            trigram_prob = trigram.get(context, {}).get(word, trigram.get(context, {}).get("<UNK>", 0))
            p += trigram_prob * lam3

            bigram_prob = bigram.get(context[1:], {}).get(word, bigram.get(context[1:], {}).get("<UNK>", 0))
            p += bigram_prob * lam2

            unigram_prob = unigram.get(word, unigram.get("<UNK>", 1e-10))
            p += unigram_prob * lam1
            l += math.log(p, 2)

    perplexity = pow(2, -l / m)

    return perplexity

for d in [1, 0.5, 0.1]:
    print(f"Delta {d}")
    unigramS = addSmoothing(trainData_, 1, d)
    bigramS = addSmoothing(trainData_, 2, d)
    trigramS = addSmoothing(trainData_, 3, d)

    print(f"Unigram: {ng_Perplex(trainData_, unigramS)} ")
    print(f"Bigram: {ng_Perplex(trainData_, bigramS)} ")
    print(f"TriGram: {ng_Perplex(trainData_, trigramS)} ")
    unigramSDev = addSmoothing(devData_, 1, d)
    bigramSDev = addSmoothing(devData_, 2, d)
    trigramSDev = addSmoothing(devData_, 3, d)
    print(f"Unigram: {ng_Perplex(devData_, unigramSDev)}")
    print(f"Bigram: {ng_Perplex(devData_, bigramSDev)}")
    print(f"Trigram: {ng_Perplex(devData_, trigramSDev)}")


print(spacer)
unigramSTest = addSmoothing(testData_, 1, d)
bigramSTest = addSmoothing(testData_, 2, d)
trigramSTest = addSmoothing(testData_, 3, d)
print(f"Unigram: {ng_Perplex(testData_, unigramSTest)}")
print(f"Bigram: {ng_Perplex(testData_, bigramSTest)}")
print(f"Trigram: {ng_Perplex(testData_, bigramSTest)}")
print(spacer)
for lambdaa in [(0.1, 0.3, 0.6), (0.7, 0.15, 0.15), (0.15, 0.7, 0.15), (0.15, 0.15, 0.7), (0.33, 0.33, 0.33)]:
    print(f"training smoothed perplexity: {smooth_Perplex(trainData_, unigram, bigram, trigram, lambdaa)}")
    print(f"dev smoothed perplexity: {smooth_Perplex(devData_, unigram, bigram, trigram, lambdaa)}")
    print(f"test smoothed perplexity: {smooth_Perplex(testData_, unigram, bigram, trigram, lambdaa)}")