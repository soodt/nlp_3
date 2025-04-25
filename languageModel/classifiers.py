import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.positive = None
        self.negative = None
        self.word_given_positive = None
        self.word_given_negative = None

    def fit(self, X, Y):

        # Calculating prior
        self.positive = np.sum(Y) / len(Y)
        self.negative = 1 - self.positive

        # Calculating conditional probabilty of each word

        num_pos = X[Y == 1].sum(axis=0)
        num_neg = X[Y == 0].sum(axis=0)
        total_pos = np.sum(X[Y == 1])
        total_neg = np.sum(X[Y == 0])
        size = X.shape[1]

        self.word_given_positive = (num_pos + 1) / (total_pos + size)
        self.word_given_negative = (num_neg + 1) / (total_neg + size)

    def predict(self, X):
        # predictions = np.zeros(X.shape[0])

        # log_prob_pos = np.log(self.word_given_positive)
        # log_prob_neg = np.log(self.word_given_negative)

        # log_prior_pos = np.log(self.positive)
        # log_prior_neg = np.log(self.negative)

        # log_positive = log_prior_pos + np.dot(X, log_prob_pos)
        # log_negative = log_prior_neg + np.dot(X, log_prob_neg)

        # log_positive = np.log(self.positive) + np.dot(X, np.log(self.word_given_positive))
        # log_negative = np.log(self.negative) + np.dot(X, np.log(self.word_given_negative))

        # predictions[log_positive > log_negative] = 1

        # return predictions.astype(int)
        predictions = np.zeros(X.shape[0])
        log_prob_pos = np.log(self.word_given_positive)
        log_prob_neg = np.log(self.word_given_negative)
        log_prior_pos = np.log(self.positive)
        log_prior_neg = np.log(self.negative)

        # Reshape log_prob_pos and log_prob_neg to match the number of features in X
        log_prob_pos = log_prob_pos.reshape(-1, X.shape[1])
        log_prob_neg = log_prob_neg.reshape(-1, X.shape[1])

        log_positive = log_prior_pos + np.dot(X, log_prob_pos.T)
        log_negative = log_prior_neg + np.dot(X, log_prob_neg.T)

        # Flatten log_positive and log_negative before indexing
        predictions[log_positive.flatten() > log_negative.flatten()] = 1

        return predictions.astype(int)

class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, l2_penalty=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, Y):
        self.initialize_parameters(X.shape[1])

        # Gradient descent
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self.sigmoid(linear_model)

            # without regularization
            derivative_without_regularization = (1 / len(X)) * np.dot(X.T, (y_predicted - Y))
            biased_without_regularization = (1 / len(X)) * np.sum(y_predicted - Y)

            # with L2 regularization
            derivative_with_regularization = derivative_without_regularization + 2 * self.l2_penalty * self.weights
            biased_with_regularization = biased_without_regularization

            # Print out gradient magnitudes
            #print("Gradient magnitude without regularization:", np.linalg.norm(dw_without_regularization))
           # print("Gradient magnitude with L2 regularization:", np.linalg.norm(dw_with_regularization))

            self.weights -= self.learning_rate * derivative_with_regularization
            self.bias -= self.learning_rate * biased_with_regularization


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)



# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
