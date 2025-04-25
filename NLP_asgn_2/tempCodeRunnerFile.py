       # Add-k Laplace smoothing
        k = 1  # Smoothing parameter
        count_ngram = self.ngrams.get(ngram, 0)
        count_context = self.vocab.get(ngram[:-1], 0)
        vocab_size = len(self.vocab)
        probability = (count_ngram + k) / (count_context + k * vocab_size)
        return probability
