import pandas as pd
import numpy as np
import string
import secrets
import time
import re
import collections


class HangmanTest:
    def __init__(self, training_file: str, testing_file: str, max_ngram=4):
        with open(training_file) as file:
            self.training_words = file.read().splitlines()

        with open(testing_file) as file:
            self.testing_words = file.read().splitlines()

        self.guessed_letters = set()
        self.max_ngram = max_ngram
        self.ngram_ps = dict()

        # 1-gram
        counts = collections.Counter("".join(self.training_words))
        self.ngram_ps[1] = pd.Series(counts) / counts.total()
        
        # n-gram
        for n in range(2, self.max_ngram+1):
            counts = collections.Counter(
                [f"<{w}>"[i:i+n] for w in self.training_words for i in range(len(w) - n + 3)])
            self.ngram_ps[n] = pd.Series(counts) / counts.total()
        
        
    def gen_contextual_ps(self, dotted: str):
        if "." not in dotted:
            return
        
        chunks = dotted.split(".") # split to runs of known letters
        print(f"chunks: {chunks}")
        dot_index = -1   # index of the dot in input
        pattern = ""
        for j in range(len(chunks) - 1):
            combined = chunks[j] + "." + chunks[j+1]
            lj = len(chunks[j])
            dot_index = dotted.index(".", dot_index+1)

            if combined == ".":
                continue

            print(f"dot_index: {dot_index}")
            if len(combined) <= self.max_ngram:
                pattern = combined
            elif lj < self.max_ngram:
                pattern = combined[:self.max_ngram]
            else:
                pattern = combined[lj+1-self.max_ngram : lj+1]

            print(f"pattern: {pattern}")

            ps = self.ngram_ps[len(pattern)]
            print(f"ps1: {ps}")
            i = pattern.index(".") # dot index in pattern
            ps = ps[ps.index.str.match(pattern.replace(".", "[a-z]"))].rename(
                index=lambda a: a[i]
            )
            print(f"ps2: {ps}")
            # input("press enter ...")
            yield dot_index-1, ps

        
    
    def guess(self, masked: str) -> str: # word input example: "_ p p _ e "
        # remove spaces, add start/end markers, and substitute . for _
        dotted = f"<{masked}>".translate(str.maketrans({"_": ".", " ": None}))
        
        p = pd.DataFrame(
            # default probabilities are those of single letters (1-gram)
            data={j: self.ngram_ps[1] for j in range(len(dotted)-2)},
            dtype=float)
        
        for column, ps in self.gen_contextual_ps(dotted):
            p.loc[:, column] = ps
        p.fillna(1e-10, inplace=True)

        # remove guessed letters (rows)
        p.drop(
            index=list(self.guessed_letters.intersection(p.index)),
            inplace=True)
        p = p.div(p.sum()) # normalize by column
        p.fillna(0., inplace=True)

        
        # remove revealed positions (columns)
        p.drop(
            columns=[j-1 for j, a in enumerate(dotted) if a.isalpha()], 
            inplace=True)

        print("p = ", p, sep="\n")
                
        # guess the letter based on overall probability
        candidates = (1-(1-p).prod(axis="columns")).nlargest(1)
        guessed_letter = np.random.choice(
            a=candidates.index,
            p=candidates.values/candidates.sum())

        return guessed_letter

 

    def start_game(self, secret_word: str) -> bool:
        # Initialize variables
        incorrect_guesses = 0
        self.guessed_letters = set()
        self.current_dictionary = self.training_words

        masked_word = "_" * len(secret_word)
        print(f"secret word: {secret_word}")

        # Main game loop
        while incorrect_guesses < 6 and "_" in masked_word:
            # Display current state of the game
            print(f"masked word: {masked_word}")
            print(f"guessed letters: {self.guessed_letters}")
            print(f"incorrect guesses: {incorrect_guesses}")
            # Get a guess from the guess function
            guess_letter = self.guess(masked_word)
            print(f"guessing: {guess_letter}")
            # Check if the guessed letter is in the secret word
            if guess_letter in secret_word:
                # Update the masked word with the correctly guessed letter
                masked_word = "".join(
                    letter if letter == guess_letter or letter in self.guessed_letters else "_"
                    for letter in secret_word
                )
            else:
                # Incorrect guess, increment the counter
                incorrect_guesses += 1
            # Add the guessed letter to the set of guessed letters
            self.guessed_letters.add(guess_letter)
            # input("press enter to continue ...")
        # Game over, display the final result
        if "_" not in masked_word:
            print(f"Congratulations! You guessed the word: {secret_word}")
            return True
        else:
            print(f"Game over! The secret word was: {secret_word}")
            return False


test = HangmanTest(
    training_file="words_train.txt",
    testing_file="words_test.txt",
    max_ngram=4)


# test.start_game("hello")

trials = 100
success = 0


for i in range(trials):
    secret_word = np.random.choice(test.testing_words)
    if test.start_game(secret_word=secret_word):
        success += 1

print(f"success rate = {success/trials}")
