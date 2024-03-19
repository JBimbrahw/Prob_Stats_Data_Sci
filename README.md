# Prob_Stats_Data_Sci
Notes from https://www.coursera.org/learn/machine-learning-probability-and-statistics/home/welcome

Notes from assignment for Week 1 (Naive Bayes Classifer):

Exercise 1:
The task is to implement the function that generates a dictionary, recording the frequency with which each word in the dataset appears as spam (1) or ham (0).

def get_word_frequency(X,Y):
    """
    Calculate the frequency of each word in a set of emails categorized as spam (1) or not spam (0).

    Parameters:
    - X (numpy.array): Array of emails, where each email is represented as a list of words.
    - Y (numpy.array): Array of labels corresponding to each email in X. 1 indicates spam, 0 indicates ham.

    Returns:
    - word_dict (dict): A dictionary where keys are unique words found in the emails, and values
      are dictionaries containing the frequency of each word for spam (1) and not spam (0) emails.
    """

This function takes the email data that has already been classified as "spam" or "ham".
It then outputs a dictionary [words][{spam: frequency, ham: frequency}]

This will be useful later when we need to determine the P(wordi | class).

Exercise 2:
In this exercise we implement the function to compute  P(word∣spam) and P(word∣ham), or more generally P(word∣class).
We are doing this because it will help us to implement the Naive Bayes classification algorithm later:

Remember that P(email|class) = 𝞹 P(word | class)

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    Calculate the conditional probability of a given word occurring in a specific class.

    Parameters:
    - word (str): The target word for which the probability is calculated.
    - cls (str): The class for which the probability is calculated, it may be 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The conditional probability of the given word occurring in the specified class.
    """

    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]
    return p_word_given_class


So this provides us with the probability of a word being in a given class.

Exercise 3: 
Here we implement the function to compute 𝑃(email∣class) where class can be either spam (1) or ham (0). We use the naive assumption that P(email|class) = 𝞹 P(word | class).
It is a naive assumption that the probabilities of a word being in an email are independent.

Iterate over every word in the email and in each step, update the probability by multiplying it with the value for 𝑃(word∣class).

Exercise 4:
In this exercise we perform both computations below to calculate the probability an email is either spam or ham:

𝑃(spam)⋅𝑃(email∣spam)
𝑃(ham)⋅𝑃(email∣ham)

P(class) is given by (class_frequency)/(number of emails) 



