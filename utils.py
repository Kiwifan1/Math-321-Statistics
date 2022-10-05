# Name: Joshua Venable
# Class: MATH321 - Experimental Statistics
# Date: 10/05/2022
# Description: Contains methods for finding the multitude of statistical
#  probabilities necessary for MATH321
# Notes:
#

import statistics as stats
from scipy.integrate import quad
import math
import pandas as pd
import numpy as np


def combination(n: int, r: int) -> float:
    """ Returns the number of combinations of n things taken r at a time

    Args:
        n (int): The number of things
        r (int): The number of things taken at a time

    Returns:
        float: The number of combinations
    """
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))


def permutation(n: int, r: int) -> float:
    """ Returns the number of permutations of n things taken r at a time

    Args:
        n (int): The number of things
        r (int): The number of things taken at a time

    Returns:
        float: The number of permutations
    """
    return math.factorial(n) / math.factorial(n - r)


def binomial(n: int, r: int, p: float) -> float:
    """ Returns the probability of r successes in n trials with probability p

    Args:
        n (int): The number of trials
        r (int): The number of successes
        p (float): The probability of success

    Returns:
        float: The probability of r successes in n trials with probability p
    """
    return combination(n, r) * (p ** r) * ((1 - p) ** (n - r))


def neg_binomial(x: float, r: int, p: float) -> float:
    """ Returns the probability of x in a negative binomial distribution with r trials and probability p

    Args:
        x (float): The value to find the probability of
        r (int): The number of trials
        p (float): The probability of success

    Returns:
        float: The probability of x in a negative binomial distribution with r trials and probability p
    """
    return combination(x - 1, r - 1) * (p ** r) * ((1 - p) ** (x - r))


def poisson(l: int, k: int) -> float:
    """ Returns the probability of k events in a Poisson distribution with lambda l

    Args:
        l (int): The lambda value
        k (int): The number of events

    Returns:
        float: The probability of k events in a Poisson distribution with lambda l
    """
    return (l ** k) * math.exp(-l) / math.factorial(k)


def cumulative_poisson(l: int, k: int) -> float:
    """ Returns the cumulative probability of k events in a Poisson distribution with lambda l

    Args:
        l (int): The lambda value
        k (int): The number of events

    Returns:
        float: The cumulative probability of k events in a Poisson distribution with lambda l
    """
    return sum([poisson(l, i) for i in range(k + 1)])


def normal(x: float, mu: float, sigma: float) -> float:
    """ Returns the probability of x in a normal distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The probability of x in a normal distribution with mean mu and standard deviation sigma
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def cumulative_normal(x: float, mu: float, sigma: float) -> float:
    """ Returns the cumulative probability of x in a normal distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The cumulative probability of x in a normal distribution with mean mu and standard deviation sigma
    """
    return (1 + math.erf((x - mu) / (sigma * math.sqrt(2)))) / 2


def exponential(x: float, lambda_: float) -> float:
    """ Returns the probability of x in an exponential distribution with lambda lambda

    Args:
        x (float): The value to find the probability of
        lambda_ (float): The lambda value

    Returns:
        float: The probability of x in an exponential distribution with lambda lambda_
    """
    return lambda_ * math.exp(-lambda_ * x)


def cumulative_exponential(x: float, lambda_: float) -> float:
    """ Returns the cumulative probability of x in an exponential distribution with lambda lambda

    Args:
        x (float): The value to find the probability of
        lambda_ (float): The lambda value

    Returns:
        float: The cumulative probability of x in an exponential distribution with lambda lambda_
    """
    return 1 - math.exp(-lambda_ * x)


def lognormal(x: float, mu: float, sigma: float) -> float:
    """ Returns the probability of x in a lognormal distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The probability of x in a lognormal distribution with mean mu and standard deviation sigma
    """
    return (1 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(x) - mu) ** 2) / (2 * sigma ** 2))


def cumulative_lognormal(x: float, mu: float, sigma: float) -> float:
    """ Returns the cumulative probability of x in a lognormal distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The cumulative probability of x in a lognormal distribution with mean mu and standard deviation sigma
    """
    return (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2)))) / 2


def phi(x: float, mu: float, sigma: float) -> float:
    """ Returns the probability of x in a phi distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The probability of x in a phi distribution with mean mu and standard deviation sigma
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def cumulative_phi(x: float, mu: float, sigma: float) -> float:
    """ Returns the cumulative probability of x in a phi distribution with mean mu and standard deviation sigma

    Args:
        x (float): The value to find the probability of
        mu (float): The mean of the distribution
        sigma (float): The standard deviation of the distribution

    Returns:
        float: The cumulative probability of x in a phi distribution with mean mu and standard deviation sigma
    """
    return (1 + math.erf((x - mu) / (sigma * math.sqrt(2)))) / 2


def gamma(x: float, alpha: int, beta: float) -> float:
    """ Returns the probability of x in a gamma distribution with k trials and alpha

    Args:
        x (float): The value to find the probability of
        alpha (int): The number of trials
        beta (float): The beta value

    Returns:
        float: The probability of x in a gamma distribution with alpha trials and beta
    """
    return (x ** (alpha - 1)) * (math.exp(-x / beta)) / (beta ** alpha * math.factorial(alpha - 1))


def cumulative_gamma(x: float, alpha: int, beta: float) -> float:
    """ Returns the cumulative probability of x in a gamma distribution with k trials and alpha

    Args:
        x (float): The value to find the probability of
        alpha (int): The number of trials
        beta (float): The beta value

    Returns:
        float: The cumulative probability of x in a gamma distribution with alpha trials and beta
    """
    return sum([gamma(i, alpha, beta) for i in range(x + 1)])

