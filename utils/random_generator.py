from numpy.random import random

class RandomNumber(object):
    full_grad = False
    full_grad_prob = 0.1

    @classmethod
    def sample(cls):
        cls.full_grad = random() < cls.full_grad_prob
