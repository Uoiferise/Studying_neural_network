import numpy as np
from random import *

epoch = [
    (1, 1, 1, 1),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 0),
    (1, 1, 0, 0)
]

w1 = np.array([[random() for _ in range(3)], [random() for _ in range(3)]])
w2 = np.array([random(), random()])


def act(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


def go(inp):
    sum_hidden = np.dot(w1, inp)
    out_hidden = [act(x) for x in sum_hidden]

    sum_end = np.dot(w2, out_hidden)
    result = act(sum_end)

    return (result, out_hidden)


def train(epoch):
    global w1, w2
    lmd = 0.01
    N = 10_000
    count = len(epoch)

    for k in range(N):
        inp = epoch[randint(0, count)]
        y, out = go(inp[:-1])

        e = y - inp[-1]
        delta = e * df(e)


def main():
    print(w1, w2, sep='\n')
    # print(go((1,1,1)))


if __name__ == '__main__':
    main()
