import numpy as np

"""
Для понимания логики расчета простейшей нейронной сети перцептрон
"""


def act(x):  # функция активации нейрона
    return 0 if x < 0.5 else 1


def go(x1, x2, x3):
    x = np.array([x1, x2, x3])  # вектор входного слоя
    w11 = [0.3, 0.3, 0]  # веса для каждой связи (входного слоя и первого нейрона первого скрытого слоя)
    w12 = [0.4, -0.5, 1]  # веса для каждой связи (входного слоя и второго нейрона первого скрытого слоя)
    weight_1 = np.array([w11, w12])  # матрица 2х3 всех весов для первого скрытого слоя
    weight_2 = np.array([-1, 1])  # вектор 1х2 весов для выходного нейрона

    sum_hidden = np.dot(weight_1, x)  # суммы значений на нейронах скрытого слоя
    out_hidden = np.array([act(x) for x in sum_hidden])  # результаты активации нейронов скрытого слоя

    sum_out = np.dot(weight_2, out_hidden)
    y = act(sum_out)

    return print(y)


go(1, 0, 1)