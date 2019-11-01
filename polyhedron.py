import numpy as np
import itertools as it
from math import factorial
import re
import fractions

def permutation(m, n):     # Количество всевозможных перестановок
    return factorial(n) / (factorial(n - m) * factorial(m))

def combinations(matr, n): # Доработанный библиотечный combinations для нашего случая
    timed = list(map(list, it.combinations(matr, n)))
    return np.array(list(timed))

def check_extreme(matr, arr, x, sym_comb, m):
    sym_comb = sym_comb.replace(']', '')   # Убираем левую скобку знаков неравенств
    sym_comb = sym_comb.replace('[', '')   # Убираем правую скобку знаков неравенств
    sym_comb = re.split("[ ,]", sym_comb)  # Из строки получаем вектор знаков неравенств
    for i in range(int(m)):                # m - кол-во неравенств
        td_answer = float("{0:.7f}".format(KahanSum(matr[i] * x))) 
        # Умножаем i-ю строку матрицы на решение, а дальше проверяем удовлетворяет ли 
        # оно неравенству с правой частью. Также округляем float на 7-х знаках, чтобы питон не бузил
        if sym_comb[i] == '>':
            if td_answer <= arr[i]:
                return 0
        elif sym_comb[i] == '>=':
            if td_answer < arr[i]:
                return 0
        elif sym_comb[i] == '<':
            if td_answer >= arr[i]:
                return 0
        elif sym_comb[i] == '<=':
            if td_answer > arr[i]:
                return 0
        elif sym_comb[i] == '=':
            if td_answer != arr[i]:
                return 0
        elif sym_comb[i] == '!=':
            if td_answer == arr[i]:
                return 0
        else:
            return 0
    return 1

def extreme_points(A, b, sym_comb):
    # A = [[a,b,c],[x,y,z],...,[...]]
    # b = [a,b,c,e,d,x,...,x,y,z]
    # sym_comb = '[<=,>=,=,>,<,!=,...,>=]' или любой другой ввод строки знаков через запятую
    # Ввод
    A = np.array(A) # Левая часть
    b = np.array(b) # Праввая часть
    m, n = A.shape  # Размер левой части - строки/столбцы
    # Обработка
    ans_comb = np.zeros((1, n)) # Создаём единичный нулевой вектор, где будем хранить ответы
    arr_comb = combinations(b, n) # Всевозможные комбинации правой части
    matr_comb = combinations(A, n) # Соответствующие им комбинации левой части
    for i in range(int(permutation(n, m))): # Количество перестановок (C^m)_n
        if np.linalg.det(matr_comb[i]) != 0:  # Если определитель равен нулю -> решений нет
            x = np.linalg.solve(np.array(matr_comb[i], dtype='float'), 
                                np.array(arr_comb[i], dtype='float'))  # Поиск решения матрицы nxn для иксов
            ans_comb = np.vstack([ans_comb, x])  # Записываем наше решение
    ans_comb = np.delete(ans_comb, 0, axis=0)    # Удаляем наш нулевой вектор (см. строка 56)
    j = 0                                        # Счётчик успешной проверки
    for i in range(len(ans_comb)):
        if check_extreme(A, b, ans_comb[j], sym_comb, m): # Проверка - является ли решение частной системы - решением общей
            j += 1                                        # Если да, то идём дальше
        else:
            ans_comb = np.delete(ans_comb, j, axis=0)     # Если нет, то удаляем решение
    # Output
    return ans_comb

def nash_equilibrium(a1):
    # Добавляем к игровой матрице условия неотрицательности каждой неизвестной
    a1 = np.concatenate((a1,np.eye(np.size(np.array(a1, dtype = float),1))), axis = 0)
    # Составляем правую часть для матричной игры
    b = np.concatenate((np.ones(np.size(np.array(a1),1)),np.zeros(np.size(np.array(a1),1))), axis = 0)
    m,n = (np.array(a1)).shape # Получаем количество неизвестных


    # Составление знаков неравенств в зависимости от размерности input // [.] - для красоты 
    c1 = '['+('<=,'*(n)+'>=,'*(m-n))[:-1]+']'    # Максимизация
    c2 = '['+('>=,'*(m))[:-1]+']'                # Минимизация

    # Транспонируем
    a2 = np.concatenate((np.transpose(np.array(a1)[:-n, :]), np.eye(n, dtype = float)), axis=0)
    max_points = extreme_points(a1, b, c1) # Массив, где надо найти максимум
    min_points = extreme_points(a2, b, c2) # Массив, где надо найти минимум
    max_size = np.size(max_points,0) # Размер массива с максимумом (только строки)
    min_size = np.size(min_points,0) # Размер массива с минимумом (только строки)

    # Поиск максимума
    max = KahanSum(max_points[0])
    max_solve = max_points[0]
    for i in range(1,max_size):
        if (KahanSum(max_points[i]) > max):
            max = KahanSum(max_points[i]) # Запомнили максимальную сумму
            max_solve = max_points[i]     # Запомнили максимальный вектор

    # Поиск минимума
    min = KahanSum(min_points[0])
    min_solve = min_points[0]
    for i in range(1,min_size):
        if (KahanSum(min_points[i]) < min):
            min = KahanSum(min_points[i]) # Запомнили минимальную сумму
            min_solve = min_points[i]     # Запомнили минимальный вектор

    # Вывод
    print("First player: ", np.true_divide(min_solve,max))
    print("Second player: ", np.true_divide(max_solve, max))
    print("Cost of the game", 1/max)

def KahanSum(input):                # Метод Кэхэна для аккуратной суммы float
    sum = 0.0
    c = 0.0
    for i in range(len(input)):     # Перебор по каждой цифре числа
        y = input[i] - c            # Сначала с = 0
        t = sum + y                 # Alas, sum is big, y small, so low-order digits of y are lost.
        c = (t - sum) - y           # (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        sum = t    
    return sum

np.set_printoptions(precision=6, suppress=True, formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})  # Чтобы вывод был аккуратным
# Manual tests
akr = [[3,6,1,4],[5,2,4,2],[1,4,3,5],[4,3,4,-1]] # Тест из интернета
# Тест из задания прака
task_test_matrix = [[4,0,6,2,2,1],[3,8,4,10,4,4],[1,2,6,5,0,0],[6,6,4,4,10,3],[10,4,6,4,0,9],[10,7,0,7,9,8]]
fake_test = [[3,1],[1,3]] # Тест Миши
nash_equilibrium(task_test_matrix)







