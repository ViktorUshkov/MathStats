import scipy.stats as stat
import numpy as np
import math
import itertools


def sign_test_exact(x, y):
    n = np.size(x)
    dif = x - y
    w = min(np.count_nonzero(dif > 0), np.count_nonzero(dif < 0))
    t = 2**(-n) * sum([math.comb(n, j) for j in range(w+1)])
    return t


def sign_test_assymp(x, y):
    n = np.size(x)
    dif = x - y
    w = min(np.count_nonzero(dif > 0), np.count_nonzero(dif < 0))
    t = (w - n / 2) / (n / 4)**0.5
    p_value = 2 * stat.norm.cdf(-np.abs(t))
    return t


def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def in_interval(t, interval):
    a, b = interval
    return a <= t <= b


def wilcoxon_test(x, y):
    dif = np.abs(x - y)
    n = np.size(dif)
    k = n // 10
    a = np.min(dif)
    b = np.max(dif)
    bounds = [a + (b - a) * i / k for i in range(k + 1)]
    intervals = list(pairwise(bounds))
    t = sum([np.sign(z) * [j for j in range(k) if in_interval(z, intervals[j])][0] for z in dif])
    t = t / (n * (n + 1) * (2 * n + 1) / 6)**0.5
    p_value = 2 * stat.norm.cdf(-np.abs(t))
    return t


def kendall_correl(x, y):
    concerted = 0
    n = np.size(x)
    for i in range(n):
        for j in range(i, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concerted += 1
    r = (concerted - (math.comb(n,2) - concerted)) / math.comb(n,2)
    t = r / (2 * (2 * n + 5) / (9 * n * (n - 1)))**0.5
    p_value = 2 * stat.norm.cdf(-np.abs(t))
    return r, t


def autocorrel(data):
    n = np.size(data)
    x1 = data
    x2 = np.zeros(n)
    for i in range(n - 1):
        x2[i] = x1[i+1]
    x2[n-1] = data[0]
    return stat.pearsonr(x1, x2)


def autocorrelation(distr):
    n = len(distr)
    counter = n * sum([distr[i] * distr[i + 1] for i in range(n - 1)]) - sum(distr) ** 2 + n * distr[0] * distr[n - 1]
    delimiter = n * sum([distr[i] ** 2 for i in range(n)]) - sum(distr) ** 2
    t = counter / delimiter
    # Проведем перенормировку
    z = (t + 1 / (n + 1)) / (n * (n - 3) / ((n + 1) * (n - 1) ** 2)) ** 0.5
    a = stat.norm.ppf(q=alpha / 2)
    print(f"z = {z}, [{a}; {-a}]")


def xu_test(x):
    x.sort()
    n = np.size(x)
    median = np.median(x)
    t = sum([(i - 1) * (x[i] - median) ** 2 for i in range(n)]) / (
            (n - 1) * sum([(x[i] - median) ** 2 for i in range(n)]))
    z = (t - 1 / 2) / ((n + 1) / (6 * (n - 1) * (n + 2))) ** 0.5
    return z, 2 * stat.norm.cdf(-np.abs(z))


def sign_test():
    n = 20
    x = stat.norm.rvs(loc=0, scale=7 ** 0.5, size=n)
    y1 = 5 * x + np.random.default_rng().uniform(low=-10, high=10, size=n)
    y2 = 5 * x + 20 * stat.expon.rvs(scale=1 / 20, size=n)
    print(f'interval: [{alpha / 2}, {1 - alpha / 2}]')
    print(f'uniform noise; exact: {sign_test_exact(x, y1)}')
    print(f'exponential noise; exact: {sign_test_exact(x, y2)}\n')


def signed_rank_test_and_corr_analysis():
    n = 500
    x = stat.norm.rvs(loc=0, scale=7 ** 0.5, size=n)
    y1 = 5 * x + np.random.default_rng().uniform(low=-10, high=10, size=n)
    y2 = 5 * x + 100 * stat.expon.rvs(scale=1/20, size=n)
    a = stat.norm.ppf(q=alpha / 2)
    b1 = stat.t.ppf(q=alpha / 2, df=n - 2)
    b2 = stat.t.ppf(q=1 - alpha / 2, df=n - 2)
    print(f'interval: [{a}, {-a}]')
    print(f'uniform noise: wilcoxon: {wilcoxon_test(x, y1)}, sign assymptotic test: {sign_test_assymp(x, y1)}')
    print(f'exponential noise: wilcoxon: {wilcoxon_test(x, y2)}, sign assymptotic test: {sign_test_assymp(x, y2)}\n')

    print('Kendall and Pearson correlation coefficients')
    r_pearson, p_pearson = pear = stat.pearsonr(x, y1)
    p = pear[0]
    t_pearson = p / (1 - p ** 2) ** 0.5
    r_kendal, t_kendall = kendall_correl(x, y1)
    print('uniform noise')
    print(f'Pearson: correlation {r_pearson}, t-value {t_pearson}, interval: [{b1}, {b2}]')
    print(f'Kendall: correlation {r_kendal}, t-value {t_kendall}, interval: [{a}, {-a}]\n')

    r_pearson, p_pearson = pear = stat.pearsonr(x, y2)
    p = pear[0]
    t_pearson = p / (1 - p ** 2) ** 0.5
    r_kendal, t_kendall = kendall_correl(x, y2)
    print('exponential noise')
    print(f'Pearson: correlation {r_pearson}, t-value {t_pearson}, interval: [{b1}, {b2}]')
    print(f'Kendall: correlation {r_kendal}, t-value {t_kendall}, interval: [{a}, {-a}]\n')


def autocorrelation_tests():
    n = 200
    x1 = stat.norm.rvs(loc=5, scale=7 ** 0.5, size=n)
    x2 = np.zeros(n)
    for i in range(n - 1):
        x2[i] = x1[i] + 0.1 * x1[i + 1]

    a = stat.norm.ppf(q=alpha / 2)
    # cor1, p_value1 = autocorrelation(x1)
    print("random sample")
    autocorrelation(x1)
    # print(f'interval: [{a}, {-a}]')
    # print(f'autocorrelation with lag 1: {cor1}\n')

    # cor2, p_value2 = autocorrelation(x2)
    print('autocorrelated sample')
    autocorrelation(x2)
    # print(f'interval: [{a}, {-a}]')
    # print(f'autocorrelation with lag 1: {cor2}\n')

    print('\nXu criteria')

    x1 = stat.norm.rvs(loc=5, scale=7 ** 0.5, size=n)
    t, p_value = xu_test(x1)
    print('same dispersion sample')
    print(f'interval: [{a}, {-a}]')
    print(f'xu statistic: {t}\n')

    x2 = np.copy(x1)
    for i in range(n // 2, n):
        x2[i] *= 1.5

    t, p_value = xu_test(x2)
    print('different dispersion sample')
    print(f'interval: [{a}, {-a}]')
    print(f'xu statistic: {t}')


if __name__ == '__main__':
    alpha = 0.05
    print('Sign test')
    sign_test()
    print('Signed-rank test')
    signed_rank_test_and_corr_analysis()
    print('Autocorrelaton')
    autocorrelation_tests()
