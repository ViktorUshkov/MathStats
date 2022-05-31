import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from sklearn import linear_model


def in_range(x, interval):
    a, b = interval
    if a <= x <= b:
        return True
    return False


def pearson_coef(x, y):
    return np.corrcoef(x, y)[1, 0]


def corr_conf_int(r):
    z = 0.5 * np.log((1 + r) / (1 - r))
    z_L = z - u_half_alpha * (n-3)**-0.5
    z_U = z + u_half_alpha * (n-3)**-0.5
    r_low = (np.exp(2 * z_L) - 1) / (np.exp(2 * z_L) + 1)
    r_high = (np.exp(2 * z_U) - 1) / (np.exp(2 * z_U) + 1)
    return r_low, r_high


def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def group_data(data, k):
    x0 = math.floor(min(data))
    xn = math.ceil(max(data))
    bounds = [x0 + i * (xn - x0) / k for i in range(k + 1)]
    intervals = list(pairwise(bounds))
    invervals_mids = [(a + b) / 2 for (a, b) in intervals]
    grouped = np.zeros(np.size(data))
    for i in range(np.size(data)):
        entry_group = [j for j in range(k) if in_range(data[i], intervals[j])][0]
        grouped[i] = invervals_mids[entry_group]
    return grouped


def print_correlation(x, y, k):
    r = pearson_coef(x, y)

    r_low, r_high = corr_conf_int(r)
    print(f'\nнесгруппированная выборка')
    print(f'r-Пирсона:{r}')
    print(f'интервал:[{r_low}; {r_high}]')

    x_group = group_data(x, k)
    y_group = group_data(y, k)
    r_group = pearson_coef(x_group, y_group)
    r_group_low, r_group_high = corr_conf_int(r_group)
    print('\nсгруппированная выборка:')
    print(f'r-Пирсона:{r_group}')
    print(f'интервал:[{r_group_low}; {r_group_high}]')


def conf_intervals_script():
    poisson_param = 6
    print(f'параметр (Пуассон) = {poisson_param}')
    poisson = np.random.default_rng().poisson(poisson_param, n)
    poission_high = np.mean(poisson) + u_half_alpha / n ** 0.5
    poission_low = np.mean(poisson) - u_half_alpha / n ** 0.5
    print(f'доверительный интервал (Пуассон):[{poission_low}; {poission_high}]\n')

    binom_param = 0.4
    print(f'параметр (Бернулли) = {binom_param}')
    binom = np.random.binomial(n=n, p=binom_param)
    m = np.sum(binom)
    binom_low = m / n - u_half_alpha * (m * (n - m) / n) ** 0.5 / n
    binom_high = m / n + u_half_alpha * (m * (n - m) / n) ** 0.5 / n
    print(f'доверительный интервал (Бернулли):[{binom_low}; {binom_high}]\n')

    exp_param = 7
    print(f'параметр (экспоненциональное) = {exp_param}')
    exp = np.random.default_rng().exponential(1 / exp_param, n)
    exp_low = (1 - u_half_alpha / n ** 0.5) / np.mean(exp)
    exp_high = (1 + u_half_alpha / n ** 0.5) / np.mean(exp)
    print(f'доверительный интервал (экспоненциальное):[{exp_low}; {exp_high}]\n')


def gauss_kernel(x, h, sample):
    return 1 / (n * h) * sum([stat.norm.pdf((x - x_i) / h) for x_i in sample])


def uniform_pdf(x, a, b):
    return 1 / (b - a) if a <= x <= b else 0


def uniform_kernel(x, h, sample):
    return 1 / (n * h) * sum([uniform_pdf((x - x_i) / h, -1, 1) for x_i in sample])


def kernel_script():
    gauss_exp = 9
    gauss_var = 14
    gauss = stat.norm.rvs(loc=gauss_exp, scale=gauss_var ** 0.5, size=n)

    h = 1.06 * np.std(gauss) / n ** 0.2
    grid = np.linspace(start=gauss_exp - 2 * gauss_var ** 0.5,
                       stop=gauss_exp + 2 * gauss_var ** 0.5,
                       num=170)
    actual_distr = np.array([stat.norm.pdf(x, loc=gauss_exp, scale=gauss_var ** 0.5) for x in grid])
    gauss_kernel_est = np.array([gauss_kernel(x, h, gauss) for x in grid])
    uniform_kernel_est = np.array([uniform_kernel(x, h, gauss) for x in grid])
    plt.plot(grid, actual_distr, '-', grid, gauss_kernel_est, '-.', grid, uniform_kernel_est, '--')
    plt.title("Распределение Гаусса N(9, 14)")
    plt.legend(['действительная плотность', 'ядро (Гаусс)', 'ядро (равномерное)'])
    plt.show()

    a = 5
    b = 7
    uniform = np.random.default_rng().uniform(low=a, high=b, size=n)

    grid = np.linspace(start=a - 1, stop=b + 1, num=130)
    actual_distr = np.array([uniform_pdf(x, a, b) for x in grid])
    uniform_kernel_est = np.array([uniform_kernel(x, h, uniform) for x in grid])
    gauss_kernel_est = np.array([gauss_kernel(x, h, uniform) for x in grid])
    plt.plot(grid, actual_distr, '-', grid, uniform_kernel_est, '--', grid, gauss_kernel_est, '-.')
    plt.title("равномерное распределение U([5, 7])")
    plt.legend(['действительная плотность', 'ядро (Гаусс)', 'ядро (равномерное)'])
    plt.show()


def linear_regression_script():
    x = stat.norm.rvs(loc=0, scale=1, size=n)
    x_group = group_data(x, k)
    gauss_noise = stat.norm.rvs(loc=0, scale=1, size=n)

    y_gauss = 2 + 5 * x + gauss_noise * 0.1
    regr_gauss = stat.linregress(x, y=y_gauss)

    slope = pearson_coef(x, y_gauss) * np.std(y_gauss) / np.std(x)
    intercept = np.mean(y_gauss) - slope * np.mean(x)

    plt.plot(x, y_gauss, '.')
    plt.plot(x, regr_gauss.intercept + regr_gauss.slope * x, '.')
    plt.legend(['данные с шумом (Гаусс)', 'регрессия'])
    plt.show()
    plt.plot(x, y_gauss - regr_gauss.intercept - regr_gauss.slope * x, '.')
    plt.axhline(0, color='red')
    plt.title("невязка (Гаусс, обычная регрессия)")
    plt.show()
    print_correlation(x, y_gauss, k)

    a = -3
    b = 3
    uniform_noise = np.random.default_rng().uniform(low=a, high=b, size=n)

    y_uniform = 2 + 5 * x + uniform_noise * 0.1
    regr_uniform = stat.linregress(x, y=y_uniform)

    plt.plot(x, y_uniform, '.')
    plt.plot(x, regr_uniform.intercept + regr_uniform.slope * x, '.')
    plt.legend(['данные с шумом (равномерное, обычная регрессия)', 'регрессия'])
    plt.show()
    plt.plot(x, y_uniform - regr_uniform.intercept - regr_uniform.slope * x, '.')
    plt.axhline(0, color='red')
    plt.title("невязка (равномерное)")
    plt.show()
    print_correlation(x, y_uniform, k)

    cauchy_noise = stat.cauchy.rvs(loc=0, scale=1, size=n)

    y_cauchy = 2 + 5 * x + cauchy_noise * 0.1
    regr_cauchy = stat.linregress(x, y=y_cauchy)

    plt.plot(x, y_cauchy, '.')
    plt.plot(x, regr_cauchy.intercept + regr_cauchy.slope * x, '.')
    plt.legend(['данные с шумом (Коши, обычная регрессия)', 'регрессия'])
    plt.show()
    plt.plot(x, y_cauchy - regr_cauchy.intercept - regr_cauchy.slope * x, '.')
    plt.axhline(0, color='red')
    plt.title("невязка (Коши)")
    plt.show()
    print_correlation(x, y_cauchy, k)


def multiple_regression_script():
    noise_var = 3

    x = stat.norm.rvs(loc=0, scale=1, size=n)
    x_powers = np.array([np.array([xi ** i for i in range(1, p + 1)]) for xi in x])
    y_gauss = 4 * np.sin(2 * x + 4) + stat.norm.rvs(loc=0, scale=noise_var ** 0.5, size=n)
    clf = linear_model.LinearRegression()
    clf.fit(x_powers, y_gauss)

    plt.plot(x, y_gauss, '.')
    plt.plot(x, x_powers @ clf.coef_ + clf.intercept_, '.')
    plt.legend(['данные с шумом (Гаусс)', 'полиномиальная регрессия'])
    plt.show()
    plt.plot(x, y_gauss - x_powers @ clf.coef_ - clf.intercept_, '.')
    plt.axhline(0, color='red')
    plt.title('невязка (Гаусс, полиномиальная регрессия)')
    plt.show()


if __name__ == '__main__':
    n = 200
    alpha = 0.05
    k = 8
    p = n // 10
    u_half_alpha = -stat.norm.ppf(alpha / 2)
    t_half_alpha = -stat.t.ppf(q=alpha / 2, df=n - 2)

    conf_intervals_script()
    kernel_script()
    linear_regression_script()
    multiple_regression_script()



