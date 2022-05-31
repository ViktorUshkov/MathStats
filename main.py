import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools


def report(data, n):
    print(f'Мат. ожидание: {np.mean(data)}')
    print(f'Дисперсия: {np.var(data)}')
    print(f'Медиана: {(data[99] + data[100]) / 2}')
    print(f'1/4 и 3/4 квантили: {np.quantile(data, 1 / 4)}, {np.quantile(data, 3 / 4)}')
    print(f'Исправленная дисперсия: {np.var(data) * n / (n - 1)}')
    print(f'Коэффициент ассиметрии: {stat.kstat(data, 3) / stat.kstat(data, 2) ** 1.5}')
    print(f'Коэффициент эксцесса: {stat.moment(data, 4) / stat.moment(data, 2) ** 2}')
    print(f'Коэффициент вариации: {np.var(data) ** 0.5 / np.mean(data)}')


def report_cauchy(data, x0):
    print(f'Медиана: {x0}')
    print(f'1/4 и 3/4 квантили: {np.quantile(data, 1 / 4)}, {np.quantile(data, 3 / 4)}')


def conf_interval_exp(data, alpha, var, n):
    point_expectation = np.mean(data)
    u_half_alpha = -stat.norm.ppf(alpha / 2)

    mu_lower = point_expectation - u_half_alpha * (var / n) ** 0.5
    mu_higher = point_expectation + u_half_alpha * (var / n) ** 0.5
    return mu_lower, mu_higher


def conf_interval_var(data, n, alpha):
    s2 = np.var(data) * n / (n - 1)
    df = n - 1
    chi2_half_alpha = stat.chi2.ppf(alpha / 2, df)
    chi2_one_minus_half_alpha = stat.chi2.ppf(1 - alpha / 2, df)

    sigma2_lower = (n - 1) * s2 / chi2_one_minus_half_alpha
    sigma2_higher = (n - 1) * s2 / chi2_half_alpha
    return sigma2_lower, sigma2_higher


def conf_interval_exp_student(data, n, alpha):
    point_expectation = np.mean(data)
    s2 = np.std(data) * n / (n - 1)
    t_half_alpha = -stat.t.ppf(q=alpha / 2, df=n - 1)

    mu_lower = point_expectation - t_half_alpha * (s2 / n) ** 0.5
    mu_higher = point_expectation + t_half_alpha * (s2 / n) ** 0.5
    return mu_lower, mu_higher


def in_range(x, interval):
    a, b = interval
    if a <= x <= b:
        return True
    return False


def pairwise(iterable):
    """s -> (s0, s1), (s1, s2), (s2, s3), ..."""
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

    _, counts = np.unique(grouped, return_counts=True)
    scaled_counts = [x for x in counts]
    plt.bar(invervals_mids, [x for x in scaled_counts], width=bounds[1] - bounds[0], edgecolor='black')
    plt.title("сгрупированная выборка Гаусса")
    plt.show()
    return grouped


def distributions_script():
    n = 200
    a = 5
    b = 7

    print('Гаусс')
    gauss_exp = 17
    gauss_var = 22
    gauss_sample = stat.norm.rvs(loc=gauss_exp, scale=gauss_var ** 0.5, size=n)
    plt.hist(gauss_sample)
    plt.title(f'Распределение Гаусса $n = 200$, $\mu = 17$, $\sigma^2 = 22$')
    report(sorted(gauss_sample), n)
    plt.show()
    print('\n')

    print('Пуассон')
    poisson_param = 9
    poisson_sample = np.random.default_rng().poisson(poisson_param, n)
    plt.hist(poisson_sample)
    plt.title(f'Распределение Пуассона $n = {n}$, $\lambda = {poisson_param}$')
    report(sorted(poisson_sample), n)
    plt.show()
    print('\n')

    print('Экспоненциальное')
    exp_param = 3
    exp_sample = np.random.default_rng().exponential(1 / exp_param, n)
    plt.hist(exp_sample)
    plt.title(f'Экспоненциальное распределение $n = {n}$, $\lambda = {exp_param}$')
    report(sorted(exp_sample), n)
    plt.show()
    print('\n')

    print('Коши')
    cauchy_shift = 0
    cauchy_scale = 2
    cauchy_sample = stat.cauchy.rvs(loc=cauchy_shift, scale=cauchy_scale, size=n)
    plt.hist(cauchy_sample)
    plt.title(f'Распределение Коши $n = {n}$, $x_0 = {cauchy_shift}$, $\gamma = {cauchy_scale}$')
    report_cauchy(cauchy_sample, cauchy_shift)
    plt.show()
    print('\n')

    print('Равномерное')
    uniform_sample = np.random.default_rng().uniform(low=a, high=b, size=n)
    plt.hist(uniform_sample)
    plt.title(f'Равномерное распределение $n = {n}$, $a = {a}$, $b = {b}$')
    report(sorted(uniform_sample), n)
    plt.show()
    print('\n')


def intervals_script():
    n = 200
    alpha = 0.05
    gauss_exp = 4
    gauss_var = 9
    gauss_sample = stat.norm.rvs(loc=gauss_exp, scale=gauss_var ** 0.5, size=n)
    mu_lower, mu_higher = conf_interval_exp(gauss_sample, alpha, gauss_var, n)
    print(f"погрешность: {alpha}")
    print(f"точечная оценка матожидания: {np.mean(gauss_sample)}")
    print("доверительный интервал с известной сигма")
    print(f'интервал: {mu_lower}, {mu_higher}\n')
    plt.hist(gauss_sample)
    plt.title("несгрупированная выборка Гаусса")
    plt.show()

    k = 8
    print(f'сгруппированная выборка Гаусса, k = {k}')
    grouped = group_data(gauss_sample, k)

    mu_lower_student, mu_higher_student = conf_interval_exp_student(gauss_sample, k, alpha)
    print("матожидание выборки Гаусса без известной дисперсии (Стьюдент)")
    print(f'интервал: {mu_lower_student}, {mu_higher_student}\n')

    mu_lower_group, mu_higher_group = conf_interval_exp(grouped, alpha, gauss_var, k)
    print(f"точечная оценка матожидания: {np.mean(grouped)}")
    print(f'интервал: {mu_lower_group}, {mu_higher_group}\n')

    sigma2_lower_group, sigma2_higher_group = conf_interval_var(grouped, k, alpha)
    print(f"точечная оценка дисперсии: {np.var(grouped)}")
    print(f'интервал: {sigma2_lower_group}, {sigma2_higher_group}\n')


if __name__ == "__main__":
    distributions_script()
    intervals_script()
