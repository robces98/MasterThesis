import statsmodels.tsa.api as smt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import math
from scipy import stats
from scipy import optimize
from statistics import mean
import numpy as np
from numpy import argsort
import pandas as pd
from pandas import Series
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import tesiLib as lib
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad

def OF(freq = 60, T = 2*60*60):
    '''
    :param freq: Frequency of trading. Must be either 60 or 30. Measured in seconds.
    :param T: Period in which the strategy is executed. Measured in seconds.
    :return: The function returns the buy and sell rates of order flow simulated following equation (3.29) in my thesis.
    uBuy, vector of length T/freq containing the rate of buy market orders at each moment in time for [0,T]
    uSell, vector of length T/freq containing the rate of sell market orders at each moment in time for [0,T]

    ###NOTES###
    - LBuy/LSell are the times within [0,T] at which the jumps happen
    - meanVBuy/meanVSell are the mean volume of buy/sell market orders
    - volumesBuy/volumesSell are the volumes of the simulated jumps LBuy/LSell
    - kappaBuy/kappaSell are the mean reverting coefficients as in (3.29)
    At each moment in T I extract a random value of the Poisson process( how many
    events in that time frame) based on the probability for that frequency
    '''
    lenght = int(T/freq)
    LBuy = np.random.poisson(freq*24/3600, lenght)  # delta+ -->24 hourly, 0.4 minutly
    LSell = np.random.poisson(freq*28/3600, lenght)  # delta- -->28 hourly, 0.46 minutly
    jumpsBuy = np.where(LBuy != 0)[0]
    jumpsSell = np.where(LSell != 0)[0]
    meanVBuy = 1.8232267397395947
    meanVSell = 1.5010693278257192
    volumesBuy = np.random.exponential(meanVBuy, len(jumpsBuy))
    volumesSell = np.random.exponential(meanVSell, len(jumpsSell))
    LBuy[jumpsBuy] = LBuy[jumpsBuy]*volumesBuy # My jumps with their own volumes
    LSell[jumpsSell] = LSell[jumpsSell]*volumesSell # My jumps with their own volumes
    uBuy = np.zeros(lenght)
    uBuy[0] = LBuy[0]
    uSell = np.zeros(lenght)
    uSell[0] = LSell[0]
    if freq == 60 :
        kappaBuy = 0.015038
        kappaSell = 0.013848
    elif freq == 30:
        kappaBuy = 0.029139
        kappaSell = 0.027272
    else:
        raise ValueError("Kappa values not found for this frequency of trading.")
    for i in range(1,lenght):
        uBuy[i] =(1-kappaBuy*freq)*uBuy[i-1] + LBuy[i]
        uSell[i] =(1-kappaSell*freq)*uSell[i-1] + LSell[i]
    uBuy = np.round(uBuy)
    uSell = np.round(uSell)
    return uBuy, uSell


def optimal_speed(freq = 60, T = 2*60*60, phi=9.9e-6, Q0=500):
    '''
    :param freq: Frequency of trading. Must be either 60 or 30. Measured in seconds.
    :param T: Period in which the strategy is executed. Measured in seconds.
    :param phi: Inventory penalty parameter representing the urgency of the trader.
    :param Q0: Initial inventory. > 0 for a sell strategy, < 0 for a buy strategy.
    :return: The function returns the results of a classic Almgren-Chriss and of a Cartea-Jaimungal optimal execution
    strategy. In particular,
    ac, vector of length T/freq containing the speed (Almgren-Chriss) of trading at each moment in time for [0,T]
    Q_ac, vector of length T/freq containing the inventory (Almgren-Chriss) at each moment in time for [0,T]
    cj, vector of length T/freq containing the speed (Cartea-Jaimungal) of trading at each moment in time for [0,T]
    Q_cj, vector of length T/freq containing the inventory (Cartea-Jaimungal) at each moment in time for [0,T]
    '''
    k = 0.1368
    b = 0.0240
    gamma = np.sqrt(phi / k) # the higher phi, the higher the urgency and the smaller the difference between ac and cj
    cj = np.zeros(int(T / freq))  # trading speed Cartea-Jaimungal strategy
    Q_cj = np.zeros(int(T / freq))  # inventory with Cartea-Jaimungal strategy
    Q_cj[0] = Q0
    ac = np.zeros(int(T / freq))  # trading speed Almgren-Chriss strategy
    Q_ac = np.zeros(int(T / freq))  # inventory with Almgren-Chriss strategy
    Q_ac[0] = Q0
    lambdaBuy = 24.0 / (3600 / freq)
    lambdaSell = 28.0 / (3600 / freq)
    etaBuy = 1.8232267397395
    etaSell = 1.5010693278257192
    if freq == 60:
        kappaBuy = 0.015038
        kappaSell = 0.013848
    elif freq == 30:
        kappaBuy = 0.029139
        kappaSell = 0.027272
    else:
        raise ValueError("Kappa values not found for this frequency of trading.")
    psiBuy = lambdaBuy * etaBuy / kappaBuy
    psiSell = lambdaSell * etaSell / kappaSell
    uBuy, uSell = OF(T, freq)
    for t in np.arange(0, int(T / freq)):
        l0 = (1 / gamma) * (math.cosh(gamma * (int(T / freq) - t)) - 1) / math.sinh(gamma * (int(T / freq) - t))
        l1Buy = 0.5 * ((math.exp(gamma * (int(T / freq) - t)) - math.exp(-kappaBuy * (int(T / freq) - t))) / (
                    kappaBuy + gamma) -
                       (math.exp(-gamma * (int(T / freq) - t)) - math.exp(-kappaBuy * (int(T / freq) - t))) / (
                                   kappaBuy - gamma)) / math.sinh(gamma * (int(T / freq) - t))
        l1Sell = 0.5 * ((math.exp(gamma * (int(T / freq) - t)) - math.exp(-kappaSell * (int(T / freq) - t))) / (
                    kappaSell + gamma) -
                        (math.exp(-gamma * (int(T / freq) - t)) - math.exp(-kappaSell * (int(T / freq) - t))) / (
                                    kappaSell - gamma)) / math.sinh(gamma * (int(T / freq) - t))
        if t != 0:
            Q_cj[t] = Q_cj[t - 1] - cj[t - 1]
            Q_ac[t] = Q_ac[t - 1] - ac[t - 1]
        ac[t] = (gamma * math.cosh(gamma * (int(T / freq) - t)) / math.sinh(gamma * (int(T / freq) - t))) * Q_ac[t]
        cj[t] = (gamma * math.cosh(gamma * (int(T / freq) - t)) / math.sinh(gamma * (int(T / freq) - t))) * Q_cj[t] - \
                (b / (2 * k)) * (l1Buy * (uBuy[t] - psiBuy) - l1Sell * (uSell[t] - psiSell) + l0 * (psiBuy - psiSell))
        if cj[t] <= 0:
            cj[t] = 0
        if Q_cj[t] <= 0:
            Q_cj[t] = 0
            cj[t] = 0
        if Q_ac[t] <= 0:
            Q_ac[t] = 0
            ac[t] = 0

    return cj, Q_cj, ac, Q_ac


def price_path(freq = 60, T = 2*60*60, phi = 9.9e-6, Q0 = 500):
    '''
    :param freq: Frequency of trading. Must be either 60 or 30. Measured in seconds.
    :param T: Period in which the strategy is executed. Measured in seconds.
    :param phi: Inventory penalty parameter representing the urgency of the trader.
    :param Q0: Initial inventory. > 0 for a sell strategy, < 0 for a buy strategy.
    :return: The function returns the path of the price of the traded asset according to (3.22) in my thesis. The latter
    is obtained for both a classic Almgren-Chriss and of a Cartea-Jaimungal execution strategy. Speeds and inventories of
    the two strategies are provided as well as a result.
    p_ac, vector of length T/freq containing the asset's price (Almgren-Chriss) at each moment in time for [0,T]
    ac, vector of length T/freq containing the speed (Almgren-Chriss) of trading at each moment in time for [0,T]
    Q_ac, vector of length T/freq containing the inventory (Almgren-Chriss) at each moment in time for [0,T]
    p_cj, vector of length T/freq containing the asset's price (Cartea-Jaimungal) at each moment in time for [0,T]
    cj, vector of length T/freq containing the speed (Cartea-Jaimungal) of trading at each moment in time for [0,T]
    Q_cj, vector of length T/freq containing the inventory (Cartea-Jaimungal) at each moment in time for [0,T]
    '''
    # SIMULATION OF PRICES
    uBuy, uSell = OF(T, freq)
    cj, Q_cj, ac, Q_ac = optimal_speed(freq=freq, T = T, phi = phi, Q0=Q0)
    k = 0.1368
    b = 0.0240
    spread = 0.537264
    ann_vol = 0.44950882
    seconds_in_a_year = 252*6*60*60
    sigma = ann_vol*np.sqrt(freq/seconds_in_a_year) #This is ann_vol * sqrt(interval of trading in years)
    p0 = 100
    p_cj = np.zeros(int(T/freq))  # vector of midprices with Cartea strategy
    p_ac = np.zeros(int(T/freq))  # vector of midprices with Almgren-Chriss strategy
    ex_cj = np.zeros(int(T/freq))  # vector of execution prices with Cartea strategy
    ex_ac = np.zeros(int(T/freq))  # vector of execution prices with Almgren-Chriss strategy
    p_cj[0] = p0
    p_ac[0] = p0
    for t in range(1, int(T/freq)):
        p_cj[t] = p_cj[t - 1] + b * (uBuy[t] - uSell[t] - cj[t]) + sigma * np.random.normal(0, 1)  # midprices computed with CJ strategy
        p_ac[t] = p_ac[t - 1] + b * (uBuy[t] - uSell[t] - ac[t]) + sigma * np.random.normal(0, 1)  # midprices computed with AC strategy
    for t in range(0, int(T/freq)):
        ex_cj[t] = p_cj[t] - (0.5 * spread + k * cj[t])  # execution prices computed with CJ strategy
        ex_ac[t] = p_ac[t] - (0.5 * spread + k * ac[t])  # execution prices computed with AC strategy

    return p_cj, p_ac, ex_cj, ex_ac, cj, Q_cj, ac, Q_ac


def cash(freq=60, T = 2*60*60, phi = 9.9e-6, Q0 = 500):
    '''
    :param freq: Frequency of trading. Must be either 60 or 30. Measured in seconds.
    :param T: Period in which the strategy is executed. Measured in seconds.
    :param phi: Inventory penalty parameter representing the urgency of the trader.
    :param Q0: Initial inventory. > 0 for a sell strategy, < 0 for a buy strategy.
    :return: The function returns the path of the cash of the trader for both AC and CJ. The initial value
    is assumed to be zero.
    X_ac, vector of length T/freq containing the cash (Almgren-Chriss) at each moment in time for [0,T]
    X_cj, vector of length T/freq containing the cash (Cartea-Jaimungal) at each moment in time for [0,T]
    '''
    p_cj, p_ac, ex_cj, ex_ac, cj, Q_cj, ac, Q_ac = price_path(freq=freq, T = T, phi=phi, Q0=Q0)
    X_cj = np.cumsum(cj*ex_cj)
    X_ac = np.cumsum(ac*ex_ac)
    return X_cj, X_ac


def relative_performance():
    '''
    :return: The function returns a measure of the relative performance of CJ with respect to AC measured as
    Saving per share (or per MWh in our case) in basis points.
    '''
    sim = 1000
    Q0 = 500
    X_cjs = np.zeros(sim)
    X_acs = np.zeros(sim)
    for i in range(sim) :
        X_cj, X_ac = cash(freq=60*1, T = 60*60, phi = 1e-4, Q0 = Q0)
        X_cjs[i] = X_cj[- 1]  # vectors of terminal cash
        X_acs[i] = X_ac[- 1]
    performance = 10000 * (X_cjs - X_acs) / X_acs
    performance = performance /(Q0) #standardization and making it per share
    plt.hist(performance, bins=30)
    plt.xlabel('Savings per MWh(bp)')
    plt.ylabel('Frequency ')
    plt.show()
    print(np.mean(performance))
    return performance