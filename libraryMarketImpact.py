import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.svar_model import SVAR
import math
from scipy import stats, optimize
import numpy as np
from numpy import argsort
import pandas as pd
from pandas import Series
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from datetime import datetime
from datetime import timedelta
import sklearn
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
import openpyxl
from numpy.linalg import matrix_power as power
from numpy.linalg import multi_dot, inv
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad

def millisecOB(t, day, month):
    '''
    :param t: Plug in dataOB.Datetime.values
    :param day: Plug in the day as an integer
    :param month: Insert the number of the month of the date (9,10,11)
    :return: Vector of time in milliseconds
    The function will transform the vector of time in the format 'format_data'
    to a vector of time in milliseconds.
    '''
    format_data = "%Y-%m-%d %H:%M:%S.%f"
    date = pd.to_datetime(t, format=format_data)
    milliseconds = (date - datetime(2021, month, day)) // timedelta(milliseconds=1)
    return np.array(milliseconds)


def millisecM(t, day, month):
    '''
    :param t: Plug in np.array(list(dataM.DateTime.values))
    :param day: Plug in the day as an integer
    :param month: Insert the number of the month of the date (9,10,11)
    :return:Vector of time in milliseconds
    The function will transform the vector of time in the format specified
    to a vector of time in milliseconds.
    '''
    milliseconds = np.zeros(len(t))
    f_yes = np.char.count(t, '.')
    date = pd.to_datetime(t[np.where(f_yes == 1)], format="%Y-%m-%dT%H:%M:%S.%fZ")
    milliseconds[np.where(f_yes == 1)] = (date - datetime(2021, month, day)) // timedelta(milliseconds=1)
    date = pd.to_datetime(t[np.where(f_yes == 0)], format="%Y-%m-%dT%H:%M:%SZ")
    milliseconds[np.where(f_yes == 0)] = (date - datetime(2021, month, day)) // timedelta(milliseconds=1)
    return milliseconds


###################################################################################################

def read_data(day, month):
    '''
    :param day: Plug in day as a string
    :return: dataOB, dataM
    The function return dataM and data OB where direction of trades are already corrected
    '''
    m = str()
    if month == 9:
        m = 'settembre'
    elif month == 10:
        m = 'ottobre'
    elif month == 11:
        m = 'novembre'
    else:
        raise ValueError
    ORDERBOOK = 'LOB_' + m + '21\LOB_Germany_Baseload_2022_2021' +f"{month:02}"+ day + '.csv'
    MSGBOOK = 'trade_' + m + r'2021\trades_Germany_Baseload_2022_2021' + f"{month:02}" + day + '.csv'
    dataM = pd.read_csv(MSGBOOK)  # columns are already named
    dataM = dataM.loc[(millisecM(np.array(list(dataM.DateTime.values)), int(day), month) > 10 * 60 * 60 * 1000) & (
            millisecM(np.array(list(dataM.DateTime.values)), int(day), month) < 16 * 60 * 60 * 1000)]
    dataOB = pd.read_csv(ORDERBOOK)  # columns are already named
    dataOB = dataOB.loc[(millisecOB(dataOB.Datetime.values, int(day), month) >= 9.9 * 60 * 60 * 1000) & (
            millisecOB(dataOB.Datetime.values, int(day), month) <= 16.1 * 60 * 60 * 1000)]
    dataOB = dataOB.rename(columns={'Unnamed: 0': 'count'})
    Signs = estim_sign(dataM, dataOB, day, month)
    Signs = Signs.rename(columns={0: 'time', 1: 'price', 2: 'est_sign'})
    dataM = fixALL(dataM, Signs)
    return dataOB, dataM

#######################################################
#PERMANENT AND TEMPORARY IMPACT ESTIMATION METHODS
#######################################################

def perm_Cart_Jaim(dataM, dataOB, day, month, freq=5 * 60 * 1000):
    '''
    :param dataM: Data set containing trades
    :param dataOB: Data set containing the order book
    :param day: Plug in the day as an integer
    :param freq: Frequency at which you want to sample (millisec). Standard is 5 minutes
    :return: Estimation of permanent impac.
    Estimation of permanent is performed as suggested by Cartea and Jaimungal.
    A regression of the difference in midprice on the net order flow return the coefficient b.
    '''
    noint = int((16 - 6) * 60 * 60 * 1000 / freq)
    bound = np.zeros(noint + 1)  # Variable for interval bound
    for j in range(0, noint + 1):
        bound[j] = 6 * 60 * 60 * 1000 + j * freq
    # (NET ORDER FLOW)
    n_buy = np.zeros(noint)  # # of buy MO
    n_sell = np.zeros(noint)  # # of sell MO
    NOF = np.zeros(noint)
    d_midprice = np.zeros(noint)
    M = np.array(list(dataM.DateTime.values))
    OB = dataOB.Datetime.values
    for l in range(0, noint):
        n_buy[l] = sum(dataM.loc[(millisecM(M, day, month) > bound[l]) &
                                 (millisecM(M, day, month) < bound[l + 1]) & (
                                         dataM.AggressorAction == 'Buy')].Volume.values)  # volume of buy MO
        n_sell[l] = sum(dataM.loc[(millisecM(M, day, month) > bound[l]) &
                                  (millisecM(M, day, month) < bound[l + 1]) & (
                                          dataM.AggressorAction == 'Sell')].Volume.values)  # volume of sell MO
        NOF[l] = - n_sell[l] + n_buy[l]

    for i in range(0, noint):
        try:
            first_mid = 0.5 * (dataOB.loc[(millisecOB(OB, day, month) >= bound[i]) &
                                          (millisecOB(OB, day, month) < bound[i + 1])].AskPrice_0.values[0] +
                               dataOB.loc[(millisecOB(OB, day, month) >= bound[i]) &
                                          (millisecOB(OB, day, month) < bound[i + 1])].BidPrice_0.values[
                                   0])  # first price displayed after that time
            last_mid = 0.5 * (dataOB.loc[(millisecOB(OB, day, month) >= bound[i]) &
                                         (millisecOB(OB, day, month) < bound[i + 1])].AskPrice_0.values[-1] +
                              dataOB.loc[(millisecOB(OB, day, month) >= bound[i]) &
                                         (millisecOB(OB, day, month) < bound[i + 1])].BidPrice_0.values[
                                  -1])  # last price displayed after that time
            d_midprice[i] = last_mid - first_mid
        except:
            pass

    # estimation of permanent impact

    cond = np.where((NOF != 0) & (d_midprice != 0))
    NOF = NOF[cond[0]]
    d_midprice = d_midprice[cond[0]]
    regr = {'d_midprice': d_midprice, 'NOF': NOF}
    df = pd.DataFrame(regr)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    permImp = sm.OLS(df['d_midprice'], df['NOF']).fit()  # OLS regression
    b = permImp.params
    r = permImp.rsquared
    # err = permImp.bse
    return r


def perm_Has_Car_Jaim(dataOB, dataM, day, month):
    '''
    :param dataOB: Data set containing the order book
    :param dataM: Data set containing trades
    :param day: Plug in the day as an integer
    :return: Estimation of permanent impact b
     The function estimate a SVAR model as proposed in Hasbrouck(1991).
    '''
    xt = np.where(dataM['AggressorAction'] == 'Buy', 1, -1) * dataM['Volume']
    M = millisecM(np.array(list(dataM.DateTime.values)), int(day), month)
    OB = dataOB.Datetime.values
    midprice = np.zeros(len(M) + 1)
    ask = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].AskPrice_0.values[-1]
    bid = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].BidPrice_0.values[-1]
    midprice[0] = 0.5 * (ask + bid)
    for l in range(0, len(M)):
        try:
            ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                             (millisecOB(OB, int(day), month) < M[l + 1])].AskPrice_0.values[0]
            bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                             (millisecOB(OB, int(day), month) < M[l + 1])].BidPrice_0.values[0]
            midprice[l + 1] = 0.5 * (ask + bid)
        except:
            ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].AskPrice_0.values[0]
            bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].BidPrice_0.values[0]
            midprice[l + 1] = 0.5 * (ask + bid)
    rt = np.diff(midprice)
    df = pd.DataFrame({'rt': rt, 'xt': xt})
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    A = np.asarray([[1, 'E'], [0, 1]])
    # B = np.asarray([['E', 0], [0, 'E']])
    model = SVAR(df, svar_type='A', A=A)  # , B=B)
    minCr = 99999
    l = 0
    for i in np.arange(1,15):
        result = model.fit(maxlags=i, trend='n')
        if result.aic <= minCr:
            minCr = result.aic
            l = i
    svar_results = model.fit(maxlags=l, trend='n')  # let's go with a one order SVAR
    svar_results.k_exog_user = 0
    A_est = svar_results.A
    # B_est = svar_results.B
    coef = svar_results.coefs
    coefs = np.array(
        [np.dot(A_est, coef[i]) for i in range(l)])  # I multiply all the regression by A to get indep errors
    svar_results.summary()
    sumB = np.sum([coefs[i, 0, 1] for i in range(l)])
    sumA = np.sum([coefs[i, 0, 0] for i in range(l)])
    b = (-A_est[0][1] + sumB) / (1 - sumA)  # (bo + B) / (1-A)
    r2 = skm.r2_score(svar_results.fittedvalues[:, 0] + svar_results.resid[:, 0],
                      svar_results.fittedvalues[:, 0])
    return b,r2, l


def temp_Cart_Jaim(dataOB):
    '''
    :param dataOB: Data set containing the order book
    :return: Estimation of temporary impact folowwing Cartea and Jaimungal method.
    '''
    # MATRIX OF CUMULATIVE VOLUMES
    dataOB.reset_index(drop=True, inplace=True)
    bids_cum = dataOB[['BidVolume_' + str(i) for i in range(10)]].apply(np.cumsum, axis=1, raw=True)
    bids_cum.dropna(axis=0, how='all', inplace=True)
    bids_cum.reset_index(drop=True, inplace=True)
    volumes = bids_cum.apply(lambda row: pd.Series(np.linspace(row[0], row[row.isnull() == False][-1], 10, dtype=int)), axis=1)
    # MATRIX OF INDEXES OF EXECUTION
    ind_ex = bids_cum.apply(lambda row: pd.Series(np.searchsorted(row, volumes.iloc[row.name], side='left')), axis=1)
    ind_ex = pd.DataFrame(ind_ex)
    tot = pd.DataFrame()
    for i in range(10):
        tot[i] = dataOB['BidVolume_' + str(i)] * dataOB['BidPrice_' + str(i)]
    # THIS IS THE MATRIX OF PRICES I WOULD PAY BY DEPLETING THE LOB UNTIL THAT POINT
    tot = tot.apply(np.cumsum, axis=1)

    # THIS IS THE VECTOR OF DELTA PRICES (BEST - EXECUTED) FOR EACH VOLUME. IF I REACH LEVEL 10 IT RETURNS NaN
    def exp(j, ex, row, v):
        best = dataOB.iloc[row.name].BidPrice_0
        index = row[j]
        if index == 0:
            return 0  # min(v[k], dataOB.iloc[row.name].BidVolume_0)*dataOB.iloc[row.name].BidPrice_0
        elif index != 10:
            executed = ex[index - 1] + min(v[j] - bids_cum.iloc[row.name]['BidVolume_' + str(index - 1)],
                                           dataOB.iloc[row.name]['BidVolume_' + str(index)]) * dataOB.iloc[row.name][
                           'BidPrice_' + str(index)]
            return best - (executed / v[j])
        else:
            return 'NaN'

    # FOR EACH ROW I RETURN THE SERIES OF DELTA PRICES
    def fun(row):
        ex = tot.iloc[row.name].values
        v = volumes.iloc[row.name].dropna()
        t = [exp(j, ex, row, v) for j in range(len(v))]
        return pd.Series(t)

    # DATAFRAME OF DELTA PRICES FOR EACH VOLUME
    delta_pr = ind_ex.apply(lambda row: fun(row), axis=1)

    def K(row):
        r = pd.Series(row, dtype='float64').dropna()
        v = volumes.iloc[row.name].dropna()
        k = stats.linregress(v, r).slope
        return k

    ks = delta_pr.apply(lambda row: K(row), axis=1)
    ks = ks[np.logical_not(np.isnan(ks))]
    ks = ks[(np.abs(stats.zscore(ks)) < 3)]  # outliers deletion
    return np.mean(ks)


def kappa_dynamics(dataOB):
    '''
    :param dataOB: Data set containing the order book
    :return: Behaviour of k throughout the day.
    Estimation of k is performed as in Cartea and Jaimungal. This function is only instrumental
    towards temp_Glas().
    '''
    # MATRIX OF CUMULATIVE VOLUMES
    bids_cum = dataOB[['BidVolume_' + str(i) for i in range(10)]].apply(np.cumsum, axis=1, raw=True)
    bids_cum.dropna(axis=0, how='all', inplace=True)
    bids_cum.reset_index(drop=True, inplace=True)
    volumes = bids_cum.apply(lambda row: pd.Series(np.arange(1, 1+min(20, int(row[row.isnull() == False][-1])), dtype=int)), axis=1)
    # MATRIX OF INDEXES OF EXECUTION
    ind_ex = bids_cum.apply(lambda row: pd.Series(np.searchsorted(row, volumes.iloc[row.name], side='left')), axis=1)
    ind_ex = pd.DataFrame(ind_ex)
    tot = pd.DataFrame()
    for i in range(10):
        tot[i] = dataOB['BidVolume_' + str(i)] * dataOB['BidPrice_' + str(i)]
    # THIS IS THE MATRIX OF PRICES I WOULD PAY BY DEPLETING THE LOB UNTIL THAT POINT
    tot = tot.apply(np.cumsum, axis=1)

    # THIS IS THE VECTOR OF DELTA PRICES (BEST - EXECUTED) FOR EACH VOLUME. IF I REACH LEVEL 10 IT RETURNS NaN
    def exp(j, ex, row, v):
        best = dataOB.iloc[row.name].BidPrice_0
        index = row[j]
        if index == 0:
            return 0  # min(v[k], dataOB.iloc[row.name].BidVolume_0)*dataOB.iloc[row.name].BidPrice_0
        else:
            executed = ex[index - 1] + min(v[j] - bids_cum.iloc[row.name]['BidVolume_' + str(index - 1)],
                                           dataOB.iloc[row.name]['BidVolume_' + str(index)]) * dataOB.iloc[row.name][
                           'BidPrice_' + str(index)]
            return best - (executed / v[j])

    # FOR EACH ROW I RETURN THE SERIES OF DELTA PRICES
    def fun(row):
        ex = tot.iloc[row.name].values
        v = volumes.iloc[row.name].dropna()
        t = [exp(j, ex, row, v) for j in range(len(v))]
        return pd.Series(t)

    # DATAFRAME OF DELTA PRICES FOR EACH VOLUME
    delta_pr = ind_ex.apply(lambda row: fun(row), axis=1)

    def K(row):
        r = pd.Series(row, dtype='float64').dropna()
        v = volumes.iloc[row.name].dropna()
        try:
            k = stats.linregress(v, r).slope
        except:
            k = 0
        return k

    ks = delta_pr.apply(lambda row: K(row), axis=1)
    return ks


def temp_Glas(days, month):
    '''
    :param days: List of days in the data set
    :param month: The month days are referring to
    :return: Estimation of temporary impact based on Glas et al. (2020)
    The function build a new artificial LOB set for each day and estimate the behaviour
    of k throughout the day. Than, based on values from the whole month it fits a curve
    to approximate the dynamic of k and it takes the integral over intervals of 1 minutes.
    '''
    m = str()
    if month == 9:
        m = 'settembre'
    elif month == 10:
        m = 'ottobre'
    elif month == 11:
        m = 'novembre'
    else:
        raise ValueError
    tot = pd.DataFrame()
    freq = 5 * 60 * 1000  # 5minute
    noint = int((16 - 10) * 60 * 60 * 1000 / freq)
    bound = np.zeros(noint + 1)  # Variable for interval bound
    for j in range(0, noint + 1):
        bound[j] = 10 * 60 * 60 * 1000 + j * freq
    for day in days:
        ORDERBOOK = 'LOB_' + m + '21\LOB_Germany_Baseload_2022_2021' + f"{month:02}" + day + '.csv'
        dataOB = pd.read_csv(ORDERBOOK)  # columns are already named
        OB = dataOB.Datetime.values
        dataOB = dataOB.rename(columns={'Unnamed: 0': 'count'})
        dataOB = dataOB.loc[(millisecOB(OB, int(day), month) >= 9.9 * 60 * 60 * 1000) & (
                    millisecOB(OB, int(day), month) < 16.1 * 60 * 60 * 1000)]
        # dataOB.reset_index(drop=True, inplace=True)
        OB = dataOB.Datetime.values
        deltaT = np.diff((millisecOB(OB, int(day), month)))
        dataOB = dataOB[:-1]
        OB = OB[:-1]
        #freq = 5 * 60 * 1000  # 5minute
        dataOB['weights'] = deltaT / freq
        #noint = int((16 - 10) * 60 * 60 * 1000 / freq)
        #bound = np.zeros(noint + 1)  # Variable for interval bound
        for j in range(0, noint + 1):
            bound[j] = 10 * 60 * 60 * 1000 + j * freq
        twDataFrame = pd.DataFrame()
        for i in range(0, noint):
            try:
                ob = dataOB.loc[(millisecOB(OB, int(day), month) >= bound[i]) &
                                (millisecOB(OB, int(day), month) < bound[i + 1])]  # .AskPrice_0.values[0]
                Po = dataOB.loc[ob.index[0] - 1]
                wo = (millisecOB(ob.Datetime.values, int(day), month)[0] - bound[
                    i]) / freq  # first slice of time (is not a full interval)
                wN = (bound[i + 1] - millisecOB(ob.Datetime.values, int(day), month)[
                    -1]) / freq  # last slice of time (is not a full interval)
                fl = Po.drop(['count', 'key', 'Datetime', 'weights']) * wo + wN * ob.drop(
                    ['count',  # here we compute the first and last slice to be added to the weighted sum
                     'key', 'Datetime', 'weights'], axis=1).tail(1).squeeze()
                tw = fl + ob.drop(['count', 'key', 'Datetime', 'weights'],
                                  # this is the time-weighted LOB row (must be ordered)
                                  axis=1)[:-1].multiply(ob.weights[:-1], axis='rows', level=None,
                                                        fill_value=None).sum(skipna=False)
                askP = tw[['AskPrice_' + str(i) for i in range(10)]]
                askV = tw[['AskVolume_' + str(i) for i in range(10)]]
                order = askP.values.argsort()
                askP = askP.replace(askP.values, askP.values[order])
                askV = askV.replace(askV.values, askV.values[order])
                bidP = tw[['BidPrice_' + str(i) for i in range(10)]]
                bidV = tw[['BidVolume_' + str(i) for i in range(10)]]
                order = bidP[bidP.isnull() == False].argsort()[::-1]
                bidP = bidP.replace(bidP[bidP.isnull() == False].values, bidP[bidP.isnull() == False].values[order])
                bidV = bidV.replace(bidV[bidV.isnull() == False].values, bidV[bidV.isnull() == False].values[order])
                df = pd.concat([askP, askV, bidP, bidV])
                twDataFrame = pd.concat([twDataFrame, df], axis=1)
            except:
                pass
        twDataFrame = twDataFrame.transpose().reset_index(drop=True)
        k = kappa_dynamics(twDataFrame)
        tot = pd.concat([tot, k], axis=1)
    tot.columns = days
    x = []
    for i in range(len(tot)):
        x = np.append(x, [bound[i] / 1000] * len(tot.columns))
    y = []
    for i in tot.columns:
        y = np.append(y, np.array(tot[str(i)]))
    regr = {'x': x, 'y': y}
    df = pd.DataFrame(regr)
    df = df.dropna()
    df = df[df.y > 0]  # delete negative spread
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    x = np.array(df.x)
    y = np.array(df.y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    inds = x.ravel().argsort()  # Sort x values and get index
    x = x.ravel()[inds].reshape(-1, 1)
    y = y[inds]  # Sort y according to x sorted index
    minAic = 999999
    order = 0
    for i in range(15):
        polynomial_features = PolynomialFeatures(degree=i)
        xp = polynomial_features.fit_transform(x)
        model = sm.OLS(y, xp).fit()
        if model.aic < minAic:
            minAic = model.aic
            order = i
    polynomial_features = PolynomialFeatures(degree=order)
    xp = polynomial_features.fit_transform(x)
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)

    fr = 1  # 1 for seconds, 60 for minutes, 3600 for hours --> interval for the integral
    kappas = []
    def fun(x):
        y = 0
        for i in range(len(model.params)):
            y = y + model.params[i] * (x ** i)
        return y
    for i in np.arange(10 * 60 * 60, 16 * 60 * 60, fr):
        res, err = quad(fun, i, i + fr)[0:2]
        kappas.append(res)
    return np.mean(kappas)



def companion_matrix(coef):
    l = len(coef)*2
    cm = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            if i == 0 or i==1:
                step = 1
                if j%2==0:
                    step=0
                cm[i][j] = coef[int(0.5*j)][i][step]
            else:
                if i == j+2:
                    cm[i][j] = 1
                else:
                    cm[i][j] = 0
    return cm


def impact_Has_Alm(dataM, dataOB, day, month):
    '''
    :param dataM: Data set containing trades
    :param dataOB: Data set containing the order book
    :param day: Plug in day as an integer
    :return: b, k. Estimation of permanent and temporary impact coefficients.
    This function estimate a VAR(p) model for the data according to Hasbrouck (1991). Then it
    estimate the impacts simulating a metaorder and following Almgren(2005)
    '''
    day = int(day)
    xt = np.where(dataM['AggressorAction'] == 'Buy', 1, -1) * dataM['Volume']
    M = millisecM(np.array(list(dataM.DateTime.values)), int(day), month)
    OB = dataOB.Datetime.values
    midprice = np.zeros(len(M) + 1)
    ask = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].AskPrice_0.values[-1]
    bid = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].BidPrice_0.values[-1]
    midprice[0] = 0.5 * (ask + bid)
    for l in range(0, len(M)):
        try:
            ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                             (millisecOB(OB, int(day), month) < M[l + 1])].AskPrice_0.values[0]
            bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                             (millisecOB(OB, int(day), month) < M[l + 1])].BidPrice_0.values[0]
            midprice[l + 1] = 0.5 * (ask + bid)
        except:
            ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].AskPrice_0.values[0]
            bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].BidPrice_0.values[0]
            midprice[l + 1] = 0.5 * (ask + bid)
    rt = np.diff(midprice)
    df = pd.DataFrame({'rt': rt, 'xt': xt})
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    A = np.asarray([[1, 'E'], [0, 1]])
    B = np.asarray([['E', 0], [0, 'E']])
    model = SVAR(df, svar_type='A', A=A)  # , B = B)
    minCr = 99999
    l = 0
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        result = model.fit(maxlags=i, trend='n')
        if result.aic <= minCr:
            minCr = result.aic
            l = i
            # print('Lag Order =', i) print('AIC : ', result.aic) print('BIC : ', result.bic)  print('FPE : ', result.fpe) print('HQIC: ', result.hqic, '\n')
    svar_results = model.fit(maxlags=l, trend='n')  # let's go with a one order SVAR
    # var_results = VAR(df).fit(l, trend='n')
    svar_results.k_exog_user = 0
    A_est = svar_results.A
    B_est = svar_results.B
    coef = svar_results.coefs
    su = svar_results.sigma_u
    tau = np.mean(np.diff(M)) / 1000  # inter-trade avg time (seconds)
    ttest = svar_results.tvalues
    ttest = np.reshape(ttest, (len(coef), 2, 2))
    for i in range(len(ttest)):
        ttest[i][0][1], ttest[i][1][0] = ttest[i][1][0], ttest[i][0][1]
    coef = np.where(abs(ttest) <= 2, 0, coef)
    if l == 1:
        return impact_VARone_AN(tau, A_est, coef)
    else:
        return impact_VARp_AN(tau, A_est, coef)


def impact_VARone_AN(tau, A_est, coef):
    '''
    ANALYTICAL VERSION
    :param M: Time of dataM file
    :param A_est: Matrix A of the sVAR model
    :param coef: Coefficients of the VAR model
    :return: Estimation of permanent and temporary impact according to Almgren(2005)
    This function is recalled through 'estimate_impact_Has_Al()' and 'temp_wholemonth_Has_Alm()'
    whenever data are modelled as a VAR(1).
    '''
    #tau = np.mean(np.diff(M)) / 1000  # inter-trade avg time (seconds)
    d = -0.2
    #dailySigma =
    T = 100
    tpost = int(30 * 60 / tau)
    vtAN = np.array([[0, 0]])  # length T+tpost+1
    # vtAN = vtAN.transpose()
    # vtAN.columns = ['rt','xt']
    ptAN = np.zeros(T + tpost + 1)  ## length T+tpost+1
    po = 100
    ptAN[0] = po
    e1 = np.array([1, 0])
    e2 = np.dot(inv(A_est), np.array([0, 1]))
    I = np.identity(2)
    for i in range(1, T + tpost + 1):  # T+TPOST LOOPS
        # err = multi_dot([np.random.normal(size=2),A_est, su, A_est.transpose()])
        if i <= T:
            vi = multi_dot([inv(I - coef[0]), I - power(coef[0], i), d * e2])
            ptAN[i] = po + d * multi_dot([e1.transpose(), inv(I - coef[0]),
                                          i * I - multi_dot([coef[0], inv(I - coef[0]), I - power(coef[0], i)]), e2])
        else:
            vi = np.dot(power(coef[0], i - T), vtAN[T])  # ???

        vtAN = np.append(vtAN, [vi], axis=0)

    JAN = (d / (T * po)) * multi_dot([e1.transpose(), inv(I - coef[0]),
                                      0.5 * T * (T + 1) * I - multi_dot([coef[0], inv(I - coef[0]),
                                                                         T * I - multi_dot([coef[0], inv(I - coef[0]),
                                                                                            I - power(coef[0], T)])]),
                                      e2])
    Ppost = ptAN[T] + multi_dot([e1.transpose(), coef[0],
                                 inv(I - coef[0]), I - power(coef[0], tpost), vtAN[T]])
    IAN = (Ppost - po) / po
    k = (JAN - 0.5 * IAN) * po * tau / d
    b = IAN * po * tau / (T * d)
    return b,k


def impact_VARp_AN(tau, A_est, coef):
    '''
    ANALYTICAL VERSION
    :param M: Time of dataM file
    :param A_est: Matrix A of the sVAR model
    :param coef: Coefficients of the VAR model
    :return: Estimation of permanent and temporary impact according to Almgren(2005)
    This function is recalled through 'estimate_impact_Has_Al()' and 'temp_wholemonth_Has_Alm()'
     whenever data are modelled as a VAR(p)
    '''
    gamma = companion_matrix(coef)
    #tau = np.mean(np.diff(M)) / 1000  # inter-trade avg time (seconds)
    d = -0.2
    #dailySigma =
    T = 100
    tpost = int(30 * 60 / tau)
    vtAN = np.zeros((len(coef) + T + tpost) * 2)
    l = len(coef)
    e1 = np.zeros(len(coef) * 2)
    e1[0] = 1
    e2 = np.dot(inv(A_est), np.array([0, 1]))
    metaOr = np.zeros(l * 2)
    metaOr[0], metaOr[1] = e2[0], e2[1]
    I = np.identity(len(gamma))
    for t in range(T + 1):
        vtAN[2 * t: 2 * t + 2 * l] = d * multi_dot([inv(I - gamma), I - power(gamma, t), metaOr])[::-1]
    for t in range(1, tpost + 1):
        vtAN[2 * (t + T): 2 * (t + T) + 2 * l] = np.dot(power(gamma, t), vtAN[2 * T: 2 * (T + l)][::-1])[::-1]
    rtAN = vtAN[1::2]
    xtAN = vtAN[::2]
    sumR = 0.5 * I * T * (T + 1) - multi_dot([gamma, inv(I - gamma),
                                              T * I - multi_dot([gamma, inv(I - gamma), I - power(gamma, T)])])
    JAN = (1 / T) * d * np.dot(e1.T, multi_dot([inv(I - gamma), sumR, metaOr]))
    PpostAN = np.dot(e1.T,
                     multi_dot([gamma, inv(I - gamma), I - power(gamma, T + tpost), vtAN[2 * T: 2 * T + 2 * l][::-1]]))
    IAN = np.sum(rtAN[:len(coef)+T]) + PpostAN
    k = (JAN - 0.5 * IAN) * tau / d
    b = IAN * tau / (T * d)
    return b, k


def impact_VARone_SIM(tau, A_est, coef):
    '''
    SIMULATED VERSION
    :param M: Time of dataM file
    :param A_est: Matrix A of the sVAR model
    :param coef: Coefficients of the VAR model
    :return: Estimation of permanent and temporary impact according to Almgren(2005)
    This function is recalled through 'estimate_impact_Has_Al()' and 'temp_wholemonth_Has_Alm()'
    whenever data are modelled as a VAR(1)
    '''
    d = -0.2
    #dailySigma = 0.001327
    T = 100
    tpost = int(30 * 60 / tau)
    vtSIM = np.array([[0, 0]])  # length T+tpost+1
    ptSIM = np.zeros(T + tpost + 1)  ## length T+tpost+1
    po = 100
    ptSIM[0] = po
    e1 = np.array([1, 0])
    e2 = np.dot(inv(A_est), np.array([0, 1]))
    I = np.identity(2)
    for i in range(1, T + tpost + 1):  # T+TPOST LOOPS
        vi = multi_dot([coef[0], np.array(vtSIM[i - 1])])
        # err = multi_dot([np.random.normal(size=2),A_est, su, A_est.transpose()])
        if i <= T:
            vi = vi + d * e2  # + err
        else:
            vi = vi  # + err
        vtSIM = np.append(vtSIM, [vi], axis=0)
        ptSIM[i] = ptSIM[i - 1] + vi[0]

    JSIM = (np.mean(ptSIM[1:T + 1]) - po) / po
    ISIM = (ptSIM[T + tpost] - po) / po
    k = (JSIM - 0.5 * ISIM) * po * tau / d
    b = ISIM * po * tau / (T * d)
    return b,k


def impact_VARp_SIM(tau, A_est, coef):
    '''
    SIMULATED VERSION
    :param M: Time of dataM file
    :param A_est: Matrix A of the sVAR model
    :param coef: Coefficients of the VAR model
    :return: Estimation of permanent and temporary impact according to Almgren(2005)
    This function is recalled through 'estimate_impact_Has_Al()' and 'temp_wholemonth_Has_Alm()'
     whenever data are modelled as a VAR(p)
    '''
    gamma = companion_matrix(coef)
    #tau = np.mean(np.diff(M)) / 1000  # inter-trade avg time (seconds)
    d = -0.2
    #dailySigma = 0.001327
    T = 100
    tpost = int(30 * 60 / tau)
    l = len(coef)
    vtSIM = np.zeros(2 * T + 2 * l + 2 * tpost)
    e1 = np.array([1, 0])
    e2 = np.dot(inv(A_est), np.array([0, 1]))
    #metaOr = np.zeros(l * 2)
    #metaOr[-1], metaOr[-2] = e2[0], e2[1]
    #for t in range(2 * l - 2, 2 * T + 2, 2):  # FROM ONE BECAUSE THE FIRST VECTOR MUST BE ZERO
    #    vtSIM[t:t + 2 * l] = np.dot(gamma, vtSIM[t - 2: t - 2 + 2 * l][::-1])[::-1] + d * metaOr
    #for t in range(2 * T + 2, 2 * T + 2 * tpost + 2, 2):
    #    vtSIM[t:t + 2 * l] = np.dot(gamma, vtSIM[t - 2: t - 2 + 2 * l][::-1])[::-1]
    metaOr = np.zeros(l * 2)
    metaOr[0], metaOr[1] = e2[0], e2[1]
    I = np.identity(len(gamma))
    for t in range(1, T + 1):  # FROM ONE BECAUSE THE FIRST VECTOR MUST BE ZERO
        vtSIM[2 * t: 2 * t + 2 * l] = (np.dot(gamma, vtSIM[2 * (t - 1): 2 * (t - 1) + 2 * l][::-1]) + metaOr * d)[::-1]
    for t in range(1, tpost + 1):
        vtSIM[2 * (t + T): 2 * (t + T) + 2 * l] = (np.dot(gamma,
                                                          vtSIM[2 * (T + t - 1): 2 * (T + t - 1) + 2 * l][::-1]))[::-1]
    rtSIM = vtSIM[1::2]
    xtSIM = vtSIM[::2]
    JSIM = np.mean(np.cumsum(rtSIM[len(coef):len(coef) + T]))
    PpostSIM = np.cumsum(rtSIM)[-1]  # + po
    ISIM = PpostSIM
    k = (JSIM - 0.5 * ISIM) * tau / d
    b = ISIM * tau / (T * d)
    return b, k


def impact_wholemonth_Has_Alm(days):
    '''
    :param days: All the days in the month
    :return: Estimation of temporary impact
    The function estimates a sVAR model (Hasbrouck, 1991) based on data from the whole month.
    Then by simulating the model + the execution of a metaorder we manage to obtain
     an estimation of the permanent and temporary impacts following Almgren(2005) proposal.
    '''
    rts = np.array([])
    xts = np.array([])
    taus = np.array([])
    for D in range(len(days)):
        day = days[D]
        dataOB, dataM = read_data(day, month)
        xt = np.where(dataM['AggressorAction'] == 'Buy', 1, -1) * dataM['Volume']
        M = millisecM(np.array(list(dataM.DateTime.values)), int(day), month)
        OB = dataOB.Datetime.values
        midprice = np.zeros(len(M) + 1)
        ask = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].AskPrice_0.values[-1]
        bid = dataOB.loc[(millisecOB(OB, int(day), month) < M[0])].BidPrice_0.values[-1]
        midprice[0] = 0.5 * (ask + bid)
        for l in range(0, len(M)):
            try:
                ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                                 (millisecOB(OB, int(day), month) < M[l + 1])].AskPrice_0.values[0]
                bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l]) &
                                 (millisecOB(OB, int(day), month) < M[l + 1])].BidPrice_0.values[0]
                midprice[l + 1] = 0.5 * (ask + bid)
            except:
                ask = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].AskPrice_0.values[0]
                bid = dataOB.loc[(millisecOB(OB, int(day), month) > M[l])].BidPrice_0.values[0]
                midprice[l + 1] = 0.5 * (ask + bid)
        rt = np.diff(midprice)
        rts = np.append(rts, rt)
        xts = np.append(xts, xt)
        taus = np.append(taus, np.mean(np.diff(M))/1000)
    df = pd.DataFrame({'rt': rts, 'xt': xts})
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    A = np.asarray([[1, 'E'], [0, 1]])
    B = np.asarray([['E', 0], [0, 'E']])
    model = SVAR(df, svar_type='A', A=A)  # , B = B)
    minCr = 99999
    l = 0
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        result = model.fit(maxlags=i, trend='n')
        if result.aic <= minCr:
            minCr = result.aic
            l = i
            # print('Lag Order =', i) print('AIC : ', result.aic) print('BIC : ', result.bic)  print('FPE : ', result.fpe) print('HQIC: ', result.hqic, '\n')
    svar_results = model.fit(maxlags=l, trend='n')  # let's go with a one order SVAR
    #var_results = VAR(df).fit(l, trend='n')
    svar_results.k_exog_user = 0
    A_est = svar_results.A
    B_est = svar_results.B
    coef = svar_results.coefs
    su = svar_results.sigma_u
    tau = np.mean(taus)
    ttest = svar_results.tvalues
    ttest = np.reshape(ttest, (len(coef), 2, 2))
    for i in range(len(ttest)):
        ttest[i][0][1], ttest[i][1][0] = ttest[i][1][0], ttest[i][0][1]
    coef = np.where(abs(ttest) <= 2, 0, coef)
    if l == 1:
        return impact_VARone_AN(tau, A_est, coef)
    else:
        return impact_VARp_AN(tau, A_est, coef)


##########################################################
#PARAMETERS ESTIMATION
##########################################################


def average_buysell_perhour(dataM):
    '''
    :param dataM: Data set containing trades
    :return: Average buy MOs per hour, average sell MOs per hour
    '''
    freq = 60 * 60 * 1000  # one hour
    noint = int((16 - 6) * 60 * 60 * 1000 / freq)
    n_buy = len(dataM.loc[(dataM.AggressorAction == 'Buy')])
    n_sell = len(dataM) - n_buy
    avg_hbuy = n_buy / noint
    avg_hsell = n_sell / noint

    return avg_hbuy, avg_hsell  # average hourly buy/sell MO


def MO_meanvolume(dataM):
    '''
    :param dataM: Data set containing trades
    :return: Mean volume of buy MOs, mean volume of sell MOs
    '''
    buy_meanVol = np.mean(dataM.loc[(dataM.AggressorAction == 'Buy')].Volume.values)
    sell_meanVol = np.mean(dataM.loc[(dataM.AggressorAction == 'Sell')].Volume.values)
    return buy_meanVol, sell_meanVol


def estimate_spread(dataOB):
    '''
    :param dataOB: Data set containing the book
    :return: Average spread
    '''
    bid = np.array(dataOB["BidPrice_0"])
    ask = np.array(dataOB["AskPrice_0"])
    spread = np.array(ask - bid)
    avg_spread = np.mean(spread, axis=0)
    return avg_spread


def realized_variance(price, time, freq):
    '''
    % Estimates Realized Variance
    %
    % USAGE:
    %   RV = realized_variance(PRICE,TIME,FREQ)
    %
    % INPUTS:
    %   PRICE            - m by 1 vector of high frequency prices
    %   TIME             - m by 1 vector of times where TIME(i) corresponds to PRICE(i). Must be in milliseconds
    %   FREQ              - frequency of the RV. Must be in milliseconds
    % OUTPUTS:
    %   RV                - Realized variance estimate
    '''


    if isinstance(price, (list, tuple, np.ndarray)) == False:
        raise ValueError("PRICE must be a m by 1 vector.")

    if any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing.")

    if (isinstance(time, (list, tuple, np.ndarray)) == False | len(time)!=len(price)):
        raise ValueError("TIME must be a m by 1 vector.")

    m = price.shape[0]
    t0 = time[0]
    tT = time[m-1]
    if np.isscalar(freq) == False:
        raise ValueError('freq must be a scalar value in milliseconds')

    ######################
    # Input Checking
    ######################
    logPrice = np.log(price)
    # Filter prices and compute the RV
    noint = int((tT - t0) / freq)
    bound = np.zeros(noint + 1)  # Variable for interval bound
    for j in range(0, noint + 1):
        bound[j] = t0 + j * freq
    ind = np.searchsorted(time, bound)
    filteredLogPrice = logPrice[ind[ind != len(logPrice)]]
    returns = np.diff(filteredLogPrice)
    rv = np.dot(returns.transpose(), returns)
    return rv


def bipower_variance(price, time, freq):
    '''
    % Estimates Realized Variance
    %
    % USAGE:
    %   RBV = realized_bipower_variance(PRICE,TIME,FREQ)
    %
    % INPUTS:
    %   PRICE            - m by 1 vector of high frequency prices
    %   TIME             - m by 1 vector of times where TIME(i) corresponds to PRICE(i). Must be in milliseconds
    %   FREQ              - frequency of the RV. Must be in milliseconds
    % OUTPUTS:
    %   RBV                - Realized bipower variance estimate
    '''


    if isinstance(price, (list, tuple, np.ndarray)) == False:
        raise ValueError("PRICE must be a m by 1 vector.")

    if any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing.")

    if (isinstance(time, (list, tuple, np.ndarray)) == False | len(time)!=len(price)):
        raise ValueError("TIME must be a m by 1 vector.")

    m = price.shape[0]
    t0 = time[0]
    tT = time[m-1]
    if np.isscalar(freq) == False:
        raise ValueError('freq must be a scalar value in milliseconds')

    ######################
    # Input Checking
    ######################
    logPrice = np.log(price)
    # Filter prices and compute the RV
    noint = int((tT - t0) / freq)
    bound = np.zeros(noint + 1)  # Variable for interval bound
    for j in range(0, noint + 1):
        bound[j] = t0 + j * freq
    ind = np.searchsorted(time, bound)
    filteredLogPrice = logPrice[ind[ind != len(logPrice)]]
    returns = np.diff(filteredLogPrice)
    ret1 = np.abs(returns[:-1])
    ret2 = np.abs(returns[1:])
    rbv = np.dot(ret1.transpose(), ret2)
    return 0.5*math.pi*rbv


###########################################################
#OTHERS
###########################################################

def fixALL(dataM, TRADES_sign):
    '''
    :param dataM: Data set containing the trades
    :param TRADES_sign: Output of 'estim_sign()'
    :return: dataM where the directions are corrected.
    '''
    def modifyALL(row):
        AA = TRADES_sign.loc[row.name].est_sign
        row.AggressorAction = AA  # change into Buy/Sell
        return row

    new = dataM.apply(lambda row: modifyALL(row), axis=1)
    return new


def estim_sign(dataM, dataOB, day, month):
    '''
    :param dataM: Data set containing trades
    :param dataOB: Data set containing the order book
    :param day: Plug in day as integer
    :return: Dataframe in which each row contain a time in millisecs, a price, and the estimated direction
    The function estimate the sign of the trade based on the distance from bid and ask. 'perc' define the
    distance from bid (or ask) in percentage of spread within which the sign is estimated. Otherwise, the
    original is kept. By setting perc = 0.5 only the trades at the midprice are not estimated.
    '''
    perc = 0.5
    def get_sign(row):
        p = row.Price
        t = np.array([row.DateTime])
        t = millisecM(t, int(day), month)
        OB = dataOB.Datetime.values
        ask = dataOB.loc[(millisecOB(OB, int(day), month) < t)].AskPrice_0.values[-1]
        bid = dataOB.loc[(millisecOB(OB, int(day), month) < t)].BidPrice_0.values[-1]
        spread = ask - bid
        up_bound = ask - perc * spread  # it has to be in the 25% upper part of the spread
        lo_bound = bid + perc * spread  # it has to be in the 25% lower part of the spread
        if spread >= 0:
            if p == bid == ask:  # spread=0 + price in the middle:
                es_si = row.AggressorAction
            elif p > up_bound:  # ABOVE THE bound
                es_si = 'Buy'
            elif p < lo_bound:  # BELOW THE bound
                es_si = 'Sell'
            else:
                es_si = row.AggressorAction
        else:
            if np.abs(p - ask) < np.abs(p - bid):
                es_si = 'Buy'
            else:
                es_si = 'Sell'
        return pd.Series([t, p, es_si])

    tr_estim_si = dataM.apply(lambda row: get_sign(row), axis=1)
    return tr_estim_si