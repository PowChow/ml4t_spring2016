"""MC2-P2: My Strategy """

import pandas as pd
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import marketsim as ms
import scipy.optimize as spo

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # Add 2sd above and below the rolling mean
    upper_band = rm + (2*rstd)
    lower_band = rm - (2*rstd)
    return upper_band, lower_band

def get_rsi_strategy(df):
    """Implementation strategy based on Relative Strength Index"""
    """Returns data frame of orders"""

    out_orders = []
    save_low_rsi = 0
    save_high_rsi = 0
    invested_short = False
    invested_long = False

    df_shift = df.shift(1)

    for i in range(0, len(df)):

        #hold lowest and highest values in dataset, double down on lowest
        if save_low_rsi > df.ix[i]['rsi']:
            save_low_rsi =  df.ix[i]['rsi']
        elif save_high_rsi < df.ix[i]['rsi']:
            save_high_rsi = df.ix[i]['rsi']

        if (df.ix[i]['rsi'] < 30) and (invested_short == False):
             out_orders.append([df.index[i],'IBM', 'BUY', 'Short', 'SELL'])
             invested_short = True
        elif (df.ix[i]['rsi'] > 30) and (invested_short == True):
            out_orders.append([df.index[i], 'IBM', 'SELL', 'Short', 'BUY'])
            invested_short = False
        elif (df.ix[i]['rsi'] > 70) and (invested_long == False):
             out_orders.append([df.index[i],'IBM', 'BUY', 'Long', 'BUY'])
             invested_long = True
        elif (df.ix[i]['rsi'] < 70) and (invested_long == True) \
            and (df.ix[i]['change_prop'] > 1.0):
            out_orders.append([df.index[i], 'IBM', 'SELL', 'Long', 'SELL'])
            invested_long = False
        else:
            pass

    # Convert orders to data frame
    df_orders = pd.DataFrame(out_orders,
                             columns=['Date', 'Symbol', 'RSI_Strat', 'Type', 'Order'])
    return df_orders

def test_run():
    # Read data
    dates = pd.date_range('2007-12-31', '2009-12-31') #Add in sample and out of sample dates
    symbols = list(['IBM'])
    df = get_data(symbols, dates, addSPY=True)
    df_rsi = df.copy()
    df_rsi.drop(['SPY'], axis=1, inplace=True)

    #COMPUTE MY STRATEGY - RELATIVE STRENGTH INDEX (RSI) & RELATIVE STRENGTH (RS)
    # 1. Compute change
    df_rsi['change'] = df_rsi['IBM'].shift(1) - df_rsi['IBM']
    df_rsi['change_prop'] = df_rsi['IBM'].shift(1) /df_rsi['IBM']

    # 2. Compute Gain
    df_rsi['gain'] = df_rsi['change'].apply(lambda x: abs(x) if x > 0 else 0)
    # 3. Compute Loss
    df_rsi['loss'] = df_rsi['change'].apply(lambda x: abs(x) if x < 0 else 0)

    # 4. Compute rolling mean Average Gain and Loss
    # #lower window to increase the sensitivity
    df_rsi['rm_gain'] = get_rolling_mean(df_rsi['gain'], window=14)
    df_rsi['rm_loss'] = get_rolling_mean(df_rsi['loss'], window=14)

    # 5. Compute RS
    df_rsi['rs'] = df_rsi['rm_gain']/df_rsi['rm_loss']

    # 6. Compute RSI
    df_rsi['rsi'] = df_rsi['rs'].apply(lambda x: 100 if x == 0.0 else 100-(100/(x+1)))

    # 7 Add signals for buy and sell
    #df_rsi.columns =['Price', 'change', 'gain', 'loss', 'rm_gain', 'rm_loss', 'rs', 'rsi']
    orders = get_rsi_strategy(df_rsi)
    orders.to_csv('./output/rsi_orders.csv')

    #df_rsi.reset_index(inplace=True)

    # Plot raw IBM values
    ax = df_rsi['IBM'].plot(title="Relative Strength Index", label='IBM')
    df_rsi['rsi'].plot(label='RSI', color='black', ax=ax)
    #TODO plot on secondary axis
    #df_rsi[['rm_gain', 'rm_loss']].plot(label='Rolling Loss and Gain', color='cyan', ax=ax)

    # Plot vertical lines for shorts
    for i in range(0, len(orders)):
        if (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['RSI_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='r', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['RSI_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['RSI_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='green', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['RSI_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        else:
            pass

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left', labels=['IBM', 'RSI'])
    # # #plt.show()
    fig = ax.get_figure()
    fig.savefig('output/rsi_strategy.png')


    # #prep orders for market simulator
    orders['Shares'] = 100
    orders.set_index('Date', inplace=True)
    orders[['Symbol', 'Order', 'Shares']].to_csv('./output/orders.csv')

    # #send order to marketsims
    ms.sims_output(sv=10000, of="./output/orders.csv", strat_name='RSI' )


if __name__ == "__main__":
    test_run()