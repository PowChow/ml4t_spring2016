"""MC2-P2: Bollinger Bands Strategy Implementation """

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

def get_bollinger_strategy(df):
    """Implementation strategy based on Bollinger Bands"""
    """Returns data frame of orders"""

    out_orders = []
    invested_short = False #indicator for deciding on exit or entry
    invested_long = False

    df_shift = df.shift(1)

    for i in range(0, len(df)):
        if (df.ix[i]['Price'] < df.ix[i]['upper_band']) and (invested_short == False) and \
                (df_shift.ix[i]['Price'] > df_shift.ix[i]['upper_band']):
            out_orders.append([df.index[i],'IBM', 'BUY', 'Short', 'SELL'])
            invested_short = True
        elif (df.ix[i]['Price'] < df.ix[i]['SMA']) and (invested_short == True) and \
                (df_shift.ix[i]['Price'] > df.ix[i]['SMA']):
            out_orders.append([df.index[i],'IBM', 'SELL', 'Short', 'BUY'])
            invested_short = False
        elif (df.ix[i]['Price'] > df.ix[i]['lower_band']) and (invested_long == False) and \
                (df_shift.ix[i]['Price'] < df_shift.ix[i]['lower_band']):
            out_orders.append([df.index[i], 'IBM', 'BUY', 'Long', 'BUY'])
            invested_long = True
        elif (df.ix[i]['Price'] > df.ix[i]['SMA']) and (invested_long == True) and \
                (df_shift.ix[i]['Price'] < df_shift.ix[i]['SMA']):
            out_orders.append([df.index[i], 'IBM', 'SELL', 'Long', 'SELL'])
            invested_long = False
        else:
            pass

    # Convert orders to data frame
    df_orders = pd.DataFrame(out_orders,
                             columns=['Date', 'Symbol', 'BB_Strat', 'Type', 'Order'])
    return df_orders

def test_run():
    # Read data
    dates = pd.date_range('2007-12-31', '2009-12-31')
    symbols = list(['IBM'])
    df = get_data(symbols, dates, addSPY=True)

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_IBM = get_rolling_mean(df['IBM'], window=20)

    # 2. Compute rolling standard deviation
    rstd_IBM = get_rolling_std(df['IBM'], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_IBM, rstd_IBM)

    # 4. Add signals for buy and sell
    combo_df = pd.concat([df['IBM'], rm_IBM, upper_band, lower_band], axis=1)
    combo_df.columns = ['Price', 'SMA', 'upper_band', 'lower_band']
    combo_df.rename(index={0:'Date'}, inplace=True)
    combo_df.to_csv('IBM.csv')

    orders = get_bollinger_strategy(combo_df)

    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df['IBM'].plot(title="Bollinger Bands", label='IBM')
    rm_IBM.plot(label='SMA', color='yellow', ax=ax)
    combo_df[['upper_band', 'lower_band']].plot(label='Bollinger Bands', color='cyan', ax=ax)

    # Plot vertical lines for shorts
    for i in range(0, len(orders)):
        if (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['BB_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='r', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['BB_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['BB_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='green', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['BB_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        else:
            pass

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left', labels=['IBM', 'SMA', 'Bollinger Bands'])
    #plt.show()
    fig = ax.get_figure()
    fig.savefig('output/bb_strategy.png')


    #prep orders for market simulator
    orders['Shares'] = 100
    orders.set_index('Date', inplace=True)
    orders[['Symbol', 'Order', 'Shares']].to_csv('output/bb_orders.csv')

    #send order to marketsims
    ms.sims_output(sv=10000, of="./output/bb_orders.csv", strat_name='BB' )


if __name__ == "__main__":
    test_run()