"""MLT: Utility code."""

import os
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')                 #TODO update this in virutal env
import matplotlib.pyplot as plt  #TODO update this in virutal env

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

##################################################################
############## PLOTTING FUNCTIONS ################################
##################################################################

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_lines_data(price_norm, actualY, predY, exitentry=True, name='default', symbol='IBM'):

    # price_norm.to_csv('price_norm.csv')
    # actualY.to_csv('actualY.csv')
    # predY.to_csv('predY.csv')
    ax = price_norm.plot(title=name, label='Norm Price')
    actualY.plot(label='Actual Price', color='green', ax=ax)
    predY.plot(label='Predicted Price', color='crimson', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Values")
    ax.legend(loc='lower left', labels=['Price Norm', 'Actual Y', 'Predicted Y'])

    print 'Actual Y stats: ', actualY.describe()
    print 'Predicted Y stats: ', predY.describe()

    #plt.show()
    fig = ax.get_figure()
    fig.savefig('./Output/%s_lines.png' % (name))

def plot_strategy(price, of='./Orders/orders.csv', name='default'):

    orders = pd.read_csv(of)
    ax = price.plot(title=name, label='Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices")

    # Plot vertical lines for shorts
    for i in range(0, len(orders)):
        if (orders.ix[i]['Strat'] == 'Short') and (orders.ix[i]['Type'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='r', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Short') and (orders.ix[i]['Type'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Long') and (orders.ix[i]['Type'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='green', linestyle='solid')
        elif (orders.ix[i]['Strat'] == 'Long') and (orders.ix[i]['Type'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        else:
            pass

    #plt.show()
    fig = ax.get_figure()
    fig.savefig('./Output/%s_strategy_exitentry.png' % (name))

def plot_backtest():
    pass