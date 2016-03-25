"""MC2-P2: My Strategy MFI """
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

def get_data_more(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close', 'High', 'Low', 'Volume'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': '%s' % symbol, 'High': '%s_High' % symbol,
                                 'Low': '%s_Low' % symbol, 'Volume': '%s_Volume' % symbol})
        df = df.join(df_temp )
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

def get_mfi_strategy(df):
    """Implementation strategy based on Relative Strength Index"""
    """Returns data frame of orders"""

    out_orders = []
    save_low_mfi = 0
    save_high_mfi = 0
    invested_short = False
    invested_long = False

    df_shift = df.shift(1)

    for i in range(0, len(df)):

        #hold lowest and highest values in dataset, double down on lowest
        if save_low_mfi > df.ix[i]['mfi']:
            save_low_mfi =  df.ix[i]['mfi']
        elif save_high_mfi < df.ix[i]['mfi']:
            save_high_mfi = df.ix[i]['mfi']

        if (df.ix[i]['mfi'] < 30) and (invested_short == False):
             out_orders.append([df.index[i],'IBM', 'BUY', 'Short', 'SELL'])
             invested_short = True
        elif (df.ix[i]['mfi'] > 35) and (invested_short == True):
            out_orders.append([df.index[i], 'IBM', 'SELL', 'Short', 'BUY'])
            invested_short = False
        elif (df.ix[i]['mfi'] > 70) and (invested_long == False):
             out_orders.append([df.index[i],'IBM', 'BUY', 'Long', 'BUY'])
             invested_long = True
        elif (df.ix[i]['mfi'] < 65) and (invested_long == True) \
            and (df.ix[i]['mf_ch_prop'] > 1.0):
            out_orders.append([df.index[i], 'IBM', 'SELL', 'Long', 'SELL'])
            invested_long = False
        else:
            pass

    # Convert orders to data frame
    df_orders = pd.DataFrame(out_orders,
                             columns=['Date', 'Symbol', 'mfi_Strat', 'Type', 'Order'])
    return df_orders

def test_run():
    #in and out sample dates
    in_dates = pd.date_range('2007-12-31', '2009-12-31') #Add in sample and out of sample dates
    out_dates = pd.date_range('2009-12-31', '2011-12-31') #Add in sample and out of sample dates

    symbols = list(['IBM'])
    df = get_data_more(symbols, in_dates, addSPY=True)
    df_mfi = df.copy()
    for col in df_mfi:
        if 'SPY' in col:
            try:
                df_mfi.drop(col, axis=1, inplace=True)
            except Exception:
                pass

    #COMPUTE MY STRATEGY - Money Flow Index(mfi) & RELATIVE STRENGTH (RS)
    #1. Compute Typical Price
    df_mfi['tprice'] = (df_mfi['IBM_High'] + df_mfi['IBM_Low'] + df_mfi['IBM']) / 3

    # 2. Raw Money Flow
    df_mfi['raw_mf'] = df_mfi['tprice'] * df_mfi['IBM_Volume']

    # 3. Calculate changes
    df_mfi['mf_ch'] = df_mfi['raw_mf'].shift(1) - df_mfi['raw_mf']
    df_mfi['mf_ch_prop'] = df_mfi['raw_mf'].shift(1) /df_mfi['raw_mf']

    # 4. Compute Gain and Loss
    df_mfi['mf_gain'] = df_mfi['mf_ch'].apply(lambda x: abs(x) if x > 0 else 0)
    df_mfi['mf_loss'] = df_mfi['mf_ch'].apply(lambda x: abs(x) if x < 0 else 0)

    # 5. Compute rolling mean Average Gain and Loss
    # lower window to increase the sensitivity
    df_mfi['rm_mf_gain'] = get_rolling_mean(df_mfi['mf_gain'], window=14)
    df_mfi['rm_mf_loss'] = get_rolling_mean(df_mfi['mf_loss'], window=14)
    
    # 5. Compute Money Flow Ratio
    df_mfi['mf_ratio'] = df_mfi['rm_mf_gain']/df_mfi['rm_mf_loss']
    
    # 6. Compute mfi
    df_mfi['mfi'] = df_mfi['mf_ratio'].apply(lambda x: 100 if x == 0.0 else 100-(100/(x+1)))
    df_mfi.to_csv('./output/mfi_df.csv')
    # 7 Add signals for buy and sell
    df_mfi.rename(columns = {'IBM': 'Price'}, inplace=True)
    orders = get_mfi_strategy(df_mfi)
    orders.to_csv('./output/mfi_orders.csv')

    # #df_mfi.reset_index(inplace=True)
    #
    # Plot raw IBM values
    ax = df_mfi['Price'].plot(title="Relative Strength Index", label='IBM')
    df_mfi['mfi'].plot(label='mfi', color='black', ax=ax)
    #TODO plot on secondary axis
    #df_mfi[['rm_gain', 'rm_loss']].plot(label='Rolling Loss and Gain', color='cyan', ax=ax)

    # Plot vertical lines for shorts
    for i in range(0, len(orders)):
        if (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['mfi_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='r', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Short') and (orders.ix[i]['mfi_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['mfi_Strat'] == 'BUY'):
            ax.axvline(orders.ix[i]['Date'], color='green', linestyle='solid')
        elif (orders.ix[i]['Type'] == 'Long') and (orders.ix[i]['mfi_Strat'] == 'SELL'):
            ax.axvline(orders.ix[i]['Date'], color='black', linestyle='solid')
        else:
            pass

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left', labels=['IBM', 'MFI'])
    # # #plt.show()
    fig = ax.get_figure()
    fig.savefig('output/mfi_strategy.png')


    # #prep orders for market simulator
    orders['Shares'] = 100
    orders.set_index('Date', inplace=True)
    orders[['Symbol', 'Order', 'Shares']].to_csv('./output/mfi_orders.csv')

    # #send order to marketsims
    ms.sims_output(sv=10000, of="./output/orders.csv", strat_name='mfi' )


if __name__ == "__main__":
    test_run()