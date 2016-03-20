"""MC2-P1: Market simulator."""

import numpy as np
import pandas as pd
from analysis import compute_portfolio_stats

from mc2_p2.marketsim import compute_portvals


def test_case(of,sd,ed,ret,avg,std,sharp,fin,sv=1000000):
    portvals = compute_portvals(orders_file = of, start_val = sv)

    if not isinstance(portvals, pd.DataFrame):
        print "Test FAILED portvals not a pd.DataFrame, type: {}".format(type(portvals))
        return
    else:
    	portvals = portvals[portvals.columns[0]] # just get the first column

    cr, adr, sddr, sr = compute_portfolio_stats(portvals)
    act = [cr,adr,sddr,sr,portvals[-1]]
    ex = [ret,avg,std,sharp,fin]

    if pd.Timestamp(sd) != portvals.index[0]:
        print "Test FAILED start-date: {}, expected: {}, actual: {}".format(of,sd,portvals.index[0])
        return
    if pd.Timestamp(ed) != portvals.index[-1]:
        print "Test FAILED end-date: {}, expected: {}, actual: {}".format(of,ed,portvals.index[-1])
        return
    if np.allclose(ex,act):
        print "Test PASSED: {}".format(of)
    else:
        print "Test FAILED: {}, expected: {}, actual: {}".format(of,ex,act)

if __name__ == "__main__":
    test_case(of="./orders/orders-short.csv",
                sd='2011-01-05',
                ed='2011-01-20',
                ret=-0.001965,
                avg=-0.000178539446839,
                std=0.00634128215394,
                sharp=-0.446948390642,
                fin=998035.0)

    test_case(of="./orders/orders.csv",
                sd='2011-01-10',
                ed='2011-12-20',
                ret=0.13386,
                avg=0.000551651296638,
                std=0.00720514136323,
                sharp=1.21540888742,
                fin=1133860.0)

    test_case(of="./orders/orders2.csv",
                sd='2011-01-14',
                ed='2011-12-14',
                ret=0.0787526,
                avg=0.000353426354584,
                std=0.00711102080156,
                sharp=0.788982285751,
                fin=1078752.6)

    test_case(of="./orders/orders3.csv",
                sd='2011-01-10',
                ed='2011-08-01',
                ret=0.05016,
                avg=0.000365289198877,
                std=0.00560508094997,
                sharp=1.03455887842,
                fin=1050160.0)

    test_case(of="./orders/orders-leverage-1.csv",
                sd='2011-01-10',
                ed='2011-06-10',
                ret=0.05016,
                avg=0.000487052265169,
                std=0.00647534272091,
                sharp=1.19402406143,
                fin=1050160.0)

    test_case(of="./orders/orders-leverage-2.csv",
                sd='2011-01-10',
                ed='2011-03-03',
                ret=0.07465,
                avg=0.00202241842159,
                std=0.00651837064888,
                sharp=4.92529481246,
                fin=1074650.0)

    test_case(of="./orders/orders-leverage-3.csv",
                sd='2011-01-10',
                ed='2011-08-01',
                ret=0.05016,
                avg=0.000365289198877,
                std=0.00560508094997,
                sharp=1.03455887842,
                fin=1050160.0)