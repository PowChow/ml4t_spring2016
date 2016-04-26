import StrategyLearner as sl


# run the code to test a learner
if __name__=="__main__":

    learner = sl.StrategyLearner(verbose = False) # constructor

    learner.addEvidence(symbol = "IBM", sd=dt.datetime(2008,1,1),
                        ed=dt.datetime(2009,1,1), sv = 10000) # training step

    df_trades = learner.testPolicy(symbol = "IBM", sd=dt.datetime(2009,1,1),
                                   ed=dt.datetime(2010,1,1), sv = 10000) # testing step