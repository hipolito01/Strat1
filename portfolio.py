# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 04:29:01 2017

@author: hipo
"""

import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt


class portfolio(object):
    
    
    def __init__(self, pnl_series=None,weights=None,leverage=1):
		
        #inputs
        self.pnl_series=pnl_series
        self.weights=weights
        self.pnl_daily=(pnl_series*weights).sum(axis=1)#total pnl for each day
        self.leverage=leverage
                                     
        #variables
        self.PnL=None
        self.sharpeRatioDaily=None#not very useful (only useful to compute the annual sharpe ratio)
        self.sharpeRatioAnnual=None 
        self.maxDD=None#max Drawdown
        self.tradedDays=None#number of days with trading activity
        self.tradedDaysPct=None#% of days with trading activity
        self.winnerDays=None
        self.averageTrade=None#average % pnl on winning days
        self.averageWin=None#average % pnl on winning days
        self.averageLoss=None#average % pnl on losing days
        self.CAGR=None#Compound Annual Growth Rate		
        self.Calmar=None#CAGR/max drawdown (the higher the Calmar Ratio the better the perfomance)
        self.durationMaxDD=None#duration (in days) of the max drawdown
        index_ = ['PnL','CAGR','Sharpe Ratio','Max Drawdown','Calmar','Max Drawdown Duration','Traded Days %','Winner Days %','Avg Trade','Avg Win', 'Avg Loss']
        columns_=['Portfolio']#one column for each "k" (lag period)
        self.results_table_df = pd.DataFrame(index=index_ , columns=columns_)                                         
        
    
    def performance(self):
        
        #we assume that the transactions were accounted for earlier, when producing
        #the pnl time series
        pnl = pnl_metrics(self.leverage,self.pnl_daily)
        pnl.compute_daily_metrics()
        
        self.results_table_df.ix[0,0]=pnl.PnL
        self.results_table_df.ix[1,0]=pnl.CAGR
        self.results_table_df.ix[2,0]=pnl.sharpeRatioAnnual
        self.results_table_df.ix[3,0]=pnl.maxDD
        self.results_table_df.ix[4,0]=pnl.Calmar
        self.results_table_df.ix[5,0]=pnl.durationMaxDD
        self.results_table_df.ix[6,0]=pnl.tradedDaysPct
        self.results_table_df.ix[7,0]=pnl.winnerDays
        self.results_table_df.ix[8,0]=pnl.averageTrade
        self.results_table_df.ix[9,0]=pnl.averageWin
        self.results_table_df.ix[10,0]=pnl.averageLoss
                               
        self.PnL=pnl.PnL
        self.sharpeRatioDaily=pnl.sharpeRatioDaily
        self.sharpeRatioAnnual=pnl.sharpeRatioAnnual 
        self.maxDD=pnl.maxDD
        self.tradedDays=pnl.tradedDays
        self.tradedDaysPct=pnl.tradedDaysPct
        self.winnerDays=pnl.winnerDays
        self.averageTrade=pnl.averageTrade
        self.averageWin=pnl.averageWin
        self.averageLoss=pnl.averageLoss
        self.CAGR=pnl.CAGR
        self.Calmar=pnl.Calmar
        self.durationMaxDD=pnl.durationMaxDD