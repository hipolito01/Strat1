#this version "simple" speeds up the screening by simplyfying the strategy to be backtested


import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt


class cross_sectional_MR_simple(object):
    #this class computes returns of simple cross sectional MR strategy: but the worst 1-day perforer
    #short the best 1-day performer (1-day lag period, no threshold)
    
    
    def __init__(self, dataClose=None):
		#inputs
                
        self.dataClose=dataClose
        
                    
        self.numberStocks=self.dataClose.shape[1]
        
        self.returns_df=None
        self.numberLags=None
        self.lags=None
        
        self.indices_df=None
        self.container_indices=None
                
        
        self.labels_df=None
        self.labels_returns_df=None
  
        
        #variables
        self.labels_pnl_df=None
        
        self.numberLags=1
       
        self.results_table_df=None        
        self.pnl_daily=None#-->daily time series of PnLs
        self.equity_curve=None
        
        #all variables from "pnl" class
        self.PnL=None
        self.sharpeRatioDaily=None#not very useful (only useful to compute the annual sharpe ratio)
        self.sharpeRatioAnnual=None 
        self.maxDD=None#max Drawdown
        self.tradedDays=None#number of days with trading activity
        self.tradedDaysPct=None#% of days with trading activity
        self.winnerDays=None
        self.averageTrade=None#average % pnl on all trades
        self.averageWin=None#average % pnl on winning days
        self.averageLoss=None#average % pnl on losing days
        self.CAGR=None#Compound Annual Growth Rate		
        self.Calmar=None#CAGR/max drawdown (the higher the Calmar Ratio the better the perfomance)
        self.durationMaxDD=None#duration (in days) of the max drawdown

        
    
    def lag_returns(self,lags=[1]):
        #This function computes the log returns for each stock and for each given "lag" period
        
        data=self.dataClose
        
        numberLags=len(lags)
        numberStocks=self.numberStocks                
        
        #"returns_df" contains the log returs for each stock and for the given lag period
        index_ = data.index
        columns_=range(numberStocks*numberLags)
        returns_df = pd.DataFrame(index=index_,columns=columns_)

        #Computing lag-log returns
        for i in range(0,numberLags):
            lag=lags[i]
            for j in range(numberStocks):
                returns_df.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(lag))
        
        
        returns_df=returns_df.ix[1:,:]#dropping NAs
        
        #Saving results
        self.returns_df=returns_df
        self.lags=lags
        self.numberLags=numberLags
        
        
    def get_indices(self):
        #this function gets for each date, the indices of the best and worst performing stocks, and then
        #converts those indices into a composite index (column 2)
        
        index_=self.returns_df.index
        columns_=range(3)
        indices_df=pd.DataFrame(index=index_,columns=columns_)
        
        indices_df.ix[:,0]=self.returns_df.idxmax(axis=1)
        indices_df.ix[:,1]=self.returns_df.idxmin(axis=1)
        indices_df.ix[:,2]=indices_df.ix[:,0]*10+indices_df.ix[:,1]
        
        indices_df=indices_df.astype(int)
        
        #saving
        self.indices_df=indices_df
    
    
    
    
    def compute_labels(self,threshold_do_nothing=0.00):
        #_____this function computes the returns of going long the worst 1-day performer, and short
        #the 1-day best performer
        #_____threshold_do_nothing is an optional input-->represents an area of indiference between
        #putting a position or doing nothing (if returns are in that area then im indiferent)
        #-->example if return is less than 0.3% i am not interested in pursuing the trade

        index_=self.returns_df.index
        columns_=range(1)
        labels_returns_df=pd.DataFrame(index=index_,columns=columns_)
        
        #shifting returns-->todays labels correspond to tomorrows returns
        returns_for_labels=self.returns_df.shift(-1).copy()
        
        for i in labels_returns_df.index:
            #compute returns from shorting the best performer and buying the worst performer
            labels_returns_df.ix[i,0]=returns_for_labels.ix[i,self.indices_df.ix[i,1]]-returns_for_labels.ix[i,self.indices_df.ix[i,0]]-threshold_do_nothing
            
        labels_df=(labels_returns_df>0)*1
          
        self.labels_returns_df=labels_returns_df
        self.labels_df=labels_df
        
    def performance(self,costPerTrade=0,tradeSize=10000):
        
        
        #Creating data frame of results
        #Using an instance of class "pnl_metrics"
        #Attributes:
        #self.PnL=None
        #self.CAGR=None#Compound Annual Growth Rate
        #self.sharpeRatioAnnual=None 
        #self.maxDD=None#max Drawdown
        #self.Calmar=None#CAGR/max drawdown (the higher the Calmar Ratio the better the perfomance)
        #self.durationMaxDD=None#duration (in days) of the max drawdown
        #self.tradedDaysPct=None#% of days with trading activity
        #self.winnerDays=None
        #self.averageWin=None#average % pnl on winning days
        #self.averageLoss=None#average % pnl on losing days
                
        index_ = ['PnL','CAGR','Sharpe Ratio','Max Drawdown','Calmar','Max Drawdown Duration','Traded Days %','Winner Days %','Avg Trade','Avg Win', 'Avg Loss']
        columns_=range(1)#one column for each "k" (lag period)
        results_table_df = pd.DataFrame(index=index_ , columns=columns_)
            
        #------Parameter
        #tradeSize=self.size
        
        
        #for i1 in range(numberLags):#looping on each lag
        #i1=0
        #a=i1*numberStocks
        #b=i1*numberStocks+numberStocks-1
        #pnl_df=pd.np.multiply(labels_df,weights_df.ix[:,a:b])#returns*weights 
        
                             
        #pnl_daily=pnl_df.sum(axis=1)*tradeSize#sum across columns (total pnl for each day)
        pnl_daily=self.labels_returns_df.copy()
        pnl_daily=pnl_daily*10000
        pnl_daily.dropna(inplace=True)#-->dropping na (from last row)
        pnl_daily=pnl_daily.sum(axis=1)#doing this crap to convert from dataframe to whatever the fuck that is (otherwise i have problems with class "pnl_metrics")
        
        #Adding transaction cost: broker fees (if any)
        brokerFees=costPerTrade*tradeSize*(pnl_daily!=0)
        pnl_daily=pnl_daily-brokerFees
        
        #computing performance metrics
        pnl = pnl_metrics(tradeSize,pnl_daily)
        pnl.compute_daily_metrics()
        #Plot
        #(pnl_daily.cumsum()).plot()
        #Printing Results:
        i1=0
        results_table_df.ix[0,i1]=pnl.PnL/tradeSize
        results_table_df.ix[1,i1]=pnl.CAGR
        results_table_df.ix[2,i1]=pnl.sharpeRatioAnnual
        results_table_df.ix[3,i1]=pnl.maxDD
        results_table_df.ix[4,i1]=pnl.Calmar
        results_table_df.ix[5,i1]=pnl.durationMaxDD
        results_table_df.ix[6,i1]=pnl.tradedDaysPct
        results_table_df.ix[7,i1]=pnl.winnerDays
        results_table_df.ix[8,i1]=pnl.averageTrade
        results_table_df.ix[9,i1]=pnl.averageWin
        results_table_df.ix[10,i1]=pnl.averageLoss
        
        #saving the variables as global variables
        self.results_table_df=results_table_df
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
                           
        #Saving into global variables
        self.results_table_df=results_table_df
        self.pnl_daily=pnl_daily
        self.equity_curve=pnl_daily.cumsum()
        