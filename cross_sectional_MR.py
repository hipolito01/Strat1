import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt
#import pandas.io.data as web#this is used to import data from yahoo finance
#import pandas_datareader as web
#import pandas.datareader.data as web
from pandas_datareader import data as web

class cross_sectional_MR(object):
    
    
    def __init__(self, stocks=None, start_date=None,end_date=None,lag=None,size=1,filter1_trigger=None,
                 momentum_trigger=None,filterFinal_threshold=None,filter2_trigger=None,
                 filter2_thresholds=None,filter2_leverage=None):
		#inputs
        self.stocks=stocks
        self.start_date=start_date
        self.end_date=end_date
        self.lag=lag
        self.filter1_trigger=filter1_trigger
        self.momentum_trigger=momentum_trigger
        self.filterFinal_threshold=filterFinal_threshold
        self.filter2_trigger=filter2_trigger
        self.filter2_thresholds=filter2_thresholds
        self.filter2_leverage=filter2_leverage
        self.size=size
                  
        
        
        
		 #variables
        self.data=None        
        self.weights_df=None
        self.returns_df=None
        self.numberLags=1
        self.numberStocks=None
        self.labels_df=None
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
        self.averageTrade=None#average % pnl on winning days
        self.averageWin=None#average % pnl on winning days
        self.averageLoss=None#average % pnl on losing days
        self.CAGR=None#Compound Annual Growth Rate		
        self.Calmar=None#CAGR/max drawdown (the higher the Calmar Ratio the better the perfomance)
        self.durationMaxDD=None#duration (in days) of the max drawdown

        
    
    def features(self):
        #Retrieving global variables        
        stocks=self.stocks 
        start_date=self.start_date
        end_date=self.end_date
        lag=self.lag
        
        #-------PULLING DATA------------
        
        #calling first stock to get the index for the df
        data1 = web.DataReader(stocks[0], data_source='yahoo',start=start_date, end=end_date)

        #creating data frame that will store the data
        index_ = data1.index
        columns_=range(len(stocks))
        data = pd.DataFrame(index=index_,columns=columns_)

        aux=0
        for i in stocks:
            data1 = web.DataReader(i, data_source='yahoo',start=start_date, end=end_date)
            data.ix[:,aux]=data1['Close']
            aux=aux+1
        
        #-------FEATURES-----------------
        numberLags=self.numberLags#we work with a single lag parameter
        numberStocks=data.shape[1]#number of columns in dataframe (each column represents one stock)        
        self.numberStocks=numberStocks
        
        #"returns_df" contains the log returs for each stock and for the given lag period
        #creating data frame that will contain the features
        index_ = data.index
        columns_=range(numberStocks*numberLags)
        returns_df = pd.DataFrame(index=index_,columns=columns_)

        #Computing lag-log returns
        for i in range(0,numberLags):
            for j in range(numberStocks):
                returns_df.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(self.lag))
        
        #deleting first rows to avoid NaNs
        returns_df = returns_df.ix[1:]
        
        
        #"weights_df" contains the weight for each stock and each day 
   
        #The weights are computed based on Khandami and Lo paper (2007):
        #Wit=-1/N*(Ri(t-k)-Rm(t-k))
        #where, Rm is the mean return accross all the stocks for lag period k
        #k is obviously a parameter of the strategy (the lag-period), but here we compute all the lag periods indicated above (so later
        #the user can use any k)

        #creating data frame that will contain the weights
        index_ = returns_df.index
        columns_=range(numberStocks*numberLags)
        weights_df = pd.DataFrame(index=index_,columns=columns_)



        for index, row in returns_df.iterrows():
            for i in range(numberLags):
                Rm=np.mean(returns_df.ix[index,i*numberStocks:i*numberStocks+numberStocks-1])#computing Rm (mean return accross all the stocks)
                normalizer=np.sum(np.absolute(-(1.0/numberStocks)*(returns_df.ix[index,i*numberStocks:i*numberStocks+numberStocks-1]-Rm)))/2
        
                for j in range(numberStocks):
                    #applying formula to get weights                    
                    weights_df.ix[index,i*numberStocks+j]=-(1*1.0/numberStocks)*(returns_df.ix[index,i*numberStocks+j]-Rm)/normalizer
            
        self.weights_df=weights_df
        self.returns_df=returns_df
        self.data=data
        
    
    def filters(self):
        #retrieving global varialbes
        weights_df=self.weights_df
        returns_df=self.returns_df
        numberStocks=self.numberStocks
        numberLags=self.numberLags
        data=self.data
        filter2_thresholds=self.filter2_thresholds
        filter2_leverage=self.filter2_leverage
        filterFinal_threshold=self.filterFinal_threshold
           
           
        
        #FILTER 1: ONLY TRADE BEST AND WORST PERFORMER OF THE LAG PERIOD (WITH WEIGHTS 1 AND -1)
        #Parameters needed:
        #filter1_trigger=1 or 0        
        if self.filter1_trigger==1:
            for index, row in returns_df.iterrows():
                for i in range(numberLags):#loop on each lag
        
                    max_val=np.max(weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])
                    min_val=np.min(weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])
        
                    for j in range(numberStocks):#loop on each stock
        
                        if weights_df.ix[index,i*numberStocks+j]==max_val:
                            weights_df.ix[index,i*numberStocks+j]=1
                        elif weights_df.ix[index,i*numberStocks+j]==min_val:
                            weights_df.ix[index,i*numberStocks+j]=-1
                        else:
                            weights_df.ix[index,i*numberStocks+j]=0           
                
        
        
        #FILTER 2:Leverage
        #FILTER: INCREASE TRADE SIZE IF DIFF BETWEEN BEST AND WORST GETS LARGER
        #Paramters needed:
        #filter2_trigger=1#if 1 it triggers
        #filter2_thresholds=[0.01,0.02,0.03]#threshold values for diff between best and worst performer
        #filter2_leverage=[1,1,1]#leverage multiples-->ratio of increase in position size
        
        if self.filter2_trigger==1:
            #Modifying weights:
            for index, row in returns_df.iterrows():
                for i in range(numberLags):#loop on each lag
                    #calculating difference between best and worst performer (for corresponding date and lag period)
                    value=np.max(returns_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])-np.min(returns_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])
        
                    if value>filter2_thresholds[2]:
                        weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]=weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]*filter2_leverage[2]
                    elif value>filter2_thresholds[1]:
                        weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]=weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]*filter2_leverage[1]
                    elif value>filter2_thresholds[0]:
                        weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]=weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]*filter2_leverage[0]

        
        
        #FILTER MOMENTUM
        #if variable momentum_trigger==1-->all the weights are inversed so the strategies goes from cross sectiona mean reversion
        #to cross sectional momentum
        #Parameter: momentum_trigger=1 or 0
        if self.momentum_trigger==1:
            weights_df=weights_df*(-1.0)
                
        
        #FILTER FINAL: THRESHOLD ON THE % DIFF BETWEEN BEST AND WORST PERFORMER; IF BELOW THRESHOLD-->NO TRADE
        #THIS FILTER SHOULD BE THE LAST ONE-->IT OVERRIDES ANY OTHER ADJUSTMENTS MADE TO THE WEIGHTS PREVIOUSLY 
        #Threshold Parameter: filterFinal_threshold=0.01#-->if set to 0, then no trade        
        for index, row in returns_df.iterrows():
            for i in range(numberLags):#loop on each lag
                
                #calculating difference between best and worst performer (for corresponding date and lag period)
                value=np.max(returns_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])-np.min(returns_df.ix[index,i*numberStocks:(i+1)*numberStocks-1])
                
                if value<=filterFinal_threshold:
                    weights_df.ix[index,i*numberStocks:(i+1)*numberStocks-1]=0#if below thresholds then all the weights are 0 (no trade on that day)
                

        #Saving changes to weights_df into global variable
        self.weights_df=weights_df


    def labels(self):
        
        #retrieving global varialbes        
        weights_df=self.weights_df
        returns_df=self.returns_df
        numberStocks=self.numberStocks
        numberLags=self.numberLags
        data=self.data

        
        #creating data frame that will contain the LABELS (forward 1-day returns for each stock)
        index_ = returns_df.index
        columns_=range(numberStocks)
        labels_df = pd.DataFrame(index=index_,columns=columns_)
            
        #Computing lag-log returns for 1 day lag (1 day forward returns)-->those are the labels independetely of the lag considered for features
        lag=1
        i=0#legacy counter for number of lags (not using it now, bc we only consider one lag period)        
        for j in range(numberStocks):
            labels_df.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(lag))

        labels_df=labels_df.shift(-1)#shifting to get correct alignements  

        #Saving changes into global variable
        self.labels_df=labels_df
 
    
    def performance(self,costPerTrade=0):
        
        #retrieving global varialbes        
        weights_df=self.weights_df
        #returns_df=self.returns_df
        numberStocks=self.numberStocks
        numberLags=self.numberLags
        labels_df=self.labels_df        
        #data=self.data        
        
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
        columns_=['N-day lag']#one column for each "k" (lag period)
        results_table_df = pd.DataFrame(index=index_ , columns=columns_)
            
        #------Parameter
        tradeSize=self.size
        
        
        #for i1 in range(numberLags):#looping on each lag
        i1=0
        a=i1*numberStocks
        b=i1*numberStocks+numberStocks-1
        pnl_df=pd.np.multiply(labels_df,weights_df.ix[:,a:b])#returns*weights 
        pnl_daily=pnl_df.sum(axis=1)*tradeSize#sum across columns (total pnl for each day)
            
        #Adding transaction cost: broker fees (if any)
        brokerFees=costPerTrade*tradeSize*(pnl_daily!=0)
        pnl_daily=pnl_daily-brokerFees
        
        #computing performance metrics
        pnl = pnl_metrics(tradeSize,pnl_daily)
        pnl.compute_daily_metrics()
        #Plot
        #(pnl_daily.cumsum()).plot()
        #Printing Results:
        results_table_df.ix[0,i1]=pnl.PnL
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
        
        
