import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt
#from pandas_datareader import data as web


class features(object):
    def __init__(self, close=None, high=None, low=None, volume=None):
        self.dataClose=close
        self.dataHigh=high
        self.dataLow=low
        self.dataVolume=volume
            
        self.numberStocks=self.dataClose.shape[1]
        
        #to be defined in the functions below
        self.all_features_df=None
        self.numberLags=None
        self.lags=None
        self.returns_diff_df=None
        self.indicators_df=None
        
        
    
    def filling_container(self, lags2=[1,2,5]):        
        
        #this function uses the following definition:
        # "dataClose" is a matrix with closing prices for a set of stocks (each column represents one particular stock)
        # S1 is the stock in the first column of "dataClose"
        # S2 is the stock in the second column of "dataClose"
        
        
        
        #1)---- 1-day log returns for S1 and S2       
        data=self.dataClose.ix[:,:1]#taking only best and worst stocks (first two columns)
        
        lags=[1]
        numberLags=1
        numberStocks=data.shape[1]#2                
        
        #"returns_df" contains the log returs for each stock and for the given lag period
        index_ = data.index
        columns_=range(numberStocks*numberLags)
        returns_df = pd.DataFrame(index=index_,columns=columns_)

        #Computing lag-log returns
        for i in range(0,numberLags):
            lag=lags[i]
            for j in range(numberStocks):
                returns_df.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(lag))
        

        
        #2)---- Average 1-day return for all the other stocks in the group (all the stocks except S1 and S2)
        
        data=self.dataClose.ix[:,2:]#taking all the stocks but the first two columns
        data.columns=range(0,data.shape[1],1)#renaming columns so the names are 0,1,2,...
                              
        lags=[1]
        numberLags=1
        numberStocks=data.shape[1]#2                
        
        #creating auxiliary DF
        index_ = data.index
        columns_=range(numberStocks*numberLags)
        returns_df2 = pd.DataFrame(index=index_,columns=columns_)
       
        #Computing lag-log returns
        for i in range(0,numberLags):
            lag=lags[i]
            for j in range(numberStocks):
                returns_df2.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(lag))
        
        #computing average and adding it to returns_df
        returns_df.ix[:,2]=returns_df2.mean(axis=1)
        #computing differences between S1 and S2 and the average of the others
        returns_df.ix[:,3]=returns_df.ix[:,0]-returns_df2.mean(axis=1)
        returns_df.ix[:,4]=returns_df2.mean(axis=1)-returns_df.ix[:,1]    
        
        #3)------ differences between S1 and S2 in 1-day,2-day and 5-day returns
    
        data=self.dataClose.ix[:,:1]#taking only best and worst stocks (first two columns)
        data.columns=range(0,data.shape[1],1)#renaming columns so the names are 0,1,2,...
        
        #lags2=[1,2,5]
        numberLags=len(lags2)
        numberStocks=data.shape[1]#2                
        
        #auxiliary DF
        index_ = data.index
        columns_=range(numberStocks*numberLags)
        returns_df2 = pd.DataFrame(index=index_,columns=columns_)

        #Computing lag-log returns
        for i in range(0,numberLags):
            lag=lags2[i]
            for j in range(numberStocks):
                returns_df2.ix[:,i*numberStocks+j]=np.log(data.ix[:,j])-np.log(data.ix[:,j].shift(lag))
           
        
        returns_df.ix[:,5]=returns_df2.ix[:,0]-returns_df2.ix[:,1]#1-day return diff
        returns_df.ix[:,6]=returns_df2.ix[:,2]-returns_df2.ix[:,3]#2-day return diff
        returns_df.ix[:,7]=returns_df2.ix[:,4]-returns_df2.ix[:,5]#5-day return diff
        
        #4)------ binary variable: 1 if 2-day return diff between S1 and S2 is positive, 0 otherwise
    
        returns_df.ix[:,8]=(returns_df.ix[:,4]>0)*1
    
        #5)------ distance S1 to high and S2 to low
        
        #Distance High to Close for S1
        returns_df[9]=self.dataHigh.ix[:,0]/self.dataClose.ix[:,0]-1
        #Distance Close to Low for S2    
        returns_df[10]=self.dataClose.ix[:,1]/self.dataLow.ix[:,1]-1

     
    
        #Saving results
        self.all_features_df=returns_df
        self.lags=lags
        self.numberLags=numberLags 
        
        
    def technical_indicators(self,bollinger_days=10,RSI_window=7,id_window=7,std_window=10):
        data=self.dataClose.ix[:,:1]#taking only best and worst stocks (first two columns)
        
        #"indicators_df" contains technical indicators
        index_ = data.index
        columns_=range(1)
        indicators_df = pd.DataFrame(index=index_,columns=columns_)
        
        #1) ratio to Exponential Moving Average (5 and 10 days)
    
        ema=pd.ewma(data,span=5)
        indicators_df=data/ema-1
        
        ema=pd.ewma(data,span=10)
        indicators_df=pd.concat([indicators_df,data/ema-1],axis=1)
        
        #2) ratio to bollinger bands
    
        #bollinger_days=10
        ema=pd.ewma(data,span=bollinger_days)
        sd=data.rolling(bollinger_days).std()
        
        indicators_df=pd.concat([indicators_df,data/(ema+2*sd)-1],axis=1)#ratio to upper bands
        indicators_df=pd.concat([indicators_df,data/(ema-2*sd)-1],axis=1)#ratio to lower bands
        
        #3)RSI
        
        #formula: RSI=100-100/(1+RS)
        #RS= Average change during win days/Average absolute change during losing days
        #averages are calculated over a window period
    
        #RSI_window=7
        data_returns=np.log(data)-np.log(data).shift(1)#1-day returns for S1 and S2
        pos=(data_returns[data_returns>0]).rolling(min_periods=1,center=False,window=RSI_window).mean()
        neg=-(data_returns[data_returns<0]).rolling(min_periods=1,center=False,window=RSI_window).mean()
        #avoiding NANs
        pos.fillna(0.00001,inplace=True)                     
        neg.fillna(0.00001,inplace=True)
        
        RS=pos/neg
        RSI=100-100/(1+RS)
    
        indicators_df=pd.concat([indicators_df,RSI],axis=1)#adding RSI for both S1 and S2
        
        #4) ID indicator
        #this is the indicator suggested in "quantitativa momentum" book (Gray and Vogel), chapter 6
        #the formula is: ID= sign(return)*(%negative periods - %positive periods)
        #the idea is that the higher the ID, there is more discrete information, and then it is better for mean reversion
        #on the contrary, the lower the ID, there is less discrete information, and then the better it is for momentum
        #id_window=7
        
        data_returns=np.log(data)-np.log(data).shift(1)#1-day returns for S1 and S2
        sign0=(data_returns>0)*1#1 or 0
        sign=(data_returns>0)*1*2-1#1 or -1
        id_indicator=sign*(1-2*sign0.rolling(id_window).sum()/id_window)
        
        indicators_df=pd.concat([indicators_df,id_indicator],axis=1)#adding ID indicator for both S1 and S2
        
                               
        #5) Ratio 1-day % change / std (10 days)
        #invento mio
        
        #std_window=7
        data_returns=np.log(data)-np.log(data).shift(1)#1-day returns for S1 and S2
        
        realized_vol=data_returns.rolling(min_periods=1,center=False,window=std_window).std()
        ratio_vol=abs(data_returns/realized_vol)
        
        indicators_df=pd.concat([indicators_df,ratio_vol],axis=1)#adding vol ratio
               
      
        #6) Ratio last volume / average volume (10 days)
        #invento mio                             
    
        data_volume=self.dataVolume.ix[:,:1]#taking only best and worst stocks (first two columns)
        vol_window=20#10 days average
        
        average_volume=data_volume.rolling(min_periods=1,center=False,window=vol_window).mean()
        ratio_volume=data_volume/average_volume
        
        indicators_df=pd.concat([indicators_df,ratio_volume],axis=1)#adding vol ratio
        

        #---------------SAVING RESULTS---------------------------------   
        self.indicators_df=indicators_df                
        self.all_features_df=pd.concat([self.all_features_df,indicators_df],axis=1)#addingt technical indicators to the features data
        self.all_features_df.columns=range(self.all_features_df.shape[1])#renaming columns: new names are 0,1,2,...
 