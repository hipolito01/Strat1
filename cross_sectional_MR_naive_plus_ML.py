#this class backtests a strategy:
#_during a backtest period
#_for each date in the backtest period, it trains the model on the "training period", and applies it to the "test period"
#using the given classifier model ("cl_production")

import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt
import scipy as scipy
import math as math

from features import *
from labels import *
from column_names import *
from computing_returns import *

#Machine Learning Models
from EnsembleClassifier import *#class for ensembles (lo baje de internet hace mucho)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score



class cross_sectional_MR_naive_plus_ML(object):
    
    
    def __init__(self, dataClose=None, dataHigh=None, dataLow=None,dataVolume=None,cl_production=None, stock_list=None,
                 data_start=None,data_end=None,backtest_start=None,backtest_end=None, train_window=None, test_window=None,
                 stock_ids=None, option=1):
        
        #inputs                
        self.dataClose=dataClose
        self.dataHigh=dataHigh
        self.dataLow=dataLow
        self.dataVolume=dataVolume
        
        self.stock_list=stock_list
        
        self.cl_production=cl_production#classifier to be used
        
        self.start_date=data_start
        self.end_date=data_end
        self.backtest_start=backtest_start
        self.backtest_end=backtest_end
        self.train_window=train_window
        self.test_window=test_window
        
        self.stock_ids=stock_ids
        
        self.option=option#option=1 if stock_ids are the indexes (0,1,20,etc)//option=2 if stock_ids are the symbols (ASR, PAC, etc)
    
    
        #variables to be defined in fucntions below            
        self.data_Close=None
        self.data_High=None
        self.data_Low=None
        self.data_Volume=None
        self.stocks=None
        
        self.features_df=None
        self.ret1_returns_df=None
        self.ret1_indices_df=None
        self.lags=None
        
        self.lab1_labels_df=None
        self.lab1_labels_returns_df=None
        
        self.features_prod=None
        self.labels_prod=None
        self.labels_returns_prod=None
        
        self.predictions_final=None
        self.labels_final=None
        self.labels_returns_final=None
        

    
        
        #all variables from "pnl" class
        self.results_table_df=None
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
        self.pnl_daily=None

        
    
    def cleaning_data(self):
        
        dataClose=self.dataClose
        dataHigh=self.dataHigh
        dataLow=self.dataLow
        dataVolume=self.dataVolume
        stock_list=self.stock_list
        
        option=self.option
        stock_ids=self.stock_ids               
        
        #stock_ids are the column numbers for each stock in "dataClose" (equivalent to the row numbers for each stock on "stock_list")
        #stocks are the IB symbol for each stock
        
        #option 1--> Input: ids for each stock
        #option 2--> Input: IB symbols for each stock       
        
        if option==1:
            stocks=[None]*len(stock_ids)
            for i in range(len(stock_ids)):
                stocks[i]=stock_list.ix[stock_ids[i],'Symbol IB']
            
        #elif option==2:
        #    stock_ids=[None]*len(stocks)
        #    for i in range(len(stocks)):
        #        stock_ids[i]=stock_list.loc[stock_list['Symbol IB']==stocks[i]].index.values[0]                
            
        #-----------------Parameters------------------------#
        #----Study Window-----
        start_date=self.start_date
        end_date=self.end_date
        
        #-------REDUCING DATA: FROM ENTIRE DATABASE TO DATA ONLY FOR THOSE STOCKS INVOLVED------------
        #creating data frame that will store the data only for those stocks involved
        index_ = dataClose.ix[start_date:end_date].index
        columns_=range(len(stock_ids))
        data_Close = pd.DataFrame(index=index_,columns=columns_)
        data_High = pd.DataFrame(index=index_,columns=columns_)
        data_Low = pd.DataFrame(index=index_,columns=columns_)
        data_Volume = pd.DataFrame(index=index_,columns=columns_)
        
        for i in range(len(stock_ids)):
            data_Close.ix[:,i]=dataClose.ix[start_date:end_date,stock_ids[i]]
            data_High.ix[:,i]=dataHigh.ix[start_date:end_date,stock_ids[i]]
            data_Low.ix[:,i]=dataLow.ix[start_date:end_date,stock_ids[i]] 
            data_Volume.ix[:,i]=dataVolume.ix[start_date:end_date,stock_ids[i]] 
                
        #Saving results
        self.data_Close=data_Close
        self.data_High=data_High
        self.data_Low=data_Low
        self.data_Volume=data_Volume
        self.stocks=stocks
        
        
    def features(self, lags=[1,2,5]):
        
        data_Close=self.data_Close
        data_High=self.data_High
        data_Low=self.data_Low
        data_Volume=self.data_Volume
                
        
        
        #------------"column_names" class--------------------------
        cols=column_names()
        cols.creating_columns(data_Close.shape[1])
        cols.indentifiers()
        
        #------------"features" class------------------------------
        features_container={}
        
        #---------------PARAMETERS-------------------
        #lags=[1,2,5]
        #--------------------------------------------
        
        for i in range(len(cols.column_names)):
            #Close
            data_Close2=data_Close.copy()
            data_Close2.columns=cols.column_names[i]
            data_Close2.sort(axis=1,inplace=True)
            #High
            data_High2=data_High.copy()
            data_High2.columns=cols.column_names[i]
            data_High2.sort(axis=1,inplace=True)
            #Low
            data_Low2=data_Low.copy()
            data_Low2.columns=cols.column_names[i]
            data_Low2.sort(axis=1,inplace=True)
            #Volume
            data_Volume2=data_Volume.copy()
            data_Volume2.columns=cols.column_names[i]
            data_Volume2.sort(axis=1,inplace=True)  
            
            feat1=features(data_Close2,data_High2,data_Low2,data_Volume2 )
            feat1.filling_container(lags)
            feat1.technical_indicators()
            features_container[i]=feat1.all_features_df
        
        #----------"computing_returns" class------------------------
        ret1=computing_returns(data_Close)
        ret1.lag_returns()#computes returns for each stock
        ret1.get_indices()#computes indices of best and worst stock for each day
        ret1.get_container_indices(cols.identifiers_df)#compute the  index of the "features container" for each day
        
        #combining the container and indices to get all the features
        index_=ret1.container_indices.index
        columns_=feat1.all_features_df.columns
        features_df=pd.DataFrame(index=index_,columns=columns_)
        
        #looping on each date
        aux=0
        for i in features_df.index:
            features_df.ix[i,:]=features_container[ret1.container_indices.ix[aux,0]].ix[i,:]
            aux=aux+1
    
        #Saving results
        self.lags=lags
        self.features_df=features_df
        self.ret1_returns_df=ret1.returns_df
        self.ret1_indices_df=ret1.indices_df
    
    def labels(self,threshold_do_nothing=0.00):
        
        ret1_returns_df=self.ret1_returns_df
        ret1_indices_df=self.ret1_indices_df
            
        #-------------------LABELS-----------------------
        lab1=labels(ret1_returns_df,ret1_indices_df)
        lab1.compute_labels(threshold_do_nothing)       
        
        #Saving results
        self.lab1_labels_df=lab1.labels_df
        self.lab1_labels_returns_df=lab1.labels_returns_df

    def cleaning_features_and_labels(self,binary_variable=8):
        
        lab1_labels_df=self.lab1_labels_df
        lab1_labels_returns_df=self.lab1_labels_returns_df
        features_df=self.features_df       
        lags=self.lags
        
        #dropping NAs from dataframes (there are NAs in the middle when missing data -->shouldnt happen often-->CHECK THAT!)
        features_df.fillna(method='ffill', inplace=True)
        lab1_labels_df.fillna(method='ffill', inplace=True)
        lab1_labels_returns_df.fillna(method='ffill', inplace=True)
        
        #features and labels production (dropping NAs from head and tails of dataframes)
        features_prod=features_df[20:-1]
        labels_prod=lab1_labels_df.ix[20:-1]
        labels_returns_prod=lab1_labels_returns_df.ix[max(lags):-1]#-->used to compute backtesting returns
        
        #Normalizing the features
        aux_f=features_prod.copy()
        aux_f=(aux_f-aux_f.mean())/aux_f.std()
        aux_f.ix[:,binary_variable]=features_prod.ix[:,binary_variable]#-->excluding binary features from normalization
        
        features_prod=aux_f
        
        #Saving results
        self.features_prod=features_prod
        self.labels_prod=labels_prod
        self.labels_returns_prod=labels_returns_prod
       
        
    def training_and_predicting(self):        
        
        features_prod=self.features_prod
        labels_prod=self.labels_prod
        labels_returns_prod=self.labels_returns_prod
        train_window=self.train_window
        test_window=self.test_window
        cl_production=self.cl_production
        backtest_start=self.backtest_start
        backtest_end=self.backtest_end
        
        
        #features_prod and labels_prod set within the study window (backtest_start to backtest_end)
        features_prod=features_prod[backtest_start:backtest_end]
        labels_prod=labels_prod[backtest_start:backtest_end]
        labels_returns_prod=labels_returns_prod[backtest_start:backtest_end]#-->used to compute backtesting returns
        
        #Creating dataframes to store
        index_ = features_prod.index[0:1]
        columns_=range(1)
        predictions_final_all = pd.DataFrame(index=index_,columns=columns_)
        labels_final_all = pd.DataFrame(index=index_,columns=columns_)
        labels_returns_final_all = pd.DataFrame(index=index_,columns=columns_)
        
        for day in range(train_window,features_prod.shape[0]-test_window,test_window):
        
            #----Splitting dataset:TRAIN and TEST-------------
            #"day" is when the test period starts (train period ends at "day - 1")
            train_start=features_prod.index[day-train_window]
            train_end=features_prod.index[day-1]
            test_start=features_prod.index[day]
            test_end=features_prod.index[day+test_window-1]
            
            #Train
            X_train=features_prod[train_start:train_end].copy()
            y_train=labels_prod[train_start:train_end].copy()
            #Test
            X_test=features_prod[test_start:test_end].copy()
            y_test=labels_prod[test_start:test_end].copy()
            y_test_returns=labels_returns_prod[test_start:test_end].copy()#-->used to compute backtesting returns (pnl)
        
            #--------------------------------------------------
        
            #----Creating and filling dataframes 'predictions_all' and 'predictions_all_prob'----
            #Creating dataframes
            index_ = y_test.index
            columns_=range(1)
            predictions_all = pd.DataFrame(index=index_,columns=columns_)
            predictions_all_prob = pd.DataFrame(index=index_,columns=columns_)
        
            #loop on each "label type" (label type=distinct pair of stocks)
            for i in range(labels_prod.shape[1]):
                #1)pulling labels for training
                y_train_production=y_train.ix[:,i]    
                #2)fitting    
                cl_production.fit(X_train, y_train_production)
                #3)predicting
                predictions_all_prob.ix[:,i*2]=cl_production.predict_proba(X_test)[:,0]
                predictions_all_prob.ix[:,i*2+1]=cl_production.predict_proba(X_test)[:,1]
                predictions_all.ix[:,i]=cl_production.predict(X_test)
        
            #getting indeces of max prob predictions
            aa=predictions_all_prob.idxmax(axis=1)
            bb=list(map(lambda x: int(x/2),aa))
        
            #pulling predictions and labels based on "max prob prediction"
            predictions_final = pd.DataFrame(index=index_,columns=columns_)
            labels_final = pd.DataFrame(index=index_,columns=columns_)
            labels_returns_final = pd.DataFrame(index=index_,columns=columns_)
        
        
            for i in range(len(predictions_all)):
                predictions_final.ix[i,0]=predictions_all.ix[i,bb[i]]
                labels_final.ix[i,0]=y_test.ix[i,bb[i]]
                labels_returns_final.ix[i,0]=y_test_returns.ix[i,bb[i]]#-->used to compute backtesting returns (pnl)
                
            
            predictions_final_all=pd.concat([predictions_final_all,predictions_final])
            labels_final_all=pd.concat([labels_final_all,labels_final])
            labels_returns_final_all=pd.concat([labels_returns_final_all,labels_returns_final])
        
            
        #Dropping NAs
        predictions_final=predictions_final_all.dropna()
        labels_final=labels_final_all.dropna()
        labels_returns_final=labels_returns_final_all.dropna()
        
        
        #Saving Results        
        self.predictions_final=predictions_final
        self.labels_final=labels_final
        self.labels_returns_final=labels_returns_final
        
        

       
        
    def computing_returns(self,costPerTrade=0.00,tradeSize=1):        
                
        #Defining positions and returns
        weights_df=self.predictions_final#if label=0-->dont trade//if label=1-->trade it
        labels_df=self.labels_returns_final
        
        #------Parameters
        #tradeSize=1
        #costPerTrade=0
        
        #------Dataframe with results
        index_ = ['PnL','CAGR','Sharpe Ratio','Max Drawdown','Calmar','Max Drawdown Duration','Traded Days %','Winner Days %','Avg Trade', 'Avg Win', 'Avg Loss']
        columns_=['PnL Metrics']#one column for each "k" (lag period)
        results_table_df = pd.DataFrame(index=index_ , columns=columns_)
        
        
        pnl_df=pd.np.multiply(labels_df,weights_df)#returns*weights 
        pnl_daily=pnl_df.sum(axis=1)*tradeSize#sum across columns (total pnl for each day)
            
        #adding transaction costs
        brokerFees=costPerTrade*tradeSize*(pnl_daily!=0)
        pnl_daily=pnl_daily-brokerFees
            
        #computing performance metrics
        pnl = pnl_metrics(tradeSize,pnl_daily)
        pnl.compute_daily_metrics()
        #Plot
        #(pnl_daily.cumsum()).plot()
        #Printing Results:
        i1=0#-->only 1 scenario
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
        self.pnl_daily=pnl_daily
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
     