import pandas as pd
import datetime as dt
import numpy as np
import random
import math
from pnl_metrics import *
from cross_sectional_MR_without_data_pull import *
import itertools


#_this class receives data as input (time series of prices for N different
#stocks)-->rows=dates/columns=stocks
#_then it groups all the stocks into all the possible combinations of x stocks (where x
#is the number of stocks in each group, by default set to 3)
#_then it calculates performance metrics for each group applying a cross-sectional
#mean reversion strategies
#_finally it ranks each group based on specific metrics and threshold values
class rankClass(object):
    
    
    def __init__(self, data=None, groupSize=None):
		#inputs
        self.data=data
        self.numberStocks=data.shape[1]
        self.groupSize=groupSize
        
    
		 #variables
        self.allCombinations=itertools.combinations(range(self.numberStocks), groupSize)
        self.listCombinations=list(self.allCombinations)
        #creating data frame that will store results
        index_=range(len(self.listCombinations))
        columns_ = ['PnL','CAGR','Sharpe_Ratio','Max_Drawdown','Calmar','Max_Drawdown_Duration','Traded_Days_%',
          'Winner_Days_%','Avg_Win', 'Avg_Loss','Group']
        self.results_table_df = pd.DataFrame(index=index_,columns=columns_)    
 
         
   
            
    
    def performance(self,lag=None,filter1_trigger=None,
                 momentum_trigger=None,filterFinal_threshold=None,filter2_trigger=None,
                 filter2_thresholds=None,filter2_leverage=None,size=1):
    
        
        #creating data frame that will store pricing data for each group (this dataframe will be 
        #re-written for every group)
        index_ = self.data.index
        columns_=range(self.groupSize)
        data1 = pd.DataFrame(index=index_,columns=columns_)    
        
        aux=0
        for group in self.listCombinations:#loop on each group
            
            #filling in data1 (matrix with pricing data for each stock in the group)
            for i in range(self.groupSize):#loop on each stock of the group
                data1.ix[:,i]=self.data.ix[:,group[i]]
                
                
            strat1=cross_sectional_MR_without_data_pull(data1,lag,size,filter1_trigger,
                         momentum_trigger,filterFinal_threshold,filter2_trigger,
                         filter2_thresholds,filter2_leverage)
            strat1.features()
            strat1.filters()
            strat1.labels()
            strat1.performance()
            
            #filling results table
            self.results_table_df.ix[aux,0]=strat1.PnL
            self.results_table_df.ix[aux,1]=strat1.CAGR
            self.results_table_df.ix[aux,2]=strat1.sharpeRatioAnnual
            self.results_table_df.ix[aux,3]=strat1.maxDD
            self.results_table_df.ix[aux,4]=strat1.Calmar
            self.results_table_df.ix[aux,5]=strat1.durationMaxDD
            self.results_table_df.ix[aux,6]=strat1.tradedDaysPct
            self.results_table_df.ix[aux,7]=strat1.winnerDays
            self.results_table_df.ix[aux,8]=strat1.averageWin
            self.results_table_df.ix[aux,9]=strat1.averageLoss
            self.results_table_df.ix[aux,10]=group
            
            aux=aux+1          
        
                
    
    def rankingBySharpe(self,threshold=None,sortingCriterium=0):
        #metric input could be any of the following (column values for results_table):
        #['PnL','CAGR','Sharpe Ratio','Max Drawdown','Calmar','Max Drawdown Duration','Traded Days %',
        #'Winner Days %','Avg Win', 'Avg Loss','Group']
        #sortingCriterium=1 if sorted in ascending order/sortingCriterium=0 if sorted in descending
        
        
        self.results_table_df=self.results_table_df.dropna()#doing this to avoid error that occurs sometime with quantile function
        threshold=self.results_table_df['Sharpe_Ratio'].quantile(threshold)
        #remove all values BELOW the threshold
        self.results_table_df=self.results_table_df[(self.results_table_df.Sharpe_Ratio>threshold)]
        
        #sort results table by the corresponding metric and criterium
        self.results_table_df = self.results_table_df.sort_values('Sharpe_Ratio', ascending=sortingCriterium)
        
        
        