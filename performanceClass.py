import pandas as pd
import datetime as dt
import numpy as np
import random
import math
from pnl_metrics import *
from cross_sectional_MR_without_data_pull import *
import itertools


#inputs:
#_dataForward: price data for the forward period (period over which the performance
#will be calculated)
#results_table_df: attribute from class "rankClass"-->gives the top performers
#groups from the backward period, to be traded in the forward period
class performanceClass(object):
    
    
    def __init__(self, dataForward=None, results_table_df=None,groupSize=3):
		#inputs
        self.data=dataForward
        self.numberStocks=dataForward.shape[1]
        self.results_table_df=results_table_df
        self.groupSize=groupSize#number of stocks in each group
        
    
        #creating data frame that will store results:one column for each group
        #(from results_table_df), one row for each date in dataForward
        index_=dataForward.index
        columns_ = range(results_table_df.shape[0])
        self.daily_pnl_table = pd.DataFrame(index=index_,columns=columns_)    
 
         
   
            
    
    def performance(self,lag=None,filter1_trigger=None,
                 momentum_trigger=None,filterFinal_threshold=None,filter2_trigger=None,
                 filter2_thresholds=None,filter2_leverage=None,size=1):
    
        
        #creating data frame that will store pricing data for each group (this dataframe will be 
        #re-written for every group)
        index_ = self.data.index
        columns_=range(self.groupSize)
        data1 = pd.DataFrame(index=index_,columns=columns_)    
        
        #loop on each group to compute its daily PnL
        aux=0
        for group in self.results_table_df.Group:#loop on each group
            
            #filling in data1 (matrix with pricing data for each stock in the group)
            for i in range(self.groupSize):#loop on each stock of the group
                data1.ix[:,i]=self.data.ix[:,group[i]]
                
                
            strat1=cross_sectional_MR_without_data_pull(data1, lag,size,filter1_trigger,
                         momentum_trigger,filterFinal_threshold,filter2_trigger,
                         filter2_thresholds,filter2_leverage)
            strat1.features()
            strat1.filters()
            strat1.labels()
            strat1.performance()
            
            #filling results table
            self.daily_pnl_table.ix[:,aux]=strat1.pnl_daily
                       
            aux=aux+1          
            
        #Adding a column with the sum of PnLs accross all the groups
        self.daily_pnl_table['Total']=self.daily_pnl_table.sum(axis=1)
       
                
    
 