# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:11:16 2017

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
#from pandas_datareader import data as web


class computing_returns(object):
    
    
    def __init__(self, close=None):
		#inputs
        self.dataClose=close
       
             
        self.numberStocks=self.dataClose.shape[1]
        
        self.returns_df=None
        self.numberLags=None
        self.lags=None
        
        self.indices_df=None
        self.container_indices=None
    
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
        
    def get_container_indices(self, identifiers_df):
        #this function uses "self.indices_df" and "cols.indentifiers_df" to get the actual indices
        #of the containers to be used in each date
        index_=self.indices_df.index
        columns_=range(1)
        container_indices=pd.DataFrame(index=index_,columns=columns_)
        
        container_indices.ix[:,0]=self.indices_df.ix[:,2].map(lambda x: list(identifiers_df.ix[:,0]).index(x))
        
        self.container_indices=container_indices