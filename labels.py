import pandas as pd
import datetime as dt
import numpy as np
from sklearn import cross_validation
import random
import math
from pnl_metrics import *
import matplotlib.pyplot as plt
#from pandas_datareader import data as web


class labels(object):
    
    
    def __init__(self, returns_df=None,indices_df=None):
		 
        #inputs
        self.returns_df=returns_df
        self.indices_df=indices_df
        
        #to be defined in the functions below
        self.labels_df=None
        self.labels_returns_df=None
        
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
        
        
 