import datetime
import pandas as pd
import csv
import datetime
import numpy as np
import math
from scipy.ndimage.interpolation import shift#shift method for numpy array

import matplotlib.pyplot as plt


class pnl_metrics(object):
    
    
    def __init__(self, capital=None, daily_pnl=None):
		#inputs
        self.daily_pnl=daily_pnl#data frame with two columns: "date" and "PnL" for each day
        self.capital=capital#capital used in the strategy
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
        
    #the below function is ONLY for DAILY pnl time series
    def compute_daily_metrics(self):        
        if np.sum(self.daily_pnl**2)!=0:
            self.PnL=np.sum(self.daily_pnl)
            #Sharpe Ratio
            self.sharpeRatioDaily=self.daily_pnl.mean(axis=0)/self.daily_pnl.std(axis=0)        
            self.sharpeRatioAnnual=self.sharpeRatioDaily*math.sqrt(252)
            #CAGR
            self.CAGR=max(0.001,(self.PnL/self.capital+1))**(252*1.0/(self.daily_pnl.count()))-1#multiplying times 1.0 to make it floating number	

            #computing max drawdown and max drawdown duration
            pnl_cumsum=self.daily_pnl.cumsum()+self.capital#adding capital to compute drawdowns
            mdd = 0
            peak = pnl_cumsum.ix[0]#initializing the variable peak (will store the max found as we move through the series)
            aux_count=0#to keep track of the number of days in the running drawdown (will count everyday we are below the peak of the equity series)
            maxdd_duration=0#to keep track of the number of days in the running drawdown if it is max seen so far
            alert_count=0#will be 1 if we are in max drawdown "valley"//wil be 0 otherwise-->set to 1 when we find a new max dd
            for x in pnl_cumsum:
                if x>peak:
                    peak=x
                    if alert_count==1:#if we were in the max drawdown seen so far
                        maxdd_duration=aux_count#save the duration of the max drawdown				        
                    aux_count=0
                    alert_count=0#setting this to zero as we are not anymore in max drawdown "valley" territory
                else:
                    aux_count=aux_count+1
                dd=(peak-x)/peak
                if dd>mdd:
                    mdd=dd
                    alert_count=1#indicates that we are currently in the max drawdown seen so far
                
            #maxDD
            self.maxDD=mdd
            #durationDD
            self.durationMaxDD=maxdd_duration
		    #Calmar Ratio
            self.Calmar=self.CAGR/self.maxDD
		    #tradedDays
            self.tradedDays=self.daily_pnl.count()-(self.daily_pnl==0).sum()
            self.tradedDaysPct=1-(self.daily_pnl==0).sum()/(self.daily_pnl.count()*1.0)#multiplying times 1.0 to force the result as floating point (otherwise i get 0)
            #winnerDays
            self.winnerDays=(self.daily_pnl>0).sum()*1.0/(self.daily_pnl.count()-(self.daily_pnl==0).sum())#winner/traded days		
		    #averageWin
            pnl_df2=((self.daily_pnl>0)*self.daily_pnl)#negative values=0
            pnl_df2 = pnl_df2.replace(0, np.NaN)#zero values are assigned an NaN
            self.averageWin=pnl_df2.mean()/self.capital
		    #averageLoss
            pnl_df2=((self.daily_pnl<0)*self.daily_pnl)#positive values=0
            pnl_df2 = pnl_df2.replace(0, np.NaN)#zero values are assigned an NaN
            self.averageLoss=pnl_df2.mean()/self.capital
           #averageTrade
            pnl_df2=self.daily_pnl
            pnl_df2 = pnl_df2.replace(0, np.NaN)#zero values are assigned an NaN
            self.averageTrade=pnl_df2.mean()/self.capital
		
		

        
       
        
