# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 07:06:36 2017

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


class column_names(object):
    
    
    def __init__(self):
		
        self.numberStocks=None
        self.column_names=None
        self.identifiers_df=None
        
        
    def creating_columns(self,numberStocks):
        self.column_names={}
        self.numberStocks=numberStocks
        if numberStocks==2:
            self.column_names[0]=[0,1]
            self.column_names[1]=[1,0]
        elif numberStocks==3:
            self.column_names[0]=[0,1,2]
            self.column_names[1]=[0,2,1]
            self.column_names[2]=[1,0,2]
            self.column_names[3]=[2,0,1]
            self.column_names[4]=[1,2,0]
            self.column_names[5]=[2,1,0]
        elif numberStocks==4:
            self.column_names[0]=[0,1,2,3]
            self.column_names[1]=[0,2,1,3]
            self.column_names[2]=[0,2,3,1]
            self.column_names[3]=[1,0,2,3]
            self.column_names[4]=[2,0,1,3]
            self.column_names[5]=[2,0,3,1]
            self.column_names[6]=[1,2,0,3]
            self.column_names[7]=[2,1,0,3]
            self.column_names[8]=[2,3,0,1]
            self.column_names[9]=[1,2,3,0]
            self.column_names[10]=[2,1,3,0]
            self.column_names[11]=[2,3,1,0]
        elif numberStocks==5:
            self.column_names[0]=[0,1,2,3,4]
            self.column_names[1]=[0,2,1,3,4]
            self.column_names[2]=[0,2,3,1,4]
            self.column_names[3]=[0,2,3,4,1]
            self.column_names[4]=[1,0,2,3,4]
            self.column_names[5]=[2,0,1,3,4]
            self.column_names[6]=[2,0,3,1,4]
            self.column_names[7]=[2,0,3,4,1]
            self.column_names[8]=[1,2,0,3,4]
            self.column_names[9]=[2,1,0,3,4]
            self.column_names[10]=[2,3,0,1,4]
            self.column_names[11]=[2,3,0,4,1]
            self.column_names[12]=[1,2,3,0,4]
            self.column_names[13]=[2,1,3,0,4]
            self.column_names[14]=[2,3,1,0,4]
            self.column_names[15]=[2,3,4,0,1]
            self.column_names[16]=[1,2,3,4,0]
            self.column_names[17]=[2,1,3,4,0]
            self.column_names[18]=[2,3,1,4,0]
            self.column_names[19]=[2,3,4,1,0]
        elif numberStocks==6:
            self.column_names[0]=[0,1,2,3,4,5]
            self.column_names[1]=[0,2,1,3,4,5]
            self.column_names[2]=[0,2,3,1,4,5]
            self.column_names[3]=[0,2,3,4,1,5]
            self.column_names[4]=[0,2,3,4,5,1]
            self.column_names[5]=[1,0,2,3,4,5]
            self.column_names[6]=[2,0,1,3,4,5]
            self.column_names[7]=[2,0,3,1,4,5]
            self.column_names[8]=[2,0,3,4,1,5]
            self.column_names[9]=[2,0,3,4,5,1]
            self.column_names[10]=[1,2,0,3,4,5]
            self.column_names[11]=[2,1,0,3,4,5]
            self.column_names[12]=[2,3,0,1,4,5]
            self.column_names[13]=[2,3,0,4,1,5]
            self.column_names[14]=[2,3,0,4,5,1]
            self.column_names[15]=[1,2,3,0,4,5]
            self.column_names[16]=[2,1,3,0,4,5]
            self.column_names[17]=[2,3,1,0,4,5]
            self.column_names[18]=[2,3,4,0,1,5]
            self.column_names[19]=[2,3,4,0,5,1]
            self.column_names[20]=[1,2,3,4,0,5]
            self.column_names[21]=[2,1,3,4,0,5]
            self.column_names[22]=[2,3,1,4,0,5]
            self.column_names[23]=[2,3,4,1,0,5]
            self.column_names[24]=[2,3,4,5,0,1]
            self.column_names[25]=[1,2,3,4,5,0]
            self.column_names[26]=[2,1,3,4,5,0]
            self.column_names[27]=[2,3,1,4,5,0]
            self.column_names[28]=[2,3,4,1,5,0]
            self.column_names[29]=[2,3,4,5,1,0]
        else:
            print('error: number of stocks must be 6 or lower (or else just add more options in class "Column_Names")')

        
    def indentifiers(self):
        index_=range(len(self.column_names))
        columns_=range(1)
        identifiers_df=pd.DataFrame(index=index_,columns=columns_)
            
        for i in range(len(self.column_names)):
            decena=self.column_names[i].index(0)
            unidad=self.column_names[i].index(1)
            identifiers_df.ix[i,0]=decena*10+unidad
            
        self.identifiers_df=identifiers_df 
                
                

  