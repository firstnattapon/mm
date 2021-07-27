import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt  
import pandas_datareader  as pdr
import ccxt
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime
st.set_option('deprecation.showPyplotGlobalUse', False)
import datetime
plt.style.use('ggplot')
from stqdm import stqdm


capital = 1000 
lowwer = 0.00  
upper = 2.00  
delta = 0.01  
n =  0.40 
f = 500     
r = 1.01

p  = np.arange( lowwer , upper , delta )
# n = ( lowwer + (upper -  lowwer) * m)
a =  f / p
d = [abs((a[i+1] - a[i]) * p[i])  if i +1< len(p) else np.nan  for i in range(len(p))]  ; d = np.array(d)

difference_array_1 = np.absolute(p-n)
index_1 = difference_array_1.argmin()
i = np.zeros(len(p) , dtype=np.float16 )
pf = np.zeros(len(p), dtype=np.float16 )
i[index_1] = f

for k in np.arange(index_1 -1 , -1 , -1 ):
    i[k] = i[k+1] + d[k]
    pf[k] = ((p[k]* r ) - p[k]) * a[k] 

for q in  np.arange( index_1 +1 , len(i)  , 1 ):
    i[q] = i[q-1] - d[q]
    pf[q] = ((p[q]* r ) - p[index_1]) * a[q] 

c = f*2 - i

difference_array_2 = np.absolute(np.nan_to_num(i) - capital)
index_2 = difference_array_2.argmin()

difference_array_3 = np.absolute(np.nan_to_num(c) - capital)
index_3 = difference_array_3.argmin()
