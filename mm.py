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
np.set_printoptions(precision=8)

col5, col7 = st.beta_columns(2)
col1,  = st.beta_columns(1)
col2,col3  = st.beta_columns((3 , 1))

with  col2.beta_expander(''):
    capital = st.number_input('capital', min_value=0, max_value=100000 , value=1000 )
    n =  st.number_input('median', min_value=0., max_value=100000., value=0.60)
    f = st.number_input('fix', min_value=0, max_value=100000 , value=500 )
    lowwer = st.number_input('lowwer', min_value=0., max_value=100000. , value=0.0 )
    upper = st.number_input('upper', min_value=0., max_value=100000. , value=2.0  )
    delta = st.number_input('delta', min_value=0., max_value=10.0 , value=0.01)
    r = st.number_input('r%', min_value=0., max_value=10., value=1.01)

p  = np.arange( lowwer , upper , delta )
# n = ( lowwer + (upper -  lowwer) * m)
a =  f / p
d = [abs((a[i+1] - a[i]) * p[i])  if i +1< len(p) else np.nan  for i in range(len(p))]  ; d = np.array(d)

difference_array_1 = np.absolute(p-n)
index_1 = difference_array_1.argmin()
i = np.zeros(len(p))
pf = np.zeros(len(p))
i[index_1] = f

for k in np.arange(index_1 -1 , -1 , -1 ):
    i[k] = i[k+1] + d[k]
    pf[k] = ((p[k]* r ) - p[k]) * a[k] 

for q in  np.arange( index_1 +1 , len(i)  , 1 ):
    i[q] = i[q-1] - d[q]
    pf[q] = ((p[q]* r ) - p[index_1]) * a[q] 

c =  capital - i

i[0] = i[1] ; i[-1] = i[-2] 
pf[index_1] = np.nan

difference_array_2 = np.absolute(np.nan_to_num(i) - capital)
index_2 = difference_array_2.argmin()

difference_array_3 = np.absolute(np.nan_to_num(i) - 0)
index_3 = difference_array_3.argmin()
 
plt.subplots(figsize=(12, 8))
plt.plot(p ,  i  , color='r') 
plt.plot(p ,  c  , color='g')

plt.axhline(0 , color='k', ls='--' , lw=0.9)
plt.axhline(capital , color='k', ls='--' , lw=0.9)
plt.axvline(p[index_1], color='k', ls='--' , lw=0.9, ymin=0.40, ymax=0.60)
plt.axvline(p[index_2], color='k', ls='--' , lw=0.9)
plt.axvline(p[index_3], color='k', ls='--' , lw=0.9)
col1.pyplot()

plt.subplots(figsize=(12, 4))
plt.plot(p , a, ls='--' )
plt.axvline(p[index_1], color='k', ls='--' , lw=0.9, ymin=0.40, ymax=0.60)
plt.axvline(p[index_2], color='k', ls='--' , lw=0.9)
plt.axvline(p[index_3], color='k', ls='--' , lw=0.9)
col5.pyplot()

col3.write('lower = {:.2f}'.format( p[index_2])) 
col3.write('50%  = {:.2f}'.format( p[index_1])) 
col3.write('upper = {:.2f}'.format( p[index_3])) 
col3.write('cf = {:.2f}'.format( pf[3] *(r-1) )) 

col6,  = st.beta_columns(1)
df = pd.DataFrame({ "????????????" : p   ,
                    "asset": a  ,
                    "?????????????????????????????????????????????": i ,
                    "??????????????????????????????????????????" : c,
                   "????????????" : pf *(r-1) })

with  col6.beta_expander('data'):
    st.write(df)

plt.subplots(figsize=(12, 4))
plt.plot(p ,  pf *(r-1)  , ls='--') 
plt.axvline(p[index_1], color='k', ls='--' , lw=0.9, ymin=0.40, ymax=0.60)
plt.axvline(p[index_2], color='k', ls='--' , lw=0.9)
plt.axvline(p[index_3], color='k', ls='--' , lw=0.9)
col7.pyplot()    
