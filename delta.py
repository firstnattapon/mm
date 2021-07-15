import pandas as pd
import numpy as np
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

class  delta :
    def __init__(self , usd = 1000 , fix_value = 0.50, p_data = 'ALPHA-PERP', timeframe = '15m' 
                 , limit  = 5000 , series_num = [None] , minimum_re = 0.005 , start_end = [179 , 193]):
        self.usd    = usd
        self.fix_value  = fix_value
        self.p_data = p_data
        self.timeframe = timeframe
        self.limit = limit
        self.series_num = series_num
        self.minimum_re = minimum_re
        self.start_end = start_end

    def get_data(self):
        exchange = ccxt.ftx({'apiKey': '', 'secret': '', 'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(self.p_data, self.timeframe, limit=self.limit)
        ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
        df = pd.DataFrame(ohlcv)
        df.t = df.t.apply(lambda x: datetime.datetime.fromtimestamp(x))
        df = df.set_index(df['t'])
        df.t = df.index.dayofyear
        df = df.loc[df.t >= self.start_end[0]]
        df = df.loc[df.t <= self.start_end[1]]
        df = df.drop(['t'], axis=1)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.drop(['open', 'high' , 'low' , 'volume'] , axis=1) 
        df = df.dropna()
        return df

    def series(self):
        series  = self.get_data()
        series['index'] =np.array([ i for i in range(len(series))])
        series['perdit'] =series['index'].apply(func= (lambda x : np.where( x in self.series_num , 1 , 0)))
        return series

    def  nav (self):
        idx_amount = 3 ; idx_close = 0 ; idx_perdit = 2
        
        nav_data = self.series()
        nav_data['amount'] =  np.nan
        for i in range(len(nav_data)): # amount
            if i == 0 :
                nav_data.iloc[i, idx_amount] =   (self.usd * self.fix_value)  / nav_data.iloc[i, idx_close]
            else :
#                 nav_data.iloc[i, idx_amount] =   np.where(nav_data.iloc[i, idx_perdit] == 0 ,  nav_data.iloc[i-1, idx_amount] , 
#                                                 (self.usd * self.fix_value) / nav_data.iloc[i , idx_close]) 
                
                  nav_data.iloc[i, idx_amount] =   np.where(nav_data.iloc[i, idx_perdit] == 1  and
                                                            (abs((nav_data.iloc[i-1, idx_amount] * nav_data.iloc[i, idx_close]) - (self.usd * self.fix_value)) 
                                                             / (self.usd * self.fix_value)) >= self.minimum_re,
                                                            (self.usd * self.fix_value) / nav_data.iloc[i , idx_close] ,  nav_data.iloc[i-1, idx_amount])                    
                
        nav_data['asset_value'] =  (nav_data['close']*nav_data['amount']) 

        nav_data['re'] =  np.nan
        for i in range(len(nav_data)): # re
            if i == 0 :
                nav_data.iloc[i, 5] =    0
            else :
                nav_data.iloc[i, 5] =    np.where(nav_data.iloc[i, idx_perdit] == 1 and 
                                                  (abs((nav_data.iloc[i-1, idx_amount] * nav_data.iloc[i, idx_close]) - (self.usd * self.fix_value)) 
                                                  / (self.usd * self.fix_value)) >= self.minimum_re
                                                  , (nav_data.iloc[i-1, 3] * nav_data.iloc[i, 0]) -  (self.usd * self.fix_value) , 0)

        nav_data['cash'] =  np.nan
        for i in range(len(nav_data)): # cash
            if i == 0 :
                nav_data.iloc[i, 6] =    (self.usd * self.fix_value)
            else :
                nav_data.iloc[i, 6] =   (nav_data.iloc[i-1, 6] + nav_data.iloc[i, 5] )

        nav_data['sumusd'] =  (nav_data['cash'] + nav_data['asset_value'])

        return nav_data

    def  mkt (self):
        mkt_data  = self.nav()
        mkt_data[':'] = ':'
        mkt_data['cash_mkt'] =  (self.usd * self.fix_value)
        mkt_data['amount_mkt'] =   (mkt_data.iloc[0, 9]  / mkt_data.iloc[0, 0])
        mkt_data['assetvalue_mkt'] =  mkt_data['close'] * mkt_data['amount_mkt']
        mkt_data['sumusd_mkt'] = (mkt_data['assetvalue_mkt']  + mkt_data['cash_mkt'])

        return mkt_data

    def cf (self):
        cf_data = self.mkt()
        cf_data[' : '] = ' : '
        cf_data['cf_usd'] =   cf_data['sumusd']  - cf_data['sumusd_mkt']
        cf_data['cf_change'] =  (cf_data['cf_usd'] /  cf_data.iloc[0 , 7]) * 100

        return cf_data

    def change (self):
        change_data = self.cf()
        change_data['  :  '] = '  :  '
        change_data['price_change'] =   ((change_data['sumusd_mkt'] - change_data.iloc[0, 12]) / change_data.iloc[0, 12]) * 100
        change_data['pv_change']  = ((change_data['sumusd'] - change_data.iloc[0 , 7] ) / change_data.iloc[0 , 7]) *100

        return change_data

    def  final (self):
        final = self.change()
        final['   :   '] = '   :   '
        final['zero_line'] =  0
        final['start_usd'] =  self.usd
        final['t'] =    final.index.dayofyear
        
        return final
  
#  streamlit
col1, col2 , col3 , col4 , col5 , col6  = st.beta_columns(6)
pair_data = col1.text_input("pair_data", "CAKE-PERP")
fix_value = float(col2.text_input("fix_value", "0.5" ))
invest =  int(col3.text_input("invest" , "1000"))
timeframe = col4.text_input("timeframe", "15m")
start = col5.date_input('start' , datetime.date(2021,7,1)) ; start = int(start.timetuple().tm_yday) #; st.sidebar.write(start)
end = col6.date_input('end', datetime.date(2021,7,31)) ; end =  int(end.timetuple().tm_yday) #; st.sidebar.write(end)

y = []
x = 0.45 
mu = 3.97 
max = 2880
n = 9999
for it in range(9999):
    x = mu * x * (1.0 - x)
    y.append(np.around( x * max))
    
if x == 0.9237416727562783 and y[-1] == 2660.0 :
    st.success('Success')
    
    series = np.unique(y)
    delta_A = delta(usd = invest , fix_value = fix_value , p_data = pair_data , timeframe =  timeframe ,series_num = series , start_end =[start , end]) 
    delta_A= delta_A.final()

    with st.beta_expander("expander"):
        options  = st.radio('options', 
                            ['cashflow_hold',
                             'rebalancing',
                             'cf_change vs price_change',
                             'port-value_change vs price_change',
                             'amount_hold vs amount_mkt' ,
                             'asset-value_hold vs asset-value_mkt' ,
                             'cash_hold vs cash_mkt' ,
                             'sumusd_hold vs sumusd_mkt'] ,index=2 )

    if options == 'cashflow_hold':plot = ['cf_change' , 'zero_line']
    elif options == 'rebalancing':plot = ['re' , "zero_line"]
    elif options == 'cf_change vs price_change':plot = ['cf_change' ,'price_change' , "zero_line"]
    elif options == 'port-value_change vs price_change':plot = ['pv_change' ,'price_change' , "zero_line"]
    elif options == 'amount_hold vs amount_mkt':plot = ['amount' ,'amount_mkt']
    elif options == 'asset-value_hold vs asset-value_mkt':plot = ['asset_value' ,'assetvalue_mkt']
    elif options == 'cash_hold vs cash_mkt':plot = ['cash' ,'cash_mkt']
    elif options == 'sumusd_hold vs sumusd_mkt':plot = ['sumusd' ,'sumusd_mkt' , "start_usd"]

    st.write('x0 :' , 0.45  , '   ,   r :' , mu , '   ,   n :' , n  ,'   ,   max :' , max)
    st.write('data :' , len(delta_A) , '   ,   start :' , start , '   ,   end :' , end ,
             '   ,   perdit :',delta_A['perdit'][-1] ,'   ,   re :' , round(delta_A['re'][-1] , 2))

    plt.subplots(figsize=(12, 8))
    for i in plot:
        plt.plot(delta_A[i] , label =i)
    plt.legend()
    st.pyplot()

    st.write('cf_change :'  , round(delta_A['cf_change'][-1] , 2),'%','   ,   cf_usd :',  round(float(delta_A['cf_usd'][-1]) , 2 ) ,'$')
    st.write('amount :'  , round(delta_A['amount'][-1] , 2) , '   ,   amount_mkt :',  round(delta_A['amount_mkt'][-1] , 2)  )
    st.write('sumusd :'  , round(delta_A['sumusd'][-1] , 2) , '   ,   sumusd_mkt :',  round(delta_A['sumusd_mkt'][-1] , 2)  )

    with st.beta_expander("data"):
        _, _ , head , _ ,   = st.beta_columns(4) 
        head.write('เริ่ม')
        st.dataframe(delta_A.head(1))
        _, _ , tail , _ ,   = st.beta_columns(4)
        tail.write('ล่าสุด')
        st.dataframe(delta_A.tail(1))
        _, _ , re , _ ,   = st.beta_columns(4)
        st.dataframe(delta_A[['close' , 're']].tail(5))
        
    st.stop()

else:
    st.stop()        
               
