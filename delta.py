import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import pandas_datareader  as pdr
import ccxt
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime


col1, col2 , col3 , col4 , col5   = st.beta_columns(5)
pair_data = col1.text_input("pair_data", "CRV/USD")
fix_value = float(col2.text_input("fix_value", "0.5" ))
invest =  int(col3.text_input("invest" , "1000"))
timeframe = col4.text_input("timeframe", "1h")
limit = int(col5.text_input("limit", "2500"))


