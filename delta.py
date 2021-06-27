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

col1.subheader('1')
col2.subheader('2')
col3.subheader('3')
col4.subheader('4')
col5.subheader('5')

pair_data = col1.text_input("pair_data", "1")
fix_value = col2.text_input("fix_value", "1")
