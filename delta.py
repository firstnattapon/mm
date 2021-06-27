import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import pandas_datareader  as pdr
import ccxt
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime


col1, col2 , col3, col4 , col5 = st.beta_columns(5)

col1.text_input('Enter some text')
col2.text_input('Enter some text')
col3.text_input('Enter some text')
col4.text_input('Enter some text')
col5.text_input('Enter some text')
