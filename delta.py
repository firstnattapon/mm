import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import pandas_datareader  as pdr
import ccxt
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime


col1, col2 , col3  = st.beta_columns(3)

col1.subheader('1')
col2.subheader('2')
col3.subheader('3')

