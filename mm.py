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
