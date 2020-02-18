#%% Imports
import streamlit as st
import numpy as np
import pandas as pd
import time

#%%
st.title('My first app')

st.write("Here's our first attempt at using data to create a table:")

#%% Dataframe
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

#%% Chart data
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

#%% Interactive widgets
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
    
option1 = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option1

#%% Sidebar widgets
option2 = st.sidebar.selectbox(
    'Which number do you like least?',
     df['second column'])

'You selected:', option2

#%%
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

df