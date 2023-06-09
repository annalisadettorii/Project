import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import streamlit as st

used_device_df = pd.read_csv('used_device_data.csv')

#renaming
used_device_df = used_device_df.rename(columns={'normalized_used_price': 'used_price', 'normalized_new_price': 'new_price'})

#cleaness
used_device_df['rear_camera_mp'].fillna(value = 0, inplace = True)
used_device_df['front_camera_mp'].fillna(value = 0, inplace = True)
used_device_df['internal_memory'].fillna(used_device_df['internal_memory'].mean(), inplace = True)
used_device_df['ram'].fillna(used_device_df['ram'].mean(), inplace = True)
used_device_df['battery'].fillna(used_device_df['battery'].mean(), inplace = True)
used_device_df['weight'].fillna(used_device_df['weight'].mean(), inplace = True)

st.header('Used phone analysis')
st.subheader('Subtitle to prepapare')

st.write('Short explanation to do')

url = 'https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data?resource=download'

st.write('Where to find the dataset: [link to kaggle](' +url+')')

if st.sidebar.checkbox('Show raw data'):
    st.write('Raw data:')
    st.write(used_device_df)

col_3, col_4, col_5 = st.columns(3)
with col_4:
    used_device_df["4g"] = used_device_df["4g"].replace({"yes": 1, "no": 0})
    used_device_df["5g"] = used_device_df["5g"].replace({"yes": 1, "no": 0})
    diag = np.triu(np.ones_like(used_device_df.corr()))
    fig = plt.figure(figsize = (9, 9))
    sns.heatmap(used_device_df.corr(), annot = True, mask=diag)
    st.write(fig)


st.code('''
remaining_variables = list(x_train.columns.values)
variables = []
RSS_list = [np.inf]
RSStest_list = []
variables_list = dict()

for i in range(1,13):
    best_RSS = np.inf
    
    for comb in combinations(remaining_variables,1):

            result = linear_reg(x_train[list(comb) + variables],y_train)
            RSStest = RSS_test(result[1], x_test[list(comb) + variables], y_test)  
            if result[0] < best_RSS:
                best_RSS = result[0]
                related_rss = RSStest
                best_feature = comb[0]

    #Updating variables for next loop
    variables.append(best_feature)
    remaining_variables.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    RSStest_list.append(related_rss)
    variables_list[i] = variables.copy() 
    
RSS_list.remove(RSS_list[0])
listt = list(variables_list.values()) 
num_variables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
forward_selection = pd.DataFrame({'num_variables': num_variables,'Variables': listt,'RSS_training': RSS_list,'RSS_test': RSStest_list})
''')

#regression model
train_df, test_df = train_test_split(used_device_df, test_size=0.2)
y_train = train_df[['used_price']].copy()
x_train1 = train_df[['new_price','release_year','screen_size','rear_camera_mp','front_camera_mp','weight','ram','4g']].copy()
reg = LinearRegression().fit(x_train1,y_train)
coefs = reg.coef_
coefs = coefs[0]


with st.expander("Choose the values to make a prediction"):
    new_price = st.number_input("Price of the new device")
    release_year = st.number_input("Release year")
    screen_size = st.number_input("Dimension of the screen")
    rear_camera_mp = st.number_input("Resolution of the rear camera")
    front_camera_mp = st.number_input("Resolution of the front camera")
    weight = st.number_input("Weight")
    ram = st.number_input("Capacity of the ram memory")
    fourg = st.text_input("Presence of the 4G technology")
    if new_price <= 0:
        error = " The price must be positive"
        st.error(error)
    a = release_year % 1
    if release_year <= 0 or a != 0:
        error = " The year must be a positive integer"
        st.error(error)
    if screen_size <= 0:
        error = " The screen size must be positive"
        st.error(error)
    if rear_camera_mp <= 0:
        error = " The resolution of the rear camera must be positive "
        st.error(error)
    if front_camera_mp <= 0:
        error = " The resolution of the front camera must be positive"
        st.error(error)
    if weight <= 0:
        error = " The weight must be positive"
        st.error(error)
    if ram <= 0:
        error = " The capacity of the ram must be positive"
        st.error(error)

    if st.button("Predict") and new_price >0 and release_year > 0 or a == 0 and screen_size > 0 and rear_camera_mp > 0 and front_camera_mp > 0 and weight > 0 and ram > 0:
        with st.spinner('Computing (=...'):
            if fourg == 'yes' or fourg == 'y':
                fourg = 1
            if fourg == 'no':
                fourg = 0
            x = np.array([[new_price, release_year, screen_size, rear_camera_mp, front_camera_mp, weight, ram, fourg]])
            y = reg.predict(x)
            st.write("The normalized price for a used device will be:")
            #st.write(f"If you consider all the variables: {prediction}")
            #st.write(f"If you only consider the model of 8 variables: {prediction}")
            st.write(y)

x_train2 = train_df.drop(['used_price', 'device_brand','os'], axis=1)