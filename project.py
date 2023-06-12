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

st.title('Used device price prediction')
''' In the digital age we live in, personal devices have become an essential tool. The rapid and constant technological advancements have led to the continuous introduction in
the market of new devices, such as smartphones, tablets, smartwhatches, smart tvs and other. As a result, the used industry \
has become increasingly relevant, offering a more affordable option for those looking to own a quality device at a lower price.

 Choosing a price for a used device may be a complex task, that\'s why for my programming project I decided to prepare a statistical analysis on predicting
the normalized price of a used device. To estimate the normalized price of a used phone. To achieve this, I decided to create a linear regression model for which I chose the
most influent variables through forward selection. The statistical analysis will be based on a dataset made of 3454 samples (devices) and 15
variables that includes detailed information about used phones, such as technical specifications and design features.'''

url = 'https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data?resource=download'

st.write('The dataset I used can be found on the following link: [link to kaggle](' +url+')')

'''The variables provided by the dataset were

- **device_brand**, the name of manufacturing brand
'''
fig = plt.figure(figsize=(5,6))
plt.title('Brands of devices')
used_device_df['device_brand'].value_counts().sort_index(ascending = False).plot(kind = 'barh', color = "coral")
plt.xlabel('Number of the devices')
plt.ylabel('Brands')
st.write(fig)
'''
The most sold brands result to be Samsung and Huawei.
- **os**, the operating system on which the device runs
'''
fig = plt.figure(figsize=(8,8))
values = used_device_df['os'].value_counts()
labels = ['Android', 'Others', 'Windows', 'iOS']
colors = ['#8EB897','#DD7596', '#B7C3F3', '#4F6272']
explode = (0, .2, .2, .2)
plt.pie(values, labels = labels, colors = colors, autopct = '%.2f%%', explode = explode, pctdistance= 0.8)
plt.title('Operative systems of the devices')
st.write(fig)
'''
The most present operative system is Android.
- **screen_size**, the size of the screen in cm
'''
fig = plt.figure()
ax = used_device_df['screen_size'].value_counts().sort_index().plot(kind = 'bar', color = "darkturquoise")   
ax.set_xticks(ax.get_xticks()[::20]) 
plt.title('Screen size of the devices')
plt.xlabel('cm')
plt.ylabel('Number of devices')
plt.show()
st.write(fig)
'''
Probably in the dataset there are more smartphones than tablets, because there is a Gaussian distribuition on the left with a long tail on the right where the devices are much bigger.
- **4g**, a string declaring whether 4G is available or not;
- **5g**, a string declaring whether 5G is available or not;
'''
col_1, col_2 = st.columns(2)
with col_1:
    fig1, ax = plt.subplots(figsize=(10,10))
    used_device_df['4g'].value_counts().sort_index().plot(kind='bar', color = "pink", ax = ax)
    ax.set_xlabel('Presence of the 4G')
    ax.set_ylabel('Number of devices')

    st.pyplot(fig1)
with col_2:
    fig2, ax = plt.subplots(figsize=(10,10))
    used_device_df['5g'].value_counts().sort_index().plot(kind='bar', color = "pink", ax = ax)
    ax.set_xlabel('Presence of the 5G')
    ax.set_ylabel('Number of devices')

    st.pyplot(fig2)
'''From this barplot we can see that more than $ 2\over{3} $ of the devices presented the 4g technologies, whereas the 5g technologies 
is present in less than 500 devices as it should be if we consider the fact that is a recent introducted technology.'''
pic = "year4g.png"
st.image(pic, caption='Presence of the 4G technology grouped by year')
'''
As I expected, the number of released devices with 4g is every year smaller and from 2016 the number is very small. 
These devices may probably be "dumb-phones" and some tablets.
- **rear_camera_mp**, the resolution of the rear camera in megapixels;
- **front_camera_mp**, the resolution of the front camera in megapixels;
- **internal_memory**, the amount of internal memory (ROM) in GB;
- **ram**, the amount of RAM in GB;
'''
col_1, col_2 = st.columns(2)
with col_1:
    fig1, ax = plt.subplots(figsize=(10,10))
    my_colors = ['b','b','b','b','b','r','r','r','b','r','b','r','r','r','r','r']
    ax0 = used_device_df['internal_memory'].value_counts().sort_index().plot(kind = 'bar', ax = ax, color = my_colors)
    ax.set_xlabel('GB')
    ax.set_ylabel('Number of devices')
    ax0.title.set_text('Amount of the internal memory')
    st.pyplot(fig1)
with col_2:
    fig2, ax = plt.subplots(figsize=(10,10))
    ax1 = used_device_df['ram'].value_counts().sort_index().plot(kind = 'bar', ax = ax, color = 'red')
    ax.set_xlabel('GB')
    ax.set_ylabel('Number of devices')
    ax1.title.set_text('Amount of the RAM')
    st.pyplot(fig2)
'''
I decided to highlight on the graph regarding the internal memory the most common measures that can be found and this seems also a 
good choice if we consider the fact that most of the devices has a internal memory with a power of 2.

If we for instance would like to consider the devices with a capacity
memory different from a power of $2$, we obtain devices that have the $75\%$ of the screen between $5cm$ and $7cm$,
the major part doesn't have 4G connection and none of these has 5G, the maximum quality of the rear cameras is 5mpx,
many devices don't have a frontal camera and the $75\%$ have an amount of RAM memory under $0,14 GB$

- **battery**, the energy capacity of the device battery in mAh;
- **weight**, the weight of the device in grams;
- **release_year**, the year when the device model was released;
'''
fig = plt.figure(figsize=(7,7))
plt.title('Release year of the devices')
used_device_df['release_year'].value_counts().sort_index().plot(kind='barh', color = "coral")
plt.xlabel('Number of devices')
plt.ylabel('Year')
st.write(fig)

'''
Not so many devices were released in 2020 probably due to the pandemics.
- **days_used**, the number of days the used/refurbished device has been used;
- **normalized_new_price**, the normalized price of a new device of the same model;
- **normalized_used_price**, the normalized price of the used/refurbished device.
'''
if st.checkbox('Show code'):
    st.write('Forward selection:')
    st.code('''
    def linear_reg(X,Y): #function to prepare the R_squared and the RSSs
        model_k = LinearRegression(fit_intercept = True).fit(X,Y)
        RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
        return RSS, model_k


    def RSS_test(model, X1, Y1): #function to test on the test set
        yhat = model.predict(X1)
        RSS_tested = ((yhat - Y1) ** 2).sum() 
        RSS_tested = RSS_tested[0]
        return RSS_tested
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
used_device_df["4g"] = used_device_df["4g"].replace({"yes": 1, "no": 0})
used_device_df["5g"] = used_device_df["5g"].replace({"yes": 1, "no": 0})
train_df, test_df = train_test_split(used_device_df, test_size=0.2, random_state = 22)
y_train = train_df[['used_price']].copy()
x_train1 = train_df[['new_price','release_year','screen_size','rear_camera_mp','front_camera_mp','weight','ram']].copy()
reg = LinearRegression().fit(x_train1,y_train)

with st.expander("Choose the values to make a prediction"):
    new_price = st.number_input("Price of the new device:")
    release_year = st.number_input("Release year:")
    screen_size = st.number_input("Dimension of the screen:")
    rear_camera_mp = st.number_input("Resolution of the rear camera:")
    front_camera_mp = st.number_input("Resolution of the front camera:")
    weight = st.number_input("Weight:")
    ram = st.number_input("Capacity of the ram memory:")
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
    if st.button("Predict") and new_price >0 and release_year > 0 or a == 0 and screen_size > 0 and rear_camera_mp > 0 and front_camera_mp > 0 and weight > 0 and ram > 0 :
        with st.spinner('Computing (=...'):
            x = np.array([[new_price, release_year, screen_size, rear_camera_mp, front_camera_mp, weight, ram]])
            y = reg.predict(x)
            st.write("The normalized price for a used device will be:")
            st.write(y)
