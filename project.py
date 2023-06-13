import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from itertools import combinations
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
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
the normalized price of a used device. To achieve this, I decided to create a linear regression model for which I chose the
most influent variables through forward selection. The statistical analysis will be based on a dataset made of 3454 samples (devices) and 15
variables that includes detailed information about used phones, such as technical specifications and design features.'''

url = 'https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data?resource=download'

st.write('The dataset I used can be found on the following link: [link to kaggle](' +url+')')

'''The variables provided by the dataset were

- **device_brand**, the name of manufacturing brand
'''
fig = plt.figure(figsize=(5,6))
used_device_df['device_brand'].value_counts().sort_index(ascending = False).plot(kind = 'barh', color = "coral")
plt.xlabel('Number of the devices')
plt.ylabel('Brands')
st.write(fig)
st.caption('Brands of devices')

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
st.write(fig)
st.caption('Operative systems of the devices')
'''
- **screen_size**, the size of the screen in cm
'''
fig = plt.figure()
ax = used_device_df['screen_size'].value_counts().sort_index().plot(kind = 'bar', color = "darkturquoise")   
ax.set_xticks(ax.get_xticks()[::20]) 
plt.xlabel('cm')
plt.ylabel('Number of devices')
st.write(fig)
st.caption('Screen size of the devices')
'''
Probably in the dataset there are more smartphones than tablets, because there is a Gaussian distribuition on the left with a long tail on the right where the devices are bigger.
- **4g**, a string declaring whether 4G is available or not
- **5g**, a string declaring whether 5G is available or not
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

'''From this barplot we can see that more than $ 2\over{3} $ of the devices present the 4G technology, whereas the 5G technology
is present in less than 500 devices as it should be if we consider the fact that it has just recently introduced.'''
pic = "year4g.png"
st.image(pic, caption='Presence of the 4G technology grouped by year')
'''
As I expected, the number of released devices without 4G is every year smaller and from 2016 the number is very small. 
These devices may probably be "dumb-phones" or some tablets.
- **rear_camera_mp**, the resolution of the rear camera in megapixels
- **front_camera_mp**, the resolution of the front camera in megapixels
- **internal_memory**, the amount of internal memory (ROM) in GB
'''

fig = plt.figure(figsize=(7,7))
my_colors = ['cadetblue','cadetblue','cadetblue','lightcoral','cadetblue','lightcoral','lightcoral','lightcoral','cadetblue','lightcoral','cadetblue','lightcoral','lightcoral','lightcoral','lightcoral','lightcoral']
used_device_df['internal_memory'].value_counts().sort_index().plot(kind = 'bar', color = my_colors)
plt.xlabel('GB')
plt.ylabel('Number of devices')
st.pyplot(fig)
st.caption('Amount of the internal memory')

'''
I decided to highlight on the graph the most common measures for the internal memory that can be found: the powers of $2$. This seems also to be a reasonable choice
if we consider that most of the devices has a internal memory with a power of $2$.

If we for instance would like to consider the devices without this feature, we would have $47$ devices that have the $75\%$ of the screen between $5cm$ and $7cm$,
for the major part doesn't have $4G$ connection, that don't have $5G$, have a maximum quality of the rear cameras of 5mpx,
for the major part don't have a frontal camera and for the $75\%$ have an amount of RAM memory under $0,14 GB.$
'''

vector = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] #first ten powers of 2
applied_mask_df = used_device_df
for i in range(0, len(vector)):
    particular_memory_mask = applied_mask_df['internal_memory'] != vector[i]
    applied_mask_df = applied_mask_df[particular_memory_mask]
if st.checkbox('Show statistics'):
    st.write(applied_mask_df.describe().T)
'''
- **ram**, the amount of RAM in GB
- **battery**, the energy capacity of the device battery in mAh
- **weight**, the weight of the device in grams
- **release_year**, the year when the device model was released
'''
fig = plt.figure(figsize=(7,7))
used_device_df['release_year'].value_counts().sort_index().plot(kind='barh', color = "coral")
plt.xlabel('Number of devices')
plt.ylabel('Year')
st.write(fig)
st.caption('Release year of the devices')

'''
Not so many devices were released in 2020 probably due to the pandemics.
- **days_used**, the number of days the used/refurbished device has been used;
- **normalized_new_price**, the normalized price of a new device of the same model;
- **normalized_used_price**, the normalized price of the refurbished device.
'''

'''If we would like for instance too see how much the variables are correlated, we should plot 
the correlation matrix:'''

#translate categorical into nnumerical variables
used_device_df["4g"] = used_device_df["4g"].replace({"yes": 1, "no": 0})
used_device_df["5g"] = used_device_df["5g"].replace({"yes": 1, "no": 0})


diag = np.triu(np.ones_like(used_device_df.corr(numeric_only = True)))
fig = plt.figure(figsize = (9, 9))
sns.heatmap(used_device_df.corr(numeric_only = True), annot = True, mask = diag)
st.write(fig)


'''From this plot we can infer that:

- the screen size is strongly correlated with the battery and the weight, probably because a big screen needs a bigger battery to last and because of this it weights more;
- the release year is strongly correlated with the days used, probably because in the years we got used to change our device very quickly because of the constant updates released;
- the normalized price of a used device is very strong correlated with the normalized price of a new model.

We can also plot the linear regression line that minimizes the mean squared error between these variables:'''

fig = plt.figure(figsize = (5,5))
sns.regplot(x = used_device_df['screen_size'],y = used_device_df['battery'],color = 'turquoise' )    
plt.xlabel('cm')
plt.ylabel('mAh')
st.write(fig)
st.caption('Correlation between screen size and battery')

fig = plt.figure(figsize = (5,5))
sns.regplot(x = used_device_df['screen_size'],y = used_device_df['weight'], color = 'tomato')    
plt.xlabel('cm')
plt.ylabel('gr')
st.write(fig)
st.caption('Correlation between screen size and weight')

fig = plt.figure(figsize = (5,5))
sns.regplot(x = used_device_df['weight'],y = used_device_df['battery'], color = 'orchid')    
plt.xlabel('gr')
plt.ylabel('mAh')
st.write(fig)
st.caption('Correlation weight size and battery')


fig = plt.figure(figsize = (5,5))
sns.regplot(x = used_device_df['release_year'],y = used_device_df['days_used'], color = 'lightgreen')    
plt.xlabel('Year')
plt.ylabel('Days')
st.write(fig)
st.caption('Correlation between release year and used days')

fig = plt.figure(figsize = (5,5))
sns.regplot(x = used_device_df['new_price'],y = used_device_df['used_price'], color = 'peachpuff')    
plt.xlabel('normalized new price')
plt.ylabel('normalized refurbished price')
st.write(fig) 
st.caption('Correlation between price of a new device and an old device')

'''After this analysis, I split randomly my dataset into a train set of $2763$ samples and a test set of $691$ samples
and I used as independent variables the screen size, the $4G$, the $5G$ , the rear camera, the front camera, the internal memory, the RAM, the battery,
the weight, the used days and the new normalized price, and as dependent variable the used price.


These are the characteristics of the model I obtained'''

#regression model
used_device_df["4g"] = used_device_df["4g"].replace({"yes": 1, "no": 0})
used_device_df["5g"] = used_device_df["5g"].replace({"yes": 1, "no": 0})
train_df, test_df = train_test_split(used_device_df, test_size=0.2, random_state = 22)
y_train = train_df[['used_price']].copy()
x_train1 = train_df[['new_price','release_year','screen_size','rear_camera_mp','front_camera_mp','weight','ram']].copy()
x_train = train_df.drop(['used_price', 'device_brand','os'], axis=1)
y_test = test_df[['used_price']].copy()
x_test = test_df.drop(['used_price', 'device_brand','os'], axis=1)

Xc = sm.add_constant(x_train) #to add the columns of ones to the matrix on the right in order to get the intercept
model = sm.OLS(y_train, Xc).fit()
st.write(model.summary())
st.write('')
'''The $R^2$ (coefficient of determination) is pretty high, which means that the model fits pretty good, but the condition number
is high too, and it is caused my a multicollinearity between the variables. Multicollinearity is usually high, because some variables almost linear dependent,
so I want to see if I can get better results by reducing the number of the variables to avoid overfitting. 

To do this, I chose to use forward selection, taking into accont the residual sum of squares of the model on the tests set as parameter for the best fit.'''
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
        RSS_list = [np.inf] #at the begginning the first RSS is always acceptable
        RSStest_list = []
        variables_list = dict()

    for i in range(1,13):
        best_RSS = np.inf
        
        for comb in combinations(remaining_variables,1):

                result = linear_reg(x_train[list(comb) + variables],y_train)
                RSStest = RSS_test(result[1], x_test[list(comb) + variables], y_test)  
                if result[0] < best_RSS: #choosing the smallest RSS
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
RSS_list = [np.inf] #at the begginning the first RSS is always acceptable
RSStest_list = []
variables_list = dict()

for i in range(1,13):
    best_RSS = np.inf
        
    for comb in combinations(remaining_variables,1):

            result = linear_reg(x_train[list(comb) + variables],y_train)
            RSStest = RSS_test(result[1], x_test[list(comb) + variables], y_test)  
            if result[0] < best_RSS: #choosing the smallest RSS
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

st.write(forward_selection)

fig = plt.figure(figsize=(5,5))
plt.title('RSS on the test set')

ax=[]
mins=[]
for i in range (1, 12):
    rsss = forward_selection['RSS_test'].loc[forward_selection['num_variables'] == i].copy() #for each number of variables we get the rss
    for m in range (len(rsss)):
        ax.append(i)
    plt.plot(ax, rsss,'.', color = 'indianred', label=i)
    mins.append(rsss.min())
    ax=[]
    
plt.xticks(np.arange(1, 12, 1))
plt.grid(visible=True)
plt.xlabel('Flexibility')
plt.ylabel('RSS')

st.write(fig)

'''As I said, I want to minimize the RSS on the test set, so the best model that does this is the model
made of $7$ variables: new_price, release_year, screen_size, rear_camera_mp, front_camera_mp, weight and ram.'''
st.write('This model has the following characteristics')

Xcc = sm.add_constant(x_train1) #to add the columns of ones to the matrix on the right in order to get the intercept
model = sm.OLS(y_train, Xcc).fit()
st.write(model.summary())

st.write('')

'''Although we still have some multicollinearity (always due to the condition number), the column $\mathbb{P}>t$ is a column
of zeros, which means that those variables shouldn't be excluded from the model.'''

st.write('')

'''It is possible with the following widget to compute the normalized expected price of a refurbished device.'''
reg = LinearRegression().fit(x_train1,y_train)
with st.expander("Choose the values to make a prediction"):
    release_year = st.slider('Release year:', 2010, 2030)
    new_price = st.number_input("Price of the new device:")
    screen_size = st.number_input("Dimension of the screen:")
    rear_camera_mp = st.number_input("Resolution of the rear camera:")
    front_camera_mp = st.number_input("Resolution of the front camera:")
    weight = st.number_input("Weight:")
    ram = st.number_input("Capacity of the ram memory:")
    if new_price <= 0:
        error = " The price must be positive"
        st.error(error)
    a = release_year % 1
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
    if st.button("Predict") and new_price >0 and a == 0 and screen_size > 0 and rear_camera_mp > 0 and front_camera_mp > 0 and weight > 0 and ram > 0 :
        x = np.array([[new_price, release_year, screen_size, rear_camera_mp, front_camera_mp, weight, ram]])
        y = reg.predict(x)
        st.write("The normalized price for a used device will be:")
        st.write(y)
