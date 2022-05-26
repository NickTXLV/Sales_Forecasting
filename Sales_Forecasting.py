#Import Required Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Get Data

df=pd.read_excel(r'C:\Users\dodso\OneDrive - Texas Tech University\Projects\Sales Forecasting\Sales Forecasting.xlsx',
                 sheet_name='Sheet1',header=1)
df.head()



#Notice that the data is 2019 and 2021, 2020 was pitched due to cvoid, however we will
#replace the year 2019 with 2020 to have sequential data

df['Year']=df['CalendarYearWkNumber'].astype(str).str[:4].replace(to_replace='2019',value='2020')
df['Week']=df['CalendarYearWkNumber'].astype(str).str[-2:]

df['Year_Week']=df['Year']+df['Week']

df=df.drop(labels=['Year','Week','CalendarYearWkNumber'],axis=1)

df['Year']=df['Year_Week'].astype(str).str[:4].astype(int)
df['Week']=df['Year_Week'].astype(str).str[-2:].astype(int)

#Get Date from Year/Week
Yw = df['Year'].astype(str) + df['Week'].astype(str) + '0'
df['Date'] = pd.to_datetime(Yw, format='%Y%U%w')

df=df[['Year_Week','Year','Week','Date','AR RUBBER LIGHTER MOQ 1000 (50)', 'ASSORTED JEWELRY $10',
       'BE RUBBER LIGHTER (1000)', 'M&M PEANUT KS (24)',
       'MARLBORO GOLD BOX KING (10)', 'PLUSH BLK TIP SHARK SR 13INCH',
       'SLRRRP JELLO SHOT ASSORTED (120)', 'TITOS HANDMADE VODKA 50ML',
       'VIKING HELEMT ']]




#Graph all product together to understand seasonality and if they can be modeled together

sns.lineplot(x='Year_Week',y='AR RUBBER LIGHTER MOQ 1000 (50)',data=df)
sns.lineplot(x='Year_Week',y='BE RUBBER LIGHTER (1000)',data=df)
sns.lineplot(x='Year_Week',y='M&M PEANUT KS (24)',data=df)
sns.lineplot(x='Year_Week',y='MARLBORO GOLD BOX KING (10)',data=df)
sns.lineplot(x='Year_Week',y='PLUSH BLK TIP SHARK SR 13INCH',data=df)
sns.lineplot(x='Year_Week',y='SLRRRP JELLO SHOT ASSORTED (120)',data=df)
sns.lineplot(x='Year_Week',y='TITOS HANDMADE VODKA 50ML',data=df)
sns.lineplot(x='Year_Week',y='VIKING HELEMT ',data=df)
plt.legend(['AR','BE','M&M','MARLBORO','Shark','Jello','Titos','Helmet'],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel('QTY')
plt.show()

#Graph select product together to understand seasonality and if they can be modeled together
#Lighters
sns.lineplot(x='Year_Week',y='AR RUBBER LIGHTER MOQ 1000 (50)',data=df)
sns.lineplot(x='Year_Week',y='BE RUBBER LIGHTER (1000)',data=df)
plt.legend(['AR','BE'],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel('QTY')
plt.show()


#Graph select product together to understand seasonality and if they can be modeled together
#Consumables

sns.lineplot(x='Year_Week',y='M&M PEANUT KS (24)',data=df)
sns.lineplot(x='Year_Week',y='MARLBORO GOLD BOX KING (10)',data=df)
sns.lineplot(x='Year_Week',y='SLRRRP JELLO SHOT ASSORTED (120)',data=df)
sns.lineplot(x='Year_Week',y='TITOS HANDMADE VODKA 50ML',data=df)
plt.legend(['M&M','MARLBORO','Jello','Titos'],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel('QTY')
plt.show()

#Graph select product together to understand seasonality and if they can be modeled together
#Toys
sns.lineplot(x='Year_Week',y='PLUSH BLK TIP SHARK SR 13INCH',data=df)
sns.lineplot(x='Year_Week',y='VIKING HELEMT ',data=df)
plt.legend(['Shark','Helmet'],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel('QTY')
plt.show()

# Graph with Sliders
fig = px.line(df,x='Year_Week',y='AR RUBBER LIGHTER MOQ 1000 (50)',title='AR Lighter')
#fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Build Lag Plots

pd.plotting.lag_plot(df['AR RUBBER LIGHTER MOQ 1000 (50)'],lag=52)

#Plot Correlation
g=sns.pairplot(df,y_vars=['AR RUBBER LIGHTER MOQ 1000 (50)'],x_vars=['BE RUBBER LIGHTER (1000)','MARLBORO GOLD BOX KING (10)','TITOS HANDMADE VODKA 50ML'])

#a=sns.pairplot(df)

#Correlation Matrix
corr=df.corr(method='pearson')
#Correlation Matrix Heat Map
h=sns.heatmap(corr,vmax=.85, center=0, square=True,linewidths=.5, cbar_kws={'shrink':.5}, annot=True, fmt= '.2f',cmap='coolwarm')
h.figure.set_size_inches(10,10)
plt.show()


#auto correlation plot
pd.plotting.autocorrelation_plot(df['TITOS HANDMADE VODKA 50ML'])

#corr.to_excel('C:/Users/dodso/OneDrive - Texas Tech University/Projects/Sales Forecasting/corr.xlsx')
#We need to build 3 different generalized models for each category


#Fill Missing values of each product
#AR Lighter only has 1 missing week, let's use back fill
df['AR RUBBER LIGHTER MOQ 1000 (50)'].fillna(method='bfill',inplace=True)

#BE lighter:
# 2 values are really low, lets set those to NAN so we can fill them in later
df['BE RUBBER LIGHTER (1000)'].replace(to_replace=[1,5],value=np.nan, inplace=True)

# Fill in missing values with data from LY
df['BETEST']=df['BE RUBBER LIGHTER (1000)'].shift(periods=53)

df['BE RUBBER LIGHTER (1000)']=df['BE RUBBER LIGHTER (1000)'].fillna(value=df['BETEST'])

df.drop(labels=['BETEST'],axis=1,inplace=True)

#Shark
#Create a few columns to identify best number to fill in
#LY Column
df['Shark_LY']=df['PLUSH BLK TIP SHARK SR 13INCH'].shift(periods=53)
#ForwardFill by shifting
df['Shark_FFill']=df['PLUSH BLK TIP SHARK SR 13INCH'].shift(periods=1)
#Backfill by shifting
df['Shark_BFill']=df['PLUSH BLK TIP SHARK SR 13INCH'].shift(periods=-1)
#Trailing 4 Week Avg
df['Trailing_4wk_Avg']=df['PLUSH BLK TIP SHARK SR 13INCH'].rolling(4).mean()
#Avg of the 4 Columns
df['Shark_Column_Avg']=(df['Shark_LY']+df['Shark_FFill']+df['Shark_BFill']+df['Trailing_4wk_Avg'])/4

#Fill with LY 
df['PLUSH BLK TIP SHARK SR 13INCH'].fillna(value=df['Shark_LY'],inplace=True)
#Fill with BackFill
df['PLUSH BLK TIP SHARK SR 13INCH'].fillna(value=df['Shark_BFill'],inplace=True)
#One NAN Value Left, Fill in with Column Avg
df['PLUSH BLK TIP SHARK SR 13INCH'].fillna(value=df['PLUSH BLK TIP SHARK SR 13INCH'].mean(),inplace=True)

#There are still low outliers due to low inventory, lets backfill/forward fill anything less than 20 units

df['PLUSH BLK TIP SHARK SR 13INCH']=np.select(
   [
    df['PLUSH BLK TIP SHARK SR 13INCH'] <20
    ],
   [
    df['Shark_FFill']
     ],
    df['PLUSH BLK TIP SHARK SR 13INCH']) 

df['PLUSH BLK TIP SHARK SR 13INCH']=np.select(
   [
    df['PLUSH BLK TIP SHARK SR 13INCH'] <20
    ],
   [
     df['Shark_BFill']
     ],
    df['PLUSH BLK TIP SHARK SR 13INCH'])  

#This created a NAN, lts fill it it with 4 week avg
df['PLUSH BLK TIP SHARK SR 13INCH'].fillna(value=df['Trailing_4wk_Avg'],inplace=True)

#Drop the Added Columns now that we have a final series for Shark
df.drop(labels=['Shark_LY', 'Shark_FFill', 'Shark_BFill',
'Trailing_4wk_Avg', 'Shark_Column_Avg'],axis=1,inplace=True)
       
 #Helmet   
#Create a few columns to identify best number to fill in  
#Create Trailing 4 Week Column
df['Helmet_Trailing4wk']=df['VIKING HELEMT '].rolling(window=4).mean()  
#LY Column   
df['Helmet_LY']=df['VIKING HELEMT '].shift(periods=53)

#Fill in missing Data with LY Column    
df['VIKING HELEMT '].fillna(value=df['Helmet_LY'], inplace=True)
#There are still low outliers due to low inventory, lets fill those in with LY

df['VIKING HELEMT ']=np.select(
    [
     df['VIKING HELEMT '] <20
     ],
    [
     df['Helmet_LY']
     ],
    default=df['VIKING HELEMT ']
               )

#This created NAN, so back fill
df['VIKING HELEMT '].fillna(method='bfill',inplace=True)

#Drop the added columns
df.drop(columns=['Helmet_Trailing4wk', 'Helmet_LY'], axis=1, inplace=True)

#Now that we have filled into missing sales, lets rerun the correlation

post_corr=df.corr(method='pearson')
new_heat=sns.heatmap(data=post_corr,vmax=.9,cmap='coolwarm',center=0,square=True,annot=True, cbar_kws={'shrink':.5},linewidths=1)
new_heat.figure.set_size_inches(10,10)
plt.show()

h=sns.heatmap(corr,vmax=.85, center=0, square=True,linewidths=.5, cbar_kws={'shrink':.5}, annot=True, fmt= '.2f',cmap='coolwarm')
h.figure.set_size_inches(8,8)
plt.show()

#ReGraph to see what the data looks like

sns.lineplot(x='Year_Week',y='AR RUBBER LIGHTER MOQ 1000 (50)',data=df)
sns.lineplot(x='Year_Week',y='BE RUBBER LIGHTER (1000)',data=df)
sns.lineplot(x='Year_Week',y='M&M PEANUT KS (24)',data=df)
sns.lineplot(x='Year_Week',y='MARLBORO GOLD BOX KING (10)',data=df)
sns.lineplot(x='Year_Week',y='PLUSH BLK TIP SHARK SR 13INCH',data=df)
sns.lineplot(x='Year_Week',y='SLRRRP JELLO SHOT ASSORTED (120)',data=df)
sns.lineplot(x='Year_Week',y='TITOS HANDMADE VODKA 50ML',data=df)
sns.lineplot(x='Year_Week',y='VIKING HELEMT ',data=df)
plt.legend(['AR','BE','M&M','MARLBORO','Shark','Jello','Titos','Helmet'],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel('QTY')
plt.show()





#Time To Forecast
# We will build generalized models than use that model to forecast individual products
#Lighters
lighter_df=pd.DataFrame()

#Combine lighter sales to build a generalized model
lighter_df['y']=df['AR RUBBER LIGHTER MOQ 1000 (50)']+df['BE RUBBER LIGHTER (1000)']
#lighter_df.insert(0,'Date',df['Date'])
lighter_df.set_index(df['Date'],inplace=True)
lighter_df.reset_index(inplace=True)
lighter_df.rename(columns= {'Date':'ds'},inplace=True)


from prophet import Prophet
m=Prophet()
m.fit(lighter_df)
future = m.make_future_dataframe(periods=52,freq='W')
forecast = m.predict(future)
##forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(
  ##  r'C:\Users\dodso\OneDrive - Texas Tech University\Projects\Sales Forecasting\Forecast.xlsx')
fig=m.plot_components(forecast)

#Measure the error of the generalize forecast

from prophet.diagnostics import cross_validation

lighter_cv=cross_validation(m, initial=105,horizon='52 days')

from prophet.diagnostics import performance_metrics
lighter_mape = performance_metrics(lighter_cv)
lighter_mape.head()

from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(lighter_cv, metric='mape')

lighter_mape= str(lighter_mape['mape'].mean())

print('The MAPE of the unadjusted model is:'+lighter_mape)

###Lets Creat individual level forecasts for each Column using a loop 
##Loop is not recommended, but the prophet model is for just 1 time series, and cant be used to generalize/push down
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
C_Forecast=['AR RUBBER LIGHTER MOQ 1000 (50)',
       'ASSORTED JEWELRY $10', 'BE RUBBER LIGHTER (1000)',
       'M&M PEANUT KS (24)', 'MARLBORO GOLD BOX KING (10)',
       'PLUSH BLK TIP SHARK SR 13INCH', 'SLRRRP JELLO SHOT ASSORTED (120)',
       'TITOS HANDMADE VODKA 50ML', 'VIKING HELEMT ']

for i in C_Forecast:
    product_df=pd.DataFrame()
    product_df['y']=df[i]
    product_df.set_index(df['Date'],inplace=True)
    product_df.reset_index(inplace=True)
    product_df.rename(columns= {'Date':'ds'},inplace=True)
    m=Prophet()
    m.fit(product_df)
    future = m.make_future_dataframe(periods=52,freq='W')
    forecast = m.predict(future)
    globals()[f'Forecast_{i}'] =pd.DataFrame(forecast)  #creates a new dataframe for each forecast column
    product_cv=cross_validation(m, initial=105,horizon='52 days')
    product_mape = performance_metrics(product_cv)
    globals()[f'Mape_{i}']= str(product_mape['mape'].mean())


#Save the model to JSON for deployment elsewhere
import json
from prophet.serialize import model_to_json, model_from_json

with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)  # Save model

with open('serialized_model.json', 'r') as fin:
    p = model_from_json(json.load(fin))  # Load model











