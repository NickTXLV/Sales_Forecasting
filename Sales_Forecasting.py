#Import Required Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
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

df=df[['Year_Week','AR RUBBER LIGHTER MOQ 1000 (50)', 'ASSORTED JEWELRY $10',
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

a=sns.pairplot(df)

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
       
 #Helmet               
        

from prophet import Prophet









