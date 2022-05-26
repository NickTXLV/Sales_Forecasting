# Sales Forecasting
We have weekly sales data from annual 2019 and 2021. 2020 will not be used due to the impacts of Covid. The task is to make a forecasting model to predict weekly sales demand for annual 2022 for each product. I decided to use the Prophet package for this project as I wanted to learn a new package in the process. 

The Steps that I will follow in this project:

1)Graph the data to understand it
  
  Line Plot and Correlation plot

2)Data Munging (Cleaning/Fill in missing values)
  For each product, engineered a column for each of the following:
  
  LY Week Sales/Forward Fill/Backward Fill/Trailing 4 Week Avg Sales/Avg of the above columns
  
  Fill into missing sales using the above columns in the same order
  
  Some weeks in the helmet product were very low due to running out of stock. Any weeks <20 units was replaced with LY Week Sales. 
  

3)Graph the data again to understand it post data munging

  Line Plot and Correlation plot

4)Forecast all products

5)Build the steps individually for total combined lighter sales then build a loop to loop through all the steps to build forecasts for the next 52 weeks for each product. 
  
  Steps:
  
  1)Create a data frame and populate the data frame with columns ds and y (as required by the package). DS is date and y is product sales. 
  
  2)Initialize the model
  
  3)Fit the model
  
  4)Make predictions for the next 52 weeks and assign the prediction to a new data frame

  5)Perform cross validation and get forecast error (MAPE) and assign the MAPE to a new data frame

  6)Save the model


Results:
Results were mixed with MAPES ranging from 28% to 95%.

Reasons for the results:
The results for each model were mixed. I did notice the products that required the most fill-ins into missing sales had the highest MAPES. Maybe this was because there was so much reprocessing that needed to be done? Ironically, the initial model for combined lighters had a MAPE of 27% beating out the models for the individual lighters. This could mean that a hierarchal model could work and be used to push down to individual products. The models can be improved upon by adding regressors to the data to identify holidays and other events that affect sales. 

