# Databricks notebook source
spark.conf.set("spark.databricks.io.cache.enabled", "true")
spark.catalog.clearCache()

# COMMAND ----------

# Installing libraries
!pip install xgboost==1.4.0
!pip install joblib

# COMMAND ----------

# Importing libraries
import joblib
import datetime
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import dateutil.relativedelta
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

pd.set_option('display.max_columns', None)

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------



##################################################     Parameters     ##################################################

##### Inputs
input_folder = "/dbfs/mnt/nindprmidasdheuno1adlprv/prod/inbound/Analytics/Category Growth/CycleRun_202309/SUPER_PREMIUM/XGBoost/"
x_train_location = input_folder + "X_train.csv"
y_train_location = input_folder + "Y_train.csv"
feed_data_location = input_folder + "Feed.csv"
actual_data_location = input_folder + "Actual.csv"
last_year_actual_location = input_folder + "Actual_LY.csv"


##### Train, Test, Prediction period
train_period = {"Start":'2021-02-01', "End":'2021-09-01'}
test_month = "2022-06-01"         # So test period would be 21 months starting from 2021-04-01 to 2022-12-01, 
                                  # but we will be testing using only 2021-04-01 to 2022-03-01.
prediction_month = "2023-06-01"   # So prediction period would be 21 months starting from 2022-04-01 to 2023-12-01.

test_period = {"Start":"2022-07-01", "End":"2023-06-01"}
# prediction_period = {"Start":"2022-04-01", "End":"2023-12-01"}



##### Outputs
output_folder = "/dbfs/mnt/nindprmidasdheuno1adlprv/prod/inbound/Analytics/Category Growth/CycleRun_202309/SUPER_PREMIUM/XGBoost/"
var_imp_folder = output_folder + "tmp/var_imp_xgb/"    # Variance importance files will be stored here (stored based on condition)
model_folder_path = output_folder + "tmp/model_xgb/"   # Models will be stored here.  (stored based on condition)
temp_excep_folder_path = output_folder + "temp_excel_folder_xgb/"  # The accuracy of each market, brand, segment of test data (stored in each loop)
result_folder = output_folder + "tmp/result_folder_xgb/"   # Output.csv will be stored here (single file)
output_xgb = output_folder + "tmp/output_xgb/"  # Raw prediction of the model on test data is stored here

temp_pandas = output_folder + "tmp/temp_pandas/"



##### Indices
start_index = 224121
end_index = 230000


# Others
segment = "SUPER_PREMIUM"

# COMMAND ----------



##################################################     Reading model data     ##################################################

X_train = pd.read_csv(x_train_location)
Y_train = pd.read_csv(y_train_location)
data_to_be_feed = pd.read_csv(feed_data_location)
actual = pd.read_csv(actual_data_location)
actual_ly = pd.read_csv(last_year_actual_location)

data_to_be_feed['MonYr'] = pd.to_datetime(data_to_be_feed['MonYr'])
actual['MonYr'] = pd.to_datetime(actual['MonYr'])
actual_ly['MonYr'] = pd.to_datetime(actual_ly['MonYr'])

print(X_train.shape)
print(Y_train.shape)
print(data_to_be_feed.shape)
print(actual.shape)
print(actual_ly.shape)


# COMMAND ----------


##################################     Defining functions for training and testing     ###################################


# We are creating different functions so that we can train and test in 
# modular way.


# COMMAND ----------


###############     Function 1     ###############
########     Function to train model     #########

def create_model(X_train, Y_train, hyperparameters):
    """
    Objective: Training a multioutput XGBoost model on train data
    
    Input:
      X_train: Train x
      Y_train: Train y
      hyperparameters: Hyperparameters.
      
    
    Returns: Multioutput XGBoost model trained on training data.
    """
    
    # 1) Fitting multi-output XGBoost model
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**hyperparameters))
    xgb_model.fit(X_train.values, Y_train.values)
    
    return xgb_model

  

# COMMAND ----------


###############     Function 2     ###############
###     Function to get feature importance     ###


def get_feature_importance(xgb_model, x_columns, y_columns):
    """
    Here we extract the feature importance values for each of the x variables, from the trained model. Model is being trained on train data,
    so the importance values are also based on train data.
    
    input:
      xgb_model: Trained XGBoost model.
    
    Returns:
      A dataframe of feature importance values
    """
    
    Final_Var_Imp = pd.DataFrame()
    for i in range(0, len(y_columns)):  # Is the Y_Var_Check_1 refering to global variable. This is a dataframe. Kaustav ###########
        importance_values = xgb_model.estimators_[i].feature_importances_.tolist()
        importance_dataframe = pd.DataFrame({"Variable":x_columns, "Value":importance_values})
        importance_dataframe['Y'] = y_columns[i]
        
        # Final_Var_Imp = Final_Var_Imp.append(importance_dataframe) # Variable importance values for each of the Y is adding one below the other.
        Final_Var_Imp = pd.concat([Final_Var_Imp, importance_dataframe], ignore_index = True)
    
    
    # Sorting variance importance values for each Y
    Final_Var_Imp = Final_Var_Imp.sort_values(['Y', 'Value'], ascending=[True, False])
    
    # Calculating cumulative sum for each of the Y
    Final_Var_Imp['Cumsum'] = Final_Var_Imp.groupby(['Y'])['Value'].transform(pd.Series.cumsum)

    # print(Final_Var_Imp)
    
    return Final_Var_Imp

# COMMAND ----------


###############       Function 3       ###############
###     Function to get predictions from model     ###

def get_predictions(xgb_model, input_x, y_column_names):
    """
    Here what we are getting predictions of the model on the input data and formatting it for better understanding.
    Two step formatting is being done;
      1) The raw predictions are numpy array, so does not have column names, we are giving it.
      2) To the raw predictions we are adding month, brand, segment information from input data, 
         on which we are getting the predictions
    
    Input:
      xgb_model: Trained XGBoost model
      input_x: A dataframe containing only x variables. input_x is generally test or prediction data.
    
    Returns:
      Formatted prediction of the model on input data
    """
    predictions = xgb_model.predict(input_x.drop(columns = ['MonYr', 'MKT', 'Brand_new', 'SEGMENT']).values)
    # Predictions will look like following.
    # [
    #   [Y1,Y2, ..., Y21]
    #   [Y1,Y2, ..., Y21]
    #   .
    #   .
    #   .
    #   [Y1,Y2, ..., Y21]
    # ]
    
    
    # Formatting the predictions for easy understanding.
    # 1) Predictions of the model will be just a numpy array, so it will not have any column names, so we are giving it.
    prediction_df = pd.DataFrame(predictions, columns = y_column_names)  
    
    # Most probably this will not be of Test, it would be of X_test, which is the test data being passed to the function. Kaustav ##############
    # 2) prediction_df has only predictions but for which brand, which month which market, of course the brand months markets of test data, 
    # so formatting the predictions by adding these information from test data
    prediction_df = pd.concat([prediction_df, input_x[['MonYr', 'MKT', 'Brand_new', 'SEGMENT']].reset_index(drop=True)], axis=1)   # Adding month/year, market, brand and segment data of the test data to predictions.
    
    # print('\n')
    # print(f"y_multixgb_df looks like this:")
    # print(prediction_df.head())
    
    return prediction_df

# COMMAND ----------


###############     Function 4, 5     ###############
###     Function to stack Yis one below other     ###


# Function 4
def stack_ys(predictions_of_a_month):
  """
  Objective:
    In the original prediction of the model there will be multiple months. In 
    predictions_of_a_month is filtered only for one month. Let's imagine the 
    month for which it is filtered is 2021-03-01. It looks like this:
         MonYr       Market       Segment       Brand         Column 1      Column 2     Column 3  ...  Column Xp     Y1     Y2      Y3  ...   Y21
      ----------   ----------   ----------    -----------    ----------    ----------   ----------     -----------   ----   -----   ----      -----
      2021-03-01       :            :               :          :                :           :              :           :      :       :         :
      2021-03-01       :            :               :          :                :           :              :           :      :       :         :  
    
    We want to create a dataframe where all the Y1 values from the 
    predictions_of_a_month sits against April 2021 (2021-04-01), Y2 
    values sits against May 2021 (2021-05-01) and so on. It looks 
    like following:
         MKT     Brand_new     SEGMENT    MonYr_New      Final_Pred_1
        -----   -----------   ---------   ----------    --------------
          :          :            :       2021-04-01       Y1 value     ___
          :          :            :       2021-04-01       Y1 value        |
          :          :            :       2021-04-01       Y1 value        |
          .          .            .            .               .           |--Y1 values
          .          .            .            .               .           |
          .          .            .            .               .           |
          :          :            :       2021-04-01       Y1 value     ---
          :          :            :       2021-05-01       Y2 value     ___
          :          :            :       2021-05-01       Y2 value        |
          :          :            :       2021-05-01       Y2 value        |
          .          .            .            .               .           |--Y2 values
          .          .            .            .               .           |
          .          .            .            .               .           |
          :          :            :       2021-05-01       Y2 value     ---
          :          :            :            :               :    
  
  Input:
    A dataframe of predictions filtered for a single month.
  
  Output:
    Formatted dataframe of predictions where predicted Yis are stacked one below the 
    other.
    
  """
  stack = pd.DataFrame(columns = ['MKT', 'Brand_new', 'SEGMENT', 'MonYr', 'Final_Pred_1'])
  for i in range(1, 19):  
      yi_df = predictions_of_a_month.loc[:, ['MKT', 'Brand_new', 'SEGMENT', 'MonYr', f'Y{i}']]
      yi_df.columns = ['MKT', 'Brand_new', 'SEGMENT', 'MonYr', 'Final_Pred_1']
      
      yi_df['MonYr'] = yi_df['MonYr'] + pd.DateOffset(months=i)
      
      # stack = stack.append(yi_df)
      stack = pd.concat([stack, yi_df], ignore_index = True)
  
  return stack


# Function 5
def stack_driver(prediction_df):
  months = prediction_df['MonYr'].unique().sort_values().tolist()
  
  formatted_prediction_df = pd.DataFrame(columns = ['MKT', 'Brand_new', 'SEGMENT', 'MonYr', 'Final_Pred_1'])
  for month in months:
    formatted_prediction_df = formatted_prediction_df.loc[formatted_prediction_df["MonYr"] <= month, :]
    
    current_month_predictions = prediction_df.loc[prediction_df["MonYr"] == month, :]
    stack = stack_ys(current_month_predictions)
    
    formatted_prediction_df = formatted_prediction_df.append(stack)
  
  return formatted_prediction_df

# COMMAND ----------


###################          Function 6          ###################
###     Function to calculate accuracy at brand-market level     ###

def brand_market_level_accuracy_growth_calculator(stacked_predictions, actual, actual_ly):
  """
  Objective:
    In this function we will use predictions of the model, actuals to calculate accuracy 
    for each of brand at each market. Additional to this information we will also calculate 
    actual growth and forecasted growth.
  
  Input:
    stacked_predictions: Prediction of the model where we have stacked the Ys one below the 
                         other
    actual: actual Y values for the same timeframe as stacked_predictions
    actual_ly: actual Y value for the 12 month old timeframe of stacked_predictions
  
  Returns:
    A dataframe containing accuracy, actual and forecasted growth for each brand, market, segment 
    present in stacked_predictions.
  """

  actual["MonYr"] = pd.to_datetime(actual["MonYr"])
  actual_ly["MonYr"] = pd.to_datetime(actual_ly["MonYr"])
  stacked_predictions["MonYr"] = pd.to_datetime(stacked_predictions["MonYr"])
  
  # Mergings
  # a) Merging predictions and actual
  predictions_actual_merged = pd.merge(stacked_predictions, actual, on = ['MKT','MonYr','Brand_new','SEGMENT'], how = "outer")
  # b) Merging above merged data and 12 months old actuals
  actual_ly["MonYr"] = actual_ly["MonYr"] + pd.DateOffset(months=12)
  predictions_actual_ly_merged = pd.merge(predictions_actual_merged, actual_ly, on = ['MKT','Brand_new', 'SEGMENT', 'MonYr'], how = "outer")
  
  
  # Aggregating at market, brand and segment level
  aggregation = predictions_actual_ly_merged.groupby(['MKT','Brand_new','SEGMENT']).agg({'Final_Pred_1' : 'sum', 'Actual_Sale' : 'sum','Actual_Sale_LY_20' : 'sum'}).reset_index()
  
  
  # Calculating Accuracy, Growth Actual and Growth Forecast at Market, brand and segment level (month/year dimension will be gone)
  accuracy_list = []
  growth_actual_list = []
  growth_forecast_list = []
  for i in range(len(aggregation)):
    row = aggregation.iloc[i, :]
    try:
      accuracy_value = 1 - (abs(row['Final_Pred_1'] - row['Actual_Sale'])/row['Actual_Sale'])
    except:
      accuracy_value = 0
    accuracy_list.append(accuracy_value)

    try:
      growth_acutal_value = (row['Actual_Sale'] - row['Actual_Sale_LY_20'])/row['Actual_Sale_LY_20']
    except:
      growth_acutal_value = 0
    growth_actual_list.append(growth_acutal_value)

    try:
      growth_forecast_value = (row['Final_Pred_1'] - row['Actual_Sale_LY_20'])/row['Actual_Sale_LY_20']
    except:
      growth_forecast_value = 0
    growth_forecast_list.append(growth_forecast_value)
    


    # if row["Actual_Sale"] != 0:
    #   accuracy_list.append(1 - (abs(row['Final_Pred_1'] - row['Actual_Sale'])/row['Actual_Sale']))
    # else:
    #   accuracy_list.append(0)
    
    # if row["Actual_Sale_LY_20"] != 0:
    #   growth_actual_list.append((row['Actual_Sale'] - row['Actual_Sale_LY_20'])/row['Actual_Sale_LY_20'])
    #   growth_forecast_list.append((row['Final_Pred_1'] - row['Actual_Sale_LY_20'])/row['Actual_Sale_LY_20'])
    # else:
    #   growth_actual_list.append(0)
    #   growth_forecast_list.append(0)
  
  aggregation['Accuracy'] = accuracy_list
  aggregation['Growth_Actual'] = growth_actual_list
  aggregation['Growth_Forecast'] = growth_forecast_list

  aggregation.sort_values(by = "Actual_Sale", ascending = False)
  
  return aggregation
  

# COMMAND ----------

########################     Defining functions for training and testing : End     #########################

# COMMAND ----------



##################################     Making the hyperparameters combinations     ###################################



parameter_range = {
    "tree_method": ["hist"],
    "colsample_bynode": np.arange(0.5, 1, 0.1),
    "colsample_bylevel": np.arange(0.5, 1, 0.1),
    "min_child_weight": np.arange(3, 12, 1),
    "max_depth": np.arange(2, 12, 1),
    "colsample_bytree": np.arange(0.5, 1, 0.1),
    "subsample": np.arange(0.5, 1, 0.1),
    "n_estimators": list(range(10, 101, 10))+[200],
    "learning_rate": np.arange(0.01, 0.31, 0.05),
    "random_state":[0]
}

param_values = list(parameter_range.values())
param_combs = list(itertools.product(*param_values))
param_combs = param_combs[start_index:end_index]
print(len(param_combs))

param_dicts = [{
    "tree_method": i[0],
    "colsample_bynode":i[1],
    "colsample_bylevel":i[2],
    "min_child_weight":i[3],
    "max_depth":i[4],
    "colsample_bytree":i[5],
    "subsample":i[6],
    "n_estimators":i[7],
    "learning_rate":i[8],
    "random_state":i[9]} for i in param_combs]



# COMMAND ----------



##################################     Looping over all hyperparamter combinations     ###################################

final_table = pd.DataFrame(columns=['MODEL','LEARNING_RATE', 'N_ESTIMATORS', 'SUBSAMPLE', 'COLSAMPLE_BYTREE', 'MAX_DEPTH', 'MIN_CHILD_WEIGHT',
                                    'COLSAMPLE_BYLEVEL', 'COLSAMPLE_BYNODE', 'TREE_METHOD', 'MKT', 'Brand_new', 'SEGMENT', 'Final_Pred_1',
                                    'Actual_Sale', 'Actual_Sale_LY_20', 'Accuracy', 'Growth_Actual','Growth_Forecast']
                          )                    

count = start_index
for parameters in param_dicts:
  # print("PARAMTERS:::::::::::::::::::::>>>", parameters)
  print(count)
  # if count > 0:
  #   break
  
  current_iteration_id = "_lr{}est{}ss{}colsbt{}maxd{}mincw{}colsbl{}colsbn{}tm{}".format(parameters['learning_rate'].round(2),
                                                                                          parameters['n_estimators'],
                                                                                          parameters['subsample'].round(2),
                                                                                          parameters['colsample_bytree'].round(2),
                                                                                          parameters['max_depth'],
                                                                                          parameters['min_child_weight'],
                                                                                          parameters['colsample_bylevel'].round(2),
                                                                                          parameters['colsample_bynode'].round(2),
                                                                                          parameters['tree_method']
                                                                                          )


  # Training model
  xgb_model = create_model(X_train, Y_train, parameters)
  
  # Getting variable importance
  Final_Var_Imp = get_feature_importance(xgb_model, X_train.columns.tolist(), Y_train.columns.tolist())
  
  # Getting predictions of the trained model 
  # In data_to_be_feed we are storing data of two dates which are test and prediction
  # dates respectively. Now we will feed this data to get_predictions function and get 
  # the predictions for both the dates at one go.
  predictions = get_predictions(xgb_model, data_to_be_feed, Y_train.columns.tolist())
  #print(predictions.head(10))
  
  
  # Testing accuracy of the trained model.
  # Steps:
  #  1) Filtering only for test_date
  #  2) Will format the predictions by stacking the ys one below the other
  #  3) Calculate accuracy of the model at brand, market, segment level
  
  # 1) Filtering only for test_date
  predictions["MonYr"] = pd.to_datetime(predictions['MonYr'])
  predictions_filtered = predictions.loc[predictions["MonYr"] == datetime.datetime.strptime(test_month, "%Y-%m-%d"), :]  # Currently in date_in_test_data 2021-03-01 is stored.
  
  # 2) Will format the predictions by stacking the ys one below the other
  stacked_ys = stack_ys(predictions_filtered)
  
  # 3) 
  stacked_ys = stacked_ys.loc[(stacked_ys["MonYr"] >= pd.to_datetime(test_period["Start"])) & (stacked_ys["MonYr"] <= pd.to_datetime(test_period["End"])), :]
  
  # 4) Calculate accuracy of the model at brand, market, segment level
  brand_market_level_accuracy_growth_df = brand_market_level_accuracy_growth_calculator(stacked_ys, actual, actual_ly)
  brand_market_level_file_name = "remaining_state_results_XGB" + current_iteration_id + ".csv"
  brand_market_level_accuracy_growth_df.to_csv(temp_excep_folder_path + brand_market_level_file_name)
  
  
  ##########  Proposed change  ###########
  # stacked_ys = stack_driver(predictions)
  
  # brand_market_level_accuracy_growth_df = brand_market_level_accuracy_growth_calculator(stacked_ys, actual, actual_ly)
  ######### - #############
  

  #######################     Getting market, brand, segments which follows some criteria     #######################

  # brand_market_level_accuracy_growth_df dataframe contains accuracy for all the 
  # market, brand, segments which were there in the test data. But we will keep 
  # only some of them, we will keep:
  #  1) Which has difference between actual and forecasted growth less than 0.3.
  #  2) Whcih we have not already covered in previous loop runs.

  # Filter 1
  # Filtering brand market level accuracy where there is less than 3% difference in actual and predicted growth.
  brand_market_level_accuracy_growth_df = brand_market_level_accuracy_growth_df[np.abs(brand_market_level_accuracy_growth_df["Growth_Actual"]-brand_market_level_accuracy_growth_df["Growth_Forecast"]) <= 0.03]

  # Filter 2
  # Filtering only those market, brand, segment which we have not yet covered. 
  temp_df_comm = brand_market_level_accuracy_growth_df.merge(final_table[["MKT", "Brand_new", "SEGMENT", "Final_Pred_1"]].rename(columns={"Final_Pred_1": "temp"}), on=["MKT", "Brand_new", "SEGMENT"], how="left")  # In temp column there will be null values which is new market, brand, segment.
  
  final_brand_market = brand_market_level_accuracy_growth_df.reset_index(drop=True).loc[temp_df_comm["temp"].isnull(), :].reset_index(drop=True)

  #####################     Filtering the accuracy dataframe : End     ######################





  #################################     Adding columns to the accuracy dataframe     ############################

  model_name = "v1_Model_" + "Remaining" + current_iteration_id + ".sav" 
  final_brand_market['MODEL'] = model_name

  final_brand_market['LEARNING_RATE'] = parameters['learning_rate']
  final_brand_market['N_ESTIMATORS'] = parameters['n_estimators']
  final_brand_market['SUBSAMPLE'] = parameters['subsample']
  final_brand_market['COLSAMPLE_BYTREE'] = parameters['colsample_bytree']
  final_brand_market['MAX_DEPTH'] = parameters['max_depth']
  final_brand_market['MIN_CHILD_WEIGHT'] = parameters['min_child_weight']
  final_brand_market['COLSAMPLE_BYLEVEL'] = parameters['colsample_bylevel']
  final_brand_market['COLSAMPLE_BYNODE'] = parameters['colsample_bynode']
  final_brand_market['TREE_METHOD'] = parameters['tree_method']

  ###############     Adding columns : End     ##################





  #############################     Updating the final_table and writing it to disk     ######################
  
  final_table = pd.concat([final_table, final_brand_market], sort=False)
  print("final_table shape: ", final_table.shape)
  # final_table.to_csv(result_folder + f"output_table{i}.csv", index=None)

  ###########################     Updating end     ########################



  # Writing the model
  model_name = "v1_Model_" + "Remaining" + current_iteration_id + ".sav" 
  joblib.dump(xgb_model, model_folder_path + model_name)
  
  # Writing all the predictions of the model
  all_predictions_file_name = "v1_Model_" + "Remaining" + current_iteration_id + ".csv"
  predictions.to_csv(output_xgb + all_predictions_file_name)
  
  # Writing the feature_importance_file
  feature_importance_file_name = "v1_VarImp_" + "Remaining" + current_iteration_id + ".csv"
  Final_Var_Imp.to_csv(var_imp_folder + feature_importance_file_name)


  # if (count % 5000) == 0:
  #   TEMP = pd.DataFrame([[1,2],[3,4]], columns = ['X', 'Y'])
  #   TEMP.to_csv(temp_pandas + f"{count}.csv", index = False)
  # elif count == end_index - 1:
  #   TEMP = pd.DataFrame([[1,2],[3,4]], columns = ['X', 'Y'])
  #   TEMP.to_csv(temp_pandas + f"{count}.csv", index = False)
  
  final_table.to_csv(result_folder + f"output_table{start_index}.csv", index=None)

  spark.conf.set("spark.databricks.io.cache.enabled", "true")
  spark.catalog.clearCache()


  count = count + 1
    


    # Writing the model, feature importance dataframe and the predictions
    # if len(final_brand_market) > 0:
    #   # Writing the model
    #   model_name = "v1_Model_" + "Remaining" + current_iteration_id + ".sav" 
    #   joblib.dump(xgb_model, model_folder_path + model_name)
      
    #   # Writing all the predictions of the model
    #   all_predictions_file_name = "v1_Model_" + "Remaining" + current_iteration_id + ".csv"
    #   predictions.to_csv(output_xgb + all_predictions_file_name)
      
    #   # Writing the feature_importance_file
    #   feature_importance_file_name = "v1_VarImp_" + "Remaining" + current_iteration_id + ".csv"
    #   Final_Var_Imp.to_csv(var_imp_folder + feature_importance_file_name)



# final_table.to_csv(result_folder + f"output_table{start_index}.csv", index=None)




# COMMAND ----------

print(1)

# COMMAND ----------

final_table.dtypes

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


