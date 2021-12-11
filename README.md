# starbucks_offer_optimization
The repo consists of exploration and modelling notebooks
The exploration notebooks perform general exploration of the data - these include the data_exploration, final_exploration, and simple_statistics. 
The data_exploration notebook explores how the final label of interes (effective or not) compares with other input features
The final_exporation notebook explores how different offer related counts like offer received, viewed, and completed compare with the other input features
Creating_model_data and Creating_model_data-xgb calculates the input features and consolidates all the features together to create the modelling data. Then it runs and evaluates the model.

The code will run in any anaconda environment. Xgboost needs to be installed for the Creating_model_data-xgb notebook using pip install xgboost
