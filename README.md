# Durhack22
We approached this task by starting from the initial notebook provided. We kept the original Random forest model but changed the features the model was trained on. We initially tried some feature combinations however we were unable to deduce any patterns from the data straight away. Due to this reason we decided a brute force approach would be most suitable for our situation since we did not want to waste too much time on the deciding an approach at the start of the project. So we decided to take average of the data grouped by the permutations of all the discrete features such as Industry and Sector and train the model on them. For example, we trained the model on data grouped by Industry and Sector or the Sector and Date. This allowed us to find which features/ feature combinations were suitable for this prediction task. 

We trained a few different models. The basic random rainforest, an XGBoost, deep neural network and an LSTM model. A brief literature review was required for this. 
The other feature engineering we did was:
data interpolation
Weekly means, 2-day means,
Moving average,
Exponential moving, average
Outlier handling,
Research on financial indices,
OBV value (partial),
Min-max scaling,
Hyperparameter tuning,
Dummy variables,
Ensemble with three models.
