•	Give a broad overview of how the optimisation and training process will work 
o	Go over the feature parameter tuning using grid search for the time-decay function parameters, which will be evaluated using Ridge re-gression with the 70% train set using a simplified rolling CV frame-work
o	Return and analyse the five best parameter combinations and outputs from the grid search.
o	Train the full XGBoost models on the 5 best parameter combinations
	Only use data from the train set
	Use a tighter rolling CV training process than with the Ridge Regression (1-24h)
	Also perform a grid search to tune the XGBoost model hy-perparameters (estimators, depth, etc.).
	Output of this model should be out-of-fold predictions (OOF) for the next day for the train AND validation set
o	We then use a logistic Regressor (or LightGBM) on the train and vali-dation set (that now include the OOF predictions of the first model) to fine-tune thresholds and turn the forecast into actionable long/short/hold outputs
•	Preliminary results would include which set of feature parameters resulted in the best model scores
