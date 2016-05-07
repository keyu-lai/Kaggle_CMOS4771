Instructions:

Our final classifer consists of three layers. 

The first two layers are implemented by script final_predictions.py. You can run "python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE" to produce a output file.

The third layer is implemented by avg_all_models_from_files.py. Just run command "python avg_all_models_from_files.py" and it will reproduce the script we used for the final result of Kaggle competition. It takes three files produced by the final_predictions.py before and used hard voting to create the final result. 

For further details, please refer to the comments of the scripts.