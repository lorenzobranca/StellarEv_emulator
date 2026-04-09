# StellarEv_emulator
Emulator for YRAC

## Training 
- main_grid_split_don is DeepOnet for only output
- main_log15_time is DeepOnet for only time prediction
- main_log15_time_diff is DeepOnet for only diff time prediction
- main_log15_timeconcatenated is DeepOnet for predicting time as an additional output (should we try to predict the time diff ?)

## Plotting 
- plot_interpolation is for checking that the interpolation when creating the dataset is working correctly (preprocessing)
- plot_combined/plot_combined_diff take (offline) the deeponet on output and deeponet on time/time_diff and plot them together (final output)

## Preprocessing 
- data_maker creation of the training set with the interpolation and the KDE resampling of the time steps

## Utils
- train_grid_split_don_pos_diff/train_grid_split_don_pos/train_grid_split_don/ contains all the training rutines
- arch_grid_split_don aarchitectures
- utils splitting of the datasets

# TODO
- making of a function that calls all the main_* and plot_combined in backhand instead of multiple files
- ensamble of models to mitigate the errors (train multiple DeepOnet on time/time_diff and take the average of the this model as time prediction)
- train splitt model for time < 1 Gyr and time > 1 Gyr
- maybe checking what happens when combining main_log15_time results for time< 1Gyr  and main_log15_time_diff results for time>1Gyr 
- why 1 Gyr? Could be different
- main_log15_time goes back in time also at the beginning, we should look out for that (dangerous)
- using a simple MLP for predicting time/time_diff?
- training a deeponet on output and time diff (need to create a main_log15_timeconcatenated_diff) 
- I believe that the problems of main_log15_time_diff at small times is due to an over/undershooting of the very first time step (need to check)
- ask for more simulations?
