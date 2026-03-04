import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
from scipy.stats import gaussian_kde

path_to_data = './../master_Prot0.dat'
directory_output_name = './preprocessing_output/'
columns_to_drop = [' Xcen',
                ' Prot (fast)',
                ' Bcoronal(fast)', ' dMdt(fast)',
                ' Prot(m1.00_zh000y264106d0_0.77139421857871FK_CALIBRATED.out)',
                ' Bcoronal(m1.00_zh000y264106d0_0.77139421857871FK_CALIBRATED.out)',
                ' dM/dt(m1.00_zh000y264106d0_0.77139421857871FK_CALIBRATED.out)']
initial_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM'] 
output_cols = ['logTeff', 'Prot_mid', 'Bcoronal_mid', 'Patm', 'tau_cz', 'dMdt_mid', 'luminosity']
# index_logTeff = output_cols.index('logTeff')
max_time = 13.8 # Gyr
val_percentage = 0.0



#### utilities functions
def build_grouped_df(keys):
    frames = [grouped.get_group(k) for k in keys]
    df_flat = pd.concat(frames).reset_index(drop=True)
    # Group by simulation, collect Age and output vector arrays
    grouped_df = (
        df_flat.groupby(initial_conditions)
        .apply(lambda g: pd.Series({
            'Age': g['Age'].values,
            'output': g[output_cols].values
        }))
        .reset_index()
    )
    return grouped_df

def log10_func(arr):
    arr[:, 1:] = np.log10(arr[:, 1:])
    return arr

def kde_func(x, max_length_new_time):
    kde = gaussian_kde(x)
    cdf_0 = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in x])
    u = np.linspace(0, 1, max_length_new_time)
    new_t = np.interp(u, cdf_0, x)
    return new_t



if __name__ == "__main__":

    data = np.loadtxt(path_to_data)
    columns = pd.read_csv(path_to_data).columns.tolist()
    df = pd.DataFrame(data, columns=columns)    
    df = df.drop(columns_to_drop, axis=1)
    df = df.rename(columns={'# Mstar': 'Mstar',
                            ' Age(Gyr)': 'Age',
                            ' [Fe/H]': 'FeH',
                            ' PMMA': 'PMMA',
                            ' PMMB': 'PMMB',
                            ' PMMM': 'PMMM',
                            ' log(Teff)': 'logTeff',
                            ' Patm': 'Patm',
                            ' tau_cz': 'tau_cz',
                            ' luminosity': 'luminosity',
                            ' Prot(mid)': 'Prot_mid', 
                            ' Bcoronal(mid)': 'Bcoronal_mid', 
                            ' dM/dt(mid)': 'dMdt_mid'
                            })
    #shuffle the dataframe
    print('The columns of the dataframe are:', df.columns)

    #Apply filtering 
    print('Filtering to max age:', max_time, 'Gyr')
    df = df[df['Age'] <= max_time]
    # result = {col: df[col].values for col in df.columns}

    #grouping the dataframe by the initial conditions
    grouped = df.groupby(initial_conditions)


    # Get unique simulation keys and shuffle them
    sim_keys = list(grouped.groups.keys())
    # Split: 10% for validation, 90% for training
    print(f"Percentage of validation simulations: {val_percentage*100:.1f}%")
    val_n = int(len(sim_keys) * val_percentage)
    train_keys = sim_keys[val_n:]

    # Build validation and training DataFrames (grouped: each row = one simulation)
    df_train = build_grouped_df(train_keys)
    print(f"Total simulations: {len(sim_keys)}")
    print(f"Training simulations: {len(train_keys)}, rows in df_train: {len(df_train)}")


    #apply the log10 
    df_train['Age'] = df_train['Age'].apply(lambda arr: np.log10(arr))
    df_train['output'] = df_train['output'].apply(lambda arr: log10_func(arr))

    #apply the kde transformation to the Age column and create the new time column from the quantiles
    max_length_new_time = max(len(arr) for arr in df_train['Age'])
    print('max_length_new_time', max_length_new_time)
    df_train['New_Age'] = df_train['Age'].apply(lambda arr: kde_func(arr, max_length_new_time))

    #create the new age array sampled from the quantile 
    array_new_age = np.zeros((len(df_train), max_length_new_time), dtype=np.float64)
    for i, arr in enumerate(df_train['New_Age']):
        array_new_age[i, :len(arr)] = arr

    #interpolate the output values to the new age array using cubic interpolation
    new_output = np.zeros((len(df_train), max_length_new_time, len(output_cols)), dtype=np.float64)
    for i, arr in enumerate(df_train['output']):
        for j in range(len(output_cols)):
            interpolator_cubic = scipy.interpolate.CubicSpline(df_train['Age'][i], arr[:, j], extrapolate=False)
            new_output[i, :, j] = interpolator_cubic(df_train['New_Age'][i])
    
    print('time_test shape:', array_new_age.shape)
    print('output_test shape:', new_output.shape)
    print('initial_conditions shape:', df_train[initial_conditions].values.shape)
    os.makedirs(directory_output_name, exist_ok=True)
    print('Saving preprocessed data to:', directory_output_name)
    np.save(os.path.join(directory_output_name, 'time.npy'), array_new_age)
    np.save(os.path.join(directory_output_name, 'output.npy'), new_output)
    np.save(os.path.join(directory_output_name, 'initial_conditions.npy'), df_train[initial_conditions].values)
