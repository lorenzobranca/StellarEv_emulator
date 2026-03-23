import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
from scipy.stats import gaussian_kde

path_to_data = '/export/scratch/lbranca/Amanda_emulator/parsed_rotevol/master_Prot0.dat'
directory_output_name = './preprocessing_new/'
directory_plot_examples = './plots_examples/new/'
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
split = False # whether to apply log10 only for age<1 Gyr or for all ages
plot_examples = True # whether to plot examples of the original and new age distributions for a few simulations
SEED = 12
# np.random.seed(SEED)

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

def split_log10(arr, split=False):
    arr = np.copy(arr)
    if split:
        #we apply log only for age<1 Gyr
        arr[arr < 1] = np.log10(arr[arr < 1])
        return arr
    else:
        return np.log10(arr)




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
    df_train['Age'] = df_train['Age'].apply(lambda arr: split_log10(arr, split=split))
    df_train['output'] = df_train['output'].apply(lambda arr: log10_func(arr))

    #apply the kde transformation to the Age column and create the new time column from the quantiles
    max_length_new_time = max(len(arr) for arr in df_train['Age'])
    # max_length_new_time = 499
    print('max_length_new_time', max_length_new_time)
    df_train['New_Age'] = df_train['Age'].apply(lambda arr: kde_func(arr, max_length_new_time))

    #create the new age array sampled from the quantile 
    array_new_age = np.zeros((len(df_train), max_length_new_time), dtype=np.float64)
    for i, arr in enumerate(df_train['New_Age']):
        array_new_age[i, :len(arr)] = arr

    #interpolate the output values to the new age array using cubic interpolation
    # new_output = np.zeros((len(df_train), max_length_new_time, len(output_cols)), dtype=np.float64)
    # for i, arr in enumerate(df_train['output']):
    #     for j in range(len(output_cols)):
    #         interpolator_cubic = scipy.interpolate.CubicSpline(df_train['Age'][i], arr[:, j], extrapolate=False)
    #         # interpolator_cubic = scipy.interpolate.Akima1DInterpolator(df_train['Age'][i], arr[:, j])
    #         new_output[i, :, j] = interpolator_cubic(df_train['New_Age'][i])
    
    def interpolate_all_outputs(row):
        interpolator = scipy.interpolate.PchipInterpolator(row['Age'], row['output'], extrapolate=False)
        return interpolator(row['New_Age'])

    # Apply the interpolation row-wise
    df_train['New_output'] = df_train.apply(interpolate_all_outputs, axis=1)

    # Stack the results back into a 3D numpy array for saving/plotting downstream
    new_output = np.stack(df_train['New_output'].values)

    print('time_test shape:', array_new_age.shape)
    print('output_test shape:', new_output.shape)
    print('initial_conditions shape:', df_train[initial_conditions].values.shape)
    os.makedirs(directory_output_name, exist_ok=True)
    print('Saving preprocessed data to:', directory_output_name)
    np.save(os.path.join(directory_output_name, 'time.npy'), array_new_age)
    np.save(os.path.join(directory_output_name, 'output.npy'), new_output)
    np.save(os.path.join(directory_output_name, 'initial_conditions.npy'), df_train[initial_conditions].values)
    df_train.to_csv(os.path.join(directory_output_name, 'df_train.csv'), index=False)

    if plot_examples:
        os.makedirs(directory_plot_examples, exist_ok=True)
        # Plot examples of the original and new age distributions for a few simulations
        num_examples = 10
        rng = np.random.default_rng(13)
        examples = rng.integers(low=0, high=len(df_train), size=num_examples)
        plt.figure(figsize=(25, 4))
        for idx, sim_idx in enumerate(examples):
            plt.subplot(1, len(examples), idx+1)
            plt.hist(df_train['Age'][sim_idx], bins=30, histtype='step',  label='Original Age', density=True)
            plt.hist(df_train['New_Age'][sim_idx], bins=30, histtype='step',  label='New Age (KDE)',density=True )
            plt.title(f'Simulation {sim_idx+1}')
            plt.xlabel('Age (Gyr)')
            plt.ylabel('Frequency')
            if idx == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(directory_plot_examples, 'age_distributions_examples.png'))
        plt.show()

        fig = plt.figure(figsize=(25, 4))
        colors = plt.cm.tab10.colors[:len(examples)]
        for j in range(len(output_cols)):
            ax = fig.add_subplot(1, len(output_cols), j+1)
            for idx, sim_idx in enumerate(examples):
                # Only add labels for the first simulation to avoid duplicate legend entries
                label_orig = 'Original Output' if idx == 0 else None
                label_new = 'New Output (Interpolated)' if idx == 0 else None
                
                ax.plot(10**df_train['Age'][sim_idx], df_train['output'][sim_idx][:, j], label=label_orig, linestyle="-", color=colors[idx])
                ax.plot(10**df_train['New_Age'][sim_idx], new_output[sim_idx, :, j], label=label_new, linestyle="--", color=colors[idx])
                # ax.scatter(10**df_train['New_Age'][sim_idx], new_output[sim_idx, :, j], label=label_new, s=15, color=colors[idx])

                
            ax.set_title(output_cols[j])
            ax.set_xlabel('Age (Gyr)')
            ax.set_ylabel(output_cols[j])
            if j == 0: 
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(directory_plot_examples, 'output_interpolation_examples.png'))
        plt.show()
