import matplotlib.pyplot as plt
import numpy as np
import os

directory_output_name = './preprocessing_output_split/'
initial_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM'] 
output_cols = ['logTeff', 'Prot_mid', 'Bcoronal_mid', 'Patm', 'tau_cz', 'dMdt_mid', 'luminosity']
directory_plot_examples = './plots/plots_examples/'
num_examples = 10


if __name__ == "__main__":
    plt.figure(figsize=(12, 6))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        plt.hist(df_train['Age'][i], bins=20, alpha=0.5, label='Original Age')
        plt.hist(df_train['New_Age'][i], bins=20, alpha=0.5, label='New Age (KDE)')
        plt.title(f'Simulation {i+1}')
        plt.xlabel('Age (Gyr)')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory_plot_examples, 'age_distributions_examples.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    for i in range(num_examples):
        for j in range(len(output_cols)):
            plt.subplot(len(output_cols), num_examples, j*num_examples + i+1)
            plt.plot(df_train['Age'][i], df_train['output'][i][:, j], label='Original Output')
            plt.plot(df_train['New_Age'][i], new_output[i, :, j], label='New Output (Interpolated)')
            plt.title(f'Simulation {i+1} - {output_cols[j]}')
            plt.xlabel('Age (Gyr)')
            plt.ylabel(output_cols[j])
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory_plot_examples, 'output_interpolation_examples.png'))
    plt.show()