import jax
jax.config.update("jax_enable_x64", True)
import scipy.io as sio
import kse_fns as kfn
import os

data_folder = 'data/paper_data'
output_folder = 'data/outputs'

if __name__ == '__main__':
    print('Load data...')
    data = sio.loadmat(os.path.join(data_folder, 'DNS.mat'))
    state = data['vv']

    print('Compute POD modes...')
    eigenvectors, eigenvalues, state_mean, state_fluctuations, _ = kfn.compute_POD_modes(state.T)

    print('Save POD modes...')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sio.savemat(os.path.join(output_folder, 'POD_modes.mat'), 
                {'eigenvectors': eigenvectors, 
                'eigenvalues': eigenvalues, 
                'state_mean': state_mean, 
                'state_fluctuations': state_fluctuations}
                )
