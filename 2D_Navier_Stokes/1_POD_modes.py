import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.io as sio

import kol_fns as kfn
import h5py
import os

data_dir = 'data/paper_data/flow_data'
out_dir = 'data/outputs'

if __name__ == '__main__':
    # Load the data 
    files = os.listdir(data_dir)
    files = [f for f in files if f.startswith('dataset') and f.endswith('.mat')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    num_sets = len(files)
    count = 0

    # Slice and stack into vectors
    for f in files:
        print(f)
        state = sio.loadmat(os.path.join(data_dir, f))['data']
        state_rfft2 = jnp.fft.rfft2(state, axes = (1,2)).astype(jnp.complex128)
        state_rfft2_s, _ = kfn.CENTRE_SLICE_VMAP(state_rfft2)
        state_stack = kfn.field2state(jnp.expand_dims(state_rfft2_s, axis = -1))
        if count == 0:
            full_stack = jnp.zeros((state_stack.shape[0] * num_sets, state_stack.shape[1]))
        full_stack = full_stack.at[count * state_stack.shape[0] : (count + 1) * state_stack.shape[0], :].set(state_stack)
        count += 1

    # Compute POD modes of the stacked vectors
    eigenvectors, eigenvalues, mean, fluctuations, _ = kfn.calc_POD_modes(state_stack)
    if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
    # Save
    with h5py.File(os.path.join(out_dir, 'POD_modes.h5'), 'w') as h5file:
        h5file.create_dataset('eigenvectors', data=eigenvectors)
        h5file.create_dataset('eigenvalues', data=eigenvalues)
        h5file.create_dataset('fluctuations', data=fluctuations)
        h5file.create_dataset('mean', data=mean) 
