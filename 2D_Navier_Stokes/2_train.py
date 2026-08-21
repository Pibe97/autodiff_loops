import jax
jax.config.update("jax_enable_x64", True)
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import time
import h5py
import scipy.io as sio
import kol_fns as kfn
import jax_helper_fns as jfn

import os
import optax, orbax
from flax.training import train_state, checkpoints, orbax_utils 

t_device = torch.device('cpu')

# Learning parameters
Nh = 192                                                        # (Latent dimension)
Nb, Nc = 128, 2                                                 # (Batch size and channels [real and imaginary components])
N_inp = jfn.Nf * (2 * jfn.Nf + 1) * Nc + (jfn.Nf - 1) * Nc + 1  # (Dimension of input vector for network)
epochs, epoch_switch = 1500, 1000                               # (Number of epochs) 
split = [0.9, 0.1]
lr_init, lr_decay, lr_switch = 5e-5, 0.1, 5
output_folder = 'data/outputs/networks/net'

if __name__ == '__main__':
    # Load the data and POD modes
    path = os.path.join(jfn.data_dir, 'POD_modes.h5')
    with h5py.File(path, 'r') as h5file:
        fluctuations = h5file['fluctuations'][:10000,:] # only loading some data, change to load more
        eigenvectors = h5file['eigenvectors'][:]
        eigenvalues = h5file['eigenvalues'][:]

    # =============================================================================
    # Shuffle and split
    idx = np.random.permutation(fluctuations.shape[0])
    fluctuations = fluctuations[idx,:]
    N_points = fluctuations.shape[0] 
    N_train = int(split[0] * N_points) + 1
    N_val = int(split[1] * N_points) 

    # Create training and test sets
    x_tot = torch.tensor(fluctuations).to(t_device)
    x_train = torch.tensor(fluctuations[:N_train, :]).to(t_device)
    x_test = torch.tensor(fluctuations[N_train:,:]).to(t_device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train)
    test_dataset = torch.utils.data.TensorDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Nb, shuffle=False, collate_fn=jfn.numpy_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Nb, shuffle=False, collate_fn=jfn.numpy_collate)

    # =============================================================================
    # Train autoencoder 
    key = np.random.randint(0, 10000)
    autoencoder, params = jfn.initialize_autoencoder(Nh, N_inp, eigenvectors, key = key)

    # Prepare optimizer and training
    scale_dict = {len(train_loader) * lr_switch : lr_decay}
    lr_schedule = optax.piecewise_constant_schedule(lr_init, scale_dict)
    tx = optax.adam(lr_schedule)
    model_state = train_state.TrainState.create(apply_fn = autoencoder.apply, params = params, tx = tx)

    # Training loop
    trained_model_state = model_state
    start_time = time.time()
    train_loss_list, test_loss_list = [], []
    print('Start training...')
    for epoch in range(epochs):
        if epoch == epoch_switch:
            print('Stage 1 done. Moving to stage 2.')
            print('Starting training - stage 2 ...')
        train_loss, div = 0, 0

        # Training loss ================================
        if epoch < epoch_switch:
            for batch in train_loader:
                trained_model_state, loss = jfn.train_step(trained_model_state, batch[0])
                train_loss += loss
                div += 1
        else:
            for batch in train_loader:
                trained_model_state, loss = jfn.train_step_phys(trained_model_state, batch[0])
                train_loss += loss
                div += 1
        train_loss /= div

        # Test loss ================================
        test_loss, div = 0, 0
        if epoch < epoch_switch:
            for batch in test_loader:
                test_loss += jfn.loss_fn(trained_model_state, trained_model_state.params, batch[0])
                div += 1
        else:
            for batch in test_loader:
                test_loss += jfn.phys_loss_fn(trained_model_state, trained_model_state.params, batch[0])
                div += 1
        test_loss /= div
        lr = lr_init + (epoch  >= lr_switch) * (lr_decay - 1.) * lr_init
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.2e}, Test loss : {test_loss:.2e}, Time: {round(time.time() - start_time, 1)}s, Learning Rate: {lr:.2e}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    print(f'Training time: {round(time.time() - start_time, 1)}s')
    print('Stage 2 done.')

    # Save network and losses      
    jfn.save_jax_state(trained_model_state, output_folder)
    sio.savemat(os.path.join(output_folder, 'losses.mat'), {'train_loss': train_loss_list,'test_loss': test_loss_list, 'key': key, 'idx': idx})
