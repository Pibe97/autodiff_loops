import jax
jax.config.update("jax_enable_x64", True)
import torch
torch.set_default_dtype(torch.float64)
import scipy.io as sio
import numpy as np
import time
import jax_helper_fns as jfn
import os
import optax
from flax.training import train_state

t_device = torch.device('cpu')

# System parameters
L, Nx, s = 39., 64, True        # System parameters
Nh = 5                          # Latent dimension
Nf = round(Nx / 3)              # Number of frequencies after dealiasing

# Learning parameters
epochs = 500                    # Number of epochs 
split = [0.9, 0.1]              # Train / test data split
init_lr, lr_decay, decay_steps = 1e-4, 0.995, 2000 # Learning rates
Nb = 2048                       # Batch size           

# Folders
data_folder = 'data/paper_data'
output_folder = 'data/outputs/networks'
net_folder = os.path.join(output_folder, 'net')

if __name__ == '__main__':
    # =============================================================================
    # Load and split the data
    # =============================================================================
    # Load POD data
    pod_data = sio.loadmat(os.path.join(data_folder, 'POD_modes.mat'))
    eigenvectors, state_mean, state_fluctuations = pod_data['eigenvectors'], pod_data['state_mean'], pod_data['state_fluctuations']

    # Shuffle the data
    idx = np.random.permutation(state_fluctuations.shape[0])
    ts_fluctuations = state_fluctuations[idx,:]

    # Split the data
    N_points = ts_fluctuations.shape[0] 
    N_train = int(split[0] * N_points) + 1
    N_val = int(split[1] * N_points) 

    # Create training and test sets
    x_train = torch.tensor(ts_fluctuations[:N_train, :]).to(t_device)
    x_test = torch.tensor(ts_fluctuations[N_train:, :]).to(t_device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train)
    test_dataset = torch.utils.data.TensorDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Nb, shuffle=False, collate_fn=jfn.numpy_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Nb, shuffle=False, collate_fn=jfn.numpy_collate)

    # =============================================================================
    # Train autoencoder
    # =============================================================================
    # initialize network and optimizer
    autoencoder, params, key = jfn.initialize_autoencoder(Nf, Nh, eigenvectors)
    lr_schedule = optax.exponential_decay(init_lr, decay_steps, lr_decay)
    tx = optax.adam(lr_schedule)
    model_state = train_state.TrainState.create(apply_fn=autoencoder.apply,params=params,tx=tx)

    # Training loop
    trained_model_state = model_state
    loss_list, test_loss_list = [], []
    start_time = time.time()

    print('Starting training - stage 1 ...')
    for epoch in range(epochs):
        if epoch == int(round(epochs / 2)):
            print('Stage 1 done. Moving to stage 2.')
            print('Starting training - stage 2 ...')

        train_loss, div = 0, 0
        if epoch < int(round(epochs / 2)):
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

        test_loss, div = 0, 0
        if epoch < int(round(epochs / 2)):
            for batch in test_loader:
                test_loss += jfn.loss_fn(trained_model_state, trained_model_state.params, batch[0])
                div += 1
        else:
            for batch in test_loader:
                test_loss += jfn.phys_loss_fn(trained_model_state, trained_model_state.params, batch[0])
                div += 1
        test_loss /= div

        lr = init_lr * lr_decay ** (trained_model_state.opt_state[-1][0] // decay_steps)        


        print(f'Epoch {epoch+1}, Loss: {train_loss:.2e}, Test loss : {test_loss:.2e}, Time: {round(time.time() - start_time, 1)}s, Learning Rate: {lr:.2e}')
        loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    print(f'Training time: {round(time.time() - start_time, 1)}s')
    print('Stage 2 done.')

    # Save
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    jfn.save_jax_state(trained_model_state, net_folder)
    sio.savemat(os.path.join(net_folder, 'losses.mat'), {'loss' : train_loss, 'test_loss' : test_loss, 'key': key, 'idx': idx})
