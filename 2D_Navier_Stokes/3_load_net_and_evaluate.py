import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py, copy
import os
import optax
from flax.training import train_state, checkpoints, orbax_utils 

import kol_fns as kfn
import jax_helper_fns as jfn

# Parameters
Nx, Ny, Ny_red, Nf = jfn.Nx, jfn.Ny, jfn.Ny_red, jfn.Nf # (Physical dimensions)
Nh, Nb, Nc = 192, 128, 2                                # (Latent dim, batch size, channels)
N_inp = Nf * (2 * Nf + 1) * Nc + (Nf - 1) * Nc + 1      # (Network input dimension)

# Net folder
net_dir = 'data/paper_data/net'

if __name__ == '__main__':
    # Load one train and one test batch
    idx = sio.loadmat(os.path.join(net_dir, 'losses.mat'))['idx'][0,:]
    idx_train = copy.deepcopy(idx[:Nb])
    idx_train.sort()
    idx_test = copy.deepcopy(idx[-Nb:])
    idx_test.sort()

    # Load POD modes
    path = os.path.join(jfn.data_dir, f'POD_modes.h5')
    with h5py.File(path, 'r') as h5file:
        fluctuations = h5file['fluctuations']
        train_batch = fluctuations[idx_train, :]
        test_batch = fluctuations[idx_test, :]
        eigenvectors = h5file['eigenvectors'][:]
        mu = h5file['mean'][:]

    # =============================================================================
    # Loading network
    # =============================================================================
    autoencoder, params = jfn.initialize_autoencoder(Nh, N_inp, eigenvectors)
    lr_schedule = optax.piecewise_constant_schedule(0.1, {1  : 0.1})
    tx = optax.adam(lr_schedule)
    state_restored = jfn.restore_jax_state(autoencoder, params, tx, os.path.join(os.getcwd(), net_dir))

    E, D = jfn.Encoder(N_inp, Nh), jfn.Decoder(N_inp, Nh)
    AE = jfn.hybrid_AE(Nh, N_inp, eigenvectors)
    params_AE = state_restored['model'].params
    params_E, params_D = state_restored['model'].params['encoder'], state_restored['model'].params['decoder']
    encoder_state = train_state.TrainState.create(apply_fn = E.apply,params = params_E,tx = tx)
    decoder_state = train_state.TrainState.create(apply_fn = D.apply,params = params_D,tx = tx)
    autoencoder_state = train_state.TrainState.create(apply_fn = AE.apply,params = params_AE,tx = tx)

    # Apply functions
    int_enc_X = lambda X : encoder_state.apply_fn({'params' : params_E}, X)
    int_dec_H = lambda H : decoder_state.apply_fn({'params' : params_D}, H)
    ae_X = lambda X : autoencoder_state.apply_fn({'params': params_AE}, X)
    enc_X = lambda X : X @ eigenvectors[:,:Nh] + int_enc_X(X @ eigenvectors)
    dec_H = lambda H : (int_dec_H(H) + jnp.hstack([H, jnp.zeros((H.shape[0], N_inp - Nh))])) @ (eigenvectors.T)

    # =============================================================================
    # Testing of network
    # =============================================================================
    # Domain
    xx = jnp.linspace(0, 2 * jnp.pi, Nx)
    yy = jnp.linspace(0, 2 * jnp.pi, Ny) 

    # 1. Inputs - output plot, with spectral frequencies ====================
    # Training data
    train_fft2 = kfn.state2field(train_batch + mu)
    train_in = jnp.fft.irfft2(train_fft2, axes = (1,2))[:,:,:,0]
    train_out_vec = ae_X(train_batch)[0] + mu
    train_out_fft2 = kfn.state2field(train_out_vec)
    train_out = jnp.fft.irfft2(train_out_fft2, axes = (1,2))[:,:,:,0]

    # Test data
    test_fft2 = kfn.state2field(test_batch + mu)
    test_in = jnp.fft.irfft2(test_fft2, axes = (1,2))[:,:,:,0]
    test_out_vec = ae_X(test_batch)[0] + mu
    test_out_fft2 = kfn.state2field(test_out_vec)
    test_out = jnp.fft.irfft2(test_out_fft2, axes = (1,2))[:,:,:,0]

    # Choose indeces for data to plot
    k = [0, 1, 2]

    fig, ax = plt.subplots(len(k), 8, figsize = (40,  3 * 40 / 8))
    for i in range(len(k)):
        _ = ax[i,0].contourf(xx, yy, train_in[k[i],:,:].T, levels = 100)
        _ = ax[i,1].contourf(xx, yy, train_out[k[i],:,:].T, levels = 100)
        cbar1 = ax[i,2].contourf(xx, yy, jnp.abs(train_in[k[i],:,:] - train_out[k[i],:,:]), levels = 100)
        cax1 = ax[i, 2].inset_axes([1.03, 0, 0.05, 1])
        _ = fig.colorbar(cbar1, cax=cax1)
        _ = ax[i, 3].scatter(jnp.arange(Nf + 1), jnp.abs(train_fft2[k[i],0,:Nf + 1,0]), color = 'b', marker='o')
        _ = ax[i, 3].scatter(jnp.arange(Nf + 1), jnp.abs(train_out_fft2[k[i],0,:Nf + 1,0]), color = 'b', marker='x')
        _ = ax[i, 3].scatter(jnp.arange(Nf + 1), jnp.abs(train_fft2[k[i],Nf,:Nf + 1,0]), color = 'r', marker='o')
        _ = ax[i, 3].scatter(jnp.arange(Nf + 1), jnp.abs(train_out_fft2[k[i],Nf,:Nf + 1,0]), color = 'r', marker='x')
        _ = ax[i, 3].set_yscale('log')
        _ = ax[i, 3].tick_params(axis = 'both', labelsize = 20)
        
        _ = ax[i,4].contourf(xx, yy, test_in[k[i],:,:].T, levels = 100)
        _ = ax[i,5].contourf(xx, yy, test_out[k[i],:,:].T, levels = 100)
        cbar2 = ax[i,6].contourf(xx, yy, jnp.abs(test_in[k[i],:,:] - test_out[k[i],:,:]), levels = 100)
        cax2 = ax[i, 6].inset_axes([1.03, 0, 0.05, 1])
        _ = fig.colorbar(cbar2, cax=cax2)
        _ = ax[i, 7].scatter(jnp.arange(Nf + 1), jnp.abs(test_fft2[k[i],0,:Nf + 1,0]), color = 'b', marker='o')
        _ = ax[i, 7].scatter(jnp.arange(Nf + 1), jnp.abs(test_out_fft2[k[i],0,:Nf + 1,0]), color = 'b', marker='x')
        _ = ax[i, 7].scatter(jnp.arange(Nf + 1), jnp.abs(test_fft2[k[i],Nf,:Nf + 1,0]), color = 'r', marker='o')
        _ = ax[i, 7].scatter(jnp.arange(Nf + 1), jnp.abs(test_out_fft2[k[i],Nf,:Nf + 1,0]), color = 'r', marker='x')
        _ = ax[i, 7].set_yscale('log')
        _ = ax[i, 7].tick_params(axis = 'both', labelsize = 20)

    _ = ax[0, 0].set_title('Original - Train', fontsize = 25)
    _ = ax[0, 1].set_title('Reconstructed', fontsize = 25)
    _ = ax[0, 2].set_title('Abs. Difference', fontsize = 25)
    _ = ax[0, 3].set_title('Spectral (hi and lo)', fontsize = 25)

    _ = ax[0, 4].set_title('Original - Test', fontsize = 25)
    _ = ax[0, 5].set_title('Reconstructed', fontsize = 25)
    _ = ax[0, 6].set_title('Abs. Difference', fontsize = 25)
    _ = ax[0, 7].set_title('Spectrum (hi and lo)', fontsize = 25)

    count = 0
    for axx in ax.flatten():
        axx.set_box_aspect(1)
        count += 1
        if count % 4 != 0:
            _ = axx.set_xticks([0, jnp.pi, 2 * jnp.pi])
            _ = axx.set_xticklabels(['0', '$\pi$', '$2\pi$'], fontsize = 20)
            _ = axx.set_yticks([0, jnp.pi, 2 * jnp.pi])
            _ = axx.set_yticklabels(['0', '$\pi$', '$2\pi$'], fontsize = 20)

    fig.tight_layout()
    plt.show(block = False)


    # 2. Temporal derivative of inputs and outputs, with spectral frequencies ====================
    # Only for test data
    dwdt_fft2 = jfn.RHS_SLICE_VMAP(test_fft2[:,:,:,0])
    dwdt = jnp.fft.irfft2(dwdt_fft2, axes = (1,2))
    dwdt_out_fft2 = jfn.RHS_SLICE_VMAP(test_out_fft2[:,:,:,0])
    dwdt_out = jnp.fft.irfft2(dwdt_out_fft2, axes = (1,2))

    fig, ax = plt.subplots(len(k),4, figsize = (20,4 * len(k)))
    for i in range(len(k)):
        _ = ax[i,0].contourf(xx,yy,dwdt[k[i],:,:].T, levels = 200)
        _ = ax[i,1].contourf(xx,yy,dwdt_out[k[i],:,:].T, levels = 200)
        cbar = ax[i,2].contourf(xx,yy,jnp.abs(dwdt[k[i],:,:].T - dwdt_out[k[i],:,:].T), levels = 200 )
        cax = ax[i, 2].inset_axes([1.03, 0, 0.05, 1])
        _ = fig.colorbar(cbar, cax=cax)
        _ = ax[i,3].scatter(jnp.arange(Nf + 1), jnp.abs(dwdt_fft2[k[i],1,:Nf + 1]), color = 'b', marker='o')
        _ = ax[i,3].scatter(jnp.arange(Nf + 1), jnp.abs(dwdt_out_fft2[k[i],1,:Nf + 1]), color = 'b', marker='x')
        _ = ax[i,3].scatter(jnp.arange(Nf + 1), jnp.abs(dwdt_fft2[k[i],Nf,:Nf + 1]), color = 'r', marker='o')
        _ = ax[i,3].scatter(jnp.arange(Nf + 1), jnp.abs(dwdt_out_fft2[k[i],Nf,:Nf + 1]), color = 'r', marker='x')
        _ = ax[i,3].set_yscale('log')

    count = 0
    for axx in ax.flatten():
        axx.set_box_aspect(1)
        count += 1
        if count % 4 != 0:
            _ = axx.set_xticks([0, jnp.pi, 2 * jnp.pi])
            _ = axx.set_xticklabels(['0', '$\pi$', '$2\pi$'], fontsize = 20)
            _ = axx.set_yticks([0, jnp.pi, 2 * jnp.pi])
            _ = axx.set_yticklabels(['0', '$\pi$', '$2\pi$'], fontsize = 20)

    _ = ax[0,0].set_title('dwdt - Original', fontsize = 25)
    _ = ax[0,1].set_title('dwdt - Reconstructed', fontsize = 25)
    _ = ax[0,2].set_title('Abs. difference', fontsize = 25)
    _ = ax[0,3].set_title('Spectrum (hi and lo)', fontsize = 25)
    fig.tight_layout()
    plt.show(block = True)
