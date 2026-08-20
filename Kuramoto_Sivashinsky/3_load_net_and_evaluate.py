import jax
jax.config.update("jax_enable_x64", True)
import scipy.io as sio
import matplotlib.pyplot as plt
import jax_helper_fns as jfn
import os
import jax.numpy as jnp
import optax
from flax.training import train_state
import kse_fns as kfn

# System parameters
L, Nx, s = 39., 64, True        # System parameters
Nh = 5                          # Latent dimension
Nf = round(Nx / 3)              # Number of frequencies after dealiasing

# Directories
data_dir = 'data/paper_data'
# net_dir = 'data/paper_data/net'
net_dir = 'data/outputs/networks/net'

if __name__ == '__main__':
    # Load POD data
    pod_data = sio.loadmat(os.path.join(data_dir, 'POD_modes.mat'))
    eigenvectors, state_mean, state_fluctuations = pod_data['eigenvectors'], pod_data['state_mean'], pod_data['state_fluctuations']

    # =============================================================================
    # Example --- Loading the saved network
    # =============================================================================
    # Optimizer - required for loading
    lr_schedule = optax.piecewise_constant_schedule(1e-3, {300 : 0.5 })
    tx = optax.adam(lr_schedule)

    # Load the network
    indir = net_dir
    autoencoder, params, _ = jfn.initialize_autoencoder(Nf, Nh, eigenvectors)
    state_restored = jfn.restore_jax_state(autoencoder, params, tx, os.path.join(os.getcwd(),indir))

    optimizer = optax.adam(lr_schedule)
    E, D = jfn.Encoder(Nf, Nh), jfn.Decoder(Nf, Nh)
    AE = jfn.hybrid_AE(Nf, Nh, eigenvectors)

    params_AE = state_restored['model'].params
    params_E, params_D = state_restored['model'].params['encoder'], state_restored['model'].params['decoder']
    encoder_state = train_state.TrainState.create(apply_fn=E.apply, params=params_E, tx=optimizer)
    decoder_state = train_state.TrainState.create(apply_fn=D.apply, params=params_D, tx=optimizer)
    autoencoder_state = train_state.TrainState.create(apply_fn=AE.apply, params=params_AE, tx=tx)

    # Encoder/Decoder
    int_enc_X = lambda X : encoder_state.apply_fn({'params' : params_E}, X)
    enc_X = lambda X : X @ eigenvectors[:,:Nh] + int_enc_X(X @ eigenvectors)
    int_dec_H = lambda H : decoder_state.apply_fn({'params' : params_D}, H)
    dec_H = lambda H : (int_dec_H(H) + jnp.hstack([H, jnp.zeros((H.shape[0], Nf - Nh))])) @ eigenvectors.T
    ae_X = lambda X : autoencoder_state.apply_fn({'params' : params_AE}, X)[0]

    # =============================================================================
    # Example --- Testing the saved network
    # =============================================================================
    # ====== Latent attractor plots ====== 
    enc_data = enc_X(state_fluctuations)

    # 2D projections
    fig, ax = plt.subplots(1,3, figsize = (15, 5))
    ax[0].scatter(enc_data[::100,0], enc_data[::100,1], s = 0.1)
    ax[0].set_xlabel('$h_1$', fontsize = 12)
    ax[0].set_ylabel('$h_2$', fontsize = 12)
    ax[1].scatter(enc_data[::100,0], enc_data[::100,2], s = 0.1)
    ax[1].set_xlabel('$h_1$', fontsize = 12)
    ax[1].set_ylabel('$h_3$', fontsize = 12)
    ax[2].scatter(enc_data[::100,1], enc_data[::100,2], s = 0.1)
    ax[2].set_xlabel('$h_2$', fontsize = 12)
    ax[2].set_ylabel('$h_3$', fontsize = 12)
    plt.show(block = False)

    # 3D projection
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(enc_data[:20000,0], enc_data[:20000,1], enc_data[:20000,2], linewidth= 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show(block = False)

    # ====== Time-series / snapshot plots ====== 
    start_index, steps = 1000, 1000

    # Autoencoder input and output
    v = state_fluctuations[start_index:start_index+steps,:]
    u = kfn.fourier_augment(v + state_mean, Nx)

    v_enc = enc_X(v)
    v_dec = dec_H(v_enc)
    u_dec = kfn.fourier_augment(v_dec + state_mean, Nx)

    # Calculate derivatives
    dudt_fft = kfn.KSE_RHS_vmap(jnp.fft.fft(u, axis = 1), L, Nx, 1)
    dudt = jnp.fft.ifft(dudt_fft, axis = 1).real
    dudt_dec_fft = kfn.KSE_RHS_vmap(jnp.fft.fft(u_dec, axis = 1), L, Nx, 1)
    dudt_dec = jnp.fft.ifft(dudt_dec_fft, axis = 1).real

    # Domain
    x, t, kx, kt = kfn.domain2(L, Nx, steps)
    xx, tt = jnp.meshgrid(x, t)
    kx_Nf = kx[1:Nf+1]

    # Individual snapshots
    fig, ax = plt.subplots(4,4, figsize = (20, 12))
    for i in range(4):
        ax[i,0].plot(x, u[i * 100,:])
        ax[i,0].plot(x, u_dec[i * 100,:], '--')
        ax[i,1].scatter(kx_Nf, jnp.abs(v[i * 100,:]))
        ax[i,1].scatter(kx_Nf, jnp.abs(v_dec[i * 100,:]))
        ax[i,1].set_yscale('log')
    for i in range(4):
        ax[i,2].plot(x, dudt[i * 100,:])
        ax[i,2].plot(x, dudt_dec[i * 100,:], '--')
        ax[i,3].scatter(kx_Nf, jnp.abs(dudt_fft[i * 100,1:Nf+1]))
        ax[i,3].scatter(kx_Nf, jnp.abs(dudt_dec_fft[i * 100,1:Nf+1]))
        ax[i,3].set_yscale('log')
    ax[0,0].set_title('Physical states')
    ax[0,1].set_title('Physical frequencies')
    ax[0,2].set_title('Physical derivatives')
    ax[0,3].set_title('Physical derivatives - frequencies')
    plt.tight_layout()
    plt.show(block = False)

    # Contour plots 
    fig, ax = plt.subplots(3, 2, figsize = (12, 8))
    ax[0,0].contourf(tt, xx, u, levels = 100)
    ax[0,0].set_title('Original time-series')

    ax[1,0].contourf(tt, xx, u_dec, levels = 100)
    ax[1,0].set_title('AE output')

    ax[2,0].plot(t, jnp.linalg.norm(u - u_dec, axis = 1) / jnp.linalg.norm(u, axis = 1))
    ax[2,0].set_title('Relative error - states')

    ax[0,1].contourf(tt, xx, dudt, levels = 100)
    ax[0,1].set_title('Derivatives of original time-series')

    ax[1,1].contourf(tt, xx, dudt_dec, levels = 100)
    ax[1,1].set_title('Derivatives of AE output')

    ax[2,1].plot(t, jnp.linalg.norm(dudt - dudt_dec, axis = 1) / jnp.linalg.norm(dudt, axis = 1))
    ax[2,1].set_title('Relative error - derivatives')

    fig.tight_layout()
    plt.show(block = True)
