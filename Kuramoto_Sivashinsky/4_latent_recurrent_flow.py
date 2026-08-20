import jax
jax.config.update("jax_enable_x64", True)
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint
import jax_helper_fns as jfn
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.experimental.ode import odeint
import kse_fns as kfn

# System parameters
L, Nx, s = 39., 64, True        # System parameters
Nh = 5                          # Latent dimension
Nf = round(Nx / 3)              # Number of frequencies after dealiasing

# Directories
data_dir = 'data/paper_data'
output_folder = 'data/outputs/recurrent_flow'
net_dir = 'data/paper_data/net'

# Recurrent flow parameters
dt, t_max = 0.1, 1000.
r, T_min, T_max, nloop, Nt_desired = 0.05, 20., 60., 20, 128

if __name__ == '__main__':
    # Load POD data
    pod_data = sio.loadmat(os.path.join(data_dir, 'POD_modes.mat'))
    eigenvectors, state_mean, state_fluctuations = pod_data['eigenvectors'], pod_data['state_mean'], pod_data['state_fluctuations']
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    # =============================================================================
    # Load autoencoder
    # =============================================================================
    # initialize network
    autoencoder, params, _ = jfn.initialize_autoencoder(Nf, Nh, eigenvectors)
    lr_schedule = optax.piecewise_constant_schedule(1e-3, {300 : 0.5 })
    tx = optax.adam(lr_schedule)
    state_restored = jfn.restore_jax_state(autoencoder, params, tx, os.path.join(os.getcwd(),net_dir))

    E, D = jfn.Encoder(Nf, Nh), jfn.Decoder(Nf, Nh)
    AE = jfn.hybrid_AE(Nf, Nh, eigenvectors)
    params_AE = state_restored['model'].params
    params_E, params_D = state_restored['model'].params['encoder'], state_restored['model'].params['decoder']
    encoder_state = train_state.TrainState.create(apply_fn=E.apply, params=params_E, tx=tx)
    decoder_state = train_state.TrainState.create(apply_fn=D.apply, params=params_D, tx=tx)
    autoencoder_state = train_state.TrainState.create(apply_fn=AE.apply, params=params_AE, tx=tx)

    # Encoder/Decoder
    int_enc_X = lambda X : encoder_state.apply_fn({'params' : params_E}, X)
    enc_X = lambda X : X @ eigenvectors[:,:Nh] + int_enc_X(X @ eigenvectors)
    int_dec_H = lambda H : decoder_state.apply_fn({'params' : params_D}, H)
    dec_H = lambda H : (int_dec_H(H) + jnp.hstack([H, jnp.zeros((H.shape[0], Nf - Nh))])) @ eigenvectors.T
    ae_X = lambda X : autoencoder_state.apply_fn({'params' : params_AE}, X)[0]

    # =============================================================================
    # Generate Guesses via Latent Recurrent Flow Analysis
    # =============================================================================
    # 1. Generate a time-series from the latent dynamics dhdt
    # ---- Pick a random point in latent space 
    h_IC = enc_X(state_fluctuations[randint(0, state_fluctuations.shape[0] - 1),:])
    dhdt = lambda h, t : kfn.dhdt(h[None,:], int_enc_X, dec_H, L, Nx, Nf, Nh, s, eigenvectors, state_mean)
    n_points = int(t_max / dt)
    t = np.linspace(0, t_max, n_points + 1)

    # ---- Integrate and save
    print('Do latent integration...')
    h_series = odeint(dhdt, h_IC, t)
    _ = sio.savemat(os.path.join(output_folder, 'trajectory.mat'), {'h_series' : h_series})

    # 2. Recurrent flow analysis on the latent time-series 
    print('Recurrent flow analysis...')
    _ = kfn.recurrent_flow(h_series, dt, T_min, T_max, nloop, r, output_folder, Nt_desired, True)
    mdic = {"h_IC" : h_IC, "t_max" : t_max, "n_points" : n_points, "t" : t, "r" : r, "T_min" : T_min, "T_max" : T_max, "nloop" : nloop}
    
    # ---- Save parameters
    print('Save parameters...')
    _ = sio.savemat(os.path.join(output_folder, 'params.mat'), mdic)

