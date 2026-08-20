import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import time
from random import randint
from jax.experimental.ode import odeint
import kol_fns as kfn
import jax_helper_fns as jfn
import os
import optax, jaxopt
from flax.training import train_state, checkpoints, orbax_utils 

# Parameters
Nf, Nh, Nc = jfn.Nf, 192, 2                    
N_inp = Nf * (2 * Nf + 1) * Nc + (Nf - 1) * Nc + 1
dt, t_max = 0.005, 200
r, T_min, T_max, nloop = 0.45, 5., 50., 100 

# Directories
data_dir = 'data/paper_data'
output_folder = 'data/outputs/recurrent_flow'
net_dir = 'data/paper_data/net'

if __name__ == '__main__':
    # =============================================================================
    # Load the data and POD modes
    # =============================================================================
    with h5py.File(os.path.join(data_dir, 'POD_modes.h5'), 'r') as h5file:
        phys_data = h5file['fluctuations'][:1000, :]
        eigenvectors = h5file['eigenvectors'][:]
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    # =============================================================================
    # Load autoencoder
    # =============================================================================
    autoencoder, params = jfn.initialize_autoencoder(Nh, N_inp, eigenvectors)
    tx = optax.adam(optax.piecewise_constant_schedule(0.1, {1  : 0.1}))
    state_restored = jfn.restore_jax_state(autoencoder, params, tx, os.path.join(os.getcwd(), net_dir))

    E, D = jfn.Encoder(N_inp, Nh), jfn.Decoder(N_inp, Nh)
    AE = jfn.hybrid_AE(Nh, N_inp, eigenvectors)
    params_AE = state_restored['model'].params
    params_E, params_D = state_restored['model'].params['encoder'], state_restored['model'].params['decoder']
    encoder_state = train_state.TrainState.create(apply_fn=E.apply,params=params_E,tx=tx)
    decoder_state = train_state.TrainState.create(apply_fn=D.apply,params=params_D,tx=tx)
    autoencoder_state = train_state.TrainState.create(apply_fn=AE.apply,params=params_AE,tx=tx)

    # Apply functions
    int_enc_X = lambda X : encoder_state.apply_fn({'params' : params_E}, X)
    int_dec_H = lambda H : decoder_state.apply_fn({'params' : params_D}, H)
    ae_X = lambda X : autoencoder_state.apply_fn({'params': params_AE}, X)
    enc_X = lambda X : X @ eigenvectors[:,:Nh] + int_enc_X(X @ eigenvectors)
    dec_H = lambda H : (int_dec_H(H) + jnp.hstack([H, jnp.zeros((H.shape[0], N_inp - Nh))])) @ (eigenvectors.T)

    # =============================================================================
    # Generate time series for Latent Recurrent flow 
    # =============================================================================
    # 1. Generate a time-series from the latent dynamics dhdt
    # ---- Pick a random point in latent space 
    h_IC = enc_X(phys_data[randint(0, phys_data.shape[0]-1),:])
    dhdt = lambda h, t : kfn.jax_HAE_dhdt(h[None,:], int_enc_X, dec_H, Nh, eigenvectors, jfn.std_mu)
    n_points = int(t_max / dt)
    t = np.linspace(0, t_max, n_points + 1)

    # ---- Integrate and save
    h_series = odeint(dhdt, h_IC, t)
    _ = sio.savemat(os.path.join(output_folder, 'trajectory.mat'), {'h_series' : h_series})

    # 2. Recurrent flow analysis on the latent time-series 
    R_mat, min_coords, num_guesses = kfn.recurrent_flow(h_series, dt, T_min, T_max, nloop, r, output_folder, 128, True)
    mdic = {"h_IC" : h_IC, "t_max" : t_max,  "dt" : dt, "n_points" : n_points, "t" : t, "r" : r, "T_min" : T_min, "T_max" : T_max, "nloop" : nloop}
    sio.savemat(os.path.join(output_folder, 'params.mat'), mdic)
