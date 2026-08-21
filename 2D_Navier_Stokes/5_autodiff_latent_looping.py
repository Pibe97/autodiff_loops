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
Nf, Nh, Nb, Nc = jfn.Nf, 192, 128, 2                    
N_inp = Nf * (2 * Nf + 1) * Nc + (Nf - 1) * Nc + 1

# Directories
data_dir = 'data/paper_data'
net_dir = 'data/paper_data/net'
guess_name = 'guess_1.mat'

guess_folder = 'data/outputs/recurrent_flow'
guess_path = os.path.join(guess_folder, guess_name)
conv_folder = os.path.join(guess_folder, 'converged')

if __name__ == '__main__':
    # =============================================================================
    # Load the data and POD modes
    # =============================================================================
    path = jfn.path
    with h5py.File(path, 'r') as h5file:
        eigenvectors = h5file['eigenvectors'][:]

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
    # L-BFGS Part
    # =============================================================================
    # Load a guess
    h0, T0 = sio.loadmat(guess_path)['h0'], sio.loadmat(guess_path)['T0'][0,0]
    Nt = h0.shape[0]
    _, _, kh, kt = kfn.jax_domain2(1, Nh, Nt)
    kkh ,kkt = jnp.meshgrid(kh,kt)

    # Prepare latent state vector
    u0_fft2 = kfn.jnp_augment_field(kfn.jnp_state2channel(dec_H(h0) + jfn.std_mu))[:,:,:,0]
    u0 = jnp.fft.irfft2(u0_fft2, axes = (1, 2))
    H0 = kfn.stack_complex(kfn.make_state(jnp.fft.fft(h0, axis = 0), T0, h0.shape[0]), Nh)

    # LBFGS step
    cost_fn = lambda H : kfn.cost_fn_latent(H, Nt, kkt, int_enc_X, dec_H, Nh, eigenvectors, jfn.std_mu)
    print(f'Initial cost: {cost_fn(H0).item():.2e}')
    lbfgs = jaxopt.LBFGS(fun = cost_fn, has_aux = False, maxiter = 1e5, verbose = True, tol = 1e-8, jit = True) 

    start_time = time.time()
    H1 = lbfgs.run(H0).params
    print(f'Cost after LBFGS: {cost_fn(H1).item()}, Time: {time.time() - start_time}')

    # Gauss Newton step
    h1_fft, T1 = kfn.extract_state(kfn.unstack_complex(H1, Nh), Nh, Nt)
    h1 = jnp.fft.ifft(h1_fft, axis = 0).real

    cost_vec = lambda H : kfn.latent_vector_residual(H, h1, kkt, int_enc_X, dec_H, eigenvectors, jfn.std_mu)
    gn = jaxopt.GaussNewton(cost_vec, maxiter = 20, verbose = True)
    H2 = gn.run(H1).params
    print(f'Cost after GN: {cost_fn(H2).item():.2e}, Time: {time.time() - start_time}')

    # Evaluate
    if cost_fn(H2).item() < 1e-10:
        print('Latent loop converged.')
        # Extract Newton solution
        h2_fft, T2 = kfn.extract_state(kfn.unstack_complex(H2, Nh), Nh, Nt)
        h2 = jnp.fft.ifft(h2_fft, axis = 0).real
        u2_fft2 = kfn.jnp_augment_field(kfn.jnp_state2channel(dec_H(h2) + jfn.std_mu))[:,:,:,0]
        u2 = jnp.fft.irfft2(u2_fft2, axes = (1,2))

        # Plot in (h1, h2) plane
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        ax.plot(h0[:,0], h0[:,1], 'r--')
        ax.plot(h2[:,0], h2[:,1], 'b')
        plt.show()

        # Save
        if not os.path.exists(conv_folder): 
            os.makedirs(conv_folder)
        sio.savemat(os.path.join(conv_folder, guess_name), {'h' : h2, 'u0' : u2, 'T0': T2})
    else:
        print('Latent loop did not converge.')

