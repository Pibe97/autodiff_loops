import jax
jax.config.update("jax_enable_x64", True)
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import jax_helper_fns as jfn
import jax.numpy as jnp
import optax, jaxopt
from flax.training import train_state
import kse_fns as kfn

# System parameters
L, Nx, s = 39., 64, True        # System parameters
Nh = 5                          # Latent dimension
Nf = round(Nx / 3)              # Number of frequencies after dealiasing

# Directories
data_dir = 'data/paper_data'
net_dir = 'data/paper_data/net'
guess_name = 'guess_10.mat'

guess_folder = 'data/outputs/recurrent_flow'
guess_path = os.path.join(guess_folder, guess_name)
conv_folder = os.path.join(guess_folder, 'converged')

if __name__ == '__main__':
    # Load POD data
    pod_data = sio.loadmat(os.path.join(data_dir, 'POD_modes.mat'))
    eigenvectors, state_mean, state_fluctuations = pod_data['eigenvectors'], pod_data['state_mean'], pod_data['state_fluctuations']

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
    # Converge the loop
    # =============================================================================
    # Load guess
    h0, T0 = sio.loadmat(guess_path)['h0'], sio.loadmat(guess_path)['T0'][0,0]
    Nt = h0.shape[0]
    u0 = kfn.fourier_augment(dec_H(h0) + state_mean, Nx)

    # Domain variables
    x, t, kx, kt = kfn.domain2(L, Nx, Nt)
    kkx, kkt_phys = jnp.meshgrid(kx, kt)
    xx, tt = jnp.meshgrid(x, t)
    h, t, kh, kt = kfn.domain2(1, Nh, Nt)
    kkh, kkt = jnp.meshgrid(kh, kt)

    # Cost function and LBFGS optimizer
    cost_fn = lambda H : kfn.cost_latent(H, Nt, kkt, int_enc_X, dec_H, L, Nx, Nf, Nh, s, eigenvectors, state_mean)
    lbfgs = jaxopt.LBFGS(fun = cost_fn, history_size= 10, has_aux = False, maxiter = 1e4, verbose = False, tol = 1e-8, jit = True) 

    # Initial state
    H0 = kfn.stack_complex(kfn.make_state1(jnp.fft.fft(h0, axis = 0), T0, Nt), Nh)
    print('Cost before optimization: ', cost_fn(H0))

    # Run LBFGS - faster if verbose = False
    H1 = lbfgs.run(H0).params
    print(f'Cost after LBFGS: {cost_fn(H1).item()}')

    # Extract LBFGS result 
    h1_fft, T1 = kfn.extract_state1(kfn.unstack_complex(H1, Nh), Nh, Nt)
    h1 = jnp.fft.ifft(h1_fft, axis = 0).real

    # Newton optimizer
    cost_vec_fn = lambda H : kfn.latent_residual(H, h1, kkt, int_enc_X, dec_H, L, Nx, Nf,s, eigenvectors, state_mean)
    gn = jaxopt.GaussNewton(cost_vec_fn, maxiter = 20, verbose = True)

    # Run Newton
    H2 = gn.run(H1).params
    print(f'Cost after Newton: {cost_fn(H2).item()}')

    # =============================================================================
    # Post processing
    # =============================================================================
    if cost_fn(H2).item() < 1e-10:
        print('Latent solution converged.')
        h2_fft, T2 = kfn.extract_state1(kfn.unstack_complex(H2, Nh), Nh, Nt)
        h2 = jnp.real(jnp.fft.ifft(h2_fft, axis = 0))
        v2 = dec_H(jnp.real(jnp.fft.ifft(h2_fft, axis = 0)))
        u2 = kfn.fourier_augment(v2 + state_mean, Nx)

        # save - useful for autodiff_physical_looping.py
        if not os.path.exists(conv_folder): 
            os.makedirs(conv_folder)
        mdic = {"H2" : H2, "u0" : u2, "T0" : T2}
        sio.savemat(os.path.join(conv_folder, guess_name), mdic)

        # Latent projections
        dim1, dim2, dim3 = 0, 1, 2
        enc_data = enc_X(state_fluctuations[::200,:])

        fig, ax = plt.subplots(1, 3, figsize = (15, 5))
        ax[0].scatter(enc_data[:,dim1], enc_data[:,dim2], s = 1, color = 'grey')
        ax[0].plot(h0[:,dim1], h0[:,dim2], color = 'r', label = 'Guess')
        ax[0].plot(h2[:, dim1], h2[:,dim2], color = 'b', label = 'Converged')
        ax[0].set_xlabel(f'$h_{dim1}$')
        ax[0].set_ylabel(f'$h_{dim2}$')
        ax[1].scatter(enc_data[:,dim1], enc_data[:,dim3], s = 1, color = 'grey')
        ax[1].plot(h0[:,dim1], h0[:,dim3], color = 'r')
        ax[1].plot(h2[:, dim1], h2[:,dim3], color = 'b')
        ax[1].set_xlabel(f'$h_{dim1}$')
        ax[1].set_ylabel(f'$h_{dim3}$')
        ax[2].scatter(enc_data[:,dim2], enc_data[:,dim3], s = 1, color = 'grey')
        ax[2].plot(h0[:,dim2], h0[:,dim3], color = 'r')
        ax[2].plot(h2[:, dim2], h2[:,dim3], color = 'b')
        ax[2].set_xlabel(f'$h_{dim2}$')
        ax[2].set_ylabel(f'$h_{dim3}$')
        plt.tight_layout()
        plt.show(block = True)

    else:
        print('No latent solution. Cost too high.')
