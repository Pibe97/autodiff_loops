import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt 
import scipy.io as sio
import matplotlib.pyplot as plt
import kse_fns as kfn
import os

# KSE parameters
L, Nx, s = 39., 64, True

# File
conv_folder = 'data/outputs/recurrent_flow/converged/'
file_name = 'guess_1.mat'

if __name__ == '__main__':
    u0, T0 = sio.loadmat(os.path.join(conv_folder, file_name))['u0'], sio.loadmat(os.path.join(conv_folder, file_name))['T0'][0,0]
    # Domain variables
    Nt = u0.shape[0]
    x, t, kx, kt = kfn.domain2(L, Nx, Nt)
    kkx, kkt = jnp.meshgrid(kx, kt)
    xx, tt = jnp.meshgrid(x, t)

    # Inital state vector
    v0 = kfn.dealiase2(jnp.fft.fft2(u0))
    X0 = kfn.make_state2(v0, T0, Nx, Nt, s)

    # Cost function
    cost_fn_phys = lambda X : kfn.phys_cost(X, Nx, Nt, kkx, kkt, s)
    print(f'Cost before optimization: {cost_fn_phys(X0).item()}')

    # LFBGS step
    lbfgs = jaxopt.LBFGS(fun = cost_fn_phys, has_aux = False, maxiter = 1e4, verbose = False, tol = 1e-7, jit = True)
    X1 = lbfgs.run(X0).params
    print(f'Cost after LBFGS: {cost_fn_phys(X1).item()}')

    # Newton step
    v1, T1 = kfn.extract_state2(X1, Nx, Nt, s)
    u1 = jnp.real(jnp.fft.ifft2(v1))
    cost_vec_fn_phys = lambda X : kfn.phys_residual(X, u1, L, Nx, Nt, s)
    gn = jaxopt.GaussNewton(cost_vec_fn_phys, maxiter = 25, verbose = True)
    X2 = gn.run(X1).params
    print(f'Cost after Newton: {cost_fn_phys(X2).item()}')

    if cost_fn_phys(X2).item() < 1e-10:
        print('Physical solution converged.')
        # Extract the converged loop
        v2, T2 = kfn.extract_state2(X2, Nx, Nt, s)
        u2 = jnp.real(jnp.fft.ifft2(v2))

        # Contour plot of decoded latent upo and the converged physical upo
        plt.close('all')
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        ax[0].contourf(tt * T0, xx, u0, levels = 100)
        ax[0].set_title(f'Decoded Latent UPO - T = {round(T1,3)}')
        ax[1].contourf(tt * T2, xx, u2, levels = 100)
        ax[1].set_title(f'Physical UPO - T = {round(T2,3)}')
        plt.show()

        sio.savemat(os.path.join(conv_folder, 'phys_' + file_name), {'u_best' : u2, 'T_best': T2})
    else:
        print('Physical solution did not converge.')

