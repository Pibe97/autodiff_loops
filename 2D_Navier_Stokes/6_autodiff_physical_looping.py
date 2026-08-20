import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.io as sio
import time, os
import kol_fns as kfn
import jax_helper_fns as jfn
import jaxopt
import matplotlib.pyplot as plt

# File
conv_folder = 'data/outputs/recurrent_flow/converged/'
file_name = 'guess_1.mat'

if __name__ == '__main__':
    u0, T0 = sio.loadmat(os.path.join(conv_folder, file_name))['u0'], sio.loadmat(os.path.join(conv_folder, file_name))['T0'][0,0]

    # Domain variables
    Nt = u0.shape[0]
    x, t, kx, kt = kfn.jax_domain2(1, jfn.Nx, Nt)
    _,_,kkt_phys = jnp.meshgrid(kx, kx, kt, indexing='ij')
    kkt_phys = kkt_phys.T

    # Initial state vector
    u0_fft2 = jnp.fft.rfft2(u0)
    U0 = kfn.field2state(jnp.expand_dims(u0_fft2, axis = -1)).reshape(-1, )
    X0 = kfn.stack_UT(U0, T0)

    # Cost function
    cost_fn_phys = lambda X : kfn.phys_residual(X, kkt_phys)
    print(f'Initial cost: {cost_fn_phys(X0).item():.2e}')

    # LBFGS step
    lbfgs = jaxopt.LBFGS(fun = cost_fn_phys, history_size= 10, has_aux = False, maxiter = 1e4, verbose = True, tol = 1e-9, jit = True) 
    start_time = time.time()
    X1 = lbfgs.run(X0).params
    print(f'Cost after LBFGS: {cost_fn_phys(X1).item()}, Time: {time.time() - start_time}')

    # Gauss Newton step
    U1, T1 = kfn.unstack_UT(X1)
    u1_fft2 = kfn.state2field(U1.reshape(kkt_phys.shape[0], -1))[:,:,:,0]
    u1 = jnp.fft.irfft2(u1_fft2, axes = (1,2))

    cost_vec_phys = lambda X : kfn.phys_vector_residual(X, u1, kkt_phys)
    gn = jaxopt.GaussNewton(cost_vec_phys, maxiter = 25, verbose = True, jit = True)
    start_time = time.time()
    X2 = gn.run(X1).params
    print(f'Cost after GN: {cost_fn_phys(X2).item():.2e}, Time: {time.time() - start_time}')

    # Evaluate
    if cost_fn_phys(X2).item() < 1e-10:
        print('Physical loop converged.')
        # Extract the converged loop
        U2, T2 = kfn.unstack_UT(X2)
        u2_fft2 = kfn.state2field(U2.reshape(kkt_phys.shape[0], -1))[:,:,:,0]
        u2 = jnp.fft.irfft2(u2_fft2, axes = (1,2))

        # Dissipation and Energy input plot of L-UPO and P-UPO 
        El, Dl = jfn.get_E_D(u0, False)
        Ep, Dp = jfn.get_E_D(u2, False)

        # Plot in (E, D) plane
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        ax.plot(El, Dl, 'r--')
        ax.plot(Ep, Dp, 'b')
        plt.savefig('data/dummy_name.png')
    else:
        print('Physical loop did not converge.')

