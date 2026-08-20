import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
import scipy.io as sio
import copy, os
from scipy.ndimage import minimum_filter

def augment_series(X):
       # Given data X, augment antisymmetry
       # Add first and middle row of zeros
       Y = jnp.vstack((jnp.zeros((1,X.shape[1])), X, jnp.zeros((1,X.shape[1]))))
       # Add antisymmetric terms
       Y = jnp.vstack((Y, - jnp.flipud(X)))
       return Y

def fourier_augment_spectral(X, Nx):
    # Time axis is the first axis
    Nf = X.shape[1]
    Y = jnp.zeros((X.shape[0], Nx), dtype = jnp.complex128)
    Y = Y.at[:,1:Nf+1].set(1j * X)
    Y = Y.at[:, 2 * Nf + 1:].set(jnp.conj(jnp.flip(Y[:, 1:Nf+1], axis = 1)))
    return Y

def fourier_augment(X, Nx):
    # Time axis is the first axis
    return jnp.real(jnp.fft.ifft(fourier_augment_spectral(X, Nx), axis = 1))


def compute_POD_modes(ts):
    # function to compute the POD modes of a time-series
    # Mean Subtraction
    ts_mean = np.mean(ts, axis=0)
    ts_fluctuations = ts - ts_mean

    # Snapshot Covariance Matrix (Set bias = False to divide by p - 1; Set rowvar = False when observations are in rows.)
    cov_matrix = np.cov(ts_fluctuations, rowvar=False, bias = False) 

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = eigh(cov_matrix)

    # Sort Eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors, eigenvalues, ts_mean, ts_fluctuations, cov_matrix

def KSE_RHS(u,L,N,s):
    _,k = domain1(L,N)
    v = - jnp.fft.fft(jnp.fft.ifft(u) * jnp.fft.ifft(k * 1j * u))
    v = dealiase1(v) + (k ** 2 - k ** 4) * u
    if s:
        v = 1j * jnp.imag(v)
    return v

KSE_RHS_vmap = jax.vmap(KSE_RHS, in_axes=(0, None, None, None))


# ================================================================
# Physical looping functions 
# Source: Matlab implementation from Omid Ashtari
# Phys. Rev. E 105, 014217 – Published 31 January, 2022
# DOI: https://doi.org/10.1103/PhysRevE.105.014217 
# 
# Tranlated here to python / jax. 
# ================================================================
def domain1(L, N):
    # Function that gives spatial and spectral discretization of the domain of the KSE
    x = L * jnp.arange(0, N) / N - L / 2
    k = jnp.hstack((jnp.arange(0, N / 2), jnp.array(0.), jnp.arange(-N / 2 + 1, 0))) * (2 * jnp.pi / L) 
    return x, k

def cost_loop(r):
    R = jnp.real(jnp.fft.ifft2(r))
    return jnp.sqrt(jnp.mean(jnp.mean(R ** 2, axis = 0)))

def residual2(u_fft2, T, kkx, kkt):
    r = - jnp.fft.fft2(jnp.real(jnp.fft.ifft2(u_fft2)) * jnp.real(jnp.fft.ifft2(1j * kkx * u_fft2)))
    r += (kkx ** 2 - kkx ** 4 - (1. / T) * 1j * kkt) * u_fft2
    r = dealiase2(r)
    return r

def dealiase1(u):
    # Dealiasing a 1D spatial field in spectral state using the 2/3 rule.
    # input:    u       vector of Fourier coefficients
    # output:   v       'u' with 1/3 of the highest frequency modes being set to 0

    N = len(u)
    Nd = round(N/3)
    v = copy.deepcopy(u)
    v = v.at[Nd+1:N-Nd].set(0.)
    return v

def dealiase1_ts_temporal(u):
    # Dealiasing a time series of 1D spatial field in spectral state using the 2/3 rule.
    # input:    u       vector of Fourier coefficients with time axis as the first axis
    # output:   v       'u' with 1/3 of the highest frequency modes being set to 0

    N = u.shape[0]
    Nd = round(N/3)
    v = copy.deepcopy(u)
    v = v.at[Nd:N-Nd + 1,:].set(0.)
    return v

def dealiase2(f): 
    Nt = f.shape[0]
    Nx = f.shape[1]

    Nt_d = int(round(Nt / 3))
    Nx_d = int(round(Nx / 3))

    f = f.at[(Nt_d + 1):Nt - Nt_d, :].set(0.)
    f = f.at[:, (Nx_d + 1):Nx - Nx_d].set(0.)
    return f

def domain2(L, Nx, Nt):
    dx = jnp.float64(L) / jnp.float64(Nx)
    dt = 1. / Nt

    x = jnp.linspace(0, L - dx, Nx) - L / 2
    t = jnp.linspace(0, 1 - dt, Nt)

    # Distinguish Nx even and odd
    if Nx % 2 == 0:
        kx = jnp.hstack((jnp.arange(0, Nx / 2), jnp.array(0.), jnp.arange(-Nx / 2 + 1, 0))) * (2 * jnp.pi / L) 
    else:
        kx = jnp.hstack((jnp.arange(0, (Nx - 1) / 2 + 1), jnp.arange(- (Nx - 1) / 2, 0))) * (2 * jnp.pi / L) 
    
    # Assume kt to be even
    kt = jnp.hstack((jnp.arange(0, Nt / 2), jnp.array(0.), jnp.arange(-Nt / 2 + 1, 0))) * (2 * jnp.pi / 1.)
    
    return x, t, kx, kt

def field2vector2(f, Nx, Nt, s): 
    Nt_d = int(round(Nt / 3))
    Nx_d = int(round(Nx / 3))

    if s == True:
        v = jnp.imag(f[0, 1:(Nx_d+1)]).T
        for i in range(1, Nx_d + 1):
            v = jnp.hstack((v, jnp.real(f[1:(Nt_d + 1), i]), jnp.imag(f[1:(Nt_d + 1), i])))
    else:
        v = jnp.real(f[0,0])
        v = jnp.hstack((v, jnp.real(f[0, 1:(Nt_d + 1)]), jnp.imag(f[0, 1:(Nt_d + 1)])))
        for i in range(1, Nx_d + 1):
            v = jnp.hstack((v, jnp.real(f[1:(Nt_d + 1), i]), jnp.imag(f[1:(Nt_d + 1), i])))
            v = jnp.hstack((v, jnp.real(f[(Nt - Nt_d):, i]), jnp.imag(f[(Nt - Nt_d):, i])))
    return v

def vector2field2(v, Nx, Nt, s): 
    Nt_d = int(round(Nt / 3))
    Nx_d = int(round(Nx / 3))
    f = jnp.zeros((Nt, Nx), dtype = jnp.complex128)
    u = copy.deepcopy(v)

    if s == True:
        f = f.at[0, 1:(Nx_d + 1)].set(1j*u[0:Nx_d].T)
        u = u[Nx_d:]

        for i in range(1, Nx_d + 1):
            f = f.at[1:(Nt_d + 1), i].set(u[:Nt_d] + 1j*u[Nt_d:2*Nt_d])
            u = u[2*Nt_d:]

        for i in range(1, Nx_d + 1):
            f = f.at[(Nt - Nt_d):, i].set(- jnp.flipud(jnp.conj(f[1:(Nt_d + 1), i])))
            f = f.at[:, Nx - i].set(- f[:, i])

    else:
        f = f.at[0,0].set(u[0])
        u = u[1:]

        f = f.at[0, 1:(Nx_d + 1)].set(u[:Nx_d].T + 1j*u[Nx_d:2*Nx_d].T)
        u = u[2*Nx_d:]

        for i in range(1, Nx_d + 1):
            f = f.at[1:(Nt_d + 1), i].set(u[:Nt_d] + 1j*u[Nt_d:2*Nt_d])
            u = u[2*Nt_d:]

            f = f.at[Nt - Nt_d:, i].set(u[:Nt_d] + 1j*u[Nt_d:2*Nt_d])
            u = u[2*Nt_d:]
        
        f = f.at[0, (Nx - Nx_d):].set(jnp.flipud(jnp.conj(f[0, 1:(Nx_d + 1)])))
        for i in range(1, Nx_d + 1):
            f = f.at[1:, Nx - i + 2].set(jnp.flip(np.conj(f[1:, i])))

    return f

def make_state2(u, T, Nx, Nt, s): 
    X = jnp.hstack((field2vector2(u, Nx, Nt, s) , T))
    return X

def extract_state2(X, Nx, Nt, s): 
    u = vector2field2(X[:-1], Nx, Nt, s)
    T = jnp.real(X[-1])
    return u, T

def phys_residual(X, u0, L, Nx, Nt, s): 
    ui, T = extract_state2(X, Nx, Nt, s)
    _, _, kx, kt = domain2(L, Nx, Nt)
    kkx, kkt = jnp.meshgrid(kx, kt)

    r = residual2(ui, T, kkx, kkt)

    t = jnp.real(jnp.fft.ifft2(1j * kkt * u0))
    du = jnp.real(jnp.fft.ifft2(ui -u0))
    c = jnp.mean(jnp.mean(du * t, axis = 0))

    return make_state2(r, c, Nx, Nt, s)

def phys_cost(X, Nx, Nt, kkx, kkt, s): 
    u, T = extract_state2(X, Nx, Nt, s)
    r = residual2(u, T, kkx, kkt)
    return cost_loop(r)

# ========================================
# ======= Latent looping functions ======= 
# ========================================
def field2vector1(f, Nt):
    Nt_d = int(round(Nt / 3))
    Nx = f.shape[1]
    v = jnp.zeros((Nx * (Nt_d ),), dtype = jnp.complex128)
    v = v.at[:Nx].set(f[0,:])
    for i in range(Nx):
        v = v.at[Nx + i * (Nt_d - 1): Nx + (i + 1) * (Nt_d - 1)].set(f[1 : Nt_d, i])
    return v

def vector2field1(v, Nx, Nt):
    Nt_d = int(round(Nt / 3))
    f = jnp.zeros((Nt, Nx), dtype = jnp.complex128)
    f = f.at[0,:].set(v[:Nx])
    for i in range(Nx):
        f = f.at[1:Nt_d, i].set(v[Nx + i * (Nt_d - 1): Nx + (i + 1) * (Nt_d - 1)])
        f = f.at[(Nt - Nt_d + 1) : , i].set(jnp.flipud(jnp.conj(v[Nx + i * (Nt_d - 1): Nx + (i + 1) * (Nt_d - 1)])))
    return f

def make_state1(u, T, Nt):
    return jnp.hstack((field2vector1(u, Nt) , T))

def extract_state1(X, Nh, Nt): 
    u = vector2field1(X[:-1], Nh, Nt)
    T = jnp.real(X[-1])
    return u, T

def stack_complex(X, Nh):
    real_parts = jnp.real(X)
    imag_parts = jnp.imag(X)[Nh:-1]
    stacked_vector = jnp.hstack((real_parts, imag_parts))
    return stacked_vector

def unstack_complex(X, Nh):
    n = int((X.shape[0] - Nh - 1) / 2 + Nh + 1)
    real_parts = X[:n]
    imag_parts = X[n:]
    imag_parts = jnp.hstack((jnp.zeros(Nh), imag_parts, 0))
    complex_vector = real_parts + 1j * imag_parts
    return complex_vector


# ======= Hybrid AE version ======
def loop_dhdt(loop_enc, encoder, decoder, L, Nx, Nf, Nh, s, U, state_mean):
    # Decode loop for KSE_RHS calculation
    dec_loop = decoder(loop_enc)
    loop_fft = fourier_augment_spectral(dec_loop + state_mean, Nx)    
    # Dealising in time
    loop_fft = 1j * (jnp.fft.ifft(dealiase1_ts_temporal(jnp.fft.fft(loop_fft, axis = 0)), axis = 0).imag)
    # Calculate KSE_RHS and rescale
    dv_tilde_dt = KSE_RHS_vmap(loop_fft, L, Nx, s)
    dv_tilde_dt_red = jnp.imag(dv_tilde_dt[:,1:Nf+1])
    dv_tilde_dt_rescaled = dv_tilde_dt_red @ U
    
    return jax.jvp(encoder, (dec_loop @ U,), (dv_tilde_dt_rescaled,))[1] + dv_tilde_dt_rescaled[:,:Nh]


def dhdt(h, encoder, decoder, L, Nx, Nf, Nh, s, U, state_mean):
    # Decode latent vector for KSE_RHS calculation
    u = decoder(h)
    u_fft = fourier_augment_spectral(u + state_mean, Nx)
    # Calculate KSE_RHS and rescale
    dv_tilde_dt = KSE_RHS(u_fft[0,:], L, Nx, s)
    dv_tilde_dt_red = jnp.imag(dv_tilde_dt[1:Nf+1])
    dv_tilde_dt_rescaled = dv_tilde_dt_red[None,:] @ U
    return jax.jvp(encoder, (u @ U,), (dv_tilde_dt_rescaled,))[1] + dv_tilde_dt_rescaled[:,:Nh]


def cost_latent(H, Nt, kkt, enc_X, dec_H, L, Nx, Nf, Nh, s, U, state_mean):
    h1, T1 = extract_state1(unstack_complex(H, Nh), Nh, Nt)
    h_loop = jnp.fft.ifft(h1, axis = 0).real 

    tangs_h1 = jnp.real(jnp.fft.ifft((1. / T1) * 1j * kkt * h1, axis = 0))

    dhdt_h1 = loop_dhdt(h_loop, enc_X, dec_H, L, Nx, Nf, Nh, s, U, state_mean)
    dhdt_h1 = jnp.fft.ifft(dealiase1_ts_temporal(jnp.fft.fft(dhdt_h1, axis = 0)), axis = 0).real
   
    return jnp.sqrt(jnp.mean(jnp.mean((tangs_h1 - dhdt_h1) ** 2, axis = 0)))


def latent_residual(H, h0, kkt, int_enc_X, dec_H,L, Nx, Nf,s, eigenvectors, state_mean):
    # latent PO2 funtion
    Nt, Nh = h0.shape
    hi_fft, T = extract_state1(unstack_complex(H, Nh), Nh, Nt)
    hi = jnp.fft.ifft(hi_fft, axis = 0).real
    # tangs and dudt
    tangs = jnp.fft.ifft((1. / T) * 1j * kkt * hi_fft, axis = 0).real
    dhdt = loop_dhdt(hi, int_enc_X, dec_H, L, Nx, Nf, Nh, s, eigenvectors, state_mean)
    # slice condition
    t = jnp.fft.ifft(1j * kkt * jnp.fft.fft(h0, axis = 0), axis = 0).real
    dh = hi - h0
    c = (dh * t).mean()
    R = stack_complex(make_state1(jnp.fft.fft(tangs-dhdt, axis = 0), c, h0.shape[0]), Nh)
    return R


# ================================================================================
# Recurrent Flow Analysis
# ================================================================================
def find_local_minima(matrix):
    # Apply a minimum filter to find minimum values in a 10x10 neighborhood
    neighborhood_min = minimum_filter(matrix, size=10, mode='constant', cval=np.inf)
    # Local minima are where the original matrix is equal to the neighborhood min
    local_minima = (matrix == neighborhood_min)
    # Get the coordinates of the local minima
    minima_coords = np.argwhere(local_minima)
    return minima_coords

def recurrent_flow(traj, dt, T_min, T_max, nloop, r, outfolder, Nt_desired, save):
    # Recurrent flow analysis
    # traj: latent trajectory
    # dt: time step in the provided trajectory
    # T_min, T_max: minimum and maximum periods (for snapshot comparison)
    # nloop: number of steps between dT_min and dT_max
    # r: recurrence threshold
    # outfolder: output folder for saving the guesses
    # save: boolean to save the guesses

    index_low = int(round(T_min / dt))
    index_step = int(round((T_max - T_min) / (dt * nloop)))

    R_mat = jnp.zeros((nloop, traj.shape[0]))
    for i in range(nloop):
        # shift trajectory and get differences
        traj_shift = jnp.roll(traj, - (index_step * i + index_low), axis = 0)
        diff = traj_shift - traj
        R_mat = R_mat.at[i,:].set(jnp.linalg.norm(diff, axis = 1) / jnp.linalg.norm(traj, axis = 1))
    min_coords = find_local_minima(R_mat)
    min_coords = min_coords[min_coords[:,0] > 0]
    min_coords = min_coords[min_coords[:,0] < nloop]
    min_coords = min_coords[R_mat[min_coords[:,0], min_coords[:,1]] < r]

    guesses_found = 0
    for coords in min_coords:
        guesses_found += 1
        # extract trajectory
        h0 = traj[coords[1]:coords[1] + index_step * coords[0] + index_low + 1, :]
        h0_fft = jnp.fft.fft(h0,axis = 0)
        # keep/increase time steps
        if h0.shape[0] < Nt_desired:
            Nt_ = h0.shape[0]
            if Nt_ % 2 == 0:
                Nt_2 = int(Nt_ / 2)
            else:
                Nt_2 = int((Nt_ - 1) / 2)
            h0_adjusted = jnp.vstack((h0_fft[:Nt_2,:], jnp.zeros((Nt_desired - h0.shape[0], h0.shape[1])), h0_fft[Nt_2:,:])) / (h0.shape[0] / Nt_desired)
        elif h0.shape[0] > Nt_desired:
            h0_adjusted = jnp.vstack((h0_fft[:int(Nt_desired / 2),:], h0_fft[-int(Nt_desired / 2):,:])) / (h0.shape[0] / Nt_desired)
        h0 = jnp.fft.ifft(h0_adjusted, axis = 0).real
        # guess period
        T0 = (index_step * coords[0] + index_low) * dt

        # save the guess
        if save:
            sio.savemat(os.path.join(outfolder, f'guess_{str(guesses_found)}.mat'), {'h0': h0, 'T0': T0})

    return R_mat, min_coords, guesses_found

def remove_every_nth(lst, n):
    return [item for i, item in enumerate(lst) if (i + 1) % n != 0]

def eval_po(h, T):
    # A function to evaluate the latent POs 
    # - whether they are fixed points or (repeated) periodic orbits
    lim = 1e-3
    Nt = h.shape[0]
    Nt_d = round(Nt/3)

    # Fourier transform of u and vector to check for 0s (remove middle 0s)
    h_f = jnp.fft.fft(h, axis = 0)
    check = jnp.max(jnp.abs(h_f),axis = 1)
    check = check[:Nt_d+1]

    # Initialise outputs
    r = 0           # repeated boolean
    T_real = T      # Period of repeated PO

    # Check if fixed point
    if jnp.max(jnp.std(h, axis = 0)) < lim:
        r = 1
        T_real = 0
        n = 1
    else:
    # Check if repeated PO 
        n = len(check)
        while r == 0 and n > 2:
            n  = n - 1
            reduced_check = check
            reduced_check = remove_every_nth(reduced_check[1:], n)
            if jnp.max(jnp.array(reduced_check)) < lim:
                r = 1
                T_real = T / n
                
    return r, n, T_real

