import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import copy
from scipy.linalg import eigh
import scipy.io as sio
from scipy.ndimage import minimum_filter
import os
from jax_helper_fns import RHS_SLICE_VMAP

# == Parameters
Nf, Nx, Ny_red, Nc = 21, 64, 33, 2

# ================================================================
# POD modes
# ================================================================
def calc_POD_modes(ts):
    # Mean subtraction
    ts_mean = np.mean(ts, axis = 0)
    ts_fluctuations = ts - ts_mean

    # Covariance matrix and eigendecomposition
    cov_matrix = np.cov(ts_fluctuations, rowvar=False, bias = False) 
    eigenvalues, eigenvectors = eigh(cov_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors, eigenvalues, ts_mean, ts_fluctuations, cov_matrix

# ================================================================
# Slicing functions
# ================================================================
def centre_slice(w):
    # Snapshot w assumed to be x by y, in Fourier space
    kx = jnp.fft.fftfreq(w.shape[0], 2 * jnp.pi / w.shape[0]) * 2 * jnp.pi
    phase =  jnp.atan2(w[1,0].imag, w[1,0].real)
    w_s = w * jnp.exp(-1j * kx[:,None] * phase)
    return w_s, phase

CENTRE_SLICE_VMAP = jax.vmap(centre_slice, in_axes = 0, out_axes = 0)

def centre_unslice(w_s, phases):
    # Add the phase back to the snapshot
    # Snapshot assumed to be x by y, in Fourier space
    kx = jnp.fft.fftfreq(w_s.shape[0], 2 * jnp.pi / w_s.shape[0]) * 2 * jnp.pi
    w = w_s * jnp.exp(+1j * kx[:,None] * phases)
    return w

CENTRE_UNSLICE_VMAP = jax.vmap(centre_unslice, in_axes = 0, out_axes = 0)

# ================================================================
# Field to vector (and vice versa) functions
# ================================================================

# Physical space functions =================================
def jax_domain2(L, Nx, Nt): 
    # Domain variables - Assume Nx and Nt to be even
    dx = jnp.float64(L) / jnp.float64(Nx)
    dt = 1. / Nt

    x = jnp.linspace(0, L - dx, Nx) - L / 2
    t = jnp.linspace(0, 1 - dt, Nt)

    kx = jnp.hstack((jnp.arange(0, Nx / 2), jnp.array(0.), jnp.arange(-Nx / 2 + 1, 0))) * (2 * jnp.pi / L) 
    kt = jnp.hstack((jnp.arange(0, Nt / 2), jnp.array(0.), jnp.arange(-Nt / 2 + 1, 0))) * (2 * jnp.pi / 1.)
    
    return x, t, kx, kt

@jax.jit
def jnp_reduce_field(f):
    # Removes all zero components of the fft2 field - jax.numpy version
    # Assume 1st dim is batch, 2nd and 3rd dims are spatial, 4th dim is channels
    out = jnp.zeros((f.shape[0], 2 * Nf + 1, Nf + 1, f.shape[-1]), dtype = jnp.complex128)
    out = out.at[:,:Nf + 1,:,:].set(f[:,:Nf + 1,:Nf + 1,:])
    out = out.at[:,Nf + 1:,:,:].set(f[:,-Nf:,:Nf + 1,:])
    return out

@jax.jit
def jnp_augment_field(f):
    # Adds zero components back to the fft2 field - jax.numpy version
    # Assume 1st dim is batch, 2nd and 3rd dims are spatial, 4th dim is channels
    out = jnp.zeros((f.shape[0], Nx, Ny_red, f.shape[-1]), dtype = jnp.complex128)
    out = out.at[:,:Nf + 1,:Nf + 1,:].set(f[:,:Nf + 1,:,:])
    out = out.at[:,-Nf:,:Nf + 1,:].set(f[:,-Nf:,:,:])
    return out

@jax.jit
def jnp_channel2state(f):
    # Turn reduced fft2 field into a vector - numpy version
    N_inp = Nf * (2 * Nf + 1) * Nc + (Nf - 1) * Nc + 1
    # Pre-allocate state
    state = jnp.zeros((f.shape[0], N_inp), dtype = jnp.float64)
    # Turn columns into a single vector (apart from the first column, which is hermitian flipped, with 0 at the beginning)
    f_ = jnp.concatenate((f[:,:,1:,:].real, f[:,:,1:,:].imag), axis = -1)
    f_ = jnp.reshape(f_, (f_.shape[0], f_.shape[1] * f_.shape[2] * f_.shape[3]))
    # Separate first column into real and imaginary parts, stack them, and concatenate with the rest
    state = state.at[:,:Nf,].set(f[:,1:Nf + 1,0,0].real)
    state = state.at[:,Nf:2 * Nf - 1,].set(f[:,2: Nf + 1,0,0].imag)
    state = state.at[:,2 * Nf - 1:,].set(f_)
    return state

@jax.jit
def jnp_state2channel(v):
    # Turn vector into reduced fft2 field - numpy version
    # Pre-allocate
    f = jnp.zeros((v.shape[0], 2 * Nf + 1, Nf + 1), dtype = jnp.complex128)
    col1 = jnp.zeros((v.shape[0], 2 * Nf + 1), dtype = jnp.complex128)

    # create first half of first column (without the 0 entry at the beginning)
    col1 = col1.at[:, 1:Nf + 1].set(v[:, :Nf] + 1j * jnp.hstack((jnp.zeros(v.shape[0])[:,None], v[:, Nf:2 * Nf - 1])))
    col1 = col1.at[:, Nf + 1:].set(jnp.fliplr(jnp.conj(col1[:, 1:Nf + 1])))

    # create other columns
    coln = jnp.reshape(v[:, 2 * Nf - 1:], (v.shape[0], 2 * Nf + 1, Nf, 2))
    coln = coln[:,:,:,0] + 1j * coln[:,:,:,1]

    # concatenate columns
    f = f.at[:,:,0].set(col1)
    f = f.at[:,:,1:].set(coln)
    return jnp.expand_dims(f, axis = -1)

@jax.jit
def field2state(f):
    # turn fft2 field into minimal vector
    return jnp_channel2state(jnp_reduce_field(f))

@jax.jit
def state2field(v):
    # turn minimal vector into fft2 field
    return jnp_augment_field(jnp_state2channel(v))

def stack_UT(U, T):
    return jnp.hstack((U, T))

def unstack_UT(UT):
    return UT[:-1], UT[-1]

# Latent space functions =================================
def field2vector(f, Nt):
    # Turn fft field into vector
    Nt_d = int(round(Nt / 3))
    Nx_ = f.shape[1]
    v = jnp.zeros((Nx_ * (Nt_d ),), dtype = jnp.complex128)
    v = v.at[:Nx_].set(f[0,:])
    for i in range(Nx_):
        v = v.at[Nx_ + i * (Nt_d - 1): Nx_ + (i + 1) * (Nt_d - 1)].set(f[1 : Nt_d, i])
    return v

def vector2field(v, Nx_, Nt):
    # Turn vector into fft field
    Nt_d = int(round(Nt / 3))
    f = jnp.zeros((Nt, Nx_), dtype = jnp.complex128)
    f = f.at[0,:].set(v[:Nx_])
    for i in range(Nx_):
        f = f.at[1:Nt_d, i].set(v[Nx_ + i * (Nt_d - 1): Nx_ + (i + 1) * (Nt_d - 1)])
        f = f.at[(Nt - Nt_d + 1) : , i].set(jnp.flipud(jnp.conj(v[Nx_ + i * (Nt_d - 1): Nx_ + (i + 1) * (Nt_d - 1)])))
    return f

def make_state(u, T, Nt):
    return jnp.hstack((field2vector(u, Nt) , T))

def extract_state(X, Nh, Nt): 
    u = vector2field(X[:-1], Nh, Nt)
    T = jnp.real(X[-1])
    return u, T

def stack_complex(X, Nh):
    # Stack real and imaginary parts of vector state
    real_parts = jnp.real(X)
    imag_parts = jnp.imag(X)[Nh:-1]
    stacked_vector = jnp.hstack((real_parts, imag_parts))
    return stacked_vector

def unstack_complex(X, Nh):
    # Unstack vector state into real and imaginary parts 
    n = int((X.shape[0] - Nh - 1) / 2 + Nh + 1)
    real_parts = X[:n]
    imag_parts = X[n:]
    imag_parts = jnp.hstack((jnp.zeros(Nh), imag_parts, 0))
    complex_vector = real_parts + 1j * imag_parts
    return complex_vector

def jax_dealiase1(u):
    # Dealiase temporal direction
    N = u.shape[0]
    Nd = round(N/3)
    v = copy.deepcopy(u)
    v = v.at[Nd:N-Nd + 1,:].set(0.)
    return v

#################################################################
# Looping functions
#################################################################
# Latent looping ================================================
def jax_HAE_dhdt(h, encoder, decoder,Nh, U, std_mu):
    # Compute latent temporal derivative for a latent time-series
    # Decode time series
    u_pod = decoder(h)
    u_fft2 = state2field(u_pod + std_mu)[:,:,:,0]

    # Calculate RHS in slice and vectorize
    dwdt = RHS_SLICE_VMAP(u_fft2)
    dwdt_vec = field2state(jnp.expand_dims(dwdt, axis = -1)) @ U
    
    return jax.jvp(encoder, (u_pod @ U,), (dwdt_vec,))[1] + dwdt_vec[:,:Nh]

def cost_fn_latent(H, Nt, kkt, encoder, decoder, Nh, U, std_mu):
    # Compute the cost function for a latent loop represented by the state vector H
    h_fft, T = extract_state(unstack_complex(H, Nh), Nh, Nt)
    h = jnp.fft.ifft(h_fft, axis = 0).real 
    # Tangents
    tangs = jnp.real(jnp.fft.ifft((1. / T) * 1j * kkt * h, axis = 0))
    # Vector field
    dhdt = jax_HAE_dhdt(h, encoder, decoder, Nh, U,std_mu)
    dhdt = jnp.fft.ifft(jax_dealiase1(jnp.fft.fft(dhdt, axis = 0)), axis = 0).real
    return  jnp.sqrt(jnp.mean((tangs - dhdt) ** 2))

def latent_vector_residual(H, h0, kkt, int_enc_X, dec_H, eigenvectors, std_mu):
    # Compute vector residual for a latent loop represented by the state vector H
    Nt, Nh = h0.shape
    hi_fft, T = extract_state(unstack_complex(H, Nh), Nh, Nt)
    hi = jnp.fft.ifft(hi_fft, axis = 0).real
    # tangs and dudt
    tangs = jnp.fft.ifft((1. / T) * 1j * kkt * hi_fft, axis = 0).real
    dhdt = jax_HAE_dhdt(hi, int_enc_X, dec_H, Nh, eigenvectors, std_mu)
    # Slice condition
    t = jnp.fft.ifft(1j * kkt * jnp.fft.fft(h0, axis = 0), axis = 0).real
    dh = hi - h0
    c = (dh * t).mean()
    R = stack_complex(make_state(jnp.fft.fft(tangs - dhdt, axis = 0), c, h0.shape[0]), Nh)
    return R

# Physical looping ================================================
def phys_residual(UT, kkt):
    # Compute the cost function for a physical loop represented by the state vector UT
    U, T = unstack_UT(UT)
    u_fft2 = state2field(U.reshape(kkt.shape[0], -1))[:,:,:,0]
    u = jnp.fft.irfft2(u_fft2, axes = (1,2))
    # tangs
    tangs = jnp.fft.ifft((1. / T) * 1j * kkt * jnp.fft.fft(u, axis = 0), axis = 0).real
    dudt = jnp.fft.irfft2(RHS_SLICE_VMAP(u_fft2), axes = (1,2))

    return jnp.sqrt(jnp.mean((tangs - dudt) ** 2))

def phys_vector_residual(UT, u0, kkt):
    # Compute vector residual for a physical loop represented by the state vector UT
    U, T = unstack_UT(UT)
    ui_fft2 = state2field(U.reshape(kkt.shape[0], -1))[:,:,:,0]
    ui = jnp.fft.irfft2(ui_fft2, axes = (1,2))
    # tangs and dudt
    tangs = jnp.fft.ifft((1. / T) * 1j * kkt * jnp.fft.fft(ui, axis = 0), axis = 0).real
    dudt = jnp.fft.irfft2(RHS_SLICE_VMAP(ui_fft2), axes = (1,2))
    res_fft2 = jnp.fft.rfft2(tangs - dudt, axes = (1,2))
    r = field2state(jnp.expand_dims(res_fft2, axis = -1)).reshape(-1, )  
    t = jnp.fft.ifft(1j * kkt * jnp.fft.fft(u0, axis = 0), axis = 0).real
    du = ui - u0
    c = jnp.sum(du * t) / jnp.sqrt(jnp.sum(t ** 2))
    return stack_UT(r, c)

#################################################################
# Latent recurrent flow
#################################################################
def find_local_minima(matrix):
    # Apply a minimum filter to find minimum values
    neighborhood_min = minimum_filter(matrix, size=10, mode='constant', cval=np.inf)
    local_minima = (matrix == neighborhood_min)
    minima_coords = np.argwhere(local_minima)
    return minima_coords

def recurrent_flow(traj, dt, T_min, T_max, nloop, r, outfolder, Nt_desired, save):
    # Code for recurrent flow analysis
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
            sio.savemat(os.path.join(outfolder, 'guess_' + str(guesses_found) + '.mat'), {'h0': h0, 'T0': T0})

    return R_mat, min_coords, guesses_found
