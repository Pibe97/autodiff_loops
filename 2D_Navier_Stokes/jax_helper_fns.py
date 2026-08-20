import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import orbax
from flax import linen as nn
from flax.training import train_state, orbax_utils
import os, h5py

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

import kol_fns as kfn

# Kolmogorov flow parameters ============================================
Re, n = 40., 4                          # Reynolds number and wave number
Nx, Ny, Nf, Ny_red = 64, 64, 21, 33     # discretization grid
Dlam = Re / (2 * n ** 2)                # Dissipation of laminar state
Elam = Re ** 2 / (4 * n ** 4)           # Energy input of laminar state

# Relevant POD folder ============================================
# data_dir = 'data/flow_data'
data_dir = 'data/paper_data/'
path = os.path.join(data_dir, 'POD_modes.h5')

if os.path.exists(path):
    print('Found path for POD mean.')
    with h5py.File(path, 'r') as h5file:
        std_mu = h5file['mean'][:]
else:
    print('Could not find path for POD mean. Run 1_POD_modes.py for this library to fully function.')
    std_mu = 0

# Vorticity equation RHS ================================================
def ForcedNSE2D(viscosity, grid, smooth):
  wave_number = n
  offsets = ((0, 0), (0, 0))
  scale = 1.
  forcing_fn = lambda grid: cfd.forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets, scale = scale)
  return spectral.equations.NavierStokes2D(
      viscosity,
      grid,
      drag=0.,
      smooth=smooth,
      forcing_fn=forcing_fn)

grid = grids.Grid((Nx, Ny), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dx, dy = grid.step
viscosity = 1. / Re

FNSE2D = ForcedNSE2D(viscosity, grid, smooth=True)
RHS_VMAP = jax.vmap(lambda x: FNSE2D.explicit_terms(x) + FNSE2D.implicit_terms(x) , in_axes = 0)

# =============================================================================

def RHS_SLICE(w_fft):
    # RHS of the vorticity equation projected onto the slice
    kx = jnp.fft.fftfreq(w_fft.shape[0], 2 * jnp.pi / w_fft.shape[0]) * 2 * jnp.pi

    x = jnp.linspace(0, 2 * jnp.pi - 2 * jnp.pi / w_fft.shape[0], w_fft.shape[0])
    sin = jnp.sin(x)[:,None]

    dwdt_fft = FNSE2D.explicit_terms(w_fft) + FNSE2D.implicit_terms(w_fft)
    dwdx_fft = 1j * (kx[:,None]) * w_fft

    return dwdt_fft - ((sin * jnp.fft.irfft2(dwdt_fft)).sum() / (sin * jnp.fft.irfft2(dwdx_fft)).sum()) * (dwdx_fft)

RHS_SLICE_VMAP = jax.vmap(RHS_SLICE, in_axes = 0)

def dphase_dt(w_fft):
    # Temporal derivative of the phase obtained from reconstruction equation
    kx = jnp.fft.fftfreq(w_fft.shape[0], 2 * jnp.pi / w_fft.shape[0]) * 2 * jnp.pi
    x = jnp.linspace(0, 2 * jnp.pi - 2 * jnp.pi / w_fft.shape[0], w_fft.shape[0])
    sin = jnp.sin(x)[:,None]

    dwdt_fft = FNSE2D.explicit_terms(w_fft) + FNSE2D.implicit_terms(w_fft)
    dwdx_fft = 1j * (kx[:,None]) * w_fft

    return (sin * jnp.fft.irfft2(dwdt_fft)).sum() / (sin * jnp.fft.irfft2(dwdx_fft)).sum()

dphase_dt_VMAP = jax.vmap(lambda x : dphase_dt(x), in_axes = 0)

def dissipation(w):
    # w is the real vorticity field
    # x, y dimensions in axes 1 and 2
    return (w ** 2).sum(axis = (1,2)) * dx * dy  / (Dlam * Re * 4 * jnp.pi ** 2)

velocity_solve = spectral.utils.vorticity_to_velocity(grid)

def energy_input(w):
    # w is a single snapshot of the spectral vorticity field
    vxhat, vyhat = velocity_solve(w)
    vx, vy = jnp.fft.irfft2(vxhat), jnp.fft.irfft2(vyhat)
    fx, _ = FNSE2D._forcing_fn_with_grid((spectral.equations._get_grid_variable(vx, grid),
                                           spectral.equations._get_grid_variable(vy, grid)))
    return (fx.data * vx).sum() * dx * dy / (Dlam * 4 * jnp.pi ** 2)

energy_input_VMAP = jax.vmap(lambda x : energy_input(x), in_axes = 0)

def get_E_D(w, is_fft2):
    if is_fft2:
        w_fft2 = w
        w = jnp.fft.irfft2(w_fft2, axes = (1,2))
    else:
        w_fft2 = jnp.fft.rfft2(w, axes = (1,2))    

    return energy_input_VMAP(w_fft2), dissipation(w)

# Helper functions for neural network ===============================================
def restore_jax_state(autoencoder, empty_params, tx, path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_state = train_state.TrainState.create(
        apply_fn=autoencoder.apply,
        params=jax.tree_util.tree_map(np.zeros_like, empty_params),  
        tx=tx,
    )
    empty_config = {'dimensions': np.array([0, 0])}
    target = {'model': empty_state, 'config': empty_config}
    state_restored = orbax_checkpointer.restore(path, item=target)
    return state_restored

def save_jax_state(state, outdir):
    config = {'dimensions': np.array([5, 3])}
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(os.path.join(os.getcwd(),outdir), ckpt, save_args=save_args)
    return

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def initialize_autoencoder(Nh, N_inp, eigenvectors, key = None):
    if not key:
        key = np.random.randint(0, 10000)
    rng = jax.random.PRNGKey(key)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (128, N_inp)) 

    # Initialize network
    autoencoder = hybrid_AE(Nh, N_inp, eigenvectors)
    params = autoencoder.init(init_rng, inp)['params']
    _ = autoencoder.apply({'params': params}, inp)

    return autoencoder, params

# Loss functions =================================================
def l2_loss(x, alpha):
    return alpha * (x ** 2).sum()

@jax.jit  
def loss_fn(state, params, batch):
    outputs = state.apply_fn({'params' : params}, batch)
    E = outputs[1]
    D = outputs[2]

    # Compute dwdt and tangent
    eps = 1e-2
    field_c = kfn.jnp_augment_field(kfn.jnp_state2channel(batch + std_mu))
    dwdt = RHS_SLICE_VMAP(field_c[:,:,:,0])
    dwdt_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(dwdt, axis = -1)))
    tang_stack = dwdt_vec / jnp.linalg.norm(dwdt_vec, axis = 1)[:,None]
    out_eps = state.apply_fn({'params' : params}, batch + eps * tang_stack)
    dir_deriv = (out_eps[0] - outputs[0]) / eps

    # Compute square loss
    field_c_real = jnp.fft.irfft2(field_c, axes = (1,2))
    field_sq_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(jnp.fft.rfft2(field_c_real[:,:,:,0] ** 2, axes = (1,2)), axis = -1)))
    field_c_out = kfn.jnp_augment_field(kfn.jnp_state2channel(outputs[0] + std_mu))
    field_c_out_real = jnp.fft.irfft2(field_c_out, axes = (1,2))
    field_out_sq_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(jnp.fft.rfft2(field_c_out_real[:,:,:,0] ** 2, axes = (1,2)), axis = -1))) 

    # Losses
    loss_rel = ((jnp.abs(outputs[0] - batch).sum(axis = 1)) / (jnp.abs(batch).sum(axis = 1)))
    loss_reg = sum(l2_loss(w, alpha = 1e-9) for w in jax.tree.leaves(params))
    loss_pod = (jnp.abs(E + D[:, :E.shape[1]]).sum(axis = 1))
    loss_tang = (jnp.abs(dir_deriv - tang_stack).sum(axis = 1))
    loss_sq = ((jnp.abs(field_out_sq_vec - field_sq_vec).sum(axis = 1)) / (jnp.abs(field_sq_vec).sum(axis = 1)))

    loss = (loss_rel + 1e-1 * loss_pod + 1e-1 * loss_tang + 1e-0 * loss_sq).mean() + loss_reg
    
    return loss

@jax.jit  
def phys_loss_fn(state, params, batch):
    outputs = state.apply_fn({'params' : params}, batch)
    E = outputs[1]
    D = outputs[2]
    field_c = kfn.jnp_augment_field(kfn.jnp_state2channel(batch + std_mu))
    field_c_out = kfn.jnp_augment_field(kfn.jnp_state2channel(outputs[0] + std_mu))

    # Compute dwdt 
    dwdt = RHS_SLICE_VMAP(field_c[:,:,:,0])
    dwdt_out = RHS_SLICE_VMAP(field_c_out[:,:,:,0])
    dwdt_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(dwdt, axis = -1)))
    dwdt_out_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(dwdt_out, axis = -1)))

    # Compute tangent 
    eps = 1e-2
    tang_stack = dwdt_vec / jnp.linalg.norm(dwdt_vec, axis = 1)[:,None]
    out_eps = state.apply_fn({'params' : params}, batch + eps * tang_stack)
    dir_deriv = (out_eps[0] - outputs[0]) / eps

    # Compute square loss
    field_c_real = jnp.fft.irfft2(field_c, axes = (1,2))
    field_sq_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(jnp.fft.rfft2(field_c_real[:,:,:,0] ** 2, axes = (1,2)), axis = -1)))
    field_c_out = kfn.jnp_augment_field(kfn.jnp_state2channel(outputs[0] + std_mu))
    field_c_out_real = jnp.fft.irfft2(field_c_out, axes = (1,2))
    field_out_sq_vec = kfn.jnp_channel2state(kfn.jnp_reduce_field(jnp.expand_dims(jnp.fft.rfft2(field_c_out_real[:,:,:,0] ** 2, axes = (1,2)), axis = -1)))
    
    loss_rel = ((jnp.abs(outputs[0] - batch).sum(axis = 1)) / (jnp.abs(batch).sum(axis = 1)))
    loss_reg = sum(l2_loss(w, alpha = 1e-9) for w in jax.tree.leaves(params))
    loss_pod = (jnp.abs(E + D[:, :E.shape[1]]).sum(axis = 1))
    loss_tang = (jnp.abs(dir_deriv - tang_stack).sum(axis = 1))
    loss_phys = (jnp.abs(dwdt_vec - dwdt_out_vec).sum(axis = 1) / (jnp.abs(dwdt_vec).sum(axis = 1)))
    loss_sq = ((jnp.abs(field_out_sq_vec - field_sq_vec).sum(axis = 1)) / (jnp.abs(field_sq_vec).sum(axis = 1)))

    loss = (loss_rel + 1e-1 * loss_pod + 1e-1 * loss_phys + 1e-2 * loss_tang + 1e-0 * loss_sq).mean() + loss_reg   

    return loss

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn, argnums = 1,has_aux=False)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit  
def train_step_phys(state, batch):
    grad_fn = jax.value_and_grad(phys_loss_fn, argnums = 1,has_aux=False)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Neural networks ===========================================================
class Encoder(nn.Module):
    Nx : int
    Nh : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(7000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(2000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(512, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        return x    


class Decoder(nn.Module):
    Nx : int
    Nh : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(2000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(3000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(7000, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        x = nn.swish(x)
        x = nn.Dense(self.Nx, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.5))(x)
        return x 


class hybrid_AE(nn.Module):
    # Hybrid autoencoder
    Nh : int        # latent dimension
    N_inp : int     # input dimension
    U : jnp.ndarray # POD eigenvectors

    def setup(self):
        self.encoder = Encoder(self.N_inp, self.Nh)
        self.decoder = Decoder(self.N_inp, self.Nh)

    def __call__(self, x):
        enc = self.encoder(x @ self.U)
        z = x @ self.U[:, :self.Nh] + enc

        dec = self.decoder(z)
        x_hat = (dec + jnp.hstack([z, jnp.zeros((x.shape[0], self.N_inp - self.Nh))])) @ ((self.U).T)

        return x_hat, enc, dec

