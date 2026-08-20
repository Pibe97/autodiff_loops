import jax
import scipy.io as sio
jax.config.update("jax_enable_x64", True)
import numpy as np
import kse_fns as kfn
import jax.numpy as jnp
import orbax
from flax import linen as nn
from flax.training import train_state, orbax_utils
import os

# Must have run 1_POD_modes.py before importing this library, or use POD_modes from paper
L, Nx, Nf = 39., 64, 21
data_dir = 'data/paper_data'
pod_dir = os.path.join(data_dir, 'POD_modes.mat')

if os.path.exists(pod_dir):
    print('Found POD modes.')
    state_mean = sio.loadmat(pod_dir)['state_mean']
else:
    print('Warning: Did not find POD modes. Please run 1_POD_modes.py to compute them, or fix path in jax_helper_fns.py')


# ==================== General helper functions ====================
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def save_jax_state(state, outdir):
    config = {'dimensions': np.array([5, 3])}
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(os.path.join(os.getcwd(),outdir), ckpt, save_args=save_args)
    return

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


# ==================== Hybrid network functions ====================
class Encoder(nn.Module):
    Nx : int
    Nh : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(32, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.Dense(self.Nh, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        return x 
    
class Decoder(nn.Module):
    Nx : int
    Nh : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(128, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nn.swish(x)
        x = nn.Dense(self.Nx, param_dtype = jnp.float64, precision = 'highest', use_bias = True, kernel_init=nn.initializers.xavier_uniform(dtype = jnp.float64), bias_init=nn.initializers.normal(stddev=0.1))(x)
        return x   

class hybrid_AE(nn.Module):
    Nx : int
    Nh : int
    U : jnp.ndarray
    def setup(self):
        self.encoder = Encoder(self.Nx, self.Nh)
        self.decoder = Decoder(self.Nx, self.Nh)
    def __call__(self, x):
        enc = self.encoder(x @ self.U)
        z = x @ self.U[:, :self.Nh] + enc
        dec = self.decoder(z)
        x_hat = (dec + jnp.hstack([z, jnp.zeros((x.shape[0], self.Nx - self.Nh))])) @ (self.U).T
        return x_hat, enc, dec 
    

def initialize_autoencoder(Nf, Nh, eigenvectors, key = None):
    if not key:
        key = np.random.randint(0, 10000)
    rng = jax.random.PRNGKey(key)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (128, Nf)) 

    # Initialize network
    autoencoder = hybrid_AE(Nf, Nh, eigenvectors)
    params = autoencoder.init(init_rng, inp)['params']
    _ = autoencoder.apply({'params': params}, inp)

    return autoencoder, params, key

    
# ==================== Loss functions ====================
@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn, argnums = 1,has_aux=False)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def train_step_phys(state, batch):
    grad_fn = jax.value_and_grad(phys_loss_fn,argnums = 1,has_aux=False)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def loss_fn(state, params, batch):
    out = state.apply_fn({'params' : params}, batch)
    # Squared loss    
    u_fft = kfn.fourier_augment_spectral(batch + state_mean, Nx)
    u = jnp.real(jnp.fft.ifft(u_fft))
    u_sq_fft = jnp.fft.fft(u ** 2).real

    u_tilde_fft = kfn.fourier_augment_spectral(out[0] + state_mean, Nx)
    u_tilde = jnp.real(jnp.fft.ifft(u_tilde_fft))
    u_tilde_sq_fft = jnp.fft.fft(u_tilde ** 2).real

    # Compute the tangent
    dudt = jnp.imag(kfn.KSE_RHS_vmap(u_fft, L, Nx, 1)[:,1:Nf+ 1])
    tang = dudt / jnp.linalg.norm(dudt, axis = 1)[:,None]

    # Compute the directional derivative via finite difference
    eps = 1e-4
    out_eps = state.apply_fn({'params' : params}, batch + eps * tang)
    dir_deriv = (out_eps[0] - out[0]) / eps

    # POD reconstruction loss
    E, D = out[1], out[2]
    

    # Losses
    loss_rel = (((out[0] - jnp.array(batch))**2).sum(axis = 1) / ((jnp.array(batch))**2 + 10**(-8)).sum(axis = 1)).mean()
    loss_tang = (((dir_deriv - tang) ** 2).sum(axis = 1)).mean() 
    loss_pod = ((E + D[:, :E.shape[1]]) ** 2).sum(axis = 1).mean()
    loss_sq = (jnp.linalg.norm(u_sq_fft - u_tilde_sq_fft, axis = 1) / jnp.linalg.norm(u_sq_fft, axis = 1)).mean()

    return loss_rel + loss_tang + loss_pod + loss_sq

def phys_loss_fn(state, params, batch):
    out = state.apply_fn({'params' : params}, batch)
    # Compute Squared loss
    u_fft = kfn.fourier_augment_spectral(batch + state_mean, Nx)
    u = jnp.real(jnp.fft.ifft(u_fft))
    u_sq_fft = jnp.fft.fft(u ** 2).real

    u_tilde_fft = kfn.fourier_augment_spectral(out[0] + state_mean, Nx)
    u_tilde = jnp.real(jnp.fft.ifft(u_tilde_fft))
    u_tilde_sq_fft = jnp.fft.fft(u_tilde ** 2).real

    # Compute tangent
    dudt = jnp.imag(kfn.KSE_RHS_vmap(u_fft, L, Nx, 1)[:,1:Nf+ 1])
    tang = dudt / jnp.linalg.norm(dudt, axis = 1)[:,None]

    # Compute the directional derivative via finite difference
    eps = 1e-4
    out_eps = state.apply_fn({'params' : params}, batch + eps * tang)
    dir_deriv = (out_eps[0] - out[0]) / eps

    # Compute dudt of output
    du_tilde_dt = jnp.imag(kfn.KSE_RHS_vmap(u_tilde_fft, L, Nx, 1)[:,1:Nf+ 1])

    # POD reconstruction loss
    E, D = out[1], out[2]

    # Losses
    loss_rel = (((out[0] - jnp.array(batch))**2).sum(axis = 1) / ((jnp.array(batch))**2 + 10**(-8)).sum(axis = 1)).mean()
    loss_tang = (((dir_deriv - tang) ** 2).sum(axis = 1)).mean() 
    loss_pod = ((E + D[:, :E.shape[1]]) ** 2).sum(axis = 1).mean()
    loss_phys = (((du_tilde_dt - dudt) ** 2).sum(axis = 1) / ((jnp.array(dudt))**2 + 10**(-8)).sum(axis = 1)).mean()
    loss_sq = (jnp.linalg.norm(u_sq_fft - u_tilde_sq_fft, axis = 1) / jnp.linalg.norm(u_sq_fft, axis = 1)).mean()

    return loss_rel + loss_tang + loss_pod + 1e-2 * loss_phys + loss_sq
