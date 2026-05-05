import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial
from typing import Optional

from .base import Network


@dataclass
class BaseHyperConfig:
    # fixed hyperparameters between simulations; changing them triggers jax.jit recompilation
    n_mitral: int
    n_granule: int
    n_granule_per_task: int

    mitral_nonlinear_func: str = "relu"
    granule_nonlinear_func: str = "sigmoid"

    n_steps_to_steady: int = 50
    n_epochs_per_pair: int = 200


@dataclass
class NeurogenesisHyperConfig(BaseHyperConfig):
    pass


@dataclass
class BaseHyperDynConfig:
    # learning parameters
    learning_rate_B: float
    learning_rate_F: float
    learning_rate_th_g: float
    learning_rate_th_m: float
    decay_rate_B: float
    decay_rate_F: float
    decay_mitral_thres: float
    decay_granule_thres: float

    # dynamical parameters
    mitral_self_excitation: float
    tau_mitral: float
    tau_granule: float
    granule_activation_scaling: float = 1.0
    mitral_activation_scaling: float = 1.0

    # weight norm
    F_norm: float = 1.0


@dataclass
class NeurogenesisHyperDynConfig(BaseHyperDynConfig):
    th_g_hi_ratio: float = 0.95


@dataclass
class BaseNetworkState:
    B_mat: jax.Array  # (n_mitral, n_granule)
    F_mat: jax.Array  # (n_granule, n_mitral)
    mitral_thres: jax.Array  # (n_mitral,)
    granule_thres: jax.Array  # (n_granule,)


@dataclass
class NeurogenesisState(BaseNetworkState):
    task_counter: int = 0
    rng: Optional[jax.Array] = None


class _ActivationMixin:
    def _setup_nonlinearities(self):
        self.mitral_nonlinear_func = self._nonlinear_activation(self.hyper.mitral_nonlinear_func)
        self.granule_nonlinear_func = self._nonlinear_activation(self.hyper.granule_nonlinear_func)

    def _nonlinear_activation(self, func_name: str) -> callable:
        if func_name == "relu":
            return jax.nn.relu
        if func_name == "piecewise_linear":
            def func(x):
                return jnp.piecewise(x, [x < -0.5, (x >= -0.5) & (x < 0.5), x >= 0.5], [0, lambda x: x+0.5, 1])
                # return jnp.piecewise(x, [x < 0, (x >= 0) & (x < 1), x >= 1], [0, lambda x: x, 1])
            return func
        if func_name == "step":
            def func(x):
                return jnp.where(x > 0, 1.0, 0.0)
            return func
        if func_name == "sigmoid":
            return jax.nn.sigmoid
        raise ValueError(f"Unsupported non-linear function: {func_name}")


class Neurogenesis1(
    _ActivationMixin,
    Network[NeurogenesisHyperConfig, NeurogenesisState, NeurogenesisHyperDynConfig],
):
    def __init__(self, hyperparameters: NeurogenesisHyperConfig):
        super().__init__(hyperparameters)
        self._setup_nonlinearities()

    def init_state(
        self, rng, hyperD: NeurogenesisHyperDynConfig, random_x: jax.Array, *args, **kwargs
    ) -> NeurogenesisState:
        n_mitral, n_granule = self.hyper.n_mitral, self.hyper.n_granule
        F_norm = hyperD.F_norm
        rng_F, rng = jax.random.split(rng)
        subkey, rng = jax.random.split(rng)
        F_mat = jax.random.uniform(rng_F, (n_granule, n_mitral), minval=0.0, maxval=1.0)
        F_mat = F_mat / (jnp.linalg.norm(F_mat, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat = jnp.zeros((n_mitral, n_granule))
        mitral_thres = jnp.zeros((n_mitral,))
        granule_thres = jnp.ones((n_granule,)) * jnp.sqrt(3*n_mitral) / 4 * F_norm # mean of x @ f when x ~ U(0, 1) and f ~ U(0, 1)*(rescale to F_norm)
        return NeurogenesisState(
            B_mat=B_mat,
            F_mat=F_mat,
            mitral_thres=mitral_thres,
            granule_thres=granule_thres,
            task_counter=0,
            rng=subkey
        )

    def _single_step_forward(self, hyperD: NeurogenesisHyperDynConfig, state: NeurogenesisState, x, m, g):
        B_mat, F_mat = state.B_mat, state.F_mat
        th_m, th_g = state.mitral_thres, state.granule_thres
        tau_m, tau_g = hyperD.tau_mitral, hyperD.tau_granule

        h_m = x + hyperD.mitral_self_excitation * m - g @ B_mat.T - th_m[None, :]
        f_m = self.mitral_nonlinear_func(h_m * hyperD.mitral_activation_scaling)
        m_new = (1 - 1 / tau_m) * m + (1 / tau_m) * f_m

        # h_g = m_new @ F_mat.T - th_g[None, :]
        h_g = m @ F_mat.T - th_g[None, :]
        f_g = self.granule_nonlinear_func(h_g * hyperD.granule_activation_scaling)
        g_new = (1 - 1 / tau_g) * g + (1 / tau_g) * f_g
        return m_new, g_new

    @partial(jax.jit, static_argnames=("self",))
    def _forward(self, hyperD: NeurogenesisHyperDynConfig, state: NeurogenesisState, x: jax.Array):
        m_0 = jnp.zeros((x.shape[0], self.hyper.n_mitral))
        g_0 = jnp.zeros((x.shape[0], self.hyper.n_granule))

        def step_fn(carry, _):
            m, g = carry
            m_new, g_new = self._single_step_forward(hyperD, state, x, m, g)
            return (m_new, g_new), None

        (m_steady, g_steady), _ = jax.lax.scan(step_fn, (m_0, g_0), None, length=self.hyper.n_steps_to_steady)
        return m_steady, g_steady

    # def _single_step_forward(self, hyperD: NeurogenesisHyperDynConfig, state: NeurogenesisState, x, hm, hg):
    #     B_mat, F_mat = state.B_mat, state.F_mat
    #     th_m, th_g = state.mitral_thres, state.granule_thres
    #     tau_m, tau_g = hyperD.tau_mitral, hyperD.tau_granule

    #     m = self.mitral_nonlinear_func(hm * hyperD.mitral_activation_scaling)
    #     g = self.granule_nonlinear_func(hg * hyperD.granule_activation_scaling)

    #     delta_hm = x - g @ B_mat.T - th_m[None, :]
    #     hm_new = (1 - 1 / tau_m) * hm + (1 / tau_m) * delta_hm
    #     delta_hg = m @ F_mat.T - th_g[None, :]
    #     hg_new = (1 - 1 / tau_g) * hg + (1 / tau_g) * delta_hg
    #     return hm_new, hg_new

    # @partial(jax.jit, static_argnames=("self",))
    # def _forward(self, hyperD: NeurogenesisHyperDynConfig, state: NeurogenesisState, x: jax.Array):
    #     hm_0 = jnp.zeros((x.shape[0], self.hyper.n_mitral))
    #     hg_0 = jnp.zeros((x.shape[0], self.hyper.n_granule))

    #     def step_fn(carry, _):
    #         hm, hg = carry
    #         hm_new, hg_new = self._single_step_forward(hyperD, state, x, hm, hg)
    #         return (hm_new, hg_new), None

    #     (hm_steady, hg_steady), _ = jax.lax.scan(step_fn, (hm_0, hg_0), None, length=self.hyper.n_steps_to_steady)
    #     m_steady = self.mitral_nonlinear_func(hm_steady * hyperD.mitral_activation_scaling)
    #     g_steady = self.granule_nonlinear_func(hg_steady * hyperD.granule_activation_scaling)
    #     return m_steady, g_steady

    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        epoch_idx,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g, lr_th_m = hyperD.learning_rate_th_g, hyperD.learning_rate_th_m
        decay_th_g, decay_th_m = hyperD.decay_granule_thres, hyperD.decay_mitral_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        task_counter = state.task_counter
        rng = state.rng
        n_mitral = self.hyper.n_mitral
        n_updated_granule = self.hyper.n_granule_per_task
        idx_list = jnp.arange(n_updated_granule) + task_counter * n_updated_granule
        idx_list = idx_list % self.hyper.n_granule

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)
        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        rng_F, rng_B, rng = jax.random.split(rng, 3)

        F_mat_updated = F_mat[idx_list, :]
        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]
        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        B_mat_updated = B_mat[:, idx_list]
        tmp1 = (m1 * (m_bar - m1))[:, None] * g1[None, :]
        tmp2 = (m2 * (m_bar - m2))[:, None] * g2[None, :]
        # tmp1 = ((m_bar - m1))[:, None] * g1[None, :]
        # tmp2 = ((m_bar - m2))[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, None)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * jnp.sqrt(n_mitral/3)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        n_updated_granule, n_mitral = self.hyper.n_granule_per_task, self.hyper.n_mitral
        task_counter = state.task_counter
        rng = state.rng
        F_norm = hyperD.F_norm
        subkey, rng = jax.random.split(rng)
        idx_list = jnp.arange(n_updated_granule) + task_counter * n_updated_granule
        idx_list = idx_list % self.hyper.n_granule
        F_mat_tmp = jax.random.uniform(subkey, (n_updated_granule, n_mitral), minval=0.0, maxval=1.0)
        F_mat_tmp = F_mat_tmp / (jnp.linalg.norm(F_mat_tmp, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat_tmp = jnp.zeros((n_mitral, n_updated_granule))
        th_g_tmp = jnp.ones(n_updated_granule) * jnp.sqrt(3*n_mitral) / 4 * F_norm
        F_mat = state.F_mat.at[idx_list, :].set(F_mat_tmp)
        B_mat = state.B_mat.at[:, idx_list].set(B_mat_tmp)
        th_g = state.granule_thres.at[idx_list].set(th_g_tmp)
        state = state.replace(F_mat=F_mat, B_mat=B_mat, granule_thres=th_g, rng=rng)

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, i)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, state)
        task_counter = new_state.task_counter + 1
        return new_state.replace(task_counter=task_counter)

class Neurogenesis_randomKSelection(Neurogenesis1):
    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        idx_list: jax.Array,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g, lr_th_m = hyperD.learning_rate_th_g, hyperD.learning_rate_th_m
        decay_th_g, decay_th_m = hyperD.decay_granule_thres, hyperD.decay_mitral_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        rng = state.rng
        n_mitral = self.hyper.n_mitral
        n_updated_granule = self.hyper.n_granule_per_task

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)
        # hg_steady12 = m_steady12 @ F_mat.T - th_g[None, :]
        # hg_steady_max = jnp.max(hg_steady12, axis=0)
        # _, selected_g_idx = jax.lax.top_k(hg_steady_max, n_updated_granule)
        # idx_list = selected_g_idx
        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        rng_F, rng_B, rng = jax.random.split(rng, 3)

        F_mat_updated = F_mat[idx_list, :]
        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]
        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        B_mat_updated = B_mat[:, idx_list]
        tmp1 = (m1 * (m_bar - m1))[:, None] * g1[None, :]
        tmp2 = (m2 * (m_bar - m2))[:, None] * g2[None, :]
        # tmp1 = ((m_bar - m1))[:, None] * g1[None, :]
        # tmp2 = ((m_bar - m2))[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, None)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * jnp.sqrt(n_mitral/3)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisState:

        task_counter = state.task_counter
        rng = state.rng
        n_updated_granule, n_mitral = self.hyper.n_granule_per_task, self.hyper.n_mitral
        F_norm = hyperD.F_norm
        subkey, rng = jax.random.split(rng)
        idx_list = jax.random.choice(subkey, jnp.arange(self.hyper.n_granule), (n_updated_granule,), replace=False)
        F_mat_tmp = jax.random.uniform(subkey, (n_updated_granule, n_mitral), minval=0.0, maxval=1.0)
        F_mat_tmp = F_mat_tmp / (jnp.linalg.norm(F_mat_tmp, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat_tmp = jnp.zeros((n_mitral, n_updated_granule))
        th_g_tmp = jnp.ones(n_updated_granule) * jnp.sqrt(3*n_mitral) / 4 * F_norm
        F_mat = state.F_mat.at[idx_list, :].set(F_mat_tmp)
        B_mat = state.B_mat.at[:, idx_list].set(B_mat_tmp)
        th_g = state.granule_thres.at[idx_list].set(th_g_tmp)
        state = state.replace(F_mat=F_mat, B_mat=B_mat, granule_thres=th_g, rng=rng)
        # state = state.replace(rng=rng)

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, idx_list)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, state)
        task_counter = new_state.task_counter + 1
        return new_state.replace(task_counter=task_counter)


@dataclass
class NeurogenesisTopKState(NeurogenesisState):
    """State for top-K no-init variant. Carries the same 3 inert idx-tracking fields
    as run_topk_no_theta.py's NeurogenesisTopKDebugState so that XLA compiles _update
    and _update_one_epoch byte-for-byte the same — without these fields, JAX 0.10's
    XLA optimizer produces subtly different fused matmul precision that compounds
    over the 300×n_pretrain inner-epoch loop into a major divergence (89° → 60°).
    """
    first_epoch_idx_list: jax.Array = None
    last_epoch_idx_list: jax.Array = None
    idx_list_changed: jax.Array = None


class Neurogenesis_topKSelection_noInit(Neurogenesis1):
    """Top-K granule selection (no init) variant — exact port of run_topk_no_theta.py
    notebook recipe so that the migrated lib reproduces topk_debug_dump_no_theta.pkl
    bit-for-bit.

    Dynamics (vs ng1/rk):
      - idx_list chosen once per pair in _update using F·m (no θ), held fixed across
        all n_epochs_per_pair inner epochs.
      - F rows L1-normalized to F_norm.
      - δth_g1 uses (2/3) coefficient (NOT sqrt(n_mitral/3) like ng1/rk).
      - task_counter incremented by 1 per pair.

    The runner pairs this with F_norm = max_dyn_range_ratio/(1-th_g_hi)/(2/3)/c²
    (= 60/c² at canonical th_g_hi=0.95, max_dyn_range_ratio=2.0). That F_norm is
    intentionally different from ng1/rk's 1/(1-th_g_hi)/c²; the two scales were
    validated independently in the topk_no_theta notebook.

    The state carries 3 inert tracking fields (first_epoch_idx_list / last_epoch_idx_list
    / idx_list_changed) — see NeurogenesisTopKState docstring for why they cannot be
    dropped without breaking numerical reproducibility on JAX 0.10 GPU.
    """

    def init_state(self, rng: jax.Array, hyperD: NeurogenesisHyperDynConfig, random_x: jax.Array):
        base_state = super().init_state(rng, hyperD, random_x)
        empty_idx = -jnp.ones((self.hyper.n_granule_per_task,), dtype=jnp.int32)
        return NeurogenesisTopKState(
            B_mat=base_state.B_mat,
            F_mat=base_state.F_mat,
            mitral_thres=base_state.mitral_thres,
            granule_thres=base_state.granule_thres,
            task_counter=base_state.task_counter,
            rng=base_state.rng,
            first_epoch_idx_list=empty_idx,
            last_epoch_idx_list=empty_idx,
            idx_list_changed=jnp.array(False),
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisTopKState,
        x1: jax.Array,
        x2: jax.Array,
        epoch_idx,
        idx_list=None,
        *args,
        **kwargs,
    ) -> NeurogenesisTopKState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g = hyperD.learning_rate_th_g
        decay_B = hyperD.decay_rate_B
        decay_th_g = hyperD.decay_granule_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        rng = state.rng
        n_mitral = self.hyper.n_mitral

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)

        first_epoch_idx_list = jax.lax.cond(
            epoch_idx == 0,
            lambda: idx_list,
            lambda: state.first_epoch_idx_list,
        )

        idx_list_changed = jax.lax.cond(
            epoch_idx == 0,
            lambda: False,
            lambda: jnp.logical_or(state.idx_list_changed, jnp.any(idx_list != state.last_epoch_idx_list)),
        )

        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        _, _, rng = jax.random.split(rng, 3)

        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]
        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp = jnp.clip(F_new_tmp, 0, None)
        # L1 normalization (notebook convention)
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True, ord=1) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        tmp1 = (m_bar - m1)[:, None] * g1[None, :]
        tmp2 = (m_bar - m2)[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = decay_B * B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, None)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        # (2/3) coefficient (notebook convention; NOT sqrt(n_mitral/3) like ng1/rk)
        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * (2 / 3)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
            first_epoch_idx_list=first_epoch_idx_list,
            last_epoch_idx_list=idx_list,
            idx_list_changed=idx_list_changed,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisTopKState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisTopKState:
        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)

        # top-K selection: F·m, NO θ subtraction. Held fixed for all inner epochs.
        hg_steady12 = m_steady12 @ state.F_mat.T
        hg_steady_max = jnp.max(hg_steady12, axis=0)
        _, selected_g_idx = jax.lax.top_k(hg_steady_max, self.hyper.n_granule_per_task)
        idx_list = selected_g_idx

        reset_state = state.replace(
            idx_list_changed=jnp.array(False),
            first_epoch_idx_list=-jnp.ones((self.hyper.n_granule_per_task,), dtype=jnp.int32),
            last_epoch_idx_list=-jnp.ones((self.hyper.n_granule_per_task,), dtype=jnp.int32),
        )

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, i, idx_list)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, reset_state)
        task_counter = new_state.task_counter + 1
        return new_state.replace(task_counter=task_counter)


class Neurogenesis_topKSelection_noInit_v2(Neurogenesis1):
    """Top-K granule selection (no init), v2 — verbatim port of the upstream
    paper-default Neurogenesis_topKSelection_noInit class plus its parent
    Neurogenesis1.init_state. Differs from the local Neurogenesis1 init by:
      - F_mat is masked by Bernoulli(p=0.5) before L2 normalization (so initial
        F is 50% sparse, matching upstream).
      - granule_thres = sqrt(n_mitral/6) * F_norm (upstream formula), not
        sqrt(3*n_mitral)/4 * F_norm (local Neurogenesis1).
      - RNG path: rng_F, rng_mask, rng = split(rng, 3); subkey, rng = split(rng).
    The _update_one_epoch / _update bodies are also verbatim from upstream:
      - F-row normalization uses L2 (default ord), not L1.
      - δth_g1 coefficient is sqrt(n_mitral/6), not (2/3).
      - top-K selection input is (m - 0.5) @ Fᵀ, not m @ Fᵀ.
      - task_counter is NOT incremented in _update.
      - State is plain NeurogenesisState (no inert idx-tracking fields).
    """

    def init_state(
        self, rng, hyperD: NeurogenesisHyperDynConfig, random_x, *args, **kwargs
    ) -> NeurogenesisState:
        n_mitral, n_granule = self.hyper.n_mitral, self.hyper.n_granule
        F_norm = hyperD.F_norm
        rng_F, rng_mask, rng = jax.random.split(rng, 3)
        mask = jax.random.bernoulli(rng_mask, p=0.5, shape=(n_granule, n_mitral))
        F_mat = jax.random.uniform(rng_F, (n_granule, n_mitral))
        F_mat = F_mat * mask
        subkey, rng = jax.random.split(rng)
        F_mat = F_mat / (jnp.linalg.norm(F_mat, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat = jnp.zeros((n_mitral, n_granule))
        mitral_thres = jnp.zeros((n_mitral,))
        granule_thres = jnp.ones((n_granule,)) * jnp.sqrt(n_mitral / 6) * F_norm
        return NeurogenesisState(
            B_mat=B_mat,
            F_mat=F_mat,
            mitral_thres=mitral_thres,
            granule_thres=granule_thres,
            task_counter=0,
            rng=subkey,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        idx_list: jax.Array = None,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g, lr_th_m = hyperD.learning_rate_th_g, hyperD.learning_rate_th_m
        decay_B = hyperD.decay_rate_B
        decay_th_g, decay_th_m = hyperD.decay_granule_thres, hyperD.decay_mitral_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        rng = state.rng
        n_mitral = self.hyper.n_mitral
        n_updated_granule = self.hyper.n_granule_per_task

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)

        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        rng_F, rng_B, rng = jax.random.split(rng, 3)

        F_mat_updated = F_mat[idx_list, :]
        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]

        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp = jnp.clip(F_new_tmp, 0, None)
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        B_mat_updated = B_mat[:, idx_list]
        tmp1 = ((m_bar - m1))[:, None] * g1[None, :]
        tmp2 = ((m_bar - m2))[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = decay_B * B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, None)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * jnp.sqrt(n_mitral / 6)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)

        hg_steady12 = (m_steady12 - 0.5) @ state.F_mat.T
        hg_steady_max = jnp.max(hg_steady12, axis=0)
        _, idx_list = jax.lax.top_k(hg_steady_max, self.hyper.n_granule_per_task)

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, idx_list=idx_list)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, state)
        return new_state


class Neurogenesis1_v2(Neurogenesis_topKSelection_noInit_v2):
    """Paper-default sequential-allocation Neurogenesis1.

    Inherits the upstream init_state from Neurogenesis_topKSelection_noInit_v2
    (Bernoulli(0.5) mask on F_mat + granule_thres = sqrt(n_mitral/6)*F_norm).

    Overrides _update_one_epoch and _update to match upstream Neurogenesis1:
      - Hebbian B rule:  (m_bar - m), without the m·(...) factor.
      - B update applies decay_rate_B (β_B):  B_new = β_B * B + delta_B.
      - δth_g1 coefficient: sqrt(n_mitral/6).
      - _update re-initializes per-task slot with Bernoulli mask + same th_g formula.
      (learning_B_sign_imbalance: upstream multiplies delta_B by (1 + B_ib·(δB<0));
       upstream notebook leaves B_ib=0 so the term is a no-op — omitted here.)
    """

    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        epoch_idx,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g, lr_th_m = hyperD.learning_rate_th_g, hyperD.learning_rate_th_m
        decay_B = hyperD.decay_rate_B
        decay_th_g, decay_th_m = hyperD.decay_granule_thres, hyperD.decay_mitral_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        task_counter = state.task_counter
        rng = state.rng
        n_mitral = self.hyper.n_mitral
        n_updated_granule = self.hyper.n_granule_per_task
        idx_list = jnp.arange(n_updated_granule) + task_counter * n_updated_granule
        idx_list = idx_list % self.hyper.n_granule

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)
        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        rng_F, rng_B, rng = jax.random.split(rng, 3)

        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]
        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        tmp1 = ((m_bar - m1))[:, None] * g1[None, :]
        tmp2 = ((m_bar - m2))[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = decay_B * B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, None)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * jnp.sqrt(n_mitral / 6)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        n_updated_granule, n_mitral = self.hyper.n_granule_per_task, self.hyper.n_mitral
        task_counter = state.task_counter
        rng = state.rng
        F_norm = hyperD.F_norm
        subkey, rng = jax.random.split(rng)
        idx_list = jnp.arange(n_updated_granule) + task_counter * n_updated_granule
        idx_list = idx_list % self.hyper.n_granule

        rng_F, rng_mask, rng = jax.random.split(rng, 3)
        mask = jax.random.bernoulli(rng_mask, p=0.5, shape=(n_updated_granule, n_mitral))
        F_mat_tmp = jax.random.uniform(rng_F, (n_updated_granule, n_mitral))
        F_mat_tmp = F_mat_tmp * mask
        F_mat_tmp = F_mat_tmp / (jnp.linalg.norm(F_mat_tmp, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat_tmp = jnp.zeros((n_mitral, n_updated_granule))
        th_g_tmp = jnp.ones(n_updated_granule) * jnp.sqrt(n_mitral / 6) * F_norm
        F_mat = state.F_mat.at[idx_list, :].set(F_mat_tmp)
        B_mat = state.B_mat.at[:, idx_list].set(B_mat_tmp)
        th_g = state.granule_thres.at[idx_list].set(th_g_tmp)
        state = state.replace(F_mat=F_mat, B_mat=B_mat, granule_thres=th_g, rng=rng)

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, i)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, state)
        task_counter = new_state.task_counter + 1
        return new_state.replace(task_counter=task_counter)


class Neurogenesis_randomKSelection_v2(Neurogenesis1_v2):
    """Paper-default random-K-selection variant.

    Inherits init_state from v2 (Bernoulli mask, sqrt(n_mitral/6) th_g).
    Overrides _update_one_epoch and _update to match upstream
    Neurogenesis_randomKSelection:
      - Hebbian B rule: (m_bar - m).
      - B update applies decay_rate_B and clips to [0, 1] (note: 1, not None).
      - δth_g1 coefficient: sqrt(n_mitral/6).
      - _update picks idx_list via random.choice (no replacement) and
        re-initializes that slot with Bernoulli mask + matching th_g formula.
    """

    @partial(jax.jit, static_argnames=("self",))
    def _update_one_epoch(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        idx_list: jax.Array,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        lr_B, lr_F = hyperD.learning_rate_B, hyperD.learning_rate_F
        lr_th_g, lr_th_m = hyperD.learning_rate_th_g, hyperD.learning_rate_th_m
        decay_B = hyperD.decay_rate_B
        decay_th_g, decay_th_m = hyperD.decay_granule_thres, hyperD.decay_mitral_thres
        F_norm = hyperD.F_norm

        F_mat, B_mat, th_m, th_g = state.F_mat, state.B_mat, state.mitral_thres, state.granule_thres
        rng = state.rng
        n_mitral = self.hyper.n_mitral
        n_updated_granule = self.hyper.n_granule_per_task

        x12 = jnp.stack([x1, x2], axis=0)
        m_steady12, g_steady12 = self._forward(hyperD, state, x12)
        m1, m2 = m_steady12[0], m_steady12[1]
        g1, g2 = g_steady12[0], g_steady12[1]
        m_bar = jnp.mean(m_steady12, axis=0)
        g_bar = jnp.mean(g_steady12, axis=0)
        g1, g2, g_bar = g1[idx_list], g2[idx_list], g_bar[idx_list]

        rng_F, rng_B, rng = jax.random.split(rng, 3)

        tmp = g1[:, None] * m1[None, :] + g2[:, None] * m2[None, :]
        delta_F = lr_F * tmp
        F_new_tmp = F_mat[idx_list, :] + delta_F
        F_new_tmp_len = jnp.linalg.norm(F_new_tmp, axis=1, keepdims=True) + 1e-8
        F_new_tmp = F_new_tmp / F_new_tmp_len * F_norm
        F_mat_new = F_mat.at[idx_list, :].set(F_new_tmp)

        tmp1 = ((m_bar - m1))[:, None] * g1[None, :]
        tmp2 = ((m_bar - m2))[:, None] * g2[None, :]
        delta_B = lr_B * (tmp1 + tmp2)
        B_new_tmp = decay_B * B_mat[:, idx_list] + delta_B
        B_new_tmp = jnp.clip(B_new_tmp, 0, 1)
        B_mat_new = B_mat.at[:, idx_list].set(B_new_tmp)

        delta_th_g1 = lr_th_g * (g_bar - 0.5) * F_norm * jnp.sqrt(n_mitral / 6)
        h1, h2 = F_mat_new[idx_list, :] @ m1, F_mat_new[idx_list, :] @ m2
        h_high, h_low = jnp.maximum(h1, h2), jnp.minimum(h1, h2)
        hi_ratio = hyperD.th_g_hi_ratio
        delta_th_g2 = (1 - decay_th_g) * (hi_ratio * h_high + (1 - hi_ratio) * h_low) - (1 - decay_th_g) * th_g[idx_list]
        new_th_g = th_g.at[idx_list].add(delta_th_g1 + delta_th_g2)

        return state.replace(
            B_mat=B_mat_new,
            F_mat=F_mat_new,
            mitral_thres=th_m,
            granule_thres=new_th_g,
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        hyperD: NeurogenesisHyperDynConfig,
        state: NeurogenesisState,
        x1: jax.Array,
        x2: jax.Array,
        random_x,
        *args,
        **kwargs,
    ) -> NeurogenesisState:
        rng = state.rng
        n_updated_granule, n_mitral = self.hyper.n_granule_per_task, self.hyper.n_mitral
        F_norm = hyperD.F_norm
        subkey, rng = jax.random.split(rng)
        idx_list = jax.random.choice(
            subkey, jnp.arange(self.hyper.n_granule), (n_updated_granule,), replace=False
        )

        rng_F, rng_mask, rng = jax.random.split(rng, 3)
        mask = jax.random.bernoulli(rng_mask, p=0.5, shape=(n_updated_granule, n_mitral))
        F_mat_tmp = jax.random.uniform(rng_F, (n_updated_granule, n_mitral))
        F_mat_tmp = F_mat_tmp * mask
        F_mat_tmp = F_mat_tmp / (jnp.linalg.norm(F_mat_tmp, axis=1, keepdims=True) + 1e-8) * F_norm
        B_mat_tmp = jnp.zeros((n_mitral, n_updated_granule))
        th_g_tmp = jnp.ones((n_updated_granule,)) * jnp.sqrt(n_mitral / 6) * F_norm
        F_mat = state.F_mat.at[idx_list, :].set(F_mat_tmp)
        B_mat = state.B_mat.at[:, idx_list].set(B_mat_tmp)
        th_g = state.granule_thres.at[idx_list].set(th_g_tmp)
        state = state.replace(F_mat=F_mat, B_mat=B_mat, granule_thres=th_g, rng=rng)

        def body_fun(i, val):
            return self._update_one_epoch(hyperD, val, x1, x2, idx_list)

        new_state = jax.lax.fori_loop(0, self.hyper.n_epochs_per_pair, body_fun, state)
        return new_state
