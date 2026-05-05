import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

from .base import Network, Xs_Generator


class Driver1:
    def __init__(
            self,
            network: Network[Any, Any, Any],
            xs_generator: Xs_Generator,
            rng: jax.Array,
            hyperD: Any = None,
            initial_state: Any = None,
            n_test_pairs: int = 100,
            n_rand_eval_pairs: int = 50,
            ):
        self.network = network
        self.xs_generator = xs_generator
        self.rng = rng
        self.hyperD = hyperD
        self.state = initial_state
        self.n_test_pairs = n_test_pairs
        self.n_rand_eval_pairs = n_rand_eval_pairs

        # Fixed random patterns for stability tracking (set after pretraining)
        self.rand_eval_xs = None   # (n_rand_eval_pairs, n_mitral)
        self.rand_eval_m0 = None   # initial representations

        self.recorder = self._m1m2_all_recorder
        # After self.run(): each record entry has shape (n_train - n_test + 1, n_test, ...);
        # index 0 = lag=-1 (pre-train snapshot), index i+1 = lag=i. rand_cos_sim[0]=1.0 and
        # rand_l2_change[0]=0.0 are mathematical identities (plot from index 1 for those metrics).
        self.record = {}

    def re_init(self, rng: jax.Array, hyperD: Any = None, initial_state: Any = None):
        """Reset the driver without triggering recompilation."""
        self.state = initial_state
        self.rng = rng
        self.hyperD = hyperD
        self.record = {}
        self.rand_eval_xs = None
        self.rand_eval_m0 = None

    def _zero_recorder(self, hyperD: Any, state: Any, test_xs1: jax.Array, test_xs2: jax.Array, *args, **kwargs):
        """Record quantities of interest after each task.
        test_xs1: (n_test_pairs, n_mitral)
        test_xs2: (n_test_pairs, n_mitral)
        """
        return {"zeros": jnp.zeros((self.n_test_pairs,))}

    def _m1m2_dist_recorder(self, hyperD: Any, state: Any, test_xs1: jax.Array, test_xs2: jax.Array, *args, **kwargs):
        test_xs = jnp.concatenate([test_xs1, test_xs2], axis=0)
        forward_out = self.network.forward(hyperD, state, test_xs)
        ms = forward_out[0]
        m1s = ms[:self.n_test_pairs, :]
        m2s = ms[self.n_test_pairs:, :]
        dists = jnp.linalg.norm(m1s - m2s, axis=1)
        return {"m1m2_dists": dists}

    def _m1m2_cos_angle_recorder(self, hyperD: Any, state: Any, test_xs1: jax.Array, test_xs2: jax.Array, *args, **kwargs):
        test_xs = jnp.concatenate([test_xs1, test_xs2], axis=0)
        forward_out = self.network.forward(hyperD, state, test_xs)
        ms = forward_out[0]
        m1s = ms[:self.n_test_pairs, :]
        m2s = ms[self.n_test_pairs:, :]
        cos_angles = jnp.sum(m1s * m2s, axis=1) / (
            jnp.linalg.norm(m1s, axis=1) * jnp.linalg.norm(m2s, axis=1) + 1e-8
        )
        return {"m1m2_cos_angles": cos_angles}

    def _m1m2_all_recorder(self, hyperD: Any, state: Any, test_xs1: jax.Array, test_xs2: jax.Array, *args, **kwargs):
        """Record all metrics: cosine angle, L2 distance, Pearson correlation, and random pattern stability."""
        test_xs = jnp.concatenate([test_xs1, test_xs2], axis=0)
        forward_out = self.network.forward(hyperD, state, test_xs)
        ms = forward_out[0]
        m1s = ms[:self.n_test_pairs, :]
        m2s = ms[self.n_test_pairs:, :]

        # Cosine angle between m1 and m2
        cos_angles = jnp.sum(m1s * m2s, axis=1) / (
            jnp.linalg.norm(m1s, axis=1) * jnp.linalg.norm(m2s, axis=1) + 1e-8
        )

        # L2 distance between m1 and m2
        l2_dists = jnp.linalg.norm(m1s - m2s, axis=1)

        # Pearson correlation between m1 and m2
        m1_c = m1s - m1s.mean(axis=1, keepdims=True)
        m2_c = m2s - m2s.mean(axis=1, keepdims=True)
        m1m2_corr = jnp.sum(m1_c * m2_c, axis=1) / (
            jnp.linalg.norm(m1_c, axis=1) * jnp.linalg.norm(m2_c, axis=1) + 1e-8
        )

        result = {
            "m1m2_cos_angles": cos_angles,
            "m1m2_l2_dists": l2_dists,
            "m1m2_corr": m1m2_corr,
        }

        # Random pattern stability: cosine similarity with initial representation
        if self.rand_eval_xs is not None and self.rand_eval_m0 is not None:
            rand_out = self.network.forward(hyperD, state, self.rand_eval_xs)
            rand_ms = rand_out[0]
            rand_cos = jnp.sum(rand_ms * self.rand_eval_m0, axis=1) / (
                jnp.linalg.norm(rand_ms, axis=1) * jnp.linalg.norm(self.rand_eval_m0, axis=1) + 1e-8
            )
            rand_l2_change = jnp.linalg.norm(rand_ms - self.rand_eval_m0, axis=1)
            result["rand_cos_sim"] = rand_cos
            result["rand_l2_change"] = rand_l2_change

        return result

    def run(self, n_pretrain_pairs: int, n_train_pairs: int, n_random_pairs: int):
        """Run the driver for a number of tasks; n_train_pairs should be larger than n_test_pairs.

        After lag-align + trim, each metric array has shape (n_train - n_test + 1, n_test, ...)
        where index 0 = lag=-1 (pre-training snapshot) and index i+1 = lag=i (post-training pair i).
        """
        self.record = {}

        if self.state is None:
            self.rng, init_rng, random_x_rng = jax.random.split(self.rng, 3)
            random_xs1, _ = self.xs_generator.generate(random_x_rng, n_random_pairs)
            self.state = self.network.init_state(init_rng, self.hyperD, random_xs1)

        self.rng, pretrain_xs_rng = jax.random.split(self.rng)
        pretrain_xs1, pretrain_xs2 = self.xs_generator.generate(pretrain_xs_rng, n_pretrain_pairs)
        # Dummy split to keep RNG chain aligned with run_topk_no_theta.py's notebook driver
        # (which does an extra `permute_rng = split(self.rng)` here, even though the permutation
        # result is discarded). Without this, test_xs / train_xs sub-keys land 1 split off and
        # test pair sequences diverge — for topk_noinit this changes lag=0 angle by ~50°.
        self.rng, _permute_rng = jax.random.split(self.rng)
        for i in range(n_pretrain_pairs):
            self.rng, random_x_rng = jax.random.split(self.rng)
            # Pass None when n_random_pairs == 0 to match notebook driver's jit trace shape.
            # Mixing None vs concrete-empty array changes XLA compile cache → numerical drift.
            if n_random_pairs > 0:
                random_xs1, _ = self.xs_generator.generate(random_x_rng, n_random_pairs)
            else:
                random_xs1 = None
            x1, x2 = pretrain_xs1[i], pretrain_xs2[i]
            self.state = self.network.update(self.hyperD, self.state, x1, x2, random_xs1)
            if (i + 1) % 100 == 0 or (i + 1) == n_pretrain_pairs:
                print(f"  pretrain {i+1}/{n_pretrain_pairs}", flush=True)

        # After pretraining: fix random evaluation patterns and record initial representations
        # NOTE: this RNG split + forward call must come AFTER test_xs/train_xs sampling so
        # that test_xs and train_xs derive from the same RNG path as the notebook driver.
        # Otherwise the rand_eval split shifts the RNG chain and test patterns differ.
        rand_eval_pending = self.n_rand_eval_pairs > 0 and self.rand_eval_xs is None

        self.rng, test_xs_rng = jax.random.split(self.rng)
        test_xs1, test_xs2 = self.xs_generator.generate(test_xs_rng, self.n_test_pairs)

        self.rng, train_xs_rng = jax.random.split(self.rng)
        train_xs1, train_xs2 = self.xs_generator.generate(train_xs_rng - self.n_test_pairs, n_train_pairs)

        # rand_eval_xs and rand_eval_m0 must be set before recorder is called (recorder reads them).
        # Done AFTER test/train xs sampling so the RNG split for rand_eval doesn't shift the chain.
        if rand_eval_pending:
            self.rng, rand_eval_rng = jax.random.split(self.rng)
            self.rand_eval_xs, _ = self.xs_generator.generate(rand_eval_rng, self.n_rand_eval_pairs)
            rand_out = self.network.forward(self.hyperD, self.state, self.rand_eval_xs)
            self.rand_eval_m0 = jnp.array(rand_out[0])

        # Pre-loop snapshot (= lag=-1 for test pair 0, lag=-2 for test pair 1, etc.).
        # Allocated size n_train+1: index 0 = pre-train, index i+1 = post-train-pair-i.
        record_dict0 = self.recorder(self.hyperD, self.state, test_xs1, test_xs2)
        for key in record_dict0:
            self.record[key] = np.zeros((n_train_pairs + 1,) + record_dict0[key].shape)
            self.record[key][0] = np.array(record_dict0[key])

        for i in range(n_train_pairs):
            self.rng, random_x_rng = jax.random.split(self.rng)
            if n_random_pairs > 0:
                random_xs1, _ = self.xs_generator.generate(random_x_rng, n_random_pairs)
            else:
                random_xs1 = None
            if i < self.n_test_pairs:
                x1, x2 = test_xs1[i], test_xs2[i]
            else:
                x1, x2 = train_xs1[i - self.n_test_pairs], train_xs2[i - self.n_test_pairs]
            self.state = self.network.update(self.hyperD, self.state, x1, x2, random_xs1)
            if (i + 1) % 100 == 0 or (i + 1) == n_train_pairs:
                print(f"  train {i+1}/{n_train_pairs}", flush=True)
            record_dict = self.recorder(self.hyperD, self.state, test_xs1, test_xs2)
            for key in record_dict:
                self.record[key][i + 1] = np.array(record_dict[key])

        # Lag-align: for test pair k, snapshot[k] = lag=-1 and snapshot[k+1] = lag=0.
        # Rolling by -k puts snapshot[k] (= lag=-1) at index 0, snapshot[k+1] (= lag=0) at index 1.
        # Trim to [:n_train - n_test + 1] so every test pair has lag=-1..lag=(n_train-n_test-1).
        for key in self.record:
            if self.record[key].shape[1] == self.n_test_pairs:
                for i in range(self.n_test_pairs):
                    self.record[key][:, i] = np.roll(self.record[key][:, i], -i, axis=0)
            self.record[key] = self.record[key][:n_train_pairs - self.n_test_pairs + 1]
