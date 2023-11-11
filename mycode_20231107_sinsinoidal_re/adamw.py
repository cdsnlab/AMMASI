# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Legacy v1 optimizer classes.

For more examples see the base class `tf.compat.v1.keras.optimizers.Optimizer`.
"""

import tensorflow.compat.v2 as tf

from keras import backend


class Optimizer:
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {"clipnorm", "clipvalue"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError(
                    "Unexpected keyword argument passed to optimizer: " + str(k)
                )
            # checks that clipnorm >= 0 and clipvalue >= 0
            if kwargs[k] < 0:
                raise ValueError(f"Expected {k} >= 0, received: {kwargs[k]}")
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    # Set this to False, indicating `apply_gradients` does not take the
    # `experimental_aggregate_gradients` argument.
    _HAS_AGGREGATE_GRAD = False

    def _create_all_weights(self, params):
        """Creates and sets all optimizer weights.

        Args:
          params: list or tuple of `Variable` objects that will be minimized
            using this optimizer.

        Returns:
          Specific weight values that are used in `get_updates`
        """
        raise NotImplementedError

    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        """Returns gradients of `loss` with respect to `params`.

        Args:
            loss: Loss tensor.
            params: List of variables.

        Returns:
            List of gradient tensors.

        Raises:
            ValueError: In case any gradient cannot be computed (e.g. if
              gradient function not implemented).
        """
        grads = backend.gradients(loss, params)
        if any(g is None for g in grads):
            raise ValueError(
                "An operation has `None` for gradient. "
                "Please make sure that all of your ops have a "
                "gradient defined (i.e. are differentiable). "
                "Common ops without gradient: "
                "backend.argmax, backend.round, backend.eval."
            )
        if hasattr(self, "clipnorm"):
            grads = [tf.clip_by_norm(g, self.clipnorm) for g in grads]
        if hasattr(self, "clipvalue"):
            grads = [
                tf.clip_by_value(g, -self.clipvalue, self.clipvalue)
                for g in grads
            ]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        Args:
            weights: a list of Numpy arrays. The number of arrays and their
              shape must match number of the dimensions of the weights of the
              optimizer (i.e. it should match the output of `get_weights`).

        Raises:
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError(
                "Length of the specified weight list ("
                + str(len(weights))
                + ") does not match the number of weights of the optimizer ("
                + str(len(params))
                + ")"
            )
        weight_value_tuples = []
        param_values = backend.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError(
                    "Optimizer weight shape "
                    + str(pv.shape)
                    + " not compatible with provided weight shape "
                    + str(w.shape)
                )
            weight_value_tuples.append((p, w))
        backend.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.

        Returns:
            A list of numpy arrays.
        """
        return backend.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, "clipnorm"):
            config["clipnorm"] = self.clipnorm
        if hasattr(self, "clipvalue"):
            config["clipvalue"] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class SGD(Optimizer):
#     """Stochastic gradient descent optimizer.

#     Includes support for momentum,
#     learning rate decay, and Nesterov momentum.

#     Args:
#         lr: float >= 0. Learning rate.
#         momentum: float >= 0. Parameter that accelerates SGD in the relevant
#           direction and dampens oscillations.
#         decay: float >= 0. Learning rate decay over each update.
#         nesterov: boolean. Whether to apply Nesterov momentum.
#     """

#     def __init__(
#         self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False, **kwargs
#     ):
#         super().__init__(**kwargs)
#         with backend.name_scope(self.__class__.__name__):
#             self.iterations = backend.variable(
#                 0, dtype="int64", name="iterations"
#             )
#             self.lr = backend.variable(lr, name="lr")
#             self.momentum = backend.variable(momentum, name="momentum")
#             self.decay = backend.variable(decay, name="decay")
#         self.initial_decay = decay
#         self.nesterov = nesterov

#     def _create_all_weights(self, params):
#         shapes = [backend.int_shape(p) for p in params]
#         moments = [backend.zeros(shape) for shape in shapes]
#         self.weights = [self.iterations] + moments
#         return moments

#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [tf.compat.v1.assign_add(self.iterations, 1)]

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (
#                 1.0
#                 / (
#                     1.0
#                     + self.decay
#                     * tf.cast(self.iterations, backend.dtype(self.decay))
#                 )
#             )
#         # momentum
#         moments = self._create_all_weights(params)
#         for p, g, m in zip(params, grads, moments):
#             v = self.momentum * m - lr * g  # velocity
#             self.updates.append(tf.compat.v1.assign(m, v))

#             if self.nesterov:
#                 new_p = p + self.momentum * v - lr * g
#             else:
#                 new_p = p + v

#             # Apply constraints.
#             if getattr(p, "constraint", None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(tf.compat.v1.assign(p, new_p))
#         return self.updates

#     def get_config(self):
#         config = {
#             "lr": float(backend.get_value(self.lr)),
#             "momentum": float(backend.get_value(self.momentum)),
#             "decay": float(backend.get_value(self.decay)),
#             "nesterov": self.nesterov,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class RMSprop(Optimizer):
#     """RMSProp optimizer.

#     It is recommended to leave the parameters of this optimizer
#     at their default values
#     (except the learning rate, which can be freely tuned).

#     Args:
#       lr: float >= 0. Learning rate.
#       rho: float >= 0.
#       epsilon: float >= 0. Fuzz factor.
#         If `None`, defaults to `backend.epsilon()`.
#       decay: float >= 0. Learning rate decay over each update.
#     """

#     def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.0, **kwargs):
#         super().__init__(**kwargs)
#         with backend.name_scope(self.__class__.__name__):
#             self.lr = backend.variable(lr, name="lr")
#             self.rho = backend.variable(rho, name="rho")
#             self.decay = backend.variable(decay, name="decay")
#             self.iterations = backend.variable(
#                 0, dtype="int64", name="iterations"
#             )
#         if epsilon is None:
#             epsilon = backend.epsilon()
#         self.epsilon = epsilon
#         self.initial_decay = decay

#     def _create_all_weights(self, params):
#         accumulators = [
#             backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
#             for p in params
#         ]
#         self.weights = accumulators
#         return accumulators

#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         accumulators = self._create_all_weights(params)
#         self.updates = [tf.compat.v1.assign_add(self.iterations, 1)]

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (
#                 1.0
#                 / (
#                     1.0
#                     + self.decay
#                     * tf.cast(self.iterations, backend.dtype(self.decay))
#                 )
#             )

#         for p, g, a in zip(params, grads, accumulators):
#             # update accumulator
#             new_a = self.rho * a + (1.0 - self.rho) * tf.square(g)
#             self.updates.append(tf.compat.v1.assign(a, new_a))
#             new_p = p - lr * g / (backend.sqrt(new_a) + self.epsilon)

#             # Apply constraints.
#             if getattr(p, "constraint", None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(tf.compat.v1.assign(p, new_p))
#         return self.updates

#     def get_config(self):
#         config = {
#             "lr": float(backend.get_value(self.lr)),
#             "rho": float(backend.get_value(self.rho)),
#             "decay": float(backend.get_value(self.decay)),
#             "epsilon": self.epsilon,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class Adagrad(Optimizer):
#     """Adagrad optimizer.

#     Adagrad is an optimizer with parameter-specific learning rates,
#     which are adapted relative to how frequently a parameter gets
#     updated during training. The more updates a parameter receives,
#     the smaller the updates.

#     It is recommended to leave the parameters of this optimizer
#     at their default values.

#     # Arguments
#         lr: float >= 0. Initial learning rate.
#         epsilon: float >= 0. If `None`, defaults to `backend.epsilon()`.
#         decay: float >= 0. Learning rate decay over each update.

#     # References
#         - [Adaptive Subgradient Methods for Online Learning and Stochastic
#         Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
#     """

#     def __init__(self, lr=0.01, epsilon=None, decay=0.0, **kwargs):
#         super().__init__(**kwargs)
#         with backend.name_scope(self.__class__.__name__):
#             self.lr = backend.variable(lr, name="lr")
#             self.decay = backend.variable(decay, name="decay")
#             self.iterations = backend.variable(
#                 0, dtype="int64", name="iterations"
#             )
#         if epsilon is None:
#             epsilon = backend.epsilon()
#         self.epsilon = epsilon
#         self.initial_decay = decay

#     def _create_all_weights(self, params):
#         shapes = [backend.int_shape(p) for p in params]
#         accumulators = [backend.zeros(shape) for shape in shapes]
#         self.weights = accumulators
#         return accumulators

#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         accumulators = self._create_all_weights(params)

#         self.updates = [tf.compat.v1.assign_add(self.iterations, 1)]

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (
#                 1.0
#                 / (
#                     1.0
#                     + self.decay
#                     * tf.cast(self.iterations, backend.dtype(self.decay))
#                 )
#             )

#         for p, g, a in zip(params, grads, accumulators):
#             new_a = a + tf.square(g)  # update accumulator
#             self.updates.append(tf.compat.v1.assign(a, new_a))
#             new_p = p - lr * g / (backend.sqrt(new_a) + self.epsilon)

#             # Apply constraints.
#             if getattr(p, "constraint", None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(tf.compat.v1.assign(p, new_p))
#         return self.updates

#     def get_config(self):
#         config = {
#             "lr": float(backend.get_value(self.lr)),
#             "decay": float(backend.get_value(self.decay)),
#             "epsilon": self.epsilon,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class Adadelta(Optimizer):
#     """Adadelta optimizer.

#     Adadelta is a more robust extension of Adagrad
#     that adapts learning rates based on a moving window of gradient updates,
#     instead of accumulating all past gradients. This way, Adadelta continues
#     learning even when many updates have been done. Compared to Adagrad, in the
#     original version of Adadelta you don't have to set an initial learning
#     rate. In this version, initial learning rate and decay factor can
#     be set, as in most other Keras optimizers.

#     It is recommended to leave the parameters of this optimizer
#     at their default values.

#     Arguments:
#       lr: float >= 0. Initial learning rate, defaults to 1.
#           It is recommended to leave it at the default value.
#       rho: float >= 0. Adadelta decay factor, corresponding to fraction of
#           gradient to keep at each time step.
#       epsilon: float >= 0. Fuzz factor.
#         If `None`, defaults to `backend.epsilon()`.
#       decay: float >= 0. Initial learning rate decay.

#     References:
#         - [Adadelta - an adaptive learning rate
#         method](http://arxiv.org/abs/1212.5701)
#     """

#     def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0.0, **kwargs):
#         super().__init__(**kwargs)
#         with backend.name_scope(self.__class__.__name__):
#             self.lr = backend.variable(lr, name="lr")
#             self.decay = backend.variable(decay, name="decay")
#             self.iterations = backend.variable(
#                 0, dtype="int64", name="iterations"
#             )
#         if epsilon is None:
#             epsilon = backend.epsilon()
#         self.rho = rho
#         self.epsilon = epsilon
#         self.initial_decay = decay

#     def _create_all_weights(self, params):
#         shapes = [backend.int_shape(p) for p in params]
#         accumulators = [backend.zeros(shape) for shape in shapes]
#         delta_accumulators = [backend.zeros(shape) for shape in shapes]
#         self.weights = accumulators + delta_accumulators
#         return accumulators, delta_accumulators

#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [tf.compat.v1.assign_add(self.iterations, 1)]
#         accumulators, delta_accumulators = self._create_all_weights(params)

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (
#                 1.0
#                 / (
#                     1.0
#                     + self.decay
#                     * tf.cast(self.iterations, backend.dtype(self.decay))
#                 )
#             )

#         for p, g, a, d_a in zip(
#             params, grads, accumulators, delta_accumulators
#         ):
#             # update accumulator
#             new_a = self.rho * a + (1.0 - self.rho) * tf.square(g)
#             self.updates.append(tf.compat.v1.assign(a, new_a))

#             # use the new accumulator and the *old* delta_accumulator
#             update = (
#                 g
#                 * backend.sqrt(d_a + self.epsilon)
#                 / backend.sqrt(new_a + self.epsilon)
#             )
#             new_p = p - lr * update

#             # Apply constraints.
#             if getattr(p, "constraint", None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(tf.compat.v1.assign(p, new_p))

#             # update delta_accumulator
#             new_d_a = self.rho * d_a + (1 - self.rho) * tf.square(update)
#             self.updates.append(tf.compat.v1.assign(d_a, new_d_a))
#         return self.updates

#     def get_config(self):
#         config = {
#             "lr": float(backend.get_value(self.lr)),
#             "rho": self.rho,
#             "decay": float(backend.get_value(self.decay)),
#             "epsilon": self.epsilon,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class Adam(Optimizer):
#     """Adam optimizer.

#     Default parameters follow those provided in the original paper.

#     Args:
#       lr: float >= 0. Learning rate.
#       beta_1: float, 0 < beta < 1. Generally close to 1.
#       beta_2: float, 0 < beta < 1. Generally close to 1.
#       epsilon: float >= 0. Fuzz factor.
#         If `None`, defaults to `backend.epsilon()`.
#       decay: float >= 0. Learning rate decay over each update.
#       amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
#         from the paper "On the Convergence of Adam and Beyond".
#     """

#     def __init__(
#         self,
#         lr=0.001,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=None,
#         decay=0.0,
#         amsgrad=False,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         with backend.name_scope(self.__class__.__name__):
#             self.iterations = backend.variable(
#                 0, dtype="int64", name="iterations"
#             )
#             self.lr = backend.variable(lr, name="lr")
#             self.beta_1 = backend.variable(beta_1, name="beta_1")
#             self.beta_2 = backend.variable(beta_2, name="beta_2")
#             self.decay = backend.variable(decay, name="decay")
#         if epsilon is None:
#             epsilon = backend.epsilon()
#         self.epsilon = epsilon
#         self.initial_decay = decay
#         self.amsgrad = amsgrad

#     def _create_all_weights(self, params):
#         ms = [
#             backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
#             for p in params
#         ]
#         vs = [
#             backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
#             for p in params
#         ]
#         if self.amsgrad:
#             vhats = [
#                 backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
#                 for p in params
#             ]
#         else:
#             vhats = [backend.zeros(1) for _ in params]
#         self.weights = [self.iterations] + ms + vs + vhats
#         return ms, vs, vhats

#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = []

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (
#                 1.0
#                 / (
#                     1.0
#                     + self.decay
#                     * tf.cast(self.iterations, backend.dtype(self.decay))
#                 )
#             )

#         with tf.control_dependencies(
#             [tf.compat.v1.assign_add(self.iterations, 1)]
#         ):
#             t = tf.cast(self.iterations, backend.floatx())
#         lr_t = lr * (
#             backend.sqrt(1.0 - tf.pow(self.beta_2, t))
#             / (1.0 - tf.pow(self.beta_1, t))
#         )

#         ms, vs, vhats = self._create_all_weights(params)
#         for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
#             m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
#             v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * tf.square(g)
#             if self.amsgrad:
#                 vhat_t = tf.maximum(vhat, v_t)
#                 p_t = p - lr_t * m_t / (backend.sqrt(vhat_t) + self.epsilon)
#                 self.updates.append(tf.compat.v1.assign(vhat, vhat_t))
#             else:
#                 p_t = p - lr_t * m_t / (backend.sqrt(v_t) + self.epsilon)

#             self.updates.append(tf.compat.v1.assign(m, m_t))
#             self.updates.append(tf.compat.v1.assign(v, v_t))
#             new_p = p_t

#             # Apply constraints.
#             if getattr(p, "constraint", None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(tf.compat.v1.assign(p, new_p))
#         return self.updates

#     def get_config(self):
#         config = {
#             "lr": float(backend.get_value(self.lr)),
#             "beta_1": float(backend.get_value(self.beta_1)),
#             "beta_2": float(backend.get_value(self.beta_2)),
#             "decay": float(backend.get_value(self.decay)),
#             "epsilon": self.epsilon,
#             "amsgrad": self.amsgrad,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))




# @register_keras_serializable()
# @keras_export(
#     "keras.optimizers.AdamW",
#     "keras.optimizers.experimental.AdamW",
#     "keras.dtensor.experimental.optimizers.AdamW",
#     v1=[],
# )
class AdamW(Optimizer):
    r"""Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to 0.9.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to 0.999.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        {{base_optimizer_keyword_args}}

    Reference:
      - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdamW",
        **kwargs
    ):
        # super().__init__(
        #     name=name,
        #     clipnorm=clipnorm,
        #     clipvalue=clipvalue,
        #     global_clipnorm=global_clipnorm,
        #     use_ema=use_ema,
        #     ema_momentum=ema_momentum,
        #     ema_overwrite_frequency=ema_overwrite_frequency,
        #     jit_compile=jit_compile,
        #     **kwargs
        # )
        super().__init__(**kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        AdamW optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build AdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
