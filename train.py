import functools
import os
import pickle
import time
from typing import Mapping, Any

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

import model
import dataset


flags.DEFINE_string('dataset_path', None, 'Single-file ASCII dataset location', short_name='d')

flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('sequence_length', 64, 'seq len')

flags.DEFINE_integer('d_model', 128, 'model width')
flags.DEFINE_integer('num_heads', 4, 'attn heads')
flags.DEFINE_integer('num_layers', 4, 'num layers')
flags.DEFINE_float('dropout_rate', 0.1, 'dropout rate')

flags.DEFINE_float('learning_rate', 3e-4, 'max lr')
flags.DEFINE_float('grad_clip_value', 1, 'max gradient')
# flags.DEFINE_string()

FLAGS = flags.FLAGS
LOG_EVERY = 100
MAX_STEPS = 10**6


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass """

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        # embedding
        tokens = data['obs']
        input_mask = jnp.greater(tokens, 0)
        seq_length = tokens.shape[1]

        # Embed the input tokens and positions
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
        token_embs = token_embedding_map(tokens)
        positional_embeddings = hk.get_parameter(
            'pos_embs', [seq_length, d_model], init=embed_init)
        input_embeddings = token_embs + positional_embeddings

        transformer = model.Transformer(
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, input_mask, is_training)

        return hk.Linear(vocab_size)(output_embeddings)

    return forward_fn


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss


class GradientUpdater:

    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state, = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }
        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


def main(_):
    # FLAGS.alsologtosterr = True
    # create dataset
    train_dataset = dataset.AsciiDataset(
        FLAGS.dataset_path, FLAGS.batch_size, FLAGS.sequence_length
    )
    vocab_size = train_dataset.vocab_size

    # setup model, loss, updater
    forward_fn = build_forward_fn(vocab_size, FLAGS.d_model, FLAGS.num_heads,
                                  FLAGS.num_layers, FLAGS.dropout_rate)
    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip_value),
        optax.adam(FLAGS.learning_rate, b1=0.9, b2=0.99)
    ) 
    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)
    # checkpoint updater

    # initialize parameters
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    # param count
    num_params = hk.data_structures.tree_size(state)
    byte_size = hk.data_structures.tree_bytes(state)
    print(f'{num_params / 1e6:.2f}M params, size: {byte_size / 1e6:.2f}MB')

    logging.info('Starting training loop')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)

        if step % LOG_EVERY == 0:
            steps_per_sec = LOG_EVERY / (time.time() - prev_time)
            prev_time = time.time()
            metrics.update({'steps_per_sec': steps_per_sec})
            logging.info({k: float(v) for k, v in metrics.items()})


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset_path')
    app.run(main)

