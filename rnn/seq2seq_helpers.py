# Copyright 2015 Google Inc. All Rights Reserved.
# Modifications copyright 2017 Jan Buys.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.util import nest

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf

import data_utils

linear = tf.nn.rnn_cell._linear  # pylint: disable=protected-access

#TODO rename: remove _ (not local method)
def _extract_embed(embedding, update_embedding=True):
  """Get a loop_function that embeds symbols.

  Args:
    embedding: list of embedding tensors for symbols.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def embed_function(symbol): 
    emb = tf.nn.embedding_lookup(embedding, symbol) 
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    if not update_embedding:
      emb = tf.stop_gradient(emb)
    return emb
  return embed_function


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def tile_embedding_attention(emb_inp, symbol_inp, initial_state, 
                             attention_states, beam_size, embedding_size):
  """Make beam_size copies of the attention states."""
  tile_emb_inp = []
  for inp in emb_inp:
    tile_emb = tf.tile(tf.reshape(inp, [1, -1]),
                       tf.pack([beam_size, 1]))
    tile_emb = tf.reshape(tile_emb, [-1, embedding_size])
    tile_emb_inp.append(tile_emb)

  tile_symbol_inp = []
  for inp in symbol_inp:
    tile_sym = tf.tile(tf.reshape(inp, [1, 1]),
                       tf.pack([beam_size, 1]))
    tile_emb = tf.reshape(tile_emb, [-1])
    tile_symbol_inp.append(tile_emb)

  tile_initial_state = tf.tile(tf.reshape(initial_state,
      [1, -1]), tf.pack([beam_size, 1]))

  attn_length = attention_states.get_shape()[1].value
  attn_size = attention_states.get_shape()[2].value
  tile_attention_states = tf.tile(attention_states,
      tf.pack([beam_size, 1, 1]))
  tile_attention_states = tf.reshape(tile_attention_states,
      [-1, attn_length, attn_size])

  return tile_emb_inp, tile_symbol_inp, tile_initial_state, tile_attention_states


def attention(query, num_heads, y_w, v, hidden, hidden_features, attention_vec_size,
              attn_length, use_global_attention=False):
  """Puts attention masks on hidden using hidden_features and query.
  
  Args:
    query: vector, usually the current decoder state. 
           2D Tensor [batch_size x state_size].
    num_heads: int. Currently always 1.
    v: attention model parameters.
    hidden: attention_states.
    hidden_features: same linear layer applied to all attention_states.
    attention_vec_size: attention embedding size.
    attn_length: number of inputs over which the attention spans.
    use_impatient_reader: make attention function dependent on previous
                          attention vector.
    prev_ds: previous weighted averaged attention vector.

  Returns:  
    atts: softmax over attention inputs.
    ds: attention-weighted averaged attention vector.
  """
  at_logits = []  # result of attention logits
  at_probs = []  # result of attention probabilities
  ds = []  # results of attention reads will be stored here.
  if nest.is_sequence(query):  # if the query is a tuple, flatten it.
    query_list = nest.flatten(query)
    for q in query_list:  # check that ndims == 2 if specified.
      ndims = q.get_shape().ndims
      if ndims:
        assert ndims == 2
    query = tf.concat(1, query_list)
  for a in xrange(num_heads):
    with tf.variable_scope("Attention_%d" % a):
      y = tf.matmul(query, y_w[a][0]) + y_w[a][1]
      y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
      # Attention mask is a softmax of v^T * tanh(...).
      if use_global_attention:
        s = tf.reduce_sum(hidden_features[a] * y, [2, 3])
      else:
        # Broadcast to add y (query vector) to all hidden_features.
        s = tf.reduce_sum(
            v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
      at_logits.append(s)
      att = tf.nn.softmax(s)
      at_probs.append(att)
      # Now calculate the attention-weighted vector d.
      d = tf.reduce_sum(
          tf.reshape(att, [-1, attn_length, 1, 1]) * hidden,
          [1, 2])
      ds.append(tf.reshape(d, [-1, attention_vec_size]))
  return at_logits, at_probs, ds


def extend_outputs_to_labels(outputs, label_inputs, label_logits,
                             label_vectors, feed_previous):
  """Include (predicted) input labels in encoder attention vectors."""
  new_outputs = []
  for i, cell_output in enumerate(outputs):
    input_label = label_inputs[i]
    if feed_previous:
      input_label = tf.argmax(label_logits[i], 1)
    label_emb = tf.nn.embedding_lookup(label_vectors, input_label)
    concat_emb = tf.concat(1, [cell_output, label_emb])
    new_outputs.append(concat_emb)
  return new_outputs


def gumbel_noise(batch_size, logit_size):
  """Computes Gumbel noise.

     When the output is added to a logit, taking the argmax will be
     approximately equivalent to sampling from the logit.
  """
  size = tf.pack([batch_size, logit_size]) 
  uniform_sample = tf.random_uniform(size, 0, 1, dtype=dtype, 
      seed=None, name=None)
  noise = -tf.log(-tf.log(uniform_sample))
  return noise


def init_thin_stack(batch_size, max_num_concepts):
  """Initializes the thin stack.
  Returns:
    thin_stack: Tensor with the stack content.
    thin_stack_head_next: Index pointers to element after stack head.
  """
  # Stack initialized to -1, points to initial state.
  thin_stack = -tf.ones(tf.pack([batch_size, max_num_concepts]),
      dtype=tf.int32)
  # Reshape to ensure dimension 1 is known.
  thin_stack = tf.reshape(thin_stack, [-1, max_num_concepts])
  # Set to 0 at position 0.
  inds = tf.transpose(tf.to_int64(tf.pack(
   [tf.range(batch_size), tf.zeros(tf.pack([batch_size]), dtype=tf.int32)])))
  delta = tf.SparseTensor(inds, tf.ones(tf.pack([batch_size]), dtype=tf.int32),
      tf.pack([tf.to_int64(batch_size), max_num_concepts]))
  new_thin_stack = thin_stack + tf.sparse_tensor_to_dense(delta)
  # Position 0 is for empty stack; position after head always >= 1.
  thin_stack_head_next = tf.ones(tf.pack([batch_size]),
      dtype=tf.int32)
  return new_thin_stack, thin_stack_head_next


def write_thin_stack(thin_stack, stack_pointers, decoder_position, batch_size,
    max_num_concepts):
  """Writes to the thin stack at the given pointers the current decoder position."""
  new_vals = tf.fill(tf.pack([batch_size]), decoder_position)
  return write_thin_stack_vals(thin_stack, stack_pointers, new_vals, batch_size,
      max_num_concepts)


def write_thin_stack_vals(thin_stack, stack_pointers, new_vals, batch_size,
    max_num_concepts):
  """Writes to the thin stack at the given pointers the current decoder position."""
  # SparseTensor requires type int64.
  stack_inds = tf.transpose(tf.to_int64(tf.pack(
     [tf.range(batch_size), stack_pointers]))) # nn_stack_pointers

  current_vals = tf.gather_nd(thin_stack, stack_inds)
  delta = tf.SparseTensor(stack_inds, new_vals - current_vals,
      tf.pack([tf.to_int64(batch_size), max_num_concepts]))
  new_thin_stack = thin_stack + tf.sparse_tensor_to_dense(delta)
  return new_thin_stack


def pure_reduce_thin_stack(thin_stack_head_next, transition_state):
  """Applies reduce to the thin stack and its head if in reduce state."""
  # Pop if current transition state is reduce. 
  stack_head_updates = tf.sparse_to_dense(data_utils.RE_STATE,
            tf.pack([data_utils.NUM_TR_STATES]), -1)
  new_thin_stack_head_next = tf.add(thin_stack_head_next,
      tf.gather(stack_head_updates, transition_state))
  return new_thin_stack_head_next


def reduce_thin_stack(thin_stack, thin_stack_head_next, batch_size,
                      max_num_concepts, decoder_position, transition_state):
  """Applies reduce to the thin stack and its head if in reduce state."""
  # Pop if current transition state is reduce. 
  stack_head_updates = tf.sparse_to_dense(data_utils.RE_STATE,
            tf.pack([data_utils.NUM_TR_STATES]), -1)
  new_thin_stack_head_next = tf.add(thin_stack_head_next,
      tf.gather(stack_head_updates, transition_state))

  return new_thin_stack_head_next


def update_buffer_head(buffer_head, predicted_attns, transition_state):
  updates = tf.sparse_to_dense(tf.pack([data_utils.GEN_STATE]),
                               tf.pack([data_utils.NUM_TR_STATES]), 
                               True, default_value=False)
  is_gen_state = tf.gather(updates, transition_state)                  

  new_buffer_head = tf.select(is_gen_state, predicted_attns, buffer_head)
  return new_buffer_head


def pure_shift_thin_stack(thin_stack_head_next, transition_state):
  """Applies shift to the thin stack and its head if in shift state."""

  # Push if previous transition state is shift (or pointer shift).
  stack_head_updates = tf.sparse_to_dense(tf.pack(
      [data_utils.GEN_STATE]),
      tf.pack([data_utils.NUM_TR_STATES]), 1)
  new_thin_stack_head_next = tf.add(thin_stack_head_next,
      tf.gather(stack_head_updates, transition_state))

  return new_thin_stack_head_next


def shift_thin_stack(thin_stack, thin_stack_head_next, batch_size,
                     max_num_concepts, decoder_position, 
                     prev_transition_state):
  """Applies shift to the thin stack and its head if in shift state."""
  # Head points to item after stack top, so always update the stack entry.
  new_thin_stack = write_thin_stack(thin_stack, thin_stack_head_next,
      decoder_position, batch_size, max_num_concepts)

  # Push if previous transition state is shift (or pointer shift).
  stack_head_updates = tf.sparse_to_dense(tf.pack(
      [data_utils.GEN_STATE]),
      tf.pack([data_utils.NUM_TR_STATES]), 1)
  new_thin_stack_head_next = tf.add(thin_stack_head_next,
      tf.gather(stack_head_updates, prev_transition_state))

  return new_thin_stack, new_thin_stack_head_next


def update_reduce_thin_stack(thin_stack, thin_stack_head_next, batch_size, 
                             max_num_concepts, decoder_position, 
                             transition_state):
  """If in reduce state, replaces the stack top with current decoder_position."""
  # Aim at head for reduce (update), head_next otherwise (no update).
  re_index_updates = tf.sparse_to_dense(data_utils.RE_STATE,
      tf.pack([data_utils.NUM_TR_STATES]), -1)
  re_stack_head = tf.add(thin_stack_head_next,
      tf.gather(re_index_updates, transition_state))

  # Update the stack.
  new_thin_stack = write_thin_stack(thin_stack, re_stack_head,
      decoder_position, batch_size, max_num_concepts)
  return new_thin_stack


def extract_stack_head_entries(thin_stack, thin_stack_head_next, batch_size):
  """Finds entries (indices) at stack head for every instance in batch."""
  stack_head_inds = tf.sub(thin_stack_head_next,
      tf.ones(tf.pack([batch_size]), dtype=tf.int32))

  # For every batch entry, get the thin stack head entry.
  stack_inds = tf.transpose(tf.pack(
      [tf.range(batch_size), stack_head_inds]))
  stack_heads = tf.gather_nd(thin_stack, stack_inds)
  return stack_heads


def mask_decoder_restrictions(logit, logit_size, decoder_restrictions, 
                              transition_state):
  """Enforces decoder restrictions determined by the transition state."""
  restrict_mask_list = []
  with tf.device("/cpu:0"): # sparse-to-dense must be on CPU for now
    for restr in decoder_restrictions:
      restrict_mask_list.append(tf.sparse_to_dense(restr,
          tf.pack([logit_size]), np.inf, default_value=-np.inf))
  mask = tf.gather(tf.pack(restrict_mask_list), transition_state)
  new_logit = tf.minimum(logit, mask)
  return new_logit

def mask_decoder_reduce(logit, thin_stack_head_next, logit_size, batch_size):
  """Ensures that we can only reduce when the stack has at least 1 item.

  For each batch entry k:
    If thin_stack_head_next == 0, #alternatively, or 1.
      let logit[k][reduce_index] = -np.inf, 
    else don't change.
  """
  # Allow reduce only if at least 1 item on stack, i.e., pointer >= 2.
  update_vals = tf.pack([-np.inf, -np.inf, 0.0])
  update_val = tf.gather(update_vals, 
      tf.minimum(thin_stack_head_next,
      2*tf.ones(tf.pack([batch_size]), dtype=tf.int32)))

  re_filled = tf.fill(tf.pack([batch_size]),
      tf.to_int64(data_utils.REDUCE_ID))
  re_inds = tf.transpose(tf.pack(
      [tf.to_int64(tf.range(batch_size)), re_filled]))
  re_delta = tf.SparseTensor(re_inds, update_val, tf.to_int64(
      tf.pack([batch_size, logit_size])))
  new_logit = logit + tf.sparse_tensor_to_dense(re_delta)
  return new_logit


def mask_decoder_only_shift(logit, thin_stack_head_next, transition_state_map,
                          logit_size, batch_size):
  """Ensures that if the stack is empty, has to GEN_STATE (shift transition)

  For each batch entry k:
    If thin_stack_head_next == 0, #alternatively, or 1.
      let logit[k][reduce_index] = -np.inf, 
    else don't change.
  """
  stack_is_empty_bool = tf.less_equal(thin_stack_head_next, 1) 
  stack_is_empty = tf.select(stack_is_empty_bool, 
                            tf.ones(tf.pack([batch_size]), dtype=tf.int32),
                            tf.zeros(tf.pack([batch_size]), dtype=tf.int32))
  stack_is_empty = tf.reshape(stack_is_empty, [-1, 1])

  # Sh and Re states are disallowed (but not root).
  state_is_disallowed_updates = tf.sparse_to_dense(
      tf.pack([data_utils.RE_STATE, data_utils.ARC_STATE]),
      tf.pack([data_utils.NUM_TR_STATES]), 1)
  logit_states = tf.gather(transition_state_map, tf.range(logit_size))
  state_is_disallowed = tf.gather(state_is_disallowed_updates, logit_states)
  state_is_disallowed = tf.reshape(state_is_disallowed, [1, -1])
  
  index_delta = tf.matmul(stack_is_empty, state_is_disallowed) # 1 if disallowed
  values = tf.pack([0, -np.inf])
  delta = tf.gather(values, index_delta)
  new_logit = logit + delta
  return new_logit

def mask_decoder_only_reduce(logit, thin_stack_head_next, transition_state_map,
                          max_stack_size, logit_size, batch_size):
  """Ensures that if the stack is empty, has to GEN_STATE (shift transition)

  For each batch entry k:
    If thin_stack_head_next == 0, #alternatively, or 1.
      let logit[k][reduce_index] = -np.inf, 
    else don't change.
  """
  # Allow reduce only if at least 1 item on stack, i.e., pointer >= 2.
  #stack_is_empty_updates = tf.pack([-np.inf, -np.inf, 0])
  stack_is_full_bool = tf.greater_equal(thin_stack_head_next, max_stack_size - 1)
  stack_is_full = tf.select(stack_is_full_bool, 
                            tf.ones(tf.pack([batch_size]), dtype=tf.int32),
                            tf.zeros(tf.pack([batch_size]), dtype=tf.int32))
  stack_is_full = tf.reshape(stack_is_full, [-1, 1])
  
  # Sh and Re states are allowed.
  state_is_disallowed_updates = tf.sparse_to_dense(
      tf.pack([data_utils.RE_STATE, data_utils.ARC_STATE, data_utils.ROOT_STATE]),
      tf.pack([data_utils.NUM_TR_STATES]), 0, 1)
  logit_states = tf.gather(transition_state_map, tf.range(logit_size))
  state_is_disallowed = tf.gather(state_is_disallowed_updates, logit_states)
  state_is_disallowed = tf.reshape(state_is_disallowed, [1, -1])
  
  index_delta = tf.matmul(stack_is_full, state_is_disallowed) # 1 if disallowed
  values = tf.pack([0, -np.inf])
  delta = tf.gather(values, index_delta)
  new_logit = logit + delta
  return new_logit


def gather_nd_lstm_states(states_c, states_h, inds, batch_size, input_size, 
    state_size):
  concat_states_c = tf.concat(1, states_c)
  concat_states_h = tf.concat(1, states_h)

  new_prev_state_c = gather_nd_states(concat_states_c,
      inds, batch_size, input_size, state_size)
  new_prev_state_h = gather_nd_states(concat_states_h,
      inds, batch_size, input_size, state_size)
  return tf.nn.rnn_cell.LSTMStateTuple(new_prev_state_c, new_prev_state_h)


def gather_nd_states(inputs, inds, batch_size, input_size, state_size):
  """Gathers an embedding for each batch entry with index inds from inputs.   

  Args:
    inputs: Tensor [batch_size, input_size, state_size].
    inds: Tensor [batch_size]

  Returns:
    output: Tensor [batch_size, embedding_size]
  """
  sparse_inds = tf.transpose(tf.pack(
      [tf.range(batch_size), inds]))
  dense_inds = tf.sparse_to_dense(sparse_inds, 
      tf.pack([batch_size, input_size]),
      tf.ones(tf.pack([batch_size])))

  output_sum = tf.reduce_sum(tf.reshape(dense_inds, 
      [-1, input_size, 1, 1]) * tf.reshape(inputs, 
        [-1, input_size, 1, state_size]), [1, 2])
  output = tf.reshape(output_sum, [-1, state_size])
  return output


def binary_select_state(state, updates, transition_state, batch_size):
  """Gathers state or zero for each batch entry."""
  update_inds = tf.gather(updates, transition_state)
  sparse_diag = tf.transpose(tf.pack(
      [tf.range(batch_size), tf.range(batch_size)]))
  dense_inds = tf.sparse_to_dense(sparse_diag,
      tf.pack([batch_size, batch_size]),
      tf.to_float(update_inds))
  new_state = tf.matmul(dense_inds, state)
  return new_state 


def hard_state_selection(attn_inds, hidden, batch_size, attn_length):
  batch_inds = tf.transpose(tf.pack(
      [tf.to_int64(tf.range(batch_size)), tf.to_int64(attn_inds)]))
  align_index = tf.to_float(tf.sparse_to_dense(batch_inds,
      tf.to_int64(tf.pack([batch_size, attn_length])), 1))
  attns = tf.reduce_sum(hidden * 
      tf.reshape(align_index, [-1, attn_length, 1, 1]), [1, 2])
  return attns


def gather_forced_att_logits(encoder_input_symbols, encoder_decoder_vocab_map, 
                             att_logit, batch_size, attn_length, 
                             target_vocab_size):
  """Gathers attention weights as logits for forced attention."""
  flat_input_symbols = tf.reshape(encoder_input_symbols, [-1])
  flat_label_symbols = tf.gather(encoder_decoder_vocab_map,
      flat_input_symbols)
  flat_att_logits = tf.reshape(att_logit, [-1])

  flat_range = tf.to_int64(tf.range(tf.shape(flat_label_symbols)[0]))
  batch_inds = tf.floordiv(flat_range, attn_length)
  position_inds = tf.mod(flat_range, attn_length)
  attn_vocab_inds = tf.transpose(tf.pack(
      [batch_inds, position_inds, tf.to_int64(flat_label_symbols)]))
 
  # Exclude indexes of entries with flat_label_symbols[i] = -1.
  included_flat_indexes = tf.reshape(tf.where(tf.not_equal(
      flat_label_symbols, -1)), [-1])
  included_attn_vocab_inds = tf.gather(attn_vocab_inds, 
      included_flat_indexes)
  included_flat_att_logits = tf.gather(flat_att_logits, 
      included_flat_indexes)

  sparse_shape = tf.to_int64(tf.pack(
      [batch_size, attn_length, target_vocab_size]))

  sparse_label_logits = tf.SparseTensor(included_attn_vocab_inds, 
      included_flat_att_logits, sparse_shape)
  forced_att_logit_sum = tf.sparse_reduce_sum(sparse_label_logits, [1])

  forced_att_logit = tf.reshape(forced_att_logit_sum, 
      [-1, target_vocab_size])

  return forced_att_logit


def gather_prev_stack_state_index(pointer_vals, prev_index, transition_state,
                                  batch_size):
  """Gathers new previous state index."""
  new_pointer_vals = tf.reshape(pointer_vals, [-1, 1])

  # Helper tensors.
  prev_vals = tf.reshape(tf.fill(
      tf.pack([batch_size]), prev_index), [-1, 1])
  trans_inds = tf.transpose(tf.pack(
      [tf.range(batch_size), transition_state]))

  # Gather new prev state for main tf.nn. Pointer vals if reduce, else prev.
  # State inds dimension [batch_size, NUM_TR_STATES]
  state_inds = tf.concat(1, [prev_vals]*6 + [new_pointer_vals, prev_vals])
  prev_state_index = tf.gather_nd(state_inds, trans_inds)
  return prev_state_index


def gather_prev_stack_aux_state_index(pointer_vals, prev_index, transition_state, 
                                      batch_size):
  """Gather new prev state index for aux rnn: as for main, but zero if shift."""
  new_pointer_vals = tf.reshape(pointer_vals, [-1, 1])

  # Helper tensors.
  prev_vals = tf.reshape(tf.fill(
      tf.pack([batch_size]), prev_index), [-1, 1])
  trans_inds = tf.transpose(tf.pack(
      [tf.range(batch_size), transition_state]))
  batch_zeros = tf.reshape(tf.zeros(
            tf.pack([batch_size]), dtype=tf.int32), [-1, 1])

  # Gather new prev state for aux tf.nn.
  # State inds dimension [batch_size, NUM_TR_STATES]
  state_inds = tf.concat(1, 
      [prev_vals, batch_zeros] + [prev_vals]*4 + [new_pointer_vals, prev_vals])
  prev_state_index = tf.gather_nd(state_inds, trans_inds)
  return prev_state_index


