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
import seq2seq_helpers

linear = tf.nn.rnn_cell._linear  # pylint: disable=protected-access

def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                output_projection=None, scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    output_projection: If not None, weights and bias to project the decoder
      cell output to the output logits.
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (logits, outputs, [], state).
  """
  with tf.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    logits = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      with tf.variable_scope("OutputProjection"):
        if output_projection is not None:
          logit = tf.matmul(output, output_projection[0]) + output_projection[1]
        else:
          logit = output
      if loop_function is not None:
        prev = logit
      logits.append(logit)
  return logits, outputs, [], state


def attention_decoder(decoder_inputs, encoder_inputs,
                      initial_state, attention_states, cell,
                      encoder_decoder_vocab_map=None,
                      decoder_vocab_sizes=None,           
                      output_size=None, num_heads=1, 
                      embed_functions=None, loop_functions=None,
                      output_projections=None, decoder_restrictions=None,
                      transition_state_map=None, dtype=tf.float32,
                      scope=None, initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size] for the
      decoder embedding inputs.
    decoder_input_symbols: A list of 1D Tensors [batch_size] for the decoder
      input symbols.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    output_projection: None or a pair (W, B) of output projection weights and
      biases.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output
      symbols for each decoder transition state.
    transition_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (logits, outputs, [], state).

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  sample_output = False
  use_nonlinear = False
  max_num_concepts = data_utils.MAX_OUTPUT_SIZE

  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0]["parse"])[0] # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    y_w = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a,
                                           [attention_vec_size]))
      y_m = tf.get_variable("AttnInputLinearW_%d" % a, 
                            [attn_size, attention_vec_size],
                            dtype=dtype)
      y_bias = tf.get_variable("AttnInputLinearBias_%d" % a, 
                               [attention_vec_size], dtype=dtype, 
                               initializer=tf.constant_initializer(
                                 0.0, dtype=dtype))
      y_w.append((y_m, y_bias))

    def attention(query):
      return seq2seq_helpers.attention(query, num_heads, y_w, v, hidden,
          hidden_features, attention_vec_size, attn_length)

    parse_logits = []
    state = initial_state
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]

    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      _, _, attns = attention(initial_state)

    for i, decoder_input in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If loop_functions is set, we use it instead of decoder_inputs.
      if loop_functions is not None and prev is not None:
        prev_symbol = tf.argmax(prev, 1)
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_functions["parse"](prev, None)
      else:
        prev_symbol = decoder_input["parse"]
        with tf.variable_scope("embed_function", reuse=True):
          inp = embed_functions["parse"](prev_symbol) 

      with tf.variable_scope("DecoderInputAttentionLinear"):
        # Merge input and previous attentions into one vector of the right size.
        x = linear([inp] + attns, cell.output_size, True)

      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True):
          _, att_weights, attns = attention(state)
      else:
        _, att_weights, attns = attention(state)

      with tf.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
        if use_nonlinear:
          output = tf.tanh(output)
        
      logit = (tf.matmul(output, output_projections["parse"][0]) 
          + output_projections["parse"][1])

      target_vocab_size = tf.shape(output_projections["parse"][0])[1]

      if loop_functions is not None and sample_output:
        logit += seq2seq_helpers.gumbel_noise(batch_size, target_vocab_size)

      if loop_functions is not None:
        prev = logit
      parse_logits.append(logit)
  logits = [{"parse": parse_logit} for parse_logit in parse_logits]
  return logits, state


def hard_attention_arc_eager_decoder(decoder_inputs, encoder_inputs,
                      initial_state, attention_states, cell, 
                      predict_end_attention=True,
                      decoder_vocab_sizes=None,           
                      output_size=None, num_heads=1, 
                      embed_functions=None, loop_functions=None,
                      output_projections=None, 
                      transition_state_map=None, dtype=tf.float32,
                      scope=None, 
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size] for the
      decoder embedding inputs.
    decoder_input_symbols: A list of 1D Tensors [batch_size] for the decoder
      input symbols.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    output_projection: None or a pair (W, B) of output projection weights and
      biases.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output
      symbols for each decoder transition state.
    transition_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (logits, outputs, [], state).

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  use_nonlinear = False
  feed_pointer_encoding = True
  max_stack_size = int(data_utils.MAX_OUTPUT_SIZE/2)

  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(attention_states)[0] # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    # Size of query vectors for attention.
    attention_vec_size = attn_size

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])

    # Define hard attention.
    num_heads = 2 if predict_end_attention else 1
    hidden_features = []
    v = []
    y_w = []

    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                          [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
      y_m_a = tf.get_variable("AttnInputLinearW_%d" % a, 
                            [attn_size, attention_vec_size],
                            dtype=dtype)
      y_bias_a = tf.get_variable("AttnInputLinearBias_%d" % a, 
                               [attention_vec_size], dtype=dtype, 
                               initializer=tf.constant_initializer(
                                 0.0, dtype=dtype))
      y_w.append((y_m_a, y_bias_a))

    def attention(query):
      return seq2seq_helpers.attention(query, num_heads, y_w, v, hidden,
          hidden_features, attention_vec_size, attn_length,
          use_global_attention=False)

    parse_logits = []
    ind_logits = []
    end_ind_logits = []
    state = initial_state
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])

    attns = tf.zeros(batch_attn_size, dtype=dtype)
    attns.set_shape([None, attn_size])
    stack_top_emb = tf.reshape(tf.zeros(tf.pack([batch_size, attn_size]), 
        dtype=tf.float32), [-1, attn_size])
    buffer_head_emb = tf.reshape(tf.zeros(tf.pack([batch_size, attn_size]), 
        dtype=tf.float32), [-1, attn_size])

    thin_stack_enc, thin_stack_head_next = seq2seq_helpers.init_thin_stack(
        batch_size, max_stack_size)
    buffer_head = tf.zeros(tf.pack([batch_size]), dtype=tf.int32)

    transition_state = tf.fill(tf.pack([batch_size]),
        data_utils.PAD_STATE)

    for i, decoder_input in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If loop_functions is set, we use it instead of decoder_inputs.
      # Don't feed index predictions.
      if loop_functions is not None and prev is not None:
        prev_symbol = tf.argmax(prev, 1)
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_functions["parse"](prev, None)
      else:
        prev_symbol = decoder_input["parse"]
        with tf.variable_scope("embed_function", reuse=True):
          inp = embed_functions["parse"](prev_symbol) 

      transition_state = tf.gather(transition_state_map, prev_symbol)

      if i == 1:
        buffer_head = seq2seq_helpers.update_buffer_head(buffer_head, attn_inds,
                transition_state)
      elif i > 1:
        thin_stack_enc = seq2seq_helpers.write_thin_stack_vals(thin_stack_enc, 
            thin_stack_head_next, buffer_head, batch_size, max_stack_size)
        thin_stack_head_next = seq2seq_helpers.pure_shift_thin_stack(
                thin_stack_head_next, transition_state)
        thin_stack_head_next = seq2seq_helpers.pure_reduce_thin_stack(
                thin_stack_head_next, transition_state)
        buffer_head = seq2seq_helpers.update_buffer_head(buffer_head, attn_inds,
                transition_state)

      # Find heads from the stack for parent-feeding.
      if i > 0:
        stack_top_enc_inds = seq2seq_helpers.extract_stack_head_entries(
            thin_stack_enc, thin_stack_head_next, batch_size)
        stack_top_emb = seq2seq_helpers.hard_state_selection(stack_top_enc_inds, 
            hidden, batch_size, attn_length)
        buffer_head_emb = seq2seq_helpers.hard_state_selection(buffer_head, 
            hidden, batch_size, attn_length)

      with tf.variable_scope("DecoderInputAttentionLinear"):
        # Merge input and previous attentions into one vector of the right size.
        if feed_pointer_encoding:
          x = linear([inp, stack_top_emb, buffer_head_emb], cell.output_size, True)
        else:
          x = linear([inp], cell.output_size, True)
       
      # Run the RNN.
      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      with tf.variable_scope("AttentionCall"):
        pointer_logits, _, _ = attention(state)
      ind_logits.append(pointer_logits[0]) 
      if predict_end_attention:
        end_ind_logits.append(pointer_logits[1]) 
     
      if loop_functions is not None:
        attn_inds = tf.to_int32(tf.argmax(pointer_logits[0], 1))
      elif i < len(decoder_inputs) - 1:
        # Corresponds to NEXT decoder input.
        attn_inds = decoder_inputs[i+1]["att"]
      else:
        attn_inds = data_utils.PAD_ID*tf.ones(tf.pack([batch_size]), tf.int32)

      attns = seq2seq_helpers.hard_state_selection(attn_inds, hidden,
          batch_size, attn_length)

      with tf.variable_scope("AttnOutputProjection"):
        # Excludes buffer_head_emb.
        output = linear([cell_output, attns, stack_top_emb], 
                        output_size, True)
        if use_nonlinear:
          output = tf.relu(output)
        
      logit = (tf.matmul(output, output_projections["parse"][0]) 
          + output_projections["parse"][1])

      target_vocab_size = tf.shape(output_projections["parse"][0])[1]

      if loop_functions is not None:
        logit = seq2seq_helpers.mask_decoder_only_shift(
                    logit, thin_stack_head_next, transition_state_map,
                    target_vocab_size, batch_size)
        logit = seq2seq_helpers.mask_decoder_only_reduce(
                    logit, thin_stack_head_next, transition_state_map,
                    max_stack_size, target_vocab_size, batch_size)
        prev = logit
      parse_logits.append(logit)
      
  if predict_end_attention:
      logits = [{"parse": parse_logit, "att": ind_logit, "endatt": end_ind_logit} 
              for parse_logit, ind_logit, end_ind_logit in zip(parse_logits, 
                  ind_logits, end_ind_logits)]
  else:
    logits = [{"parse": parse_logit, "att": ind_logit} 
              for parse_logit, ind_logit in zip(parse_logits, ind_logits)]

  return logits, state


def hard_attention_decoder(decoder_inputs, encoder_inputs,
                      initial_state, attention_states, cell, 
                      predict_end_attention=False,
                      encoder_decoder_vocab_map=None,
                      decoder_vocab_sizes=None,           
                      output_size=None, num_heads=1, 
                      embed_functions=None, loop_functions=None,
                      output_projections=None, decoder_restrictions=None,
                      transition_state_map=None, dtype=tf.float32,
                      scope=None, initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size] for the
      decoder embedding inputs.
    decoder_input_symbols: A list of 1D Tensors [batch_size] for the decoder
      input symbols.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    output_projection: None or a pair (W, B) of output projection weights and
      biases.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output
      symbols for each decoder transition state.
    transition_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (logits, outputs, [], state).

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  use_nonlinear = False
  feed_pointer_encoding = True
  max_num_concepts = data_utils.MAX_OUTPUT_SIZE

  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0]["parse"])[0] # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    # Size of query vectors for attention.
    attention_vec_size = attn_size

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])

    # Define hard attention.
    num_heads = 2 if predict_end_attention else 1
    hidden_features = []
    v = []
    y_w = []

    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                          [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
      y_m_a = tf.get_variable("AttnInputLinearW_%d" % a, 
                            [attn_size, attention_vec_size],
                            dtype=dtype)
      y_bias_a = tf.get_variable("AttnInputLinearBias_%d" % a, 
                               [attention_vec_size], dtype=dtype, 
                               initializer=tf.constant_initializer(
                                 0.0, dtype=dtype))
      y_w.append((y_m_a, y_bias_a))

    def attention(query):
      return seq2seq_helpers.attention(query, num_heads, y_w, v, hidden,
          hidden_features, attention_vec_size, attn_length,
          use_global_attention=False)

    parse_logits = []
    ind_logits = []
    end_ind_logits = []
    state = initial_state
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])
    attns = tf.zeros(batch_attn_size, dtype=dtype)
    attns.set_shape([None, attn_size])

    for i, decoder_input in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If loop_functions is set, we use it instead of decoder_inputs.
      if loop_functions is not None and prev is not None:
        prev_symbol = tf.argmax(prev, 1)
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_functions["parse"](prev, None)
      else:
        prev_symbol = decoder_input["parse"]
        with tf.variable_scope("embed_function", reuse=True):
          inp = embed_functions["parse"](prev_symbol) 

      with tf.variable_scope("DecoderInputAttentionLinear"):
        # Merge input and previous attentions into one vector of the right size.
        if feed_pointer_encoding:
          x = linear([inp, attns], cell.output_size, True)
        else:
          x = linear([inp], cell.output_size, True)

      # Run the RNN.
      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      with tf.variable_scope("AttentionCall"):
        pointer_logits, _, _ = attention(state)
      ind_logits.append(pointer_logits[0]) 
      if predict_end_attention:
        end_ind_logits.append(pointer_logits[1]) 
     
      if loop_functions is not None:
        attn_inds = tf.argmax(pointer_logits[0], 1)
      elif i < len(decoder_inputs) - 1:
        # Corresponds to NEXT decoder input.
        attn_inds = decoder_inputs[i+1]["att"]
      else:
        attn_inds = data_utils.PAD_ID*tf.ones(tf.pack([batch_size]), tf.int32)

      attns = seq2seq_helpers.hard_state_selection(attn_inds, hidden,
          batch_size, attn_length)
      
      with tf.variable_scope("AttnOutputProjection"):
        output = linear([cell_output, attns], output_size, True)
        if use_nonlinear:
          output = tf.relu(output)
        
      logit = (tf.matmul(output, output_projections["parse"][0]) 
          + output_projections["parse"][1])

      if loop_functions is not None:
        prev = logit
      parse_logits.append(logit)
  if predict_end_attention:
      logits = [{"parse": parse_logit, "att": ind_logit, "endatt": end_ind_logit} 
              for parse_logit, ind_logit, end_ind_logit in zip(parse_logits, 
                  ind_logits, end_ind_logits)]
  else:
    logits = [{"parse": parse_logit, "att": ind_logit} 
              for parse_logit, ind_logit in zip(parse_logits, ind_logits)]

  return logits, state


def attention_pointer_decoder(decoder_inputs, encoder_inputs,
                      initial_state, attention_states, cell,
                      feed_alignment=False, feed_post_alignment=False,
                      predict_end_attention=False,
                      encoder_decoder_vocab_map=None,
                      decoder_vocab_sizes=None,           
                      output_size=None, num_heads=1, 
                      embed_functions=None, loop_functions=None,
                      output_projections=None, decoder_restrictions=None,
                      transition_state_map=None, 
                      decoder_embedding_sizes=None,           
                      dtype=tf.float32,
                      scope=None, initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size] for the
      decoder embedding inputs.
    decoder_input_symbols: A list of 1D Tensors [batch_size] for the decoder
      input symbols.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    output_projection: None or a pair (W, B) of output projection weights and
      biases.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output
      symbols for each decoder transition state.
    transition_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (logits, outputs, [], state).

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size
  assert not (feed_alignment and feed_post_alignment)

  use_nonlinear = False
  max_num_concepts = data_utils.MAX_OUTPUT_SIZE

  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0]["parse"])[0] # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    # Size of query vectors for attention.
    attention_vec_size = attn_size

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])

    # Define soft attention.
    k = tf.get_variable("SoftAttnW",
                        [1, 1, attn_size, attention_vec_size])
    hidden_features_attn = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
    v_attn = tf.get_variable("SoftAttnV", [attn_size])
    y_m = tf.get_variable("SoftAttnInputLinearW", 
                          [attn_size, attn_size],
                          dtype=dtype)
    y_bias = tf.get_variable("SoftAttnInputLinearBias", 
                             [attn_size], dtype=dtype, 
                             initializer=tf.constant_initializer(
                               0.0, dtype=dtype))

    def attention(query):
      return seq2seq_helpers.attention(query, 1, [(y_m, y_bias)], [v_attn], 
          hidden, [hidden_features_attn], attn_size, attn_length,
          use_global_attention=False)

    # Define hard attention.
    num_heads = 2 if predict_end_attention else 1
    hidden_features = []
    v = []
    y_w = []
    attn_q_size = attn_size + decoder_embedding_sizes["parse"] if feed_post_alignment else attn_size

    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                          [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
      y_m_a = tf.get_variable("AttnInputLinearW_%d" % a, 
                            [attn_q_size, attention_vec_size],
                            dtype=dtype)
      y_bias_a = tf.get_variable("AttnInputLinearBias_%d" % a, 
                               [attention_vec_size], dtype=dtype, 
                               initializer=tf.constant_initializer(
                                 0.0, dtype=dtype))
      y_w.append((y_m_a, y_bias_a))

    def pointer_attention(query):
      return seq2seq_helpers.attention(query, num_heads, y_w, v, hidden,
          hidden_features, attention_vec_size, attn_length,
          use_global_attention=False)

    parse_logits = []
    ind_logits = []
    end_ind_logits = []
    state = initial_state
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)]

    for a in attns: # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    pointer_attns = tf.zeros(batch_attn_size, dtype=dtype)
    pointer_attns.set_shape([None, attn_size])
    if initial_state_attention:
      _, _, attns = attention(initial_state)

    for i, decoder_input in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If loop_functions is set, we use it instead of decoder_inputs.
      if loop_functions is not None and prev is not None:
        prev_symbol = tf.argmax(prev, 1)
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_functions["parse"](prev, None)
      else:
        prev_symbol = decoder_input["parse"]
        with tf.variable_scope("embed_function", reuse=True):
          inp = embed_functions["parse"](prev_symbol) 

      with tf.variable_scope("DecoderInputAttentionLinear"):
        # Merge input and previous attentions into one vector of the right size.
        if feed_alignment:
          x = linear([inp, pointer_attns] + attns, cell.output_size, True)
        else:
          x = linear([inp] + attns, cell.output_size, True)

      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True):
          _, att_weights, attns = attention(state)
      else:
        _, att_weights, attns = attention(state)

      if not feed_post_alignment:
        with tf.variable_scope("PointerAttentionCall"):
          pointer_logits, _, _ = pointer_attention(state)
        ind_logits.append(pointer_logits[0]) 
        if predict_end_attention:
          end_ind_logits.append(pointer_logits[1]) 
     
      if feed_alignment:
        if loop_functions is not None:
          attn_inds = tf.argmax(pointer_logits[0], 1)
        elif i < len(decoder_inputs) - 1:
          # Corresponds to NEXT decoder input.
          attn_inds = decoder_inputs[i+1]["att"]
        else:
          attn_inds = data_utils.PAD_ID*tf.ones(tf.pack([batch_size]), tf.int32)

        pointer_attns = seq2seq_helpers.hard_state_selection(attn_inds, hidden,
            batch_size, attn_length)

        with tf.variable_scope("AttnOutputProjection"):
          output = linear([cell_output, pointer_attns] + attns, output_size, True)
          if use_nonlinear:
            output = tf.relu(output)
      else:
        with tf.variable_scope("AttnOutputProjection"):
          output = linear([cell_output] + attns, output_size, True)
          if use_nonlinear:
            output = tf.relu(output)
          
      logit = (tf.matmul(output, output_projections["parse"][0]) 
          + output_projections["parse"][1])

      if feed_post_alignment:
        # Embed parse argmax.
        if loop_functions is not None:
          parse_out_symbol = tf.argmax(logit, 1)
        else:
          if i < len(decoder_inputs) - 1:
            # Corresponds to NEXT decoder input.
            parse_out_symbol = decoder_inputs[i+1]["parse"]
          else:
            parse_out_symbol = data_utils.PAD_ID*tf.ones(tf.pack([batch_size]), tf.int32)
        
        if loop_functions is not None:
          with tf.variable_scope("output_loop_function_parse", reuse=True):
            parse_out = loop_functions["parse"](logit, None)
        else:
          with tf.variable_scope("output_embed_function_parse", reuse=True):
            parse_out = embed_functions["parse"](parse_out_symbol) 
       
        with tf.variable_scope("PointerAttentionCall"):
          pointer_logits, _, _ = pointer_attention((state.h, state.c,
              parse_out))
        ind_logits.append(pointer_logits[0]) 
 
      if loop_functions is not None:
        prev = logit
      parse_logits.append(logit)
  if predict_end_attention:
    logits = [{"parse": parse_logit, "att": ind_logit, "endatt": end_ind_logit} 
              for parse_logit, ind_logit, end_ind_logit in zip(parse_logits, 
                  ind_logits, end_ind_logits)]
  else:
    logits = [{"parse": parse_logit, "att": ind_logit}
              for parse_logit, ind_logit in zip(parse_logits, ind_logits)]
  return logits, state


def attention_stack_decoder(decoder_inputs, encoder_inputs,
    initial_state, attention_states, cell, aux_cell, mem_cell,
    use_aux_stack=True, use_memory_stack=False, 
    output_size=None, num_heads=1, embed_functions=None, 
    loop_functions=None,
    output_projections=None, decoder_restrictions=None,
    transition_state_map=None, dtype=tf.float32, scope=None,
    initial_state_attention=False):
  """Stack RNN attention-based decoder with recursively-composed subtree
    representations computer through an auxiliary stack.

    Same arguments as attention_decoder, except sample_outputs.

  Returns:
    A tuple of the form (logits, outputs, [], state).

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  max_num_concepts = data_utils.MAX_OUTPUT_SIZE
  assert len(decoder_restrictions) == data_utils.NUM_TR_STATES

  with tf.variable_scope(scope or "attention_stack_decoder"):
    # The batch size is variable in the graph, so handle with care.
    batch_size = tf.shape(decoder_inputs[0]["parse"])[0]
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    y_w = []
    attention_vec_size = attn_size  # size of query vectors for attention
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a,
                                           [attention_vec_size]))
      y_m = tf.get_variable("AttnInputLinearW_%d" % a, 
                            [attn_size, attention_vec_size],
                            dtype=dtype)
      y_bias = tf.get_variable("AttnInputLinearBias_%d" % a, 
                               [attention_vec_size], dtype=dtype, 
                               initializer=tf.constant_initializer(
                                 0.0, dtype=dtype))
      y_w.append((y_m, y_bias))


    def attention(query):
      return seq2seq_helpers.attention(query, num_heads, y_w, v, hidden,
          hidden_features, attention_vec_size, attn_length)

    assert cell.state_size.c == cell.state_size.h
    state_size = cell.state_size.h 
    states_c = [tf.reshape(initial_state.c, [-1, 1, state_size])]
    states_h = [tf.reshape(initial_state.h, [-1, 1, state_size])]
    state = initial_state
    linear_state = initial_state

    if use_aux_stack:
      # Declare initial aux state variable, fill for each item in batch.
      initial_aux_state_var_c = tf.get_variable("InitialAuxStateC", [state_size])
      initial_aux_state_c = tf.tile(tf.reshape(initial_aux_state_var_c, 
          [1, -1]), tf.pack([batch_size, 1]))
      initial_aux_state_var_h = tf.get_variable("InitialAuxStateH", [state_size])
      initial_aux_state_h = tf.tile(tf.reshape(initial_aux_state_var_h, 
          [1, -1]), tf.pack([batch_size, 1]))
      aux_states_c = [tf.reshape(initial_aux_state_c, [-1, 1, state_size])]
      aux_states_h = [tf.reshape(initial_aux_state_h, [-1, 1, state_size])]

    outputs = []
    parse_logits = []
    prev = None
    attns = [tf.zeros(tf.pack([batch_size, attn_size]),
                                            dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # ensure the second shape of attention vectors is set
      a.set_shape([None, attn_size])
    if initial_state_attention:
      _, _, attns = attention(initial_state)

    thin_stack, thin_stack_head_next = seq2seq_helpers.init_thin_stack(
        batch_size, max_num_concepts)
    transition_state = tf.fill(tf.pack([batch_size]),
        data_utils.PAD_STATE)

    for i, decoder_input in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If loop_functions is set, we use it instead of decoder_inputs.
      if loop_functions is not None and prev is not None:
        prev_symbol = tf.argmax(prev, 1)
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_functions["parse"](prev, None)
      else:
        prev_symbol = decoder_input["parse"]
        with tf.variable_scope("embed_function", reuse=True):
          inp = embed_functions["parse"](prev_symbol) 

      prev_transition_state = transition_state
      transition_state = tf.gather(transition_state_map, prev_symbol)

      if i > 0:
        thin_stack_head_next = seq2seq_helpers.reduce_thin_stack(
            thin_stack, thin_stack_head_next, batch_size, max_num_concepts,
            i, transition_state)

      if use_aux_stack:
        # First run inp through a linear layer.
        with tf.variable_scope("DecoderInputLinear"):
          inp_lin = linear([inp], cell.output_size, True)

        if i > 0:
          # Select previous aux cell output for reduce state, else given input.
          aux_updates = tf.sparse_to_dense(data_utils.RE_STATE,
              tf.pack([data_utils.NUM_TR_STATES]), 1)
          aux_inp = seq2seq_helpers.binary_select_state(aux_cell_output,
              aux_updates, transition_state, batch_size)

          inp_updates = tf.ones([data_utils.NUM_TR_STATES],
              dtype=np.int32) - aux_updates
          new_inp = seq2seq_helpers.binary_select_state(inp_lin, inp_updates,
              transition_state, batch_size)
          new_inp += aux_inp
        else:
          new_inp = inp_lin

      if i > 0:
        # Compute previous state indexes.
        pointer_vals = seq2seq_helpers.extract_stack_head_entries(thin_stack,
            thin_stack_head_next, batch_size)

        # Add one to account for initial state.
        pointer_vals = tf.add(pointer_vals,
            tf.ones(tf.pack([batch_size]), dtype=tf.int32))

        prev_state_index = seq2seq_helpers.gather_prev_stack_state_index(
            pointer_vals, i, transition_state, batch_size)
        if use_aux_stack:
          prev_aux_state_index = seq2seq_helpers.gather_prev_stack_aux_state_index(
              pointer_vals, i, transition_state, batch_size)
      else:
        prev_state_index = tf.fill(tf.pack([batch_size]), i)
        if use_aux_stack:
          prev_aux_state_index = tf.fill(tf.pack([batch_size]), i)

      # Extract previous states (main and aux).
      new_prev_state = seq2seq_helpers.gather_nd_lstm_states(states_c, 
          states_h, prev_state_index, batch_size, i + 1, state_size)
      
      if use_aux_stack:
        new_prev_aux_state = seq2seq_helpers.gather_nd_lstm_states(aux_states_c,
            aux_states_h, prev_state_index, batch_size, i + 1, state_size)

      # Merge input and previous attentions into one vector of the right size.
      with tf.variable_scope("DecoderInputAttentionLinear"):
        if use_memory_stack and use_aux_stack:
          x = linear([new_inp], cell.output_size, True)
        elif use_memory_stack:
          x = linear([inp], cell.output_size, True)
        elif use_aux_stack:
          x = linear([new_inp] + attns, cell.output_size, True)
        else:
          x = linear([inp] + attns, cell.output_size, True)

      if i > 0:
        thin_stack, thin_stack_head_next = seq2seq_helpers.shift_thin_stack(
            thin_stack, thin_stack_head_next, batch_size, max_num_concepts,
            i, prev_transition_state)
        thin_stack = seq2seq_helpers.update_reduce_thin_stack(thin_stack,
            thin_stack_head_next, batch_size, max_num_concepts, i,
            transition_state)

      # Run the RNN.
      with tf.variable_scope("StackDecoderMainCell"):
        cell_output, state = cell(x, new_prev_state)
      states_h.append(tf.reshape(state.h, [-1, 1, state_size]))
      states_c.append(tf.reshape(state.c, [-1, 1, state_size]))

      if use_memory_stack:
        with tf.variable_scope("DecoderInputMemLinear"):
          l_x = linear([inp] + attns, cell.output_size, True)
        linear_cell_output, linear_state = mem_cell(l_x, linear_state)

      if use_aux_stack:
        # Run the auxiliary RNN.
        with tf.variable_scope("StackDecoderAuxCell"):
          aux_cell_output, aux_state = aux_cell(new_inp, new_prev_aux_state)
        aux_states_h.append(tf.reshape(aux_state.h, [-1, 1, state_size]))
        aux_states_c.append(tf.reshape(aux_state.c, [-1, 1, state_size]))

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True):
          _, att_weights, attns = attention(state)
      else:
        _, att_weights, attns = attention(state)

      with tf.variable_scope("AttnOutputProjection"):
        if use_memory_stack:
          output = linear([linear_cell_output, cell_output] + attns, 
                          output_size, True)
        else:
          output = linear([cell_output] + attns, output_size, True)
        outputs.append(output)

      logit = tf.matmul(output, output_projections["parse"][0]) + output_projections["parse"][1]

      target_vocab_size = tf.shape(output_projections["parse"][0])[1]

      logit = seq2seq_helpers.mask_decoder_restrictions(logit,
          target_vocab_size, decoder_restrictions, transition_state)
      logit = seq2seq_helpers.mask_decoder_reduce(logit, thin_stack_head_next,
          target_vocab_size, batch_size)

      if loop_functions is not None:
        prev = logit
      parse_logits.append(logit)
  logits = [{"parse": parse_logit} for parse_logit in parse_logits]
  return logits, state

