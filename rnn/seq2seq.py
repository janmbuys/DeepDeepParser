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

"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.
      Attention optional.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf

import data_utils
import seq2seq_helpers
import seq2seq_decoders

linear = tf.nn.rnn_cell._linear  # pylint: disable=protected-access

def embedding_attention_decoder(decoder_type, decoder_inputs,
                                encoder_inputs,
                                initial_state, attention_states,
                                cell, aux_cell, mem_cell,
                                decoder_vocab_sizes,
                                decoder_embedding_sizes,
                                decoder_restrictions=None,
                                num_heads=1, 
                                predict_span_end_pointers=False,
                                output_projections=None,
                                transition_state_map=None,
                                encoder_decoder_vocab_map=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=tf.float32, scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and multiple decoder models.

  Args:
    decoder_type: int indicating decoder to be used for encoder-decoder models 
      - constants defined in data_utils.py.
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    encoder_label_inputs: A list of 1D int32 Tensors of shape [batch_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    aux_cell: rnn_cell.RNNCell defining the cell function and size. Auxiliary
      decoder LSTM cell. 
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output 
      symbols for each decoder transition state.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    transition_state_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state. 
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (logits, outputs, pointer_logits, state), where:
      logits: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        output logits.
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_cell_size] (if output_projection is not None)
        containing the outputs that are fed to the loss function.
        Also fed to loss function.
      label_logits: List of the same length as encoder_inputs for label logits.
      label_outputs: List of the same length as encoder_inputs for label output
        vectors for loss function.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  output_size = cell.output_size
  for key, output_projection in output_projections.iteritems():
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([decoder_vocab_sizes[key]])

  with tf.variable_scope(scope or "embedding_attention_decoder"):
    embeddings = {}
    embed_functions = {}
    if feed_previous:
      loop_functions = {}
    else:
      loop_functions = None

    for key, vocab_size in decoder_vocab_sizes.iteritems(): 
      embedding = tf.get_variable("decoder_input_embedding_{0}".format(key),
        [vocab_size, decoder_embedding_sizes[key]],
        initializer=tf.random_uniform_initializer(-np.sqrt(3),
            np.sqrt(3)))
      embeddings[key] = embedding 
      embed_functions[key] = seq2seq_helpers._extract_embed(embedding)

      if feed_previous:
        loop_functions[key] = seq2seq_helpers._extract_argmax_and_embed(
            embedding)

    if decoder_type == data_utils.STACK_DECODER_STATE:
      return seq2seq_decoders.attention_stack_decoder(
          decoder_inputs, encoder_inputs, initial_state, 
          attention_states, cell, aux_cell, mem_cell,
          use_aux_stack=True, output_size=output_size, 
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions, 
          output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.PURE_STACK_DECODER_STATE:
      return seq2seq_decoders.attention_stack_decoder(
          decoder_inputs, encoder_inputs, initial_state, 
          attention_states, cell, 
          aux_cell, mem_cell, use_aux_stack=False, output_size=output_size, 
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions, output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.MEMORY_STACK_DECODER_STATE:
      return seq2seq_decoders.attention_stack_decoder(
          decoder_inputs, encoder_inputs, initial_state, 
          attention_states, cell, 
          aux_cell, mem_cell, use_aux_stack=True, use_memory_stack=True, 
          output_size=output_size, num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions, output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.LINEAR_POINTER_DECODER_STATE:
      return seq2seq_decoders.attention_pointer_decoder(
          decoder_inputs, encoder_inputs,
          initial_state, attention_states, cell, feed_alignment=False,
          feed_post_alignment=False, 
          predict_end_attention=predict_span_end_pointers, 
          output_size=output_size,
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions,
          output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          decoder_embedding_sizes=decoder_embedding_sizes,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.LINEAR_FEED_POINTER_DECODER_STATE:
      return seq2seq_decoders.attention_pointer_decoder(
          decoder_inputs, encoder_inputs,
          initial_state, attention_states, cell, feed_alignment=True, 
          feed_post_alignment=False,
          predict_end_attention=predict_span_end_pointers, 
          output_size=output_size,
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions,
          output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          decoder_embedding_sizes=decoder_embedding_sizes,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.HARD_ATTENTION_DECODER_STATE:
      return seq2seq_decoders.hard_attention_decoder(
          decoder_inputs, encoder_inputs,
          initial_state, attention_states, cell, 
          predict_end_attention=predict_span_end_pointers, 
          output_size=output_size,
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions,
          output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.HARD_ATTENTION_ARC_EAGER_DECODER_STATE:
      return seq2seq_decoders.hard_attention_arc_eager_decoder(
          decoder_inputs, encoder_inputs,
          initial_state, attention_states, cell, 
          predict_end_attention=predict_span_end_pointers, 
          output_size=output_size,
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions,
          output_projections=output_projections,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    elif decoder_type == data_utils.ATTENTION_DECODER_STATE:
      return seq2seq_decoders.attention_decoder(
          decoder_inputs, encoder_inputs,
          initial_state, attention_states, cell, 
          output_size=output_size,
          num_heads=num_heads,
          embed_functions=embed_functions,
          loop_functions=loop_functions,
          output_projections=output_projections,
          decoder_restrictions=decoder_restrictions,
          transition_state_map=transition_state_map,
          initial_state_attention=initial_state_attention)
    else:
      return seq2seq_decoders.rnn_decoder(emb_inp, initial_state, cell, 
          loop_function=loop_function, output_projection=output_projection)


def embedding_attention_seq2seq(decoder_type, encoder_inputs, 
                                decoder_inputs,
                                fw_cell, bw_cell, 
                                dec_cell, dec_aux_cell, dec_mem_cell,
                                encoder_vocab_sizes,
                                decoder_vocab_sizes, encoder_embedding_sizes,
                                decoder_embedding_sizes, 
                                predict_span_end_pointers=False,
                                decoder_restrictions=None,
                                num_heads=1, output_projections=None,
                                word_vectors=None,
                                transition_state_map=None,
                                encoder_decoder_vocab_map=None,
                                use_bidirectional_encoder=False,
                                feed_previous=False, dtype=tf.float32,
                                scope=None, initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Args:
    decoder_type: int indicating decoder to be used for encoder-decoder models 
      - constants defined in data_utils.py.
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    encoder_label_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_pointer_inputs: A list of 1D int32 Tensors of shape [batch_size].
    fw_cell: rnn_cell.RNNCell defining the cell function and size. Forward
      encoder cell.
    bw_cell: rnn_cell.RNNCell defining the cell function and size. Backward
      encoder cell.
    dec_cell: rnn_cell.RNNCell defining the cell function and size. Decoder
      cell.
    dec_aux_cell: rnn_cell.RNNCell defining the cell function and size. Decoder
      auxiliary LSTM cell.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer; length of the embedding vector for each symbol.
    label_embedding_size: Integer; length of the label embedding vector for each symbol.
    use_input_labels: Boolean; encoder should include input labels in input.
    predict_input_labels: Boolean; encoder should predict label for each input
      token.
    decoder_restrictions: List of (dense) 1D int32 Tensors of allowed output 
      symbols for each decoder transition state.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    label_output_projection; None or label prediction output weights and biases.
    word_vectors: 2D Tensor shape [source_vocab_size, embedding_size] of encoder
      embedding vectors.
    label_vectors: 2D Tensor shape [label_vocab_size, embedding_size] of encoder
      label embedding vectors.
    transition_state_map: Constant 1D int Tensor size output_vocab_size. Maps
      each word to its transition state. 
    use_bidirectional_encoder: Boolean; predict output for each encoder input 
      token, no seperate decoder.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (logits, outputs, pointer_logits, label_logits,
    label_outputs, state), where:
      logits: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        output logits.
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_cell_size] (if output_projection is not None)
        containing the outputs that are fed to the loss function.
        Also fed to loss function.
      label_logits: List of the same length as encoder_inputs for label logits.
      label_outputs: List of the same length as encoder_inputs for label output
        vectors for loss function.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope(scope or "embedding_attention_seq2seq"):
    assert word_vectors is not None
    encoder_input_size = fw_cell.output_size
    encoder_output_size = fw_cell.output_size
    
    b = tf.get_variable("input_proj_b", [encoder_input_size])
    emb_layer = [b for _ in encoder_inputs]

    for input_type in encoder_inputs[0].iterkeys():
      # Defines encoder input projection layer.
      w = tf.get_variable("input_proj_w_{0}".format(input_type),  
          [encoder_embedding_sizes[input_type], encoder_input_size],
          initializer=tf.uniform_unit_scaling_initializer())
      for i, encoder_input in enumerate(encoder_inputs):
        emb_inp = tf.nn.embedding_lookup(word_vectors[input_type], 
            encoder_input[input_type])
        # Linear combination of the inputs.
        emb_layer[i] = tf.add(emb_layer[i], tf.matmul(emb_inp, w))

    if use_bidirectional_encoder:
      # Encoder state is final backward state.
      encoder_outputs, _, encoder_state = tf.nn.bidirectional_rnn(fw_cell,
          bw_cell, emb_layer, dtype=dtype, scope="embedding_encoder")
      encoder_output_size *= 2
    else:
      encoder_outputs, encoder_state = tf.nn.rnn(
          fw_cell, emb_layer, dtype=dtype)

    if decoder_type == data_utils.NO_ATTENTION_DECODER_STATE:
      attention_states = None
    else:
      # First calculate a concatenation of encoder outputs to put attention on.
      top_states = [tf.reshape(e, [-1, 1, encoder_output_size])
                    for e in encoder_outputs]
      attention_states = tf.concat(1, top_states)

    # Decoder.
    if isinstance(feed_previous, bool):
      logits, state = embedding_attention_decoder(
          decoder_type, decoder_inputs, encoder_inputs,
          encoder_state, attention_states, dec_cell, dec_aux_cell, dec_mem_cell,
          decoder_vocab_sizes, decoder_embedding_sizes, 
          decoder_restrictions, num_heads=num_heads,
          predict_span_end_pointers=predict_span_end_pointers,
          output_projections=output_projections,
          transition_state_map=transition_state_map,
          encoder_decoder_vocab_map=encoder_decoder_vocab_map,
          feed_previous=feed_previous, 
          initial_state_attention=initial_state_attention)
      return logits, state

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with tf.variable_scope(tf.get_variable_scope(),
                                         reuse=reuse):
        logits, state = embedding_attention_decoder(
            decoder_type, decoder_inputs, encoder_inputs, 
            encoder_state, attention_states, dec_cell, dec_aux_cell,
            dec_mem_cell, decoder_vocab_sizes, decoder_embedding_sizes, 
            decoder_restrictions, num_heads=num_heads,
            predict_span_end_pointers=predict_span_end_pointers,
            output_projections=output_projections,
            transition_state_map=transition_state_map,
            encoder_decoder_vocab_map=encoder_decoder_vocab_map,
            feed_previous=feed_previous_bool, 
            update_embedding_for_previous=not feed_previous_bool,
            initial_state_attention=initial_state_attention)
        return [logits, state] 

    outputs_and_state = tf.cond(feed_previous, lambda: decoder(True),
                                               lambda: decoder(False))
    return outputs_and_state[0], outputs_and_state[1]


def sequence_loss_by_example(key, logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
    log_perp_list = []
    weight_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if key == "parse" or key == "att" or key == "endatt":
        weight_key = "parse"
      elif key == "ind":
        weight_key = "ind"
      else:  
        weight_key = "predicate"
      crossent = softmax_loss_function(logit[key], target[key])
      log_perp_list.append(crossent * weight[weight_key])
      weight_list.append(weight[weight_key])
    log_perps = tf.add_n(log_perp_list)
    total_size = tf.add_n(weight_list)
    total_size += 1e-12  # just to avoid division by 0 for all-0 weights
    if average_across_timesteps:
      log_perps /= total_size
  return log_perps, total_size


def sequence_loss(key, logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with tf.name_scope(name, "sequence_loss", logits + targets + weights):
    cost_per_example, total_size = sequence_loss_by_example(key,
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function)
    cost = tf.reduce_sum(cost_per_example)
    total_size = tf.reduce_sum(total_size)
    if average_across_batch and not average_across_timesteps:
      return cost / total_size
    elif average_across_batch:
      batch_size = tf.shape(next(targets[0].itervalues()))[0]
      return cost / tf.cast(batch_size, tf.float32)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, 
                       buckets, seq2seq, forward_only,
                       softmax_loss_function=None,
                       average_across_timesteps=True,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    encoder_label_inputs: A list of Tensors to feed the encoder; second
      seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; 3rd seq2seq input.
    decoder_pointer_inputs: A list of Tensors to feed the decoder; 4th seq2seq
      input.
    label_targets: A list of 1D batch-sized int32 Tensors (desired label sequence).
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    pointer_targets: A list of 1D batch-sized int32 Tensors (desired pointer 
      output sequence).
    pointer_weights: List of 1D batch-sized float-Tensors to weight the pointer
      targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    forward_only: boolean, set True for decoding.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    pointer_softmax_loss_function: Loss function for pointer prediction.
    label_softmax_loss_function: Loss function for label prediction.
    average_across_timesteps: Boolean, if True average the loss for each
      timestep. 
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with tf.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with tf.variable_scope(tf.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_logits, _ = seq2seq(
            encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
        outputs.append(bucket_logits) 

        bucket_targets = targets[:bucket[1]]
        bucket_weights = weights[:bucket[1]]
        
        loss = 0
        for key in bucket_targets[0].iterkeys():
          loss += sequence_loss(key,
            bucket_logits, bucket_targets, bucket_weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function)
        losses.append(loss)

  return outputs, losses

