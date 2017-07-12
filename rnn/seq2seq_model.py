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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import seq2seq
import data_utils

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/pdf/1412.2007v2.pdf
  """

  def __init__(self, buckets, source_vocab_sizes, target_vocab_sizes,
               size, source_embedding_sizes, target_embedding_sizes,
               target_data_types, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, decoder_type, use_lstm=True,
               average_loss_across_timesteps=True,
               forward_only=False, feed_previous=False,
               predict_span_end_pointers=False, use_adam=False, 
               restrict_decoder_structure=False,
               transition_vocab_sets=None,
               transition_state_map=None, encoder_decoder_vocab_map=None,
               use_bidirectional_encoder=False,
               pretrained_word_embeddings=None, word_embeddings=None,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.buckets = buckets
    self.batch_size = batch_size
    self.decoder_type = decoder_type

    self.transition_vocab_sets = transition_vocab_sets
    if transition_state_map is None:
      self.transition_state_map = None
    else:
      self.transition_state_map = tf.constant(transition_state_map)
    self.encoder_decoder_vocab_map = tf.constant(encoder_decoder_vocab_map)
    self.use_stack_decoder = decoder_type == data_utils.STACK_DECODER_STATE
    self.average_loss_across_timesteps = average_loss_across_timesteps
    self.input_keep_prob = tf.placeholder(tf.float32,
        name="input_keep_probability")
    self.output_keep_prob = tf.placeholder(tf.float32,
        name="output_keep_probability")

    if not use_adam:
      self.learning_rate = tf.Variable(
          float(learning_rate), trainable=False, dtype=dtype)
      self.learning_rate_decay_op = self.learning_rate.assign(
          self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    
    self.embedding_weights = {}
    for source_type in source_embedding_sizes.iterkeys():
      self.embedding_weights[source_type] = tf.Variable(
          tf.constant(0.0, shape=[source_vocab_sizes[source_type], 
               source_embedding_sizes[source_type]]),
          trainable=(source_type <> 'em'), 
          name=source_type + "_encoder_embeddings")
      if source_type == 'en':
        assert word_embeddings is not None
        assert source_embedding_sizes['en'] == word_embeddings.shape[1]
        self.embedding_weights['en'].assign(word_embeddings)
      elif source_type == 'em':
        assert pretrained_word_embeddings is not None
        assert source_embedding_sizes['em'] == pretrained_word_embeddings.shape[1]
        self.embedding_weights['em'].assign(pretrained_word_embeddings)
      else:
        init_vectors = np.random.uniform(-np.sqrt(3), np.sqrt(3),
            (source_vocab_sizes[source_type], 
             source_embedding_sizes[source_type]))
        self.embedding_weights[source_type].assign(init_vectors)

    output_projections = {}
    for target_type in target_vocab_sizes.iterkeys():
      vocab_size = target_vocab_sizes[target_type]
      w = tf.get_variable(target_type + "_proj_w", [size, vocab_size],
          initializer=tf.uniform_unit_scaling_initializer(), dtype=dtype)
      w_t = tf.transpose(w)
      b = tf.get_variable(target_type + "_proj_b", [vocab_size], dtype=dtype)
      output_projections[target_type] = (w, b)

    def full_loss(logits, labels):
      labels = tf.reshape(labels, [-1])
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels)

    def full_output_loss(inputs, labels):
      logits = tf.nn.xw_plus_b(inputs, w, b)
      labels = tf.reshape(labels, [-1])
      return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

    softmax_loss_function = full_loss

    def create_cell(use_dropout=True):
      # Create the internal cell for our RNN.
      if use_lstm:
        cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=False,
            state_is_tuple=True, 
            initializer=tf.uniform_unit_scaling_initializer())
      else:
        cell = tf.nn.rnn_cell.GRUCell(size)
      if use_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
            self.input_keep_prob, self.output_keep_prob)
      return cell

    with tf.variable_scope("encoder_fw"):
      fw_cell = create_cell()
    with tf.variable_scope("encoder_bw"):
      bw_cell = create_cell()

    with tf.variable_scope("decoder_main"):
      dec_cell = create_cell()
    with tf.variable_scope("decoder_aux"):
      dec_aux_cell = create_cell(False)
    if self.decoder_type == data_utils.MEMORY_STACK_DECODER_STATE:
      with tf.variable_scope("decoder_lin_mem"):
        dec_mem_cell = create_cell()
    else:
      dec_mem_cell = None

    self.decoder_restrictions = []
    num_decoder_restrictions = 0
    if restrict_decoder_structure:
      num_decoder_restrictions = data_utils.NUM_TR_STATES
    for i in xrange(num_decoder_restrictions):
      self.decoder_restrictions.append(tf.placeholder(tf.int32, shape=[None],
          name="restrictions{0}".format(i)))

    if self.transition_vocab_sets is None:
      self.decoder_transition_map = None
    else:
      self.decoder_transition_map = data_utils.construct_transition_map(
          self.transition_vocab_sets, False)   

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return seq2seq.embedding_attention_seq2seq(self.decoder_type,
          encoder_inputs, decoder_inputs, 
          fw_cell, bw_cell, dec_cell, dec_aux_cell, dec_mem_cell,
          source_vocab_sizes, target_vocab_sizes, source_embedding_sizes,
          target_embedding_sizes, 
          predict_span_end_pointers=predict_span_end_pointers,
          decoder_restrictions=self.decoder_restrictions,
          output_projections=output_projections,
          word_vectors=self.embedding_weights,
          transition_state_map=self.transition_state_map,
          encoder_decoder_vocab_map=self.encoder_decoder_vocab_map,
          use_bidirectional_encoder=use_bidirectional_encoder,
          feed_previous=do_decode, dtype=dtype)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    # For now assume that we only have embedding inputs, and single sequence
    # of target weights.

    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append({})
      for key in source_vocab_sizes.iterkeys():
        self.encoder_inputs[-1][key] = tf.placeholder(tf.int32, shape=[None],
            name="encoder_{0}_{1}".format(key, i))

    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append({})
      for key in target_data_types:
        self.decoder_inputs[-1][key] = tf.placeholder(tf.int32, shape=[None],
            name="decoder_{0}_{1}".format(key, i))

    for i in xrange(buckets[-1][1] + 1):
      self.target_weights.append({})
      for key in target_data_types:
        if key == "parse" or key == "predicate" or key == "ind":
          self.target_weights[-1][key] = tf.placeholder(dtype, shape=[None],
              name="weight_{0}_{1}".format(key, i))
    
    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]
    
    # Training outputs and losses.
    self.outputs, self.losses = seq2seq.model_with_buckets(
        self.encoder_inputs, self.decoder_inputs, targets, 
        self.target_weights, buckets,
        lambda x, y: seq2seq_f(x, y, feed_previous), forward_only,
        softmax_loss_function=softmax_loss_function,
        average_across_timesteps=self.average_loss_across_timesteps)
    
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      if use_adam:
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-02)
      else:
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        if max_gradient_norm > 0:
          clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                           max_gradient_norm)
          self.gradient_norms.append(norm)
          self.updates.append(opt.apply_gradients(
              zip(clipped_gradients, params), global_step=self.global_step))
        else:
          self.gradient_norms.append(tf.zeros([1]))
          self.updates.append(opt.apply_gradients(
              zip(gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only, input_keep_prob=1.0, output_keep_prob=1.0, 
           decoder_vocab=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of enconder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs["en"]) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs["en"]), encoder_size))
    if len(decoder_inputs["parse"]) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs["parse"]), decoder_size))
    if len(target_weights["parse"]) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights["parse"]), decoder_size))
    
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      for key in encoder_inputs.iterkeys():
        input_feed[self.encoder_inputs[l][key].name] = encoder_inputs[key][l]

    for l in xrange(decoder_size):
      for key in decoder_inputs.iterkeys():
        input_feed[self.decoder_inputs[l][key].name] = decoder_inputs[key][l]
      for key in target_weights.iterkeys():
        input_feed[self.target_weights[l][key].name] = target_weights[key][l]
 
    # Since our targets are decoder inputs shifted by one, we need one more.
    for key in decoder_inputs.iterkeys():
      last_target = self.decoder_inputs[decoder_size][key].name
      input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
     
    if self.decoder_restrictions:
      assert self.transition_vocab_sets is not None
      if len(self.decoder_restrictions) == 1:
        assert decoder_vocab is not None
        decoder_restrictions = [list(decoder_vocab.union(*self.transition_vocab_sets[1:]))]
      else:
        assert len(self.decoder_restrictions) == data_utils.NUM_TR_STATES
        decoder_restrictions = self.decoder_transition_map

      for l in xrange(len(decoder_restrictions)):
        input_feed[self.decoder_restrictions[l].name] = np.array(
            decoder_restrictions[l], dtype=int)

    # Add dropout probabilities to input feed.
    assert input_keep_prob >= 0.0 and input_keep_prob <= 1.0
    input_feed[self.input_keep_prob.name] = input_keep_prob
    assert output_keep_prob >= 0.0 and output_keep_prob <= 1.0
    input_feed[self.output_keep_prob.name] = output_keep_prob

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],
                     self.losses[bucket_id]]
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Dicts of output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      # Gradient norm, loss, no outputs.
      return outputs[1], outputs[2], None
    else:
      # No gradient norm, loss, outputs.
      return None, outputs[0], outputs[1:decoder_size+1]

  def get_batch(self, data, data_types, bucket_id, batch_number, 
      singleton_keep_prob=1.0, singleton_sets=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the btch for.
      batch_number: integer, which batch in the bucket to get.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = {}, {}

    for key in data_types[0]:
      encoder_inputs[key] = []
    for key in data_types[1]:
      decoder_inputs[key] = []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for batch_pos in xrange(self.batch_size):
      if batch_number == -1:
        encoder_input_data, decoder_input_data = random.choice(data[bucket_id])
      else:
      # input_data is list (over types) of sequences
        encoder_input_data, decoder_input_data = \
          data[bucket_id][min(self.batch_size*batch_number + batch_pos,
            len(data[bucket_id])-1)]

      for k, key in enumerate(data_types[0]):
        encoder_input = encoder_input_data[k]

        # Keep or replace all singletons per sentence.
        if (singleton_keep_prob > 0 and singleton_sets is not None 
            and singleton_sets.has_key(key)):
          for i in xrange(len(encoder_input)):
            if encoder_input[i] in singleton_sets[key]:
              unk_singletons = (singleton_keep_prob < 1.0 
                  and random.random() > singleton_keep_prob)
              if unk_singletons:
                encoder_input[i] = data_utils.UNK_ID

        # Encoder inputs are padded.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs[key].append(list(encoder_input + encoder_pad))

      for k, key in enumerate(data_types[1]):
        #decoder_input = [] # TODO for batch evaluation
        decoder_input = decoder_input_data[k]
        if key == "att" or key == "endatt":
          decoder_input = [min(inp, encoder_size - 1) for inp in decoder_input]
                
        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input)
        decoder_inputs[key].append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * (decoder_pad_size - 1))

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = {}, {}, {}

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for k, key in enumerate(data_types[0]):
      batch_encoder_inputs[key] = []
      for length_idx in xrange(encoder_size):
        batch_encoder_inputs[key].append(
            np.array([encoder_inputs[key][batch_idx][length_idx]
                     for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for k, key in enumerate(data_types[1]):
      batch_decoder_inputs[key] = []
      if key == "parse" or key == "predicate" or key == "ind":
        batch_weights[key] = []

      for length_idx in xrange(decoder_size):
        batch_input = np.array([decoder_inputs[key][batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32)
        # Remove -1 indexes
        if key == "start" or key == "end" or key == "ind":
          batch_input = np.maximum(batch_input, np.zeros(self.batch_size,
                                                         dtype=np.int32))
        batch_decoder_inputs[key].append(batch_input)
        # target weights customized for certain keys. 
        if key == "parse" or key == "predicate" or key == "ind":
          # Create target_weights to be 0 for targets that are padding.
          batch_weight = np.ones(self.batch_size, dtype=np.float32)
          for batch_idx in xrange(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
              target = decoder_inputs["parse"][batch_idx][length_idx + 1]
            if (length_idx == decoder_size - 1 or target == data_utils.PAD_ID
                or (key == "predicate" and target == data_utils.REDUCE_ID)
                or (key == "ind" and target <> data_utils.OPEN_ID 
                    and target <> data_utils.CLOSE_ID)):
              batch_weight[batch_idx] = 0.0

          batch_weights[key].append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

