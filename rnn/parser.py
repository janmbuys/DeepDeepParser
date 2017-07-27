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

"""Binary for training transducer models and decoding from them.

Running this program without --decode will find the training data in
the directory specified as --data_dir and then start training a model saving
checkpoints to --train_dir. Following neural MT convension, the input is
referred to as English (en) and the output as French (fr).

Running with --decode starts an interactive loop so you can see how
the current checkpoint performs the transduction.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/pdf/1412.2007v2.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

tf.app.flags.DEFINE_string("data_dir", "parsing-data", "Data directory")
tf.app.flags.DEFINE_string("embedding_vectors", "vectors.en", "Pre-trained word embeddings")
tf.app.flags.DEFINE_string("train_dir", "working", "Training directory.")
tf.app.flags.DEFINE_string("train_name", "train", 
                           "Training set file name.")
tf.app.flags.DEFINE_string("dev_name", "dev", "Dev set file name.")
tf.app.flags.DEFINE_string("checkpoint_file", "",
                           "Checkpoint file name for decoding.")
tf.app.flags.DEFINE_float("gpu_memory_fraction", 1.0, 
                          "GPU memory fraction.")

tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("decode_dev", False,
                            "Set to True to decode dev set.")
tf.app.flags.DEFINE_boolean("score_only", False,
                            "Score pairs instead of decoding.")
tf.app.flags.DEFINE_boolean("decode_train", False,
                            "Set to True to decode the train set.")
tf.app.flags.DEFINE_boolean("eval_dev", False,
                            "Set to True to evaluate the dev set ppl.")
tf.app.flags.DEFINE_boolean("eval_train", False,
                            "Set to True to evaluate the train set ppl.")
tf.app.flags.DEFINE_boolean("batch_decode", False,
                            "Set to True to evaluate in batches.")

tf.app.flags.DEFINE_boolean("use_adam", True,
                            "Use Adam instead of pure SGD.")
tf.app.flags.DEFINE_boolean("use_lstm", True,
                            "Set to True for LSTM cells instead of GRU cells.")

tf.app.flags.DEFINE_boolean("train_per_epoch", True,
                            "Pass through all training data in an epoch.")
tf.app.flags.DEFINE_integer("epochs_per_checkpoint", 10,
                            "How many training epochs to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("num_train_eval_batches", 20,
                            "How many training batches to decode.")

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much, for SGD.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("input_drop_prob", 0.0,
                          "Dropout on input.")
tf.app.flags.DEFINE_float("output_drop_prob", 0.0,
                          "Dropout on output.")
tf.app.flags.DEFINE_float("singleton_keep_prob", 0.5,
                          "Probability to include encoder singletons in vocabs.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("input_embedding_size", 256,
                            "Size of input embedding layer.")
tf.app.flags.DEFINE_integer("output_embedding_size", 128,
                            "Size of output embedding layer.")
tf.app.flags.DEFINE_integer("pretrained_embedding_size", 100,
                            "Size of pretrained input embedding.")
tf.app.flags.DEFINE_integer("tag_embedding_size", 32,
                            "Size of input tag embeddings.")

tf.app.flags.DEFINE_boolean("use_stack_decoder", False,
                            "Use stack LSTM decoder.")
tf.app.flags.DEFINE_boolean("use_pure_stack_decoder", False,
                            "Use pure stack LSTM decoder.")
tf.app.flags.DEFINE_boolean("use_memory_stack_decoder", False,
                            "Use pure memory stack LSTM decoder.")

tf.app.flags.DEFINE_boolean("no_attention", False,
                            "Set to True to disable attention.")
tf.app.flags.DEFINE_boolean("use_parent_feed_decoder", False,
                            "Use parent feeding for LSTM decoding.")
tf.app.flags.DEFINE_boolean("use_forced_attention_decoder", False,
                            "Use mixed output pointer-network.")
tf.app.flags.DEFINE_boolean("predict_span_end_pointers", False,
                            "Use pointer decoder with span end prediction.")
tf.app.flags.DEFINE_boolean("use_linear_point_decoder", False,
                            "Use linear decoder with pointers.")
tf.app.flags.DEFINE_boolean("use_linear_feed_point_decoder", False,
                            "Use linear decoder with pointers and feeding.")

tf.app.flags.DEFINE_boolean("use_hard_attention_decoder", False,
                            "Use hard attention linear decoder.")
tf.app.flags.DEFINE_boolean("use_hard_attention_parent_feed_decoder", False,
                            "Use hard attention parent feed (amr) linear decoder.")
tf.app.flags.DEFINE_boolean("use_hard_attention_arc_eager_decoder", False,
                            "Use hard attention decoder with arc eager transition features.")

tf.app.flags.DEFINE_boolean("restrict_decoder_structure", False,
                            "Enforce decoder output structure to be well-formed.")
tf.app.flags.DEFINE_boolean("offset_pointers", False,
                            "Offset pointer positions by 1.")
tf.app.flags.DEFINE_boolean("use_encoder_tags", False,
                            "Use input POS and NE label features in encoder.")
tf.app.flags.DEFINE_boolean("use_bidirectional_encoder", True,
                            "Use bidirectional encoder.")
tf.app.flags.DEFINE_boolean("initialize_word_vectors", False,
                            "Initialize with external input word vectors.")
tf.app.flags.DEFINE_boolean("use_pretrained_word_vectors", False,
                            "Use fixed external input word vectors.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

def read_mrs_source_data(buckets, source_paths, max_size=None,
    any_length=False, offset_target=-1):
  # Read in only input files (for decoding)
  source_inputs = [data_utils.read_ids_file(path, max_size) 
                   for path in source_paths]
  
  data_set = [[] for _ in buckets]
  data_list = []
  # Assume everything is well-aligned.
  for i in xrange(len(source_inputs[0])): # over examples
    # List of sequences of each type.
    source_ids = [source_input[i] for source_input in source_inputs]
    # Assume first target type predicts EOS.

    found_bucket = False
    for bucket_id, (source_size, target_size) in enumerate(buckets):
      if len(source_ids[0]) < source_size:
        data_set[bucket_id].append([source_ids, []])
        data_list.append([source_ids, [], bucket_id])
        found_bucket = True
        break
    if any_length and not found_bucket:
      # Crop examples that are larger than the largest bucket.
      source_size, target_size = buckets[-1][0], buckets[-1][1]
      if len(source_ids[0]) >= source_size:
        source_ids = [source_id[:source_size] for source_id in source_ids]
      bucket_id = len(buckets) - 1
      data_set[bucket_id].append([source_ids, []])
      data_list.append([source_ids, [], bucket_id])
  return data_set, data_list


def read_mrs_data(buckets, source_paths, target_paths, max_size=None,
    any_length=False, offset_target=-1):
  # Read in all files seperately.
  source_inputs = [data_utils.read_ids_file(path, max_size) 
                   for path in source_paths]
  target_inputs = [data_utils.read_ids_file(path, max_size) 
                   for path in target_paths]
  
  data_set = [[] for _ in buckets]
  data_list = []
  # Assume everything is well-aligned.
  for i in xrange(len(source_inputs[0])): # over examples
    # List of sequences of each type.
    source_ids = [source_input[i] for source_input in source_inputs]
    # Assume first target type predicts EOS.
    # Not checking pointer ranges: do that inside tf graph.
    target_ids = [target_inputs[0][i] + [data_utils.EOS_ID]]
    for j, target_input in enumerate(target_inputs[1:]):
      if offset_target > 0 and j + 1 == offset_target:
        target_ids.append([data_utils.PAD_ID] + target_input[i] 
                          + [data_utils.PAD_ID])
      else:
        target_ids.append(target_input[i] + [data_utils.PAD_ID])

    found_bucket = False
    for bucket_id, (source_size, target_size) in enumerate(buckets):
      if len(source_ids[0]) < source_size and len(target_ids[0]) < target_size:
        data_set[bucket_id].append([source_ids, target_ids])
        data_list.append([source_ids, target_ids, bucket_id])
        found_bucket = True
        break
    if any_length and not found_bucket:
      # Crop examples that are larger than the largest bucket.
      source_size, target_size = buckets[-1][0], buckets[-1][1]
      if len(source_ids[0]) >= source_size:
        source_ids = [source_id[:source_size] for source_id in source_ids]
      if len(target_ids[0]) >= target_size:
        target_ids = [target_id[:target_size] for target_id in target_ids]
      bucket_id = len(buckets) - 1
      data_set[bucket_id].append([source_ids, target_ids])
      data_list.append([source_ids, target_ids, bucket_id])
  return data_set, data_list


def construct_embeddings(word_vectors, vocab, embedding_size):
  embedding = np.random.uniform(-np.sqrt(3), np.sqrt(3), (len(vocab),
      embedding_size))
  if word_vectors is not None:
    for word, vector in word_vectors.iteritems():
      if vocab.has_key(word):
        embedding[vocab[word],:len(vector)] = vector
  return embedding


def create_model(buckets, session, forward_only, source_vocab_sizes,
    target_vocab_sizes, source_embedding_sizes, target_embedding_sizes,
    target_data_types, pretrained_word_embeddings, word_embeddings, 
    encoder_decoder_vocab_map=None,
    restricted_vocab_sets=None, restricted_state_map=None):
  """Create translation model and initialize or load parameters in session."""
  average_loss_across_timesteps = False
  for bucket in buckets:
    assert bucket[1] < data_utils.MAX_OUTPUT_SIZE
  decoder_type = data_utils.ATTENTION_DECODER_STATE
  if FLAGS.no_attention:
    decoder_type = data_utils.NO_ATTENTION_DECODER_STATE
  elif FLAGS.use_linear_point_decoder:
    decoder_type = data_utils.LINEAR_POINTER_DECODER_STATE
  elif FLAGS.use_linear_feed_point_decoder:
    decoder_type = data_utils.LINEAR_FEED_POINTER_DECODER_STATE
  elif FLAGS.use_hard_attention_decoder:
    decoder_type = data_utils.HARD_ATTENTION_DECODER_STATE
  elif FLAGS.use_hard_attention_parent_feed_decoder:
    decoder_type = data_utils.HARD_ATTENTION_PARENT_FEED_DECODER_STATE
  elif FLAGS.use_hard_attention_arc_eager_decoder:
    decoder_type = data_utils.HARD_ATTENTION_ARC_EAGER_DECODER_STATE
  elif FLAGS.use_parent_feed_decoder:
    decoder_type = data_utils.PARENT_FEED_DECODER_STATE
  elif FLAGS.use_pure_stack_decoder:
    decoder_type = data_utils.PURE_STACK_DECODER_STATE
    assert FLAGS.restrict_decoder_structure
  elif FLAGS.use_memory_stack_decoder:
    decoder_type = data_utils.MEMORY_STACK_DECODER_STATE
    assert FLAGS.restrict_decoder_structure
  elif FLAGS.use_stack_decoder:
    decoder_type = data_utils.STACK_DECODER_STATE
    assert FLAGS.restrict_decoder_structure
 
  feed_previous = False if FLAGS.score_only else forward_only
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      buckets, source_vocab_sizes, target_vocab_sizes,
      FLAGS.size, source_embedding_sizes, target_embedding_sizes,
      target_data_types,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, 
      decoder_type, use_lstm=FLAGS.use_lstm, 
      average_loss_across_timesteps=average_loss_across_timesteps, 
      forward_only=forward_only, 
      feed_previous=feed_previous,
      predict_span_end_pointers=FLAGS.predict_span_end_pointers,
      use_adam=FLAGS.use_adam, 
      restrict_decoder_structure=FLAGS.restrict_decoder_structure, 
      transition_vocab_sets=restricted_vocab_sets, 
      transition_state_map=restricted_state_map, 
      encoder_decoder_vocab_map=encoder_decoder_vocab_map,
      use_bidirectional_encoder=FLAGS.use_bidirectional_encoder,
      pretrained_word_embeddings=pretrained_word_embeddings,
      word_embeddings=word_embeddings,
      dtype=dtype)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    if FLAGS.checkpoint_file:
      checkpoint_path = FLAGS.train_dir + "/" + FLAGS.checkpoint_file
    else:
      checkpoint_path = ckpt.model_checkpoint_path
    print("Reading model parameters from %s" % checkpoint_path)
    model.saver.restore(session, checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def decode_mrs_dev(model, sess, dev_list, data_types, rev_vocabs, dev_name,
    write_output_gold, restricted_vocab_sets=None, restricted_vocab_dict=None,
    score_only=False):
  dev_output_path = os.path.join(FLAGS.train_dir, dev_name + ".output")
  dev_gold_path = os.path.join(FLAGS.train_dir, dev_name + ".gold")

  model.batch_size = 1
  eval_loss = 0
  eval_loss2 = 0
  total_out_size = 0.0
  total_diff_size = 0.0
  prop_output_lengths = [len(pair[1]) + 0.0 for pair in dev_list]
  prop_output_lengths = [l / (sum(prop_output_lengths) + 0.001)
                         for l in prop_output_lengths]
  output_lines = {}
  gold_output_lines = {}
  for key in data_types[1]:
    output_lines[key] = []  
    gold_output_lines[key] = []  
  output_scores = []

  for i, pair_ids in enumerate(dev_list):
    bucket_id = pair_ids[2]
    # Decode one sentence at a time.
    if score_only:
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(pair_ids[0], pair_ids[1])]}, 
          data_types, bucket_id, 0)
      decoder_vocab = None
      _, step_loss, output_logits = model.step(sess, encoder_inputs, 
          decoder_inputs, target_weights,
      bucket_id, True, decoder_vocab=decoder_vocab)
      print("step loss %.2f" % (step_loss))
      step_loss_str = "%.9f" % step_loss
      output_scores.append(step_loss_str)
    else:
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(pair_ids[0], [[]]*len(data_types[1]))]}, 
          data_types, bucket_id, 0)

      if restricted_vocab_dict is None:
        decoder_vocab = None
      else:
        decoder_vocab = data_utils.extract_decoder_vocab(pair_ids[0],
                restricted_vocab_dict)

      _, step_loss, output_logits = model.step(sess, encoder_inputs, 
          decoder_inputs, target_weights,
      bucket_id, True, decoder_vocab=decoder_vocab)

      outputs = {}        
      for target_key in data_types[1]:
        outputs[target_key] = [int(np.argmax(logit[target_key], axis=1))
                               for logit in output_logits]

      eval_loss += step_loss * prop_output_lengths[i]
      eval_loss2 += step_loss / len(dev_list)

      # If there is an EOS or PAD symbol in output, cut them at that point.
      if data_utils.EOS_ID in outputs["parse"]:
        eos_ind = outputs["parse"].index(data_utils.EOS_ID)
        for key in data_types[1]:
          outputs[key] = outputs[key][:eos_ind]
      elif data_utils.PAD_ID in outputs["parse"]:
        pad_ind = outputs["parse"].index(data_utils.PAD_ID)
        for key in data_types[1]:
          outputs[key] = outputs[key][:pad_ind]

      for key in data_types[1]:
        if rev_vocabs.has_key(key):
          out_line = " ".join(
              [rev_vocabs[key][token] for token in outputs[key]])
        else:
          out_line = " ".join([str(token) for token in outputs[key]])
        output_lines[key].append(out_line)
      
      if write_output_gold:
        gold_outputs = {}
        for k, key in enumerate(data_types[1]):
          gold_outputs[key] = pair_ids[1][k]

        # If there is an EOS or PAD symbol in output, cut them at that point.
        if data_utils.EOS_ID in gold_outputs["parse"]:
          eos_ind = gold_outputs["parse"].index(data_utils.EOS_ID)
          for key in data_types[1]:
            gold_outputs[key] = gold_outputs[key][:eos_ind]
        elif data_utils.PAD_ID in outputs["parse"]:
          pad_ind = gold_outputs["parse"].index(data_utils.PAD_ID)
          for key in data_types[1]:
            gold_outputs[key] = gold_outputs[key][:pad_ind]

          for key in data_types[1]:
            if rev_vocabs.has_key(key):
              out_line = " ".join([rev_vocabs[key][token] for token in gold_outputs[key]])
            else:
              out_line = " ".join([str(token) for token in gold_outputs[key]])
            gold_output_lines[key].append(out_line)

  if score_only:
    data_utils.write_output_file(dev_output_path + ".scores", output_scores)
  else:   
    for key in data_types[1]:
      data_utils.write_output_file(dev_output_path + "." + key, output_lines[key])
      if write_output_gold:
        data_utils.write_output_file(dev_gold_path + "." + key, gold_output_lines[key])

  eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
  eval_ppx2 = math.exp(eval_loss2) if eval_loss < 300 else float('inf')
  print("  dev perplexity %.2f" % (eval_ppx))
  print("  dev perplexity (2) %.2f" % (eval_ppx2))
  model.batch_size = FLAGS.batch_size # restore batch size

#TODO get this to actually work
def decode_batch_mrs_dev(model, sess, dev_list, data_types, rev_vocabs, 
    dev_name, write_output_gold, buckets, restricted_vocab_sets=None, 
    restricted_vocab_dict=None, score_only=False):
  dev_output_path = os.path.join(FLAGS.train_dir, dev_name + ".output")
  dev_gold_path = os.path.join(FLAGS.train_dir, dev_name + ".gold")

  model.batch_size = FLAGS.batch_size
  eval_loss = 0
  eval_loss2 = 0
  total_out_size = 0.0
  total_diff_size = 0.0
  prop_output_lengths = [len(pair[1]) + 0.0 for pair in dev_list]
  prop_output_lengths = [l / sum(prop_output_lengths)
                         for l in prop_output_lengths]
  output_lines = {}
  gold_output_lines = {}
  for key in data_types[1]:
    output_lines[key] = []  
    gold_output_lines[key] = []  
  output_scores = []

  bucket_id = len(buckets) - 1 
  # Place all in the same bucket:
  dev_set = {}
  for i in xrange(len(buckets)):
    dev_set[i] = []

  bucket_id_list = []
  for pair_ids in dev_list:
    dev_set[bucket_id].append((pair_ids[0], []*len(data_types[1]))) 
    bucket_id_list.append(pair_ids[2])

  batch_total = int(math.ceil((len(dev_list)+0.0)/FLAGS.batch_size))

  for i in xrange(batch_total):
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        dev_set,
        data_types, bucket_id, i)

    if restricted_vocab_dict is None:
      decoder_vocab = None
    else:
      decoder_vocab = data_utils.extract_decoder_vocab(pair_ids[0],
              restricted_vocab_dict)
    batch_bucket_id = max(bucket_id_list[i*FLAGS.batch_size:min((i+1)*FLAGS.batch_size, len(bucket_id_list))])

    _, step_loss, output_logits = model.step(sess, encoder_inputs, 
        decoder_inputs, target_weights,
    batch_bucket_id, True, decoder_vocab=decoder_vocab)

    outputs = {}        
    for target_key in data_types[1]:
      outputs[target_key] = [int(np.argmax(logit[target_key][0], axis=1))
                             for logit in output_logits]

    eval_loss += step_loss * prop_output_lengths[i]
    eval_loss2 += step_loss / len(dev_list)

    # If there is an EOS or PAD symbol in output, cut them at that point.
    if data_utils.EOS_ID in outputs["parse"]:
      eos_ind = outputs["parse"].index(data_utils.EOS_ID)
      for key in data_types[1]:
        outputs[key] = outputs[key][:eos_ind]
    elif data_utils.PAD_ID in outputs["parse"]:
      pad_ind = outputs["parse"].index(data_utils.PAD_ID)
      for key in data_types[1]:
        outputs[key] = outputs[key][:pad_ind]

    for key in data_types[1]:
      if rev_vocabs.has_key(key):
        out_line = " ".join([rev_vocabs[key][token] for token in outputs[key]])
      else:
        out_line = " ".join([str(token) for token in outputs[key]])
      output_lines[key].append(out_line)
    
  for key in data_types[1]:
    data_utils.write_output_file(dev_output_path + "." + key, output_lines[key])

  eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
  eval_ppx2 = math.exp(eval_loss2) if eval_loss < 300 else float('inf')
  print("  dev perplexity %.2f" % (eval_ppx))
  print("  dev perplexity (2) %.2f" % (eval_ppx2))


def get_data_types():
  source_data_types = ["en"]
  vocab_data_types = ["en"]
  if FLAGS.use_encoder_tags:
    source_data_types += ["ne", "pos"]
    vocab_data_types += ["ne", "pos"]
  if FLAGS.use_pretrained_word_vectors:
    source_data_types += ["em"]
    vocab_data_types += ["em"]

  copy_data_types = []
  vocab_data_types += ["parse"]
  target_data_types = ["parse"]

  if (FLAGS.use_linear_point_decoder
        or FLAGS.use_linear_feed_point_decoder
        or FLAGS.use_hard_attention_decoder 
        or FLAGS.use_hard_attention_arc_eager_decoder
        or FLAGS.use_hard_attention_parent_feed_decoder):
    target_data_types += ["att"]
    copy_data_types = ["att"]
    if FLAGS.predict_span_end_pointers:
      target_data_types.append("endatt")
      copy_data_types.append("endatt")

  return source_data_types, target_data_types, vocab_data_types, copy_data_types

"""Split source and target vocab sizes, init embedding sizes."""
def get_vocab_embed_sizes(vocab_data_types, source_data_types, 
    target_data_types, vocabs):
  source_vocab_sizes = {} 
  target_vocab_sizes = {}
  source_embedding_sizes = {}
  target_embedding_sizes = {}

  for data_type in vocab_data_types:
    if data_type in source_data_types:
      source_vocab_sizes[data_type] = len(vocabs[data_type])
      if data_type == 'en':
        source_embedding_sizes[data_type] = FLAGS.input_embedding_size
      elif data_type == 'em':
        source_embedding_sizes[data_type] = FLAGS.pretrained_embedding_size
      else:
        source_embedding_sizes[data_type] = FLAGS.tag_embedding_size
    elif data_type in target_data_types:
      target_vocab_sizes[data_type] = len(vocabs[data_type])
      target_embedding_sizes[data_type] = FLAGS.output_embedding_size
  return source_vocab_sizes, target_vocab_sizes, source_embedding_sizes, target_embedding_sizes


def train():
  """Train an MRS transducer model using the given data."""
  # Prepare MRS data.
  print("Preparing MRS data in %s" % (FLAGS.data_dir))

  source_data_types, target_data_types, vocab_data_types, copy_data_types = get_data_types()
  data_types = [source_data_types, target_data_types]

  if FLAGS.use_pretrained_word_vectors or FLAGS.initialize_word_vectors:
    word_vector_path = FLAGS.embedding_vectors
    word_vector_vocab_path = os.path.join(FLAGS.data_dir, "vocab.em")
    word_vectors = data_utils.read_word_vectors(word_vector_path,
        word_vector_vocab_path)

  # Prepare training data.
  vocab_paths = []
  singleton_vocab_paths = []

  for data_type in vocab_data_types:
    _, vocab_path, singleton_vocab_path = data_utils.prepare_mrs_data(
        FLAGS.data_dir, FLAGS.data_dir, data_type, FLAGS.train_name, True)
    vocab_paths.append(vocab_path)
    singleton_vocab_paths.append(singleton_vocab_path)
 
  # Prepare dev data.
  for data_type in vocab_data_types:
    data_utils.prepare_mrs_data(FLAGS.data_dir, FLAGS.data_dir, data_type, 
        FLAGS.dev_name, False)

  # Read in bucket sizes. 
  bucket_path = os.path.join(FLAGS.data_dir, "buckets")
  buckets = data_utils.read_buckets(bucket_path)

  # Read data into buckets and compute their sizes.
  print ("Reading development and training data.")
  source_ids_data_types = [data_type + ".ids" 
                           for data_type in source_data_types]
  target_ids_data_types = [(data_type + ".ids" if data_type in vocab_data_types
      else data_type) for data_type in target_data_types]

  train_source_paths = [(FLAGS.data_dir + "/" + FLAGS.train_name + "." 
                         + data_type) for data_type in source_ids_data_types]
  train_target_paths = [(FLAGS.data_dir + "/" + FLAGS.train_name + "." 
                         + data_type) for data_type in target_ids_data_types]
  dev_source_paths = [(FLAGS.data_dir + "/" + FLAGS.dev_name + "." 
                         + data_type) for data_type in source_ids_data_types]
  dev_target_paths = [(FLAGS.data_dir + "/" + FLAGS.dev_name + "." 
                         + data_type) for data_type in target_ids_data_types]
  
  if FLAGS.use_linear_point_decoder and FLAGS.offset_pointers:
    offset_target = target_data_types.index("att")
  else:
    offset_target = -1

  train_set, train_list = read_mrs_data(buckets, train_source_paths, 
      train_target_paths, offset_target=offset_target)
  dev_set, dev_list = read_mrs_data(buckets, dev_source_paths, 
      dev_target_paths, any_length=True, offset_target=offset_target)
 
  vocabs = {}
  rev_vocabs = {}
  singleton_sets = {}

  # Load vocabularies.
  for i, data_type in enumerate(vocab_data_types):
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_paths[i])
    vocabs[data_type] = vocab
    rev_vocabs[data_type] = rev_vocab
    if data_type <> 'em':
      singleton_vocab, _ = data_utils.initialize_vocabulary(
          singleton_vocab_paths[i])
      singletons = set()
      for word in singleton_vocab.iterkeys():
        if vocab.has_key(word):
          singletons.add(vocab[word]) 
      singleton_sets[data_type] = singletons

  if FLAGS.use_forced_attention_decoder:
    # Read word to concept map.
    input_map_path = os.path.join(FLAGS.data_dir, "concepts.map")
    encoder_decoder_vocab_map = data_utils.encoder_decoder_vocab_map_to_token_ids(
        input_map_path, vocabs["en"] , vocabs["lexeme"])
  else: 
    encoder_decoder_vocab_map = []  

  restricted_vocab_sets = data_utils.id_vocab_sets(vocabs["parse"])
  restricted_state_map = [data_utils.map_restricted_state(word) 
                          for word in rev_vocabs["parse"]]


  source_vocab_sizes, target_vocab_sizes, source_embedding_sizes, target_embedding_sizes = get_vocab_embed_sizes(vocab_data_types, source_data_types, target_data_types, vocabs)

  if FLAGS.initialize_word_vectors:
    word_embeddings = construct_embeddings(word_vectors, vocabs["en"],
        FLAGS.input_embedding_size)
  else:
    word_embeddings = construct_embeddings(None, vocabs["en"],
        FLAGS.input_embedding_size)

  if FLAGS.use_pretrained_word_vectors:
    pretrained_word_embeddings = construct_embeddings(word_vectors,
            vocabs["em"], FLAGS.pretrained_embedding_size)
  else:
    pretrained_word_embeddings = None

  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
  train_total_size = float(sum(train_bucket_sizes))

  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]

  # Number of full batches.
  batch_total = [math.floor(len(train_set[i])/FLAGS.batch_size) 
                 for i in xrange(len(buckets))]
  batches_per_epoch = sum(batch_total)

  current_step = 0
  current_epoch = 0
  previous_losses = []

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  while True:
    with tf.Graph().as_default() as g:
      with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        first_step = True
        # Create model.
        print("Creating layer of %d units." % (FLAGS.size))
        model = create_model(buckets, sess, False, source_vocab_sizes, 
            target_vocab_sizes, source_embedding_sizes, target_embedding_sizes,
            target_data_types, pretrained_word_embeddings, word_embeddings,
            encoder_decoder_vocab_map=encoder_decoder_vocab_map,
            restricted_vocab_sets=restricted_vocab_sets,
            restricted_state_map=restricted_state_map)

        step_time, loss = 0.0, 0.0
        batch_counter = [0 for _ in buckets]
        for s in train_set:
          random.shuffle(s)

        if FLAGS.train_per_epoch:
          steps_per_checkpoint = sum(batch_total)*FLAGS.epochs_per_checkpoint
        else:
          steps_per_checkpoint = FLAGS.steps_per_checkpoint
        print("%d batches per epoch. %d steps per checkpoint." % \
            (batches_per_epoch, steps_per_checkpoint))

        # This is the training loop.
        while first_step or (current_step % steps_per_checkpoint <> 0):
          first_step = False
          # Choose a bucket according to data distribution. We pick a random 
          # number in [0, 1] and use the corresponding interval in 
          # train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])
          if FLAGS.train_per_epoch and \
              batch_counter[bucket_id] >= batch_total[bucket_id]:
            continue

          # Get a batch and make a step.
          start_time = time.time()
          batch_number = batch_counter[bucket_id] if FLAGS.train_per_epoch else -1
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, data_types, bucket_id, 
              batch_number, FLAGS.singleton_keep_prob, singleton_sets)

          _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, 
              target_weights, bucket_id,
              False, 1 - FLAGS.input_drop_prob, 1 - FLAGS.output_drop_prob)
          step_time += (time.time() - start_time) / steps_per_checkpoint
          loss += step_loss / steps_per_checkpoint
          current_step += 1
          batch_counter[bucket_id] += 1

          if FLAGS.train_per_epoch and current_step % batches_per_epoch == 0:
            # Reset batch counters, randomize.
            batch_counter = [0 for _ in buckets]
            current_epoch += 1
            for s in train_set:
              random.shuffle(s)

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            rate = FLAGS.learning_rate if FLAGS.use_adam else model.learning_rate.eval()

            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), rate, step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if not FLAGS.use_adam and len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            if FLAGS.eval_dev: #TODO check if ppl calculation working
              # Estimate dev ppl based on 1 batch per bucket
              eval_loss = 0.0
              prop_output_lengths = []
              for bucket_set in dev_set:
                prop_output_lengths.append(sum(
                    [len(pair[1]) for pair in bucket_set]))
              prop_output_lengths = [l / sum(prop_output_lengths)
                                     for l in prop_output_lengths]

              for bucket_id in xrange(len(buckets)):
                if len(dev_set[bucket_id]) == 0:
                  continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    dev_set, data_types, bucket_id, -1)
                _, step_loss, _ = model.step(sess,
                    encoder_inputs, decoder_inputs, target_weights,
                    bucket_id, True)
                eval_loss += step_loss * prop_output_lengths[bucket_id]
              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  Dev eval: perplexity %.2f" % eval_ppx)
              sys.stdout.flush()

    # Run evals on development set and print their perplexity.
    with tf.Graph().as_default() as gg:
      with tf.Session(graph=gg, config=tf.ConfigProto(gpu_options=gpu_options)) as dev_sess:
        if FLAGS.decode_dev or FLAGS.decode_train:
          # Create the model for decoding.
          print("Creating decoding model.")
          dev_model = create_model(buckets, dev_sess, True, source_vocab_sizes, 
            target_vocab_sizes, source_embedding_sizes, target_embedding_sizes,
            target_data_types, pretrained_word_embeddings, word_embeddings, 
            restricted_vocab_sets=restricted_vocab_sets,
            encoder_decoder_vocab_map=encoder_decoder_vocab_map,
            restricted_state_map=restricted_state_map)

        if FLAGS.decode_train:
          # Decoding a number of training batches, write input and output.
          print("Decoding training data.")
          decode_train_list = []
          for _ in xrange(FLAGS.num_train_eval_batches):
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            batch_number = int(np.random.random_sample()*batch_total[bucket_id])
            for batch_pos in xrange(FLAGS.batch_size):
              inp, outp = train_set[bucket_id][FLAGS.batch_size*batch_number + batch_pos]
              decode_train_list.append([inp, outp, bucket_id])
          decode_mrs_dev(dev_model, dev_sess, decode_train_list,
              data_types, rev_vocabs, 
              FLAGS.train_name + "." + str(current_epoch), 
              True, restricted_vocab_sets)
        if FLAGS.decode_dev:
          print("Decoding " + FLAGS.dev_name)
          decode_mrs_dev(dev_model, dev_sess, dev_list, data_types,
                     rev_vocabs,
                     FLAGS.dev_name + "." + str(current_epoch), 
                     False, restricted_vocab_sets)


def mrs_decode(score_only=False):
  """Decode from a trained MRS transducer model."""  
  # Prepare MRS data.
  print("Preparing MRS data in %s" % (FLAGS.data_dir))

  source_data_types, target_data_types, vocab_data_types, copy_data_types = get_data_types()
  data_types = [source_data_types, target_data_types]

  vocab_paths = [os.path.join(FLAGS.data_dir, "vocab." + data_type)
                 for data_type in vocab_data_types]

  if FLAGS.use_pretrained_word_vectors or FLAGS.initialize_word_vectors:
    word_vector_path = FLAGS.embedding_vectors
    word_vector_vocab_path = os.path.join(FLAGS.data_dir, "vocab.em")
    word_vectors = data_utils.read_word_vectors(word_vector_path,
        word_vector_vocab_path)

  # Prepare dev data 
  for data_type in source_data_types:
    data_utils.prepare_mrs_data(FLAGS.data_dir, FLAGS.data_dir, data_type, 
       FLAGS.dev_name, False)

  # Read in bucket sizes. 
  bucket_path = os.path.join(FLAGS.data_dir, "buckets")
  buckets = data_utils.read_buckets(bucket_path)

  vocabs = {}
  rev_vocabs = {}

  # Load vocabularies.
  for i, data_type in enumerate(vocab_data_types):
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_paths[i])
    vocabs[data_type] = vocab
    rev_vocabs[data_type] = rev_vocab
  
  source_ids_data_types = [data_type + ".ids" 
                           for data_type in source_data_types]

  dev_source_paths = [(FLAGS.data_dir + "/" + FLAGS.dev_name + "." 
                         + data_type) for data_type in source_ids_data_types]

  if FLAGS.use_forced_attention_decoder:
    # Read word to concept map.
    input_map_path = os.path.join(FLAGS.data_dir, "concepts.map")
    encoder_decoder_vocab_map = data_utils.encoder_decoder_vocab_map_to_token_ids(
        input_map_path, vocabs["en"] , vocabs["lexeme"])
  else: 
    encoder_decoder_vocab_map = []  

  restricted_vocab_sets = data_utils.id_vocab_sets(vocabs["parse"])
  restricted_state_map = [data_utils.map_restricted_state(word) 
                          for word in rev_vocabs["parse"]]


  source_vocab_sizes, target_vocab_sizes, source_embedding_sizes, target_embedding_sizes = get_vocab_embed_sizes(vocab_data_types, source_data_types, target_data_types, vocabs)

  if FLAGS.initialize_word_vectors:
    word_embeddings = construct_embeddings(word_vectors, vocabs["en"],
        FLAGS.input_embedding_size)
  else:
    word_embeddings = construct_embeddings(None, vocabs["en"],
        FLAGS.input_embedding_size)

  if FLAGS.use_pretrained_word_vectors:
    pretrained_word_embeddings = construct_embeddings(word_vectors,
            vocabs["em"], FLAGS.pretrained_embedding_size)
  else:
    pretrained_word_embeddings = None

  restricted_state_map = [data_utils.map_restricted_state(word) 
                          for word in rev_vocabs["parse"]]

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  with tf.Graph().as_default() as g:
    with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      # Create model and load parameters.
      t0 = time.time()
      model = create_model(buckets, sess, True, source_vocab_sizes, 
          target_vocab_sizes, source_embedding_sizes, target_embedding_sizes,
          target_data_types, pretrained_word_embeddings, word_embeddings,
          encoder_decoder_vocab_map=encoder_decoder_vocab_map,
          restricted_vocab_sets=restricted_vocab_sets,
          restricted_state_map=restricted_state_map)
      t1 = time.time()
      create_duration = t1 - t0
      print("model creation time %.4f" % (create_duration))

      # Decode the dev set.
      assert FLAGS.decode_dev, "Only supporting decoding from text files."
      print("Decoding " + FLAGS.dev_name)
      if FLAGS.use_linear_point_decoder and FLAGS.offset_pointers:
        offset_target = target_data_types.index("att")
      else:
        offset_target = -1

      dev_set, dev_list = read_mrs_source_data(buckets, dev_source_paths, 
          any_length=True, offset_target=offset_target)
      
      t0 = time.time()
      if FLAGS.batch_decode:
        decode_batch_mrs_dev(model, sess, dev_list, data_types, rev_vocabs, 
            FLAGS.dev_name, False, buckets, restricted_vocab_sets, score_only=score_only)
      else:
        decode_mrs_dev(model, sess, dev_list, data_types, rev_vocabs, 
            FLAGS.dev_name, False, restricted_vocab_sets, score_only=score_only)
      t1 = time.time()
      duration = t1 - t0
      print("decoding time %.4f" % (duration))
    

def main(_):
  if FLAGS.score_only:
    mrs_decode(True)
  elif FLAGS.decode:
    mrs_decode()
  else:
    train()


if __name__ == "__main__":
  tf.app.run()
