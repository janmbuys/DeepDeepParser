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

"""Utilities for preprocessing data and vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import random
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

GEN_STATE = 0
PAD_STATE = 1
RE_STATE = 2
ARC_STATE = 3
ROOT_STATE = 4

NUM_TR_STATES = 5

NO_ATTENTION_DECODER_STATE = 0
ATTENTION_DECODER_STATE = 1
LINEAR_POINTER_DECODER_STATE = 2
HARD_ATTENTION_DECODER_STATE = 3
LINEAR_FEED_POINTER_DECODER_STATE = 4
HARD_ATTENTION_ARC_EAGER_DECODER_STATE = 5
STACK_DECODER_STATE = 6
PURE_STACK_DECODER_STATE = 7
MEMORY_STACK_DECODER_STATE = 8

MAX_OUTPUT_SIZE = 300 #TODO parameterize

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def space_tokenizer(sentence):
  """Tokenize only on whitespace."""
  words = sentence.strip().split()
  return words

def create_vocabulary(vocabulary_path, singleton_vocabulary_path, data_path,
                      max_vocabulary_size, singleton_keep_prob, col_format,
                      tokenizer=None, normalize_digits=False, dic=None, col=1):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, space_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    singleton_keep_prob: probability to include singletons in the vocabulary.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    if dic is not None:
      print("Dic is not None.")
    vocab = {}
    singleton_vocab = set()
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        if not line.strip():
          continue
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        if col_format:
          tokens = [line.strip().split()[col]]
        else:
          tokens = tokenizer(line) if tokenizer else space_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          elif not word in _START_VOCAB:
            if dic is not None and dic.has_key(word):
              vocab[word] = 1
            elif word in singleton_vocab:
              vocab[word] = 2
            elif not col_format or word <> b"_":
              singleton_vocab.add(word)
      for word in singleton_vocab:
        if word not in vocab and random.random() < singleton_keep_prob:
          vocab[word] = 1
      singletons = set()
      for word in vocab:
        if vocab[word] == 1 and map_restricted_state(word) == GEN_STATE:
          singletons.add(word)
      vocab_list = [w for w in _START_VOCAB]
      if col_format:
        vocab_list.append(b"_")
      vocab_list += sorted(vocab, key=vocab.get, reverse=True)
      if max_vocabulary_size > 0 and len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
      with gfile.GFile(singleton_vocabulary_path, mode="wb") as vocab_file:
        if not singletons:
          vocab_file.write(b"")
        for w in singletons:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def read_buckets(bucket_path):
  """Read bucket sizes from file."""
  if gfile.Exists(bucket_path):
    lines = []
    with gfile.GFile(bucket_path, mode="rb") as f:
      lines.extend(f.readlines())
    buckets = []
    for line in lines:
      entry = line.strip().split(b" ")
      buckets.append((int(entry[0]), int(entry[1])))
    return buckets
  else:
    raise ValueError("Bucket file %s not found.", bucket_path)


def read_word_vectors(word_vector_path, vocabulary_path):
  """Read word vectors from file."""
  if gfile.Exists(word_vector_path):
    vocab_list = []
    lines = []
    with gfile.GFile(word_vector_path, mode="rb") as f:
      lines.extend(f.readlines())
    vector_dict = dict()
    for line in lines:
      entry = line.strip().split(b" ")
      word = entry[0] # preserve case
      vocab_list.append(word)
      vector_dict[word] = map(float, entry[1:])
    # Add unk entry
    assert not vector_dict.has_key('_UNK')
    vocab_list.append('_UNK')
    vector_dict['_UNK'] = [0.0 for _ in range(len(vector_dict['</s>']))]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + b"\n")
    return vector_dict
  else:
    raise ValueError("Vocabulary file %s not found.", word_vector_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False, has_unk=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, space_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = space_tokenizer(sentence)
  tokens = []
  for w in words:
    if normalize_digits:
      w = _DIGIT_RE.sub(b"0", w)
    if vocabulary.has_key(w):
      tokens.append(vocabulary[w])
    elif w.startswith(b":") and vocabulary.has_key(b":op("):
      # Special UNK handling for single sequence model.
      if w.endswith(b"("):
        tokens.append(vocabulary[b":op("])
      elif w.endswith(b"()") or w.endswith(b"(*)"):
        tokens.append(vocabulary[b":op()"])
      elif has_unk:
        tokens.append(UNK_ID)
      else:
        tokens.append(len(vocabulary)-1)
    elif has_unk:
      tokens.append(UNK_ID)
    else:
      tokens.append(len(vocabulary)-1)
  return tokens


def dict_to_token_ids(en_fr_dict, en_vocab, fr_vocab):
  ids_dict = dict()
  for word, words in en_fr_dict.iteritems():
    en_id = en_vocab.get(word, UNK_ID)
    if en_id <> UNK_ID:
      ids_dict[en_id] = [fr_vocab.get(w, UNK_ID) for w in words]

  return ids_dict


def dict_data_to_token_ids(dict_path, null_path, en_vocab, fr_vocab):
  with gfile.GFile(dict_path, mode="rb") as dict_file:
    en_fr_dict = dict()
    for line in dict_file:
      entry = line.strip().split(b"\t")
      if len(entry) > 1:
        en_fr_dict[entry[0]] = entry[1:]
    vocab_dict = dict_to_token_ids(en_fr_dict, en_vocab, fr_vocab)
  with gfile.GFile(null_path, mode="rb") as null_vocab_file:
    null_vocab = []
    for line in null_vocab_file:
      if line.strip():
        null_vocab.append(line.strip())
    null_ids = [fr_vocab.get(w, UNK_ID) for w in null_vocab]
    null_ids.append(UNK_ID)
  return null_ids, vocab_dict


def id_vocab_sets(fr_vocab):
  id_sets = [set() for _ in xrange(NUM_TR_STATES)]
  for word, ind in fr_vocab.iteritems():
    state = map_restricted_state(word)
    id_sets[state].add(ind)
  return id_sets


def map_restricted_state(word):
  state = GEN_STATE
  if ((word.startswith(b":") and word.endswith(b"("))
      or word.startswith(b"LA:") or word.startswith(b"RA:") 
      or word.startswith(b"UA:") or word.startswith(b"STACK*")):
    state = ARC_STATE
  elif word == b"ROOT":  
    state = ROOT_STATE
  elif word == b")" or word == b"RE":
    state = RE_STATE
  elif word in [_PAD, _EOS]:
    state = PAD_STATE
  return state


def construct_transition_map(vocab_sets, restrict_vocab):
  transitions = [range(0, 5) for _ in xrange(5)]

  restrictions = []
  for indexes in transitions:
    restr = vocab_sets[indexes[0]]
    for k in indexes[1:]:
      restr = restr.union(vocab_sets[k])
    restr_list = list(restr)
    restr_list.sort()
    restrictions.append(restr_list)
  return restrictions


def extract_decoder_vocab(sent, vocab_dict):
  decoder_vocab = set()
  for en_id in sent:
    if type(en_id) == tuple:
      en_id = en_id[0]
    if vocab_dict.has_key(en_id):
      for fr_id in vocab_dict[en_id]:
        if fr_id <> UNK_ID:
          decoder_vocab.add(fr_id)
  return decoder_vocab


def encoder_decoder_vocab_map_to_token_ids(map_path, source_vocab, target_vocab):
  with gfile.GFile(map_path, mode="rb") as map_file:
    concept_map = [-1 for _ in source_vocab]
    for line in map_file:
      word, concept = line.strip().split(b"\t")[0], line.strip().split(b"\t")[1]
      if source_vocab.has_key(word):
        word_id = source_vocab[word]
        if target_vocab.has_key(concept):
          concept_id = target_vocab[concept]
        else:
          concept_id = UNK_ID 
        concept_map[word_id] = concept_id
  return concept_map


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False, has_unk=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, space_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits, has_unk)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_mrs_data(data_dir, source_dir, data_type, set_name, 
    create_vocab):
  """Get data into data_dir, create vocabulary, convert to ids."""
  if data_type == 'em':
    path = os.path.join(source_dir, set_name + ".en")
  else:
    path = os.path.join(source_dir, set_name + "." + data_type)
  ids_path = os.path.join(data_dir, set_name + "." + data_type + ".ids")
  vocab_path = os.path.join(data_dir, "vocab." + data_type)
  sing_vocab_path = os.path.join(data_dir, "singleton-vocab." + data_type)
 
  if create_vocab and data_type <> 'em':
    create_vocabulary(vocab_path, sing_vocab_path, path, 0, 1.0, False,
                      space_tokenizer)

  data_to_token_ids(path, ids_path, vocab_path, space_tokenizer, False, 
      data_type <> 'em')
  return ids_path, vocab_path, sing_vocab_path


def copy_mrs_data(data_dir, source_dir, data_type, set_name):
  """Create copy of data file."""
  source_path = os.path.join(source_dir, set_name + "." + data_type)
  target_path = os.path.join(data_dir, set_name + "." + data_type + ".pnt")

  if not gfile.Exists(target_path):
    with gfile.GFile(source_path, mode="rb") as source_file:
      with gfile.GFile(target_path, mode="w") as target_file:
        for line in source_file:
          line = line.strip()
          target_file.write(line + "\n")
  return target_path


def read_ids_file(source_path, max_size):
  source_input = []
  print("Reading ids file:", source_path)
  with gfile.GFile(source_path, mode="r") as source_file:
    source = source_file.readline()
    counter = 0
    while source and (not max_size or counter < max_size):
      counter += 1
      source_ids = [int(x) for x in source.split()]
      source_input.append(source_ids) 
      source = source_file.readline()
  return source_input 


def write_output_file(output_path, output_lines):
  with gfile.GFile(output_path, mode="w") as output_file:
    for line in output_lines:
      output_file.write(line + "\n") 


