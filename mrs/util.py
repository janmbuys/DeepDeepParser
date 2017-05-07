# Copyright 2017 Jan Buys.
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

import re
import os
import string

class ConstantVars():
  def __init__(self):
    self.names = []
    self.numbers = []

def prefix_sim(a, b):
  delta = 0.1
  l = min(len(a), len(b))
  prefix_length = len(os.path.commonprefix([a, b]))
  return (prefix_length + delta)/(min(len(a), len(b)) + delta)

def prefix_sim_long(a, b):
  delta = 0.1
  prefix_length = len(os.path.commonprefix([a, b]))
  return (prefix_length + delta)/(max(len(a), len(b)) + delta)

def is_quoted(item):
  return len(item) > 2 and item[0] == '"' and item[-1] == '"'

def remove_quotes(s):
  return re.sub(r'^\"(\S+)\"$', r'\1', s)

''' Remove concept id's and string quotes. '''
def clean_concept(s, remove_ids):
  if remove_ids:
    s = re.sub(r'^([^\s\d]+)-\d\d$', r'\1', s)
  s = remove_quotes(s)
  return s

''' Non-content-bearing punctuation. '''
def is_punct(w):
  punct = ['\'', '\"', '.', ',', ':', '-']
  return w in punct

def clean_punct(w):
  if w[0] == '"' or w[0] == "'":
    w = w[1:]
  if w[-1] == "'" or w[-1] == '"':
    w = w[:-1] 
  if w <> '' and w[0] in string.punctuation:
    w = w[1:]
  while w <> '' and w[-1] <> '.' and w[-1] in string.punctuation:
    w = w[:-1]
  if w <> '' and w[-1] == '.' and '.' not in w[:-1]:
    w = w[:-1]
  return w


def index_sort(align_ind):
  new_ind = range(len(align_ind))
  new_ind = sorted(new_ind, key = lambda i: align_ind[i])
  return new_ind


def read_lexicon(lex_filename):
  lex_map = {}
  lex_file = open(lex_filename, 'r')
  state = False
  orth = ''
  for line in lex_file: 
    if state:
      if line.strip() <> '_':
        lex_map[orth] = line.strip()
    else:
      orth = line.strip()
    state = not state
  return lex_map


def separate_brackets(amr_str):
  new_str = ''
  open_quotes = False
  for ch in amr_str:
    if ch == '\"':
      open_quotes = not open_quotes
    if ch == '(' and not open_quotes:
      new_str += ch + ' '
    elif ch == ')' and not open_quotes:
      new_str += ' ' + ch
    elif ch == ' ' and open_quotes: # replaces spaces inside quotes
      new_str += '_'
    else:
      new_str += ch
  return new_str 

