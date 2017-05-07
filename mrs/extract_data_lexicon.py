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

import sys

import delphin.mrs as mrs
import delphin.mrs.simplemrs as simplemrs
import delphin.mrs.simpledmrs as simpledmrs

import graph as mrs_graph
import util as mrs_util

if __name__=='__main__':
  assert len(sys.argv) >= 2
  data_name = sys.argv[1]
  input_dir = data_name + '/'
  working_dir = data_name + '-working/'
  out_dir = data_name + '-working/'

  set_name = 'train'
  suffix = '.raw'

  erg_dir = erg_name + '/' 
  erg_pred_map = mrs_util.read_lexicon(working_dir + 'predicates.erg.lexicon')
  erg_const_map = mrs_util.read_lexicon(working_dir + 'constants.erg.lexicon')

  sent_file = open(input_dir + set_name + suffix, 'r')
  sentences_raw = [sent for sent in sent_file.read().split('\n')[:-1]]

  sdmrs_file = open(input_dir + set_name + '.sdmrs', 'r')
  simple_dmrs_strs = [sent for sent in sdmrs_file.read().split('\n')[:-1]]
  carg_file = open(input_dir + set_name + '.carg', 'r')
  dmrs_carg_strs = [sent for sent in carg_file.read().split('\n')[:-1]]

  pred_dict = {}
  const_dict = {}

  for i, simple_dmrs_str in enumerate(simple_dmrs_strs):
    graph = mrs_graph.parse_dmrs(simple_dmrs_str, sentence_str=sentences_raw[i]) 

    # Adds constants. 
    carg_list = dmrs_carg_strs[i].split()
    carg_inds = carg_list[::3]
    carg_preds = carg_list[1::3]
    carg_values = carg_list[2::3]

    for ind, pred, const in zip(carg_inds, carg_preds, carg_values):
      # Finds head node of CARG.
      found = False
      if (pred[0] == '"' and pred[-1] == '"') or pred[0] == '_':
        continue
      for j, node in enumerate(graph.nodes):
        if node.ind == ind and node.concept == pred:
          graph.nodes[j].constant = const
          if const[0] =='"' and const[-1] == '"':
            const = const[1:-1]
          const = mrs_util.clean_punct(const)
          ind_start, ind_end = int(ind.split(':')[0]), int(ind.split(':')[1])
          const_raw = sentences_raw[i][ind_start:ind_end]
          const_raw = mrs_util.clean_punct(const_raw)
          if const_raw == '':
            continue
          if const_dict.has_key(const_raw):
            if const_dict[const_raw].has_key(const):
              const_dict[const_raw][const] += 1
            else:
              const_dict[const_raw][const] = 1
          else:
            const_dict[const_raw] = {}
            const_dict[const_raw][const] = 1
          found = True
          
    # Extracts lexical dict.
    for node in graph.nodes:
      if (node.alignment and node.concept.startswith('_')
          and '/' not in node.concept):
        ind_start = int(node.ind.split(':')[0])
        ind_end = int(node.ind.split(':')[1])
        pred = node.concept[:node.concept.index('_', 1)]
        pred_raw = sentences_raw[i][ind_start:ind_end]
        pred_raw = mrs_util.clean_punct(pred_raw)
        if pred_raw == '':
          continue
        if pred_raw[0].isupper(): # lowercase if only first letter is upper
          if pred_raw[0].lower() + pred_raw[1:] == pred_raw.lower():
            pred_raw = pred_raw.lower()
        if pred_dict.has_key(pred_raw):
          if pred_dict[pred_raw].has_key(pred):
            pred_dict[pred_raw][pred] += 1
          else:
            pred_dict[pred_raw][pred] = 1
        else:
          pred_dict[pred_raw] = {}
          pred_dict[pred_raw][pred] = 1

  # Extracts 1-best map, disambiguate with the ERG.
  pred_map = {}
  for pred_raw, dic in pred_dict.iteritems():
    max_count = max(dic.values())
    found_pred = False
    if erg_pred_map.has_key(pred_raw):
      erg_pred = erg_pred_map[pred_raw]
      if dic.has_key(erg_pred) and dic[erg_pred] == max_count:
        pred_map[pred_raw] = erg_pred
        found_pred = True
    for pred, count in dic.iteritems():
      if found_pred:
        break
      if count == max_count:
        pred_map[pred_raw] = pred
        found_pred = True

  for pred_raw, pred in erg_pred_map.iteritems():
    if not pred_map.has_key(pred_raw):
      pred_map[pred_raw] = pred

  const_map = {}
  for const_raw, dic in const_dict.iteritems():
    max_count = max(dic.values())
    found_const = False
    if erg_const_map.has_key(const_raw):
      erg_const = erg_const_map[const_raw]
      if dic.has_key(erg_const) and dic[erg_const] == max_count:
        const_map[const_raw] = erg_const
        found_const = True
    for const, count in dic.iteritems():
      if found_const:
        break
      if count == max_count:
        const_map[const_raw] = const
        found_const = True

  for const_raw, const in erg_const_map.iteritems():
    if not const_map.has_key(const_raw):
      const_map[const_raw] = const

  pred_out_file = open(out_dir + 'predicates.lexicon', 'w')
  for orth, pred in pred_map.iteritems():
    pred_out_file.write(orth + '\n' + pred + '\n')
  pred_out_file.close()

  const_out_file = open(out_dir + 'constants.lexicon', 'w')
  for orth, const in const_map.iteritems():
    const_out_file.write(orth + '\n' + const + '\n')
  const_out_file.close()



