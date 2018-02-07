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

import os 
import re
import string
import sys
import codecs

import graph as mrs_graph

def clean_quoted(word):
  constant = word.replace('"', '')
  constant = constant.replace('\'', '')
  constant = constant.replace(':', '')
  constant = constant.replace('(', '_')
  constant = constant.replace(')', '_')
  constant = constant.strip()
  constant = constant.replace(' ', '_')
  return constant


def read_span(filename):
  token_starts = []
  token_ends = []
  inds_file = open(filename, 'r')  
  for line in inds_file:
    inds = line.strip().split(' ')
    token_starts.append(map(int, [ind.split(':')[0] for ind in inds]))
    token_ends.append(map(int, [ind.split(':')[1] for ind in inds]))
  return token_starts, token_ends


def read_ints(filename):
    sentences = []
    sentence_file = open(filename, 'r')  
    for line in sentence_file:
      sentences.append(int(line.strip()))
    return sentences


def read_ids(filename):
    sentences = []
    sentence_file = open(filename, 'r')  
    for line in sentence_file:
      sentences.append(line[:line.index('.txt')])
    return sentences


def read_tokens(filename):
    sentences = []
    sentence_file = codecs.open(filename, 'r', 'utf-8')
    for line in sentence_file:
      sentences.append(line.strip().split(' '))
    return sentences


def linear_to_mrs(base_path, data_path, mrs_path, set_name, convert_train,
                  is_inorder=False, is_arceager=False,
                  is_arceager_buffer_shift=False,
                  is_lexicalized=True, is_no_span=False,
                  recover_end_spans=False, is_amr=False, is_epe=False, 
                  domain=''):
  graphs = mrs_graph.read_linear_dmrs_graphs(mrs_path + '.lin', True,
      is_inorder, is_arceager, is_arceager_buffer_shift, is_no_span)

  for i, graph in enumerate(graphs):
    if graph.nodes:
      graphs[i].spanned = [False for _ in graph.nodes]
      graphs[i].find_span_tree(graph.root_index)

  if recover_end_spans: 
    for i, graph in enumerate(graphs):
      for j in xrange(len(graphs[i].nodes)): # clear any end alignments
        graphs[i].nodes[j].alignment_end = graphs[i].nodes[j].alignment
      graphs[i].recover_end_spans(graph.root_index, -1)      
 
  if convert_train:
    # Keeps token-level spans.
    for i, graph in enumerate(graphs):
      for j, node in enumerate(graph.nodes):
        if node.alignment_end < node.alignment:
          node.alignment_end = node.alignment # correct negative spans
        graphs[i].nodes[j].ind = (str(node.alignment) + ':' 
            + str(node.alignment_end))
        if node.concept.endswith('_CARG'):
          concept = node.concept[:node.concept.index('_CARG')]
          graphs[i].nodes[j].concept = concept
          graphs[i].nodes[j].constant = 'CARG'
 
  # Reads untokenized sentences.
  sentences = []
  sentences_unstripped = []
  if is_epe:
    sentence_file = codecs.open(base_path + '.txt', 'r', 'utf-8')
  else:
    sentence_file = open(data_path + '.raw', 'r') 

  for line in sentence_file:
    sentences_unstripped.append(line.replace('\n', ' '))
    sentences.append(line.strip())

  # Reads const and pred candidate tokens.
  const_tokens = read_tokens(base_path + '.lex.const')
  pred_tokens = read_tokens(base_path + '.lex.pred')
  pos_tokens = read_tokens(base_path + '.pos')
  ne_tokens = read_tokens(base_path + '.ne')
  if is_amr:
    nom_tokens = read_tokens(amr_base_path + '.lex.nom')
  if is_epe:
    offsets = read_ints(base_path + '.off')
    file_ids = read_ids(base_path + '.ids')

  # Reads token span indexes.
  token_starts, token_ends = read_span(base_path + '.span')
  const_token_starts, const_token_ends = read_span(base_path + '.span.const')  
  pred_token_starts, pred_token_ends = read_span(base_path + '.span.pred')

  for i, graph in enumerate(graphs):
    for j, node in enumerate(graph.nodes):
      if node.alignment_end < node.alignment:
        node.alignment_end = node.alignment # correct negative sized spans
      align = min(len(token_starts[i]) -1, node.alignment)
      set_span = False
      if node.concept.endswith('_CARG'):
        concept = node.concept[:node.concept.index('_CARG')]
        graphs[i].nodes[j].concept = concept
        constant = const_tokens[i][align]
        if constant:
          graphs[i].nodes[j].constant = '"' + constant + '"'
          span_start = const_token_starts[i][align]
          span_end = const_token_ends[i][align]
      elif is_amr and node.concept == 'CONST':
        constant = const_tokens[i][align]
        if re.match(r'^\d+(\.\d+)?$', constant): # match numbers
          graphs[i].nodes[j].constant = constant
        else:
          graphs[i].nodes[j].constant = '"' + constant + '"'
        span_start = const_token_starts[i][align]
        span_end = const_token_ends[i][align] 
      elif is_amr and node.concept[0] == '"' and node.concept[-1] == '"':
        constant = node.concept[1:-1]
        if re.match(r'^\d+(\.\d+)?$', constant): # match numbers
          graphs[i].nodes[j].constant = constant
        else:
          graphs[i].nodes[j].constant = '"' + constant + '"'
        span_start = 0
        span_end = 0
      elif is_amr and not is_lexicalized and node.concept.startswith('_'):
        if node.concept.startswith('_p_'):
          pred = pred_tokens[i][align][1:]
          concept = pred + '-' + node.concept[3:]
        else:
          pred = nom_tokens[i][align][1:] 
          concept = pred
        span_start = pred_token_starts[i][align]
        span_end = pred_token_ends[i][align]
        graphs[i].nodes[j].concept = concept
      elif is_amr and node.concept.startswith('_'):
        sense_index = node.concept.index('_', 1)
        pred = node.concept[1:sense_index]
        suffix = node.concept[sense_index:]
        if suffix.startswith('_p_'):
          concept = pred + '-' + suffix[3:]
        else:
          concept = pred
        graphs[i].nodes[j].concept = concept
      elif ((not is_lexicalized and node.concept.startswith('_')) 
            or (is_lexicalized and 'u_unknown' in node.concept)):
        pred = pred_tokens[i][align]
        if pred.startswith('_'): 
          if node.concept.startswith('_+') or node.concept.startswith('_-'):
            concept = pred + node.concept[1:] 
          else:
            concept = pred + node.concept 
        else: 
          # Dictionary overrules prediction.
          concept = '_' + pred + '_u_unknown' 
        span_start = pred_token_starts[i][align]
        span_end = pred_token_ends[i][align]
        graphs[i].nodes[j].concept = concept
      if not set_span:
        if node.alignment >= len(token_starts[i]):
          span_start = token_starts[i][-1] 
        else:
          span_start = token_starts[i][node.alignment]
        if node.alignment_end >= len(token_ends[i]):
          span_end = token_ends[i][-1] 
        else:
          span_end = token_ends[i][node.alignment_end]

      if node.alignment >= len(pos_tokens[i]):
        graphs[i].nodes[j].pos = pos_tokens[i][-1]
        graphs[i].nodes[j].ne = ne_tokens[i][-1]
      else:
        graphs[i].nodes[j].pos = pos_tokens[i][node.alignment]
        graphs[i].nodes[j].ne = ne_tokens[i][node.alignment]
      # Post-process span for punctuation.
      if (span_end + 1 < len(sentences[i]) 
          and sentences[i][span_end] in string.punctuation
          and sentences[i][span_end+1].isspace()):
        span_end += 1
      graphs[i].nodes[j].ind = str(span_start) + ':' + str(span_end)

  # Write out char-level EDM.
  if not is_no_span:
    edm_out_file = open(mrs_path + '.edm', 'w')
    for graph in graphs:
      if graph is None or len(graph.nodes) == 0:
        edm_out_file.write('NONE\n')
      else:
        str_enc = graph.edm_ch_str().encode('utf-8', 'replace')
        edm_out_file.write(str_enc + '\n')
    edm_out_file.close()

    epe_out_file = open(mrs_path + '.dmrs.json', 'w')
    offset = 0
    for i, graph in enumerate(graphs):
      if is_epe:
        offset = offsets[i]
      if not (graph is None or len(graph.nodes) == 0):
        epe_out_file.write((graph.json_parse_str(i) + '\n').encode('utf-8', 'replace'))
      if not is_epe:
        offset += len(sentences_unstripped[i])
    epe_out_file.close()    

    if is_epe:
      # write out to seperate files:
      type_map = {'train': 'training', 'dev': 'development', 'test': 'evaluation'}
      file_id = ''
      file_i = 0 
      out_file = None
      for i, graph in enumerate(graphs):
        offset = offsets[i]
        if not (graph is None or len(graph.nodes) == 0):
          if file_ids[i] == file_id:
            file_i += 1
          else:  
            file_id = file_ids[i] 
            file_i = 0 
            filename = ('epe-results/' + domain + '/' + type_map[set_name] 
                         + '/' + file_id + '.epe')
            out_file = open(filename, 'w')
          enc_str = (graph.json_str(file_i, sentences_unstripped[i], offset) + '\n').encode('utf-8', 'replace')
          out_file.write(enc_str)

  # Writes out AMR for Smatch evaluation.
  amr_out_file = open(mrs_path + '.amr', 'w')
  for i, graph in enumerate(graphs):
    if graph is None or len(graph.nodes) == 0:
      amr_out_file.write('( n1 / _UNK )\n\n')
    else:
      graph.correct_concept_names()
      if is_amr:
        graph.restore_op_indexes()    
        graph.restore_original_constants(graph.root_index, 'focus')
      if is_amr and graph.nodes[graph.root_index].constant <> '':
        concept = graph.nodes[graph.root_index].constant
        if concept[0] == '"' and concept[-1] == '"':
          concept = concept[1:-1]
        amr_out_file.write('( n1 / ' + concept + ' )\n\n')
      else:
        amr_out_file.write(graph.amr_graph_str(graph.root_index, 1, is_amr).encode('ascii', 'replace') + '\n\n')
  amr_out_file.close()


def point_to_linear(amr_path, copy_only, shift_pointer, no_pointer, no_endspan, 
                    with_endspan):
  if no_pointer or copy_only:
    input_names = ['parse']
  elif with_endspan:
    input_names = ['parse', 'att', 'endatt']
  else:
    input_names = ['parse', 'att']

  input_dmrs = {}
  for name in input_names:
    mrs_file = open(amr_path + '.' + name, 'r')
    lines = [] 
    for line in mrs_file:
      lines.append(line.strip().split(' '))
    input_dmrs[name] = lines

  out_file = open(amr_path + '.lin', 'w')
  for i, parse_line in enumerate(input_dmrs['parse']):
    if copy_only:
      out_file.write(parse_line)
      continue
    dmrs = []
    start_ind = 0
    if not no_pointer:
      assert len(input_dmrs['att'][i]) == len(parse_line)
    if with_endspan:
      assert len(input_dmrs['endatt'][i]) == len(parse_line)
    for j, parse_symbol in enumerate(parse_line):
      if no_pointer:
        ind = 0
      elif shift_pointer:
        if (j + 1 >= len(input_dmrs['att'][i]) or
            len(input_dmrs['att'][i][j+1]) == 0):
          ind = 0
        else:
          ind = max(0, int(input_dmrs['att'][i][j+1]))
      else:
        if len(input_dmrs['att'][i][j]) == 0:
          ind = 0
        else:
          ind = max(0, int(input_dmrs['att'][i][j]))
      if parse_symbol == ')' or parse_symbol == 'RE':
        dmrs.append(parse_symbol)
        if with_endspan:
          if len(input_dmrs['endatt'][i][j]) == 0:
            end_ind = 0
          else:
            end_ind = max(0, int(input_dmrs['endatt'][i][j]))
          dmrs.append(str(end_ind) + '>')
        elif not no_endspan:
          dmrs.append(str(ind) + '>')
      elif (parse_symbol.startswith(':') or parse_symbol.startswith('LA:') 
          or parse_symbol.startswith('RA:') or parse_symbol.startswith('UA:') 
          or parse_symbol.startswith('STACK*') or parse_symbol == 'ROOT'):
        dmrs.append(parse_symbol)
      else: # shift
        dmrs.append('<' + str(ind))
        dmrs.append(parse_symbol)
    out_file.write(' '.join(dmrs) + '\n')
  out_file.close()


if __name__=='__main__':
  assert len(sys.argv) >= 5
  data_name = sys.argv[1]
  set_name = sys.argv[2]

  convert_train = len(sys.argv) >= 6 and sys.argv[5] == '-t'

  is_inorder = '-inorder' in sys.argv[5:]
  is_arceager = '-arceager' in sys.argv[5:]
  is_arceager_buffer_shift = '-arceagerbuffershift' in sys.argv[5:]
  is_lexicalized = '-unlex' not in sys.argv[5:]
  is_no_span = '-nospan' in sys.argv[5:]

  copy_only = len(sys.argv) >= 6 and '-copy' in sys.argv[5:]
  shift_pointer = len(sys.argv) >= 6 and '-shift' in sys.argv[5:]
  no_pointer = len(sys.argv) >= 6 and '-nopointer' in sys.argv[5:]
  no_endspan = len(sys.argv) >= 6 and '-noendspan' in sys.argv[5:]
  with_endspan = len(sys.argv) >= 6 and '-withendspan' in sys.argv[5:]

  is_amr = '-amr' in sys.argv[5:]
  is_epe = '-epe' in sys.argv[5:]
  recover_end_spans = False

  amr_dir = data_name + '/' #TODO temp
  #amr_dir = data_name + '-working/'
  if is_epe:
    domain = data_name[4:]
  else:
    domain = ''
  amr_file_name = set_name
  amr_base_path = amr_dir + amr_file_name
  amr_data_path = data_name + '/' + amr_file_name

  working_dir = sys.argv[3] + '/'
  if sys.argv[4] == '-':
    amr_path = working_dir + amr_file_name
  else:
    amr_path = working_dir + amr_file_name + '.' + sys.argv[4]

  point_to_linear(amr_path, copy_only, shift_pointer, no_pointer, no_endspan, with_endspan)

  linear_to_mrs(amr_base_path, amr_data_path, amr_path, set_name, convert_train,
                is_inorder, is_arceager, is_arceager_buffer_shift, 
                is_lexicalized, is_no_span, recover_end_spans, is_amr, 
                is_epe, domain)

