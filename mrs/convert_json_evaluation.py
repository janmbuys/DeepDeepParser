# Copyrjght 2018 Jan Buys.
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

import graph as mrs_graph
import sentence

def is_const_predicate(concept):
  const_set = set(["named", "card", "named_n", "mofy", "yofc", "ord", 
      "dofw", "dofm", "fraction", "season", "excl", "year_range", 
      "numbered_hour", "holiday", "timezone_p", "polite"])
  return concept in const_set

if __name__=='__main__':
  assert len(sys.argv) > 3
  input_dir = sys.argv[1] + '/' # for gold input
  working_dir = sys.argv[2] + '/' # for predicted dmrs and output
  set_name = sys.argv[3]

  sent_file = open(input_dir + set_name + '.txt', 'r')
  sentences_raw = [sent for sent in sent_file.read().split('\n')[:-1]]

  # Read tokenization json
  tok_file = open(input_dir + set_name + '.tok.json', 'r')
  tok_json_strs = [sent for sent in tok_file.read().split('\n')[:-1]]
  sentences = {}
  for json_str in tok_json_strs:
    sent = sentence.Sentence.parse_json_line(json_str)
    sentences[sent.sent_ind] = sent

  # Read predicted graph json
  sys_dmrs_file = open(working_dir + set_name + '.dmrs.json', 'r')
  dmrs_json_strs = [sent for sent in sys_dmrs_file.read().split('\n')[:-1]]
  graphs = []
  for json_str in dmrs_json_strs:
    graph = mrs_graph.MrsGraph.parse_json_line(json_str)
    sent = sentences[graph.parse_ind]
    raw_str = sentences_raw[sent.sent_ind-1]
    for node in graph.nodes:
      token = sent.sentence[node.alignment]
      # spans
      start_ind = token.char_start
      end_ind = sent.sentence[node.alignment_end].char_end
      if is_const_predicate(node.concept):           
        const_end_ind = token.const_char_end
        if token.is_const:
          constant = token.const_lexeme
        else:  
          constant = raw_str[start_ind:end_ind]
          if ' ' in constant:
            constant = constant[:constant.index(' ')]
        if constant[0] != '"':
          constant = '"' + constant
        if constant[-1] == '"': # first remove end qoute
          constant = constant[:-1]
        if len(constant) > 1 and constant[-1] in '.,':
          constant = constant[:-1]
        node.constant = constant + '"'
      elif node.concept.startswith("_"): 
        if (node.concept != '_u_unknown' and token.pos == '-LRB-'
              and node.alignment + 1 < len(sent.sentence)):
          token = sent.sentence[node.alignment + 1]
        if token.is_pred:
          node.concept = '_' + token.lemma + node.concept
        else:  
          node.concept = ('_' + token.original_word.lower() + '/' 
                          + token.pos.lower() + '_u_unknown')
        
      node.ind = str(start_ind) + ":" + str(end_ind)

    graphs.append(graph)

  # Read gold json
  gold_dmrs_file = open(input_dir + set_name + '.dmrs.orig.json', 'r')
  gold_dmrs_json_strs = [sent for sent in gold_dmrs_file.read().split('\n')[:-1]]
  gold_graphs = []
  for json_str in gold_dmrs_json_strs:
    graph = mrs_graph.MrsGraph.parse_orig_json_line(json_str)

    gold_graphs.append(graph)

  # write out
  system_orig_dmrs_file = open(working_dir + set_name + '.dmrs.orig.json', 'w')
  for i, graph in enumerate(graphs):
    system_orig_dmrs_file.write((graph.json_orig_parse_str(i, False) + '\n').encode('utf-8', 'replace'))

  system_edm_out_file = open(working_dir + set_name + '.dmrs.edm', 'w')
  for graph in graphs:
    system_edm_out_file.write(graph.edm_ch_str(True) + '\n')

  gold_edm_out_file = open(working_dir + set_name + '.dmrs.orig.edm', 'w')
  for graph in gold_graphs:
    gold_edm_out_file.write(graph.edm_ch_str(True) + '\n')


