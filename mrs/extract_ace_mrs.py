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
import delphin.mrs.eds as eds

import graph as mrs_graph

if __name__=='__main__':
  assert len(sys.argv) >= 4, 'Invalid number of arguments (at least 3 required).'
  set_name = sys.argv[1]
  working_dir = sys.argv[2] + '/'
  extract_dmrs = sys.argv >= 4 and '-dmrs' in sys.argv[3:]
  extract_eds = sys.argv >= 4 and '-eds' in sys.argv[3:]
  assert extract_dmrs or extract_eds
  if extract_dmrs:
    mrs_type = 'dmrs'
  else:  
    mrs_type = 'eds'

  extract_dmrs_files = sys.argv >= 4 and '-extract_dmrs_files' in sys.argv[3:]
  extract_mrs_files = sys.argv >= 4 and '-extract_mrs_files' in sys.argv[3:]
  ignore_unparsed = sys.argv >= 4 and '-ignore_unparsed' in sys.argv[3:]

  filename = working_dir + set_name + '.mrs'
  in_file = open(filename, 'r')

  mrs_strs = []
  sentences = []

  sentence_str = ''
  simple_mrs_str = ''
  state = 0 
    
  for line in in_file:
    line = line.strip()
    if state == 0:
      if line.startswith('[ LTOP:'):
        simple_mrs_str = line + ' '
        state = 1
      else:
        if line.startswith('SKIP:'):
          sentence_str = line[line.index(':') + 2:]
          if not ignore_unparsed:
            sentences.append(sentence_str)
            mrs_strs.append('')
        elif line.startswith('SENT:'):
          sentence_str = line[line.index(':') + 2:]
    elif state == 1:
      if line:
        simple_mrs_str += line + ' '
        if line.startswith('HCONS:'):
          state = 0
          sentences.append(sentence_str)
          mrs_strs.append(simple_mrs_str)
      else:
        state = 0
        sentences.append(sentence_str)
        mrs_strs.append(simple_mrs_str)

  print len(sentences), "sentences"
  assert len(mrs_strs) == len(sentences)
  graphs = []
  simple_strs = []
  const_strs = []
  hyphen_sent_strs = []
  nohyphen_sent_strs = []

  for i, mrs_str in enumerate(mrs_strs):
     token_inds = {}
     token_starts = []
     token_ends = []
     token_start = 0

     state = True
     counter = 0
     for k, char in enumerate(sentences[i]):
       if state:
         if char.isspace():
           token_end = k
           token_inds[str(token_start) + ':' + str(token_end)] = counter
           token_starts.append(token_start)
           token_ends.append(token_end)
           counter += 1
           state = False
       elif not char.isspace():
         state = True
         token_start = k
     if state:
       token_end = len(sentences[i])
       token_inds[str(token_start) + ':' + str(token_end)] = counter
       token_starts.append(token_start)
       token_ends.append(token_end)

     if mrs_str:
       # Writes out mrs string to its own file.
       # This is required so that we can convert MRS to EDS using LOGON.
       if extract_mrs_files:
         mrs_file = open(working_dir + 'mrs/' + set_name + str(i) + '.mrs', 'w')
         mrs_file.write(mrs_str)
         mrs_file.close()

       # TODO Delphin passes utf-8 as encoding parameter, but somewhere it changes to ascii...
       # pydelphin/delphin/mrs/util.py:94
       # /usr/lib/python2.7/xml/etree/ElementTree.py
       simple_mrs_code = mrs_str.decode('utf-8', 'replace')
       mrs_str = simple_mrs_code.encode('ascii', 'replace')
       if extract_eds:
         mrx_str = mrs.convert(mrs_str, 'simplemrs', 'mrx')
         eds_object = mrs.mrx.loads(mrx_str)

         try:
           eds_str = eds.dumps(eds_object)
         except (KeyError, TypeError, IndexError) as err:
           if not ignore_unparsed:
             graphs.append(None) 
             simple_strs.append('')
             const_strs.append('')
             hyphen_sent_strs.append('')
             nohyphen_sent_strs.append('')
           continue

         eds_lines = eds_str.split('\n')
         single_eds_str = eds_lines[0][1:eds_lines[0].index(':')]
         for line in eds_lines[1:]:
           if line.strip() <> '}':
             single_eds_str += ' ; ' + line.strip()

         graph = mrs_graph.parse_eds(single_eds_str) 
         simple_dmrs_str = single_eds_str
         dmrs_const_str = ''
       else:  
         dmrs_xml_str = mrs.convert(mrs_str, 'simplemrs', 'dmrx')
         dmrs_object = mrs.dmrx.loads(dmrs_xml_str)

         try:
           simple_dmrs_str = simpledmrs.dumps(dmrs_object)
         except (KeyError, TypeError) as err:
           if not ignore_unparsed:
             graphs.append(None) 
             simple_strs.append('')
             const_strs.append('')
             hyphen_sent_strs.append('')
             nohyphen_sent_strs.append('')
           continue

         simple_mrs = simplemrs.loads_one(mrs_str)
         graph = mrs_graph.parse_dmrs(simple_dmrs_str) 

         if graph.root_index == -1:
           graph.root_index = 0

         dmrs_const_str = ''
         # Add constants  
         for ep in simple_mrs.eps():
           if ep.args.has_key('CARG'):
             dmrs_const_str += (str(ep.lnk)[1:-1] + ' ' + str(ep.pred) + ' ' 
                         + ep.args['CARG'] + ' ')

             # Find head node
             found = False
             pred = str(ep.pred)
             for j, node in enumerate(graph.nodes):
               if node.ind == str(ep.lnk)[1:-1] and node.concept == pred:
                 graph.nodes[j].constant = ep.args['CARG']
                 found = True
             if not found:
               print pred, str(ep.lnk)[1:-1]

       nohyphen_sent_str = sentences[i]
       for j in xrange(1, len(nohyphen_sent_str)-1):
         if (nohyphen_sent_str[j] == '-' and nohyphen_sent_str[j-1] <> ' ' and
             nohyphen_sent_str[j+1] <> ' '):
           nohyphen_sent_str = nohyphen_sent_str[:j] + ' ' + nohyphen_sent_str[j+1:]

       graph.find_span_tree(graph.root_index)
       graphs.append(graph)
       simple_strs.append(simple_dmrs_str)
       const_strs.append(dmrs_const_str)
       hyphen_sent_strs.append(sentences[i])
       nohyphen_sent_strs.append(nohyphen_sent_str)
     else:
       if not ignore_unparsed:
         graphs.append(None) 
         simple_strs.append('') 
         const_strs.append('') 
         hyphen_sent_strs.append('')
         nohyphen_sent_strs.append('')
   
  if extract_dmrs_files:
    # Writes out the dmrs's so that it can be used for training data.
    sent_out_file = open(working_dir + set_name + '.hraw', 'w')
    print working_dir + set_name + '.hraw'
    print len(hyphen_sent_strs)
    for sentence_str in hyphen_sent_strs:
      sent_out_file.write(sentence_str + '\n')
    sent_out_file.close()

    sent_out_file = open(working_dir + set_name + '.raw', 'w')
    for sentence_str in nohyphen_sent_strs:
      sent_out_file.write(sentence_str + '\n')
    sent_out_file.close()

    lin_out_file = open(working_dir + set_name + '.sdmrs', 'w')
    print len(simple_strs)
    for simple_dmrs_str in simple_strs:
      lin_out_file.write(simple_dmrs_str + '\n')
    lin_out_file.close()

    lin_out_file = open(working_dir + set_name + '.carg', 'w')
    for dmrs_const_str in const_strs:
      lin_out_file.write(dmrs_const_str + '\n')
    lin_out_file.close()

  # Writes out char-level EDM.
  edm_out_file = open(working_dir + set_name + '.' + mrs_type + '.edm', 'w')
  for graph in graphs:
    if graph is None:
      edm_out_file.write('NONE\n')
    else:
      edm_out_file.write(graph.edm_ch_str() + '\n')
  edm_out_file.close()

  # Writes out AMR for Smatch evaluation (modifies graph).
  amr_out_file = open(working_dir + set_name + '.' + mrs_type + '.amr', 'w')
  for graph in graphs:
    if graph is None or len(graph.nodes) == 0:
      amr_out_file.write('( n1 / _UNK )\n\n')
    else:
      graph.correct_concept_names()
      graph.correct_node_names()
      amr_out_file.write(graph.amr_graph_str(graph.root_index, 1) + '\n\n')
  amr_out_file.close()


