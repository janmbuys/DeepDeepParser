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

import graph as mrs_graph

if __name__=='__main__':
  assert len(sys.argv) >= 4
  input_dir = sys.argv[1] + '/'
  working_dir = sys.argv[2] + '/'
  data_type = sys.argv[3] # dmrs or eds

  set_names = ['train', 'dev', 'test']
  suffix = '.raw'

  for set_name in set_names:
    sent_file = open(input_dir + set_name + suffix, 'r')
    sentences_raw = [sent for sent in sent_file.read().split('\n')[:-1]]

    # Reads token indexes.
    inds_filename = working_dir + set_name + '.span'
    inds_file = open(inds_filename, 'r')  
    token_inds = []
    token_starts = []
    token_ends = []
    for line in inds_file:
      inds = line.strip().split(' ')
      token_ind = {}
      for i, ind in enumerate(inds):
        token_ind[ind] = i
      token_inds.append(token_ind)
      token_starts.append(map(int, [ind.split(':')[0] for ind in inds]))
      token_ends.append(map(int, [ind.split(':')[1] for ind in inds]))

    if data_type == 'dmrs':
      sdmrs_file = open(input_dir + set_name + '.sdmrs', 'r')
      simple_strs = [sent for sent in sdmrs_file.read().split('\n')[:-1]]
      carg_file = open(input_dir + set_name + '.carg', 'r')
      dmrs_carg_strs = [sent for sent in carg_file.read().split('\n')[:-1]]
    else:
      eds_file = open(input_dir + set_name + '.eds', 'r')
      simple_strs = [sent for sent in eds_file.read().split('\n')[:-1]]
  
    # Files to write output to.
    lin_amr_out_file = open(working_dir + set_name + '.amr.lin', 'w')
    lin_dmrs_out_file = open(working_dir + set_name + '.dmrs.lin', 'w')
    unlex_dmrs_out_file = open(working_dir + set_name + '.dmrs.unlex.lin', 'w')
    nospan_dmrs_out_file = open(working_dir + set_name + '.dmrs.nospan.lin', 'w')
    nospan_unlex_dmrs_out_file = open(working_dir + set_name + '.dmrs.nospan.unlex.lin', 'w')
    point_dmrs_out_file = open(working_dir + set_name + '.dmrs.point.lin', 'w')

    lin_dmrs_ae_out_file = open(working_dir + set_name + '.dmrs.ae.lin', 'w')
    lin_dmrs_ae_io_out_file = open(working_dir + set_name + '.dmrs.ae.io.lin', 'w')
    lin_dmrs_unlex_ae_io_out_file = open(working_dir + set_name + '.dmrs.ae.io.unlex.lin', 'w')
    lin_dmrs_nospan_unlex_ae_io_out_file = open(working_dir + set_name + '.dmrs.ae.io.nospan.unlex.lin', 'w')
    point_dmrs_ae_io_out_file = open(working_dir + set_name + '.dmrs.ae.io.point.lin', 'w')
    end_point_dmrs_ae_io_out_file = open(working_dir + set_name + '.dmrs.ae.io.endpoint.lin', 'w')
     
    lin_dmrs_unlex_ae_ioc_out_file = open(working_dir + set_name + '.dmrs.ae.ioc.unlex.lin', 'w')
    lin_dmrs_nospan_unlex_ae_ioc_out_file = open(working_dir + set_name + '.dmrs.ae.ioc.nospan.unlex.lin', 'w')
    point_dmrs_ae_ioc_out_file = open(working_dir + set_name + '.dmrs.ae.ioc.point.lin', 'w')
    end_point_dmrs_ae_ioc_out_file = open(working_dir + set_name + '.dmrs.ae.ioc.endpoint.lin', 'w')
    
    lin_dmrs_unlex_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.unlex.lin', 'w')
    lin_dmrs_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.lin', 'w')
    lin_dmrs_nospan_unlex_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.nospan.unlex.lin', 'w')
    lin_dmrs_action_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.action.lin', 'w')
    lin_dmrs_concept_unlex_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.concept.unlex.lin', 'w')
    lin_dmrs_morph_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.morph.lin', 'w')
    point_dmrs_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.point.lin', 'w')
    end_point_dmrs_ae_ao_out_file = open(working_dir + set_name + '.dmrs.ae.ao.endpoint.lin', 'w')
    
    lin_dmrs_unlex_ae_out_file = open(working_dir + set_name + '.dmrs.ae.unlex.lin', 'w')
    nospan_dmrs_ae_out_file = open(working_dir + set_name + '.dmrs.ae.nospan.lin', 'w')
    nospan_unlex_dmrs_ae_out_file = open(working_dir + set_name + '.dmrs.ae.nospan.unlex.lin', 'w')
    point_dmrs_ae_out_file = open(working_dir + set_name + '.dmrs.ae.point.lin', 'w')

    lin_preds_out_file = open(working_dir + set_name + '.preds.lin', 'w')
    lin_preds_unlex_out_file = open(working_dir + set_name + '.preds.unlex.lin', 'w')
    lin_preds_nospan_out_file = open(working_dir + set_name + '.preds.nospan.lin', 'w')
    lin_preds_nospan_unlex_out_file = open(working_dir + set_name + '.preds.nospan.unlex.lin', 'w')
    lin_preds_point_out_file = open(working_dir + set_name + '.preds.point.lin', 'w')
    

    lin_dmrs_io_out_file = open(working_dir + set_name + '.dmrs.io.lin', 'w')
    lin_dmrs_unlex_io_out_file = open(working_dir + set_name + '.dmrs.io.unlex.lin', 'w')
    lin_dmrs_ioha_out_file = open(working_dir + set_name + '.dmrs.ioha.lin', 'w')
    point_dmrs_ioha_out_file = open(working_dir + set_name + '.dmrs.ioha.ind', 'a')

    edm_out_file = open(working_dir + set_name + '.edm', 'w')
    edmu_out_file = open(working_dir + set_name + '.edmu', 'w')
    amr_out_file = open(working_dir + set_name + '.amr', 'w')

    for i, simple_str in enumerate(simple_strs):
      if data_type == 'dmrs': 
        graph = mrs_graph.parse_dmrs(simple_str, token_inds[i], 
                                     token_starts[i], token_ends[i], 
                                     sentences_raw[i])

        # Adds constants.
        carg_list = dmrs_carg_strs[i].split()
        carg_inds = carg_list[::3]
        carg_preds = carg_list[1::3]
        carg_values = carg_list[2::3]

        for ind, pred, const in zip(carg_inds, carg_preds, carg_values):
          # Find head node of each CARG.
          found = False
          if (pred[0] == '"' and pred[-1] == '"') or pred[0] == '_':
            continue
          for i, node in enumerate(graph.nodes):
            if node.ind == ind and node.concept == pred:
              graph.nodes[i].constant = const
              found = True
          if not found:
            print "Constant not found:", pred, const
      else:
        graph = mrs_graph.parse_eds(simple_str, token_inds[i], 
                                    token_starts[i], token_ends[i], 
                                    sentences_raw[i])


      graph.find_span_tree(graph.root_index)

      graph.find_alignment_spans(graph.root_index)
      graph.find_span_edge_directions()

      # Validate alignments.
      for i, node in enumerate(graph.nodes):
        if graph.spanned[i] and node.alignment >= 0:
          if node.constant:
            graph.nodes[i].is_aligned = True
        elif node.concept.startswith('_'): # lexical concepts
          graph.nodes[i].is_aligned = True
    
      # Writes output to files.
      lin_amr_out_file.write(':focus( ' 
          + graph.linear_amr_str(graph.root_index) + ' )\n')

      span_start = str(graph.nodes[graph.root_index].alignment)
      span_end = str(graph.nodes[graph.root_index].alignment_end)
      lin_dmrs_out_file.write(':/H( <' + span_start + ' ')
      lin_dmrs_out_file.write(graph.dmrs_str(graph.root_index) + ' ) ' + span_end + '>\n')
      
      unlex_dmrs_out_file.write(':/H( <' + span_start + ' ')
      unlex_dmrs_out_file.write(graph.dmrs_str(graph.root_index, False) + ' ) ' + span_end + '>\n')

      nospan_dmrs_out_file.write(':/H( ' + graph.dmrs_str(graph.root_index,
          True, False) + ' )\n')
      nospan_unlex_dmrs_out_file.write(':/H( ' 
          + graph.dmrs_str(graph.root_index, False, False) + ' )\n')

      point_dmrs_out_file.write(span_start + ' ' +
              graph.dmrs_point_str(graph.root_index) + span_end + '\n')
       
      lin_dmrs_ae_out_file.write(graph.dmrs_arceager_str(graph.root_index, '/H'))
      lin_dmrs_ae_out_file.write(') ' + span_end + '>\n')

      lin_dmrs_unlex_ae_out_file.write(graph.dmrs_arceager_str(graph.root_index,
        '/H', False))
      lin_dmrs_unlex_ae_out_file.write(') ' + span_end + '>\n')

      ae_io_str, _, _, _, _, _, _ = graph.dmrs_arceager_oracle_str('inorder', True)
      lin_dmrs_ae_io_out_file.write(ae_io_str + '\n')

      unlex_ae_io_str, unlex_ae_io_nospan_str, _, _, _, ae_io_point_str, ae_io_end_point_str = graph.dmrs_arceager_oracle_str('inorder', False)
      lin_dmrs_unlex_ae_io_out_file.write(unlex_ae_io_str + '\n')
      lin_dmrs_nospan_unlex_ae_io_out_file.write(unlex_ae_io_nospan_str + '\n')
      point_dmrs_ae_io_out_file.write(ae_io_point_str + '\n')
      end_point_dmrs_ae_io_out_file.write(ae_io_end_point_str + '\n')
     
      unlex_ae_ioc_str, unlex_ae_ioc_nospan_str, _, _, _, ae_ioc_point_str, ae_ioc_end_point_str = graph.dmrs_arceager_oracle_str('cleaninorder', False)
      lin_dmrs_unlex_ae_ioc_out_file.write(unlex_ae_ioc_str + '\n')
      lin_dmrs_nospan_unlex_ae_ioc_out_file.write(unlex_ae_ioc_nospan_str + '\n')
      point_dmrs_ae_ioc_out_file.write(ae_ioc_point_str + '\n')
      end_point_dmrs_ae_ioc_out_file.write(ae_ioc_end_point_str + '\n')

      unlex_ae_ao_str, unlex_ae_ao_nospan_str, ae_ao_action_str, unlex_ae_ao_concept_str, ae_ao_morph_str, ae_ao_point_str, ae_ao_end_point_str = graph.dmrs_arceager_oracle_str('alignorder', False)
      lex_ae_ao_str, _, _, _, _, _, _ = graph.dmrs_arceager_oracle_str('alignorder', True)
      lin_dmrs_unlex_ae_ao_out_file.write(unlex_ae_ao_str + '\n')
      lin_dmrs_ae_ao_out_file.write(lex_ae_ao_str + '\n')
      lin_dmrs_nospan_unlex_ae_ao_out_file.write(unlex_ae_ao_nospan_str + '\n')
      lin_dmrs_concept_unlex_ae_ao_out_file.write(unlex_ae_ao_concept_str + '\n')
      lin_dmrs_action_ae_ao_out_file.write(ae_ao_action_str + '\n')
      lin_dmrs_morph_ae_ao_out_file.write(ae_ao_morph_str + '\n')
      point_dmrs_ae_ao_out_file.write(ae_ao_point_str + '\n')
      end_point_dmrs_ae_ao_out_file.write(ae_ao_end_point_str + '\n')

      nospan_dmrs_ae_out_file.write(graph.dmrs_arceager_str(graph.root_index,
          '/H', True, False) + ')\n')
      nospan_unlex_dmrs_ae_out_file.write(graph.dmrs_arceager_str(
          graph.root_index, '/H', False, False) + ')\n')

      point_dmrs_ae_out_file.write(graph.dmrs_arceager_point_str(
          graph.root_index, True) + span_end + '\n')

      io_lin_str, _ = graph.dmrs_inorder_str(graph.root_index)
      lin_dmrs_io_out_file.write(io_lin_str + '\n')
      io_unlex_lin_str, _ = graph.dmrs_inorder_str(graph.root_index, False)
      lin_dmrs_unlex_io_out_file.write(io_unlex_lin_str + '\n')

      ioha_lin_str, ioha_ind_str, _ = graph.dmrs_inorder_ha_str(graph.root_index, '0')
      lin_dmrs_ioha_out_file.write(ioha_lin_str + '\n')
      point_dmrs_ioha_out_file.write(ioha_ind_str + '\n')

      preds, preds_nospan, preds_point = graph.ordered_predicates_str() 

      lin_preds_out_file.write(preds + '\n')
      lin_preds_nospan_out_file.write(preds_nospan + '\n')
      lin_preds_point_out_file.write(preds_point + '\n')

      preds_unlex, preds_nospan_unlex, _ = graph.ordered_predicates_str(False) 

      lin_preds_unlex_out_file.write(preds_unlex + '\n')
      lin_preds_nospan_unlex_out_file.write(preds_nospan_unlex + '\n')
       
      edm_out_file.write(graph.edm_ch_str(True) + '\n')
      edmu_out_file.write(graph.edm_ch_str(False) + '\n')

      if len(graph.nodes) == 0:
        amr_out_file.write('( n1 / _UNK )\n\n')
      else:
        graph.correct_constants()
        graph.correct_concept_names()
        graph.correct_node_names()
        amr_out_file.write(graph.amr_graph_str(graph.root_index, 1) + '\n\n')

