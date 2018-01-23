# Copyrjght 2017 Jan Buys.
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

  compute_statistics = False

  offset = 0 
  for set_name in set_names:
    sent_file = open(input_dir + set_name + suffix, 'r')
    sentences_raw = [sent for sent in sent_file.read().split('\n')[:-1]]

    # Reads token indexes.
    inds_filename = input_dir + set_name + '.span' # temp
    #inds_filename = working_dir + set_name + '.span'
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
    epe_out_file = open(working_dir + set_name + '.json', 'w')
    epe_orig_parse_out_file = open(working_dir + set_name + '.dmrs.orig.json', 'w')
    epe_parse_out_file = open(working_dir + set_name + '.dmrs.json', 'w')
    amr_out_file = open(working_dir + set_name + '.amr', 'w')

    # accumulate statistics
    num_nodes = 0
    num_phrasal_nodes = 0.0
    num_overlapping_nodes = 0.0
    num_spans = 0.0
    num_multi_spans = 0.0

    num_outside_phrases = 0.0
    num_inside_phrases = 0.0
    num_anchored_phrases = 0.0
    num_phrase_phrase_edges = 0.0
    num_phrase_edges = 0.0
    num_word_phrase_edges = 0.0
    num_word_anchored_edges = 0.0
    num_edges = 0.0
    num_self_edges = 0.0

    surface1 = 0.0
    surface2 = 0.0
    surfacem = 0.0
    abstract1 = 0.0
    abstract2 = 0.0
    abstract5 = 0.0
    abstract10 = 0.0
    abstract20 = 0.0
    abstractm = 0.0

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
          for j, node in enumerate(graph.nodes):
            if node.ind == ind and node.concept == pred:
              graph.nodes[j].constant = const
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
      for j, node in enumerate(graph.nodes):
        if graph.spanned[j] and node.alignment >= 0:
          if node.constant:
            graph.nodes[j].is_aligned = True
        elif node.concept.startswith('_'): # lexical concepts
          graph.nodes[j].is_aligned = True
    
      if compute_statistics:
        num_nodes += len(graph.nodes)
        num_overlapping_nodes += graph.num_overlapping_nodes()
        num_phrasal_nodes += graph.num_phrasal_nodes()
        num_self_edges += graph.num_self_edges()

        num_multi, num_span = graph.num_unaries()
        num_multi_spans += num_multi
        num_spans += num_span

        num_outside, num_inside, num_anchored, num_phrase_phrase, num_phrase, num_edge = graph.num_phrase_phrase_spans()
        num_word_phrase, num_word_anchored, _ = graph.num_word_phrase_spans()
        s1, s2, sm, a1, a2, a5, a10, a20, am = graph.num_predicate_types()
        
        surface1 += s1
        surface2 += s2
        surfacem += sm
        abstract1 += a1
        abstract2 += a2
        abstract5 += a5
        abstract10 += a10
        abstract20 += a20
        abstractm += am

        num_outside_phrases += num_outside
        num_inside_phrases += num_inside
        num_anchored_phrases += num_anchored
        num_phrase_phrase_edges += num_phrase_phrase
        num_phrase_edges += num_phrase
        num_edges += num_edge
        num_word_phrase_edges += num_word_phrase
        num_word_anchored_edges += num_word_anchored
      
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

      if len(graph.nodes) > 0:
        epe_out_file.write((graph.json_str(i, sentences_raw[i], offset) + '\n').encode('utf-8', 'replace'))
        #offset += len(sentences_raw[i])
        epe_parse_out_file.write((graph.json_parse_str(i) + '\n').encode('utf-8', 'replace'))
        epe_orig_parse_out_file.write((graph.json_orig_parse_str(i) + '\n').encode('utf-8', 'replace'))

      if len(graph.nodes) == 0:
        amr_out_file.write('( n1 / _UNK )\n\n')
      else:
        graph.correct_constants()
        graph.correct_concept_names()
        graph.correct_node_names()
        amr_out_file.write(graph.amr_graph_str(graph.root_index, 1) + '\n\n')
    
    if compute_statistics:
      print("Partially overlapping nodes %.2f" %
              (num_overlapping_nodes/num_nodes)) 
      print("Phrasal nodes %.4f" %
              (num_phrasal_nodes/num_nodes)) 

      print("Spans vs nodes %.4f" %
              (num_spans/num_nodes)) 
      print("Multi Spans %.4f" %
              (num_multi_spans/num_spans)) 

      num_surface = surface1 + surface2 + surfacem
      num_abstract = abstract1 + abstract2 + abstractm 
      total_preds = num_surface + num_abstract
      print("Abstract predicates: %.4f" % (num_abstract/total_preds))
      print("Surface predicates: %.4f" % (num_surface/total_preds))
      print("Abstract span 1: %.4f" % (abstract1/num_abstract))
      print("Abstract span 2: %.4f" % (abstract2/num_abstract))
      print("Abstract span 5: %.4f" % (abstract5/num_abstract))
      print("Abstract span 10: %.4f" % (abstract10/num_abstract))
      print("Abstract span 20: %.4f" % (abstract20/num_abstract))
      print("Abstract span m: %.4f" % (abstractm/num_abstract))
      print("Surface span 1: %.4f" % (surface1/num_surface))
      print("Surface span 2: %.4f" % (surface2/num_surface))
      print("Surface span 3: %.4f" % (surfacem/num_surface))

      print("Outside phrases: %.4f" % (num_outside_phrases/num_phrase_phrase_edges))
      print("Inside phrases: %.4f" % (num_inside_phrases/num_phrase_phrase_edges)) 
      print("Anchored phrases: %.4f" % (num_anchored_phrases/num_phrase_phrase_edges))
      print("Phrase phrase vs phrase edges: %.4f" % (num_phrase_phrase_edges/num_phrase_edges))
      print("Phrase phrase vs all edges: %.4f" % (num_phrase_phrase_edges/num_edges))
      print("Phrase edges vs all edges: %.4f" % (num_phrase_edges/num_edges))
      print("Word phrase vs all edges: %.4f" % (num_word_phrase_edges/num_edges))
      print("Word anchored phrases: %.4f" % (num_word_anchored_edges/num_word_phrase_edges))
      print("Self edges: %.4f" % (num_self_edges/num_edges))

