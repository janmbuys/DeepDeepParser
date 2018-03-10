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
import string
import json

def strip_unknown_concept(s):
  if s[0] == '_' and '/' in s and '_u_unknown' in s:
    s= '_u_unknown'
  if s.endswith('_rel'):
    s = s[:s.index('_rel')]
  return s

def strip_lexeme(s):
  if s[0] == '_' and '/' in s and '_u_unknown' in s:
    s = '_u_unknown'
  elif s[0] == '_':
    if '_' in s[1:]:
      sind = s.index('_', 1)
      lex = s[:sind]
      if '+' in lex[:-1]:
        s = '_' + s[s.index('+'):]
      elif '-' in lex[:-1]:
        s = '_' + s[s.index('-'):]
      else:
        s = s[sind:]
    else:
      print 'keeping lexeme', s
  if s.endswith('_rel'):
    s = s[:s.index('_rel')]
  return s

class MrsNode():
  def __init__(self, name, concept='', alignment=-1, alignment_end=-1):
    self.name = name
    self.concept = concept
    self.alignment = alignment
    self.alignment_end = alignment_end
    self.is_aligned = False
    self.ne = None
    self.pos = None
    self.ind = ''
    self.constant = ''
    self.pred_type = ''
    self.recover_mrs_str = ''
    self.features = []
    self.morph_tag = ''
    self.edge_names = [] # these are child nodes
    self.edges = []
    self.relations = []
    self.heads = []
    self.span_edges = [] # child nodes
    self.span_left_edges = [] # left node inds
    self.span_right_edges = [] # right node inds
    self.span_relations = []
    self.span_reent = [] # bools
    self.spanned_min_alignment = -1
    self.spanned_max_alignment = -1

  def alignment_str(self):
    assert self.alignment >=0 and self.alignment_end >= 0
    return str(self.alignment) + ':' + str(self.alignment_end)

  def __str__(self):
    s = self.name + ' ' + self.concept + ' ' + str(self.alignment) + ' '
    if self.constant:
      s += '(' + self.constant + ')'
    for i in xrange(len(self.edge_names)):
      s += (' :' + self.relations[i] + ' ' + str(self.edges[i]) + ' ' 
          + self.edge_names[i])
    return s
 
  def append_edge(self, child_index, relation):
    self.edges.append(child_index)
    self.relations.append(relation)

class MrsGraph():
  def __init__(self, nodes, root_index=-1, parse_ind=-1):
    self.nodes = nodes
    self.root_index = root_index
    self.parse_ind = parse_ind
    self.spanned = [False for _ in nodes]

  @classmethod
  def parse_orig_json_line(cls, json_line):
    mrs = json.loads(json_line)
    parse_id = mrs["id"] # need to return to ensure correct alignment 
    nodes = []
    top_ind = -1
    nodes_index = {}

    # First parse nodes
    for node in mrs["nodes"]:
      node_id = node["id"] - 1
      props = node["properties"] 
      if node.has_key("top") and node["top"]:
        top_ind = node_id 
      concept = props["predicate"]
      graph_node = MrsNode(str(node_id), concept)

      start_ind = node["start"]
      end_ind = node["end"]
      graph_node.ind = str(start_ind) + ":" + str(end_ind)
      if props.has_key("constant"):
        const = props["constant"]
        if const[0] != '"':
          const = '"' + const
        if const[-1] != '"':
          const = const + '"' 
        graph_node.constant = const

      nodes_index[node_id] = len(nodes)
      # ignore features for now

      nodes.append(graph_node)
    # Then add edges 
    for node in mrs["nodes"]:
      parent_ind = nodes_index[node["id"] - 1]
      if node.has_key("edges"):
        for edge in node["edges"]:
          child_ind = nodes_index[edge["target"] - 1]
          label = edge["label"]
          if label == "/EQ":
            label = "/EQ/U"
          nodes[parent_ind].append_edge(child_ind, label)
          nodes[child_ind].heads.append(parent_ind)

    return cls(nodes, top_ind, parse_id)


  @classmethod
  def parse_json_line(cls, json_line):
    mrs = json.loads(json_line)
    parse_id = mrs["id"] # need to return to ensure correct alignment 
    nodes = []
    top_ind = -1
    nodes_index = {}

    # First parse nodes
    for node in mrs["nodes"]:
      node_id = node["id"] - 1
      props = node["properties"] 
      if node.has_key("top") and node["top"]:
        top_ind = node_id 
      if props.has_key("abstract") and props["abstract"]:
        concept = props["predicate"]
      else:
        concept = '_' + props["predicate"]

      graph_node = MrsNode(str(node_id), concept, node["start"]-1, node["end"]-1)
      nodes_index[node_id] = len(nodes)
      # ignore features for now

      nodes.append(graph_node)
    # Then add edges 
    for node in mrs["nodes"]:
      parent_ind = nodes_index[node["id"] - 1]
      if node.has_key("edges"):
        for edge in node["edges"]:
          child_ind = nodes_index[edge["target"] - 1]
          label = edge["label"]
          if label == "/EQ":
            label = "/EQ/U"
          nodes[parent_ind].append_edge(child_ind, label)
          nodes[child_ind].heads.append(parent_ind)

    return cls(nodes, top_ind, parse_id)


  def correct_node_names(self):
    for j, node in enumerate(self.nodes):
      self.nodes[j].name = 'n' + node.name

  def correct_concept_names(self):
    for j, node in enumerate(self.nodes):
      concept = re.sub(r'\/', '|', node.concept)  
      concept = concept.replace('(', '_')
      concept = concept.replace(')', '_')
      concept = concept.replace(':', '')
      concept = concept.replace('.', '')
      if concept == '':
        concept = '_'
      self.nodes[j].concept = concept

  def correct_constants(self):
    for j, node in enumerate(self.nodes):
      # Transformations to not break smatch evaluator.
      if node.constant:
        constant = node.constant.replace('"', '') 
        constant = constant.replace(':', '_')
        constant = constant.replace('(', '_')
        constant = constant.replace(')', '_')
        constant = constant.strip()
        constant = constant.replace(' ', '_')
        self.nodes[j].constant = '"' + constant + '"'

  def restore_op_indexes(self):
    # Index ops for each common parent in node order.
    # Assumes that ops dependents don't have multiple parents.
    for i, node in enumerate(self.nodes):
      ind = 1
      for j, k in enumerate(self.nodes[i].span_edges):
        rel = self.nodes[i].span_relations[j]
        if rel.startswith('op'):
          self.nodes[i].span_relations[j] = 'op' + str(ind)
          ind += 1

  def restore_original_constants(self, i, relation):
    if i  == -1:
      return
    elif (relation in ['polarity', 'polite', 'mode'] or
          self.nodes[i].concept == '-'):
      self.nodes[i].constant = self.nodes[i].concept
    for j, child_index in enumerate(self.nodes[i].span_edges):
      if not self.nodes[i].span_reent[j]:
        rel = self.nodes[i].span_relations[j]
        self.restore_original_constants(child_index, rel)

  def append_edge(self, node_index, child_index, relation):
    if node_index == -1:
      self.root_index = child_index    
    else:
      self.nodes[node_index].append_edge(child_index, relation)
      self.nodes[child_index].heads.append(node_index)
 
  def get_undirected_dependencies(self):
    deps = [set() for _ in self.nodes]
    for i, node in enumerate(self.nodes):
      for edge in node.edges:
        deps[i].add(edge)
        deps[edge].add(i)
    return deps

  def ordered_predicates_str(self, lexicalized=True):
    preds = [node.concept for node in self.nodes]
    if not lexicalized:
      preds = map(strip_lexeme, preds)
    else:
      preds = map(strip_unknown_concept, preds)
    preds = [pred + '_CARG' if self.nodes[i].constant else pred    
             for i, pred in enumerate(preds)]
    inds = [node.alignment for node in self.nodes]

    out_str = ''
    pred_str = ''
    ind_str = ''
    for i, ind in enumerate(inds):
      out_str += '<' + str(ind) + ' ' + preds[i] + ' '
      pred_str += preds[i] + ' '
      ind_str += str(ind) + ' '
    return out_str, pred_str, ind_str

  def find_span_tree(self, i):
    self.spanned[i] = True
    for k, j in enumerate(self.nodes[i].edges):
      # Checks if reverse edge has been added.
      if not self.spanned[j] or (i not in self.nodes[j].span_edges): 
        rel = self.nodes[i].relations[k]
        self.nodes[i].span_edges.append(j)
        self.nodes[i].span_relations.append(rel)
        self.nodes[i].span_reent.append(self.spanned[j])
      if not self.spanned[j]:
        self.find_span_tree(j)
    for j in self.nodes[i].heads:
      # Checks if edge has been added.
      if not self.spanned[j]:
        # This will let reentrancy always refer to a node above:   
          #or (i not in self.nodes[j].span_edges):
        k = self.nodes[j].edges.index(i)
        rel = self.nodes[j].relations[k]
        rel = rel if rel.endswith('/U') else rel + '-of'
        self.nodes[i].span_edges.append(j)
        self.nodes[i].span_relations.append(rel)
        self.nodes[i].span_reent.append(self.spanned[j])
      if not self.spanned[j]:
        self.find_span_tree(j)

  def find_alignment_spans(self, i):
    inds = [self.nodes[i].alignment]
    for k in xrange(len(self.nodes[i].span_edges)):
      j = self.nodes[i].span_edges[k]
      if not self.nodes[i].span_reent[k]:
        self.find_alignment_spans(j)
        inds.append(self.nodes[j].spanned_min_alignment)
        inds.append(self.nodes[j].spanned_max_alignment)
    self.nodes[i].spanned_min_alignment = min(inds)
    self.nodes[i].spanned_max_alignment = max(inds)


  def num_phrasal_nodes(self):
    count = 0
    for i, node_a in enumerate(self.nodes):
      if node_a.alignment_end >= node_a.alignment + 1: # orginally 2
        count += 1
    return count

  def num_self_edges(self):
    # consider phrase to phrase edges
    edge_count = 0
   
    for i, node_a in enumerate(self.nodes):
      for k, child_index in enumerate(node_a.edges):
        node_b = self.nodes[child_index]
        if node_a.alignment == node_b.alignment and node_a.alignment_end == node_b.alignment_end:
          edge_count += 1

    return edge_count
 

  def num_word_phrase_spans(self):
    # consider phrase to phrase edges
    edge_count = 0
    anchored_count = 0
    total_edge_count = 0
   
    for i, node_a in enumerate(self.nodes):
      total_edge_count += len(node_a.edges)
      if node_a.alignment_end == node_a.alignment:
        for k, child_index in enumerate(node_a.edges):
          node_b = self.nodes[child_index]
          if node_b.alignment_end >= node_b.alignment + 2:
            edge_count += 1
            if (node_b.alignment == node_a.alignment 
                or node_b.alignment_end == node_a.alignment_end):
              anchored_count += 1
 
    return edge_count, anchored_count, total_edge_count


  def num_phrase_phrase_spans(self):
    # consider phrase to phrase edges
    outside_count = 0
    anchored_count = 0
    inside_count = 0
    edge_count = 0
    all_edge_count = 0
    total_edge_count = 0

    for i, node_a in enumerate(self.nodes):
      total_edge_count += len(node_a.edges)
      if node_a.alignment_end >= node_a.alignment + 1:
        for k, child_index in enumerate(node_a.edges):
            all_edge_count += 1
            node_b = self.nodes[child_index]
            if node_b.alignment_end >= node_b.alignment + 1:
              edge_count += 1
              if (node_b.alignment_end < node_a.alignment 
                  or node_b.alignment > node_a.alignment_end):
                outside_count += 2
              elif (node_b.alignment == node_a.alignment 
                    or node_b.alignment_end == node_a.alignment_end):
                anchored_count += 1
              else:
                inside_count += 1
                #print node_a.alignment, node_a.alignment_end, node_b.alignment, node_b.alignment_end
 
    return outside_count, anchored_count, inside_count, edge_count, all_edge_count, total_edge_count


  def num_predicate_types(self):
    surface1 = 0
    surface2 = 0
    surfacem = 0
    abstract1 = 0
    abstract2 = 0
    abstract5 = 0
    abstract10 = 0
    abstract20 = 0
    abstractm = 0

    for i, node_a in enumerate(self.nodes):
      if node_a.concept[0] == '_':
        if node_a.alignment == node_a.alignment_end:
          surface1 += 1
        elif node_a.alignment + 1 == node_a.alignment_end:
          surface2 += 1
          #print node_a.concept
        else:  
          surfacem += 1
      else:
        if node_a.alignment == node_a.alignment_end:
          abstract1 += 1
        elif node_a.alignment + 1 == node_a.alignment_end:
          abstract2 += 1
        else:  
          abstractm += 1
          if node_a.alignment + 4 <= node_a.alignment_end:
            abstract5 += 1
          if node_a.alignment + 9 <= node_a.alignment_end:
            abstract10 += 1
          if node_a.alignment + 19 <= node_a.alignment_end:
            abstract20 += 1

    return surface1, surface2, surfacem, abstract1, abstract2, abstract5, abstract10, abstract20, abstractm


  def num_unaries(self):
    ind_map = {} #TODO self-edges
    for i, node_a in enumerate(self.nodes):
      ind = str(node_a.alignment) + ':' + str(node_a.alignment_end)
      if ind_map.has_key(ind):
        ind_map[ind] += 1
      else:
        ind_map[ind] = 1
    multi_spans = 0
    for s in ind_map.iterkeys():
      if ind_map[s] > 1:
        multi_spans += 1

    return multi_spans, len(ind_map) #len(self.nodes)
  
  def num_overlapping_nodes(self):
    count = 0
    for i, node_a in enumerate(self.nodes):
      for j in range(i+1, len(self.nodes)):
        node_b = self.nodes[j]
        if node_a.alignment < node_b.alignment:
          if node_b.alignment_end > node_a.alignment_end and node_a.alignment_end > node_b.alignment:
            count += 1
        elif node_a.alignment > node_b.alignment:
          if node_a.alignment_end > node_b.alignment_end and node_b.alignment_end > node_a.alignment:
            count += 1
    return count


  def find_span_edge_directions(self):
    for i in xrange(len(self.nodes)):
      node_ind = self.nodes[i].alignment
      left_edges = []
      right_edges = []
      for k in xrange(len(self.nodes[i].span_edges)):
        if (not self.nodes[i].span_reent[k] 
            and self.nodes[self.nodes[i].span_edges[k]].alignment < node_ind):
          left_edges.append(k)
        else: # includes same alignment and re-entrancies
          right_edges.append(k)

      def order_func(k):
        span_ind = self.nodes[i].span_edges[k]
        return (self.nodes[span_ind].alignment, 
                self.nodes[span_ind].spanned_min_alignment,
                1 if len(self.nodes[span_ind].span_edges) == 0 else 0,
                self.nodes[span_ind].spanned_max_alignment,
                self.nodes[span_ind].alignment_end)
      left_edges = sorted(left_edges, key=order_func)
      right_edges = sorted(right_edges, key=order_func)

      self.nodes[i].span_left_edges = left_edges
      self.nodes[i].span_right_edges = right_edges

  def recover_end_spans(self, i, head_i):
    start = self.nodes[i].alignment
    end = self.nodes[i].alignment
    is_aligned = start <> -1
    children = set()
    for k in xrange(len(self.nodes[i].span_edges)):
      if not self.nodes[i].span_reent[k]: 
        edge_index = self.nodes[i].span_edges[k]
        children.add(edge_index)
        child_start, child_end = self.recover_end_spans(edge_index, i)
        if child_start <> child_end:
          print child_start, child_end
        if not is_aligned and child_start >= 0:
          start = child_start
          end = child_end
          is_aligned = True
        elif child_start >= 0:
          # Disjoint spans.
          if child_start >= end:
            end = child_end
          elif child_end <= start:
            start = child_start
          # Covering spans.
          elif child_start <= start and child_end >= end:
            start, end = child_start, child_end
          # Partially overlapping spans.
          elif child_start < start and child_end <= end:
            start = child_start
          elif child_start >= start and child_end > end:
            end = child_end

    if (self.nodes[i].concept.endswith('_CARG') 
        or self.nodes[i].concept.startswith('_')):
      self.nodes[i].alignment_end = self.nodes[i].alignment
    elif is_aligned:
      self.nodes[i].alignment_end = end

    # Span-inverted heads.
    for j in self.nodes[i].heads:
      if j <> head_i and not (self.nodes[j].concept.endswith('_CARG') 
          or self.nodes[j].concept.startswith('_')): 
        # Only if not yet spanned, i.e. have no children in span tree.
        if self.nodes[j].alignment == -1:
          self.nodes[j].alignment = start 
          self.nodes[j].alignment_end = end

    return start, end

  def alignment_order_node_traversal(self):
    traversal = self.inorder_node_traversal(self.root_index)
    alignments = [self.nodes[i].alignment for i in traversal]
    sorted_traversal = [t for (a, r, t) in sorted(zip(
        alignments, range(len(traversal)), traversal))]
    sorted_alignments = [self.nodes[i].alignment for i in sorted_traversal]
    return sorted_traversal

  """ Given a list of items and a list of indexes, Sort the list such that all 
      occurances of each index are grouped together, in the position of the 
      longest, or first original group. """
  def clean_inorder_node_traversal(self):
    traversal = self.inorder_node_traversal(self.root_index)
    alignments = [self.nodes[i].alignment for i in traversal]
    alignment_list = sorted(list(set(alignments)))
    # For every alignment index, records best group length and index.
    groups_per_alignment = {} 
    for a in alignment_list:
      groups_per_alignment[a] = (0, -1)

    align = alignments[0]
    group_start = 0
    for j in xrange(1, len(alignments) + 1):
      if j == len(alignments) or alignments[j] <> align:
        group_length = j - group_start
        if group_length > groups_per_alignment[align][0]: 
          groups_per_alignment[align] = (group_length, group_start)
        if j < len(alignments):
          align = alignments[j]
          group_start = j

    # Finds index of best group for each alignment.
    alignment_group_indexes = [groups_per_alignment[a][1] for a in alignments]
    sorted_traversal = [t for (a, r, t) in sorted(zip(
        alignment_group_indexes, range(len(traversal)), traversal))]
    return sorted_traversal
   
  def inorder_node_traversal(self, i):
    s = []
    for k in self.nodes[i].span_left_edges:
      assert not self.nodes[i].span_reent[k]
      s += self.inorder_node_traversal(self.nodes[i].span_edges[k])
    s.append(i)
    for k in self.nodes[i].span_right_edges:
      if not self.nodes[i].span_reent[k]: 
        s += self.inorder_node_traversal(self.nodes[i].span_edges[k])
    return s

  def dmrs_arceager_oracle_str(self, alignment_order='inorder',
                               include_lexemes=True):
    if alignment_order == 'alignorder':
      node_order = self.alignment_order_node_traversal()
    elif alignment_order == 'cleaninorder':
      node_order = self.clean_inorder_node_traversal()
    elif alignment_order == 'inorder':
      node_order = self.inorder_node_traversal(self.root_index)
    else:
      assert False, 'Invalid alignment ordering.'
    alignment_order = [self.nodes[i].alignment for i in node_order]

    actions, concepts, morph_tags, indexes, end_indexes = self.arceager_oracle(node_order,
        include_lexemes)
    withspan_str = ''
    nospan_str = ''
    action_str = ''
    concept_str = ''
    morph_tag_str = ''
    point_str = ''
    end_point_str = ''
    last_shift_index = -1

    for i, action in enumerate(actions):
      if i == len(actions) - 1 and action == 'SH':
        break # doesn't write EOS
      if action == 'SH' or action == 'I':
        withspan_str += '<' + str(indexes[i]) + ' ' + concepts[i] + ' '
        nospan_str += concepts[i] + ' '
        action_str += 'SH '
        concept_str += concepts[i] + ' '
        morph_tag_str += morph_tags[i] + ' '
        point_str += str(indexes[i]) + ' ' 
        end_point_str += str(end_indexes[i]) + ' ' 
        last_shift_index = indexes[i]
      elif action == 'RE':
        withspan_str += 'RE ' + str(end_indexes[i]) + '> '
        nospan_str += 'RE ' 
        action_str += 'RE '
        concept_str += '_ '
        morph_tag_str += '_ '
        point_str += str(indexes[i]) + ' ' 
        end_point_str += str(end_indexes[i]) + ' ' 
      else:  
        withspan_str += actions[i] + ' '
        nospan_str += actions[i] + ' '
        action_str += actions[i] + ' '
        concept_str += '_ '
        morph_tag_str += '_ '
        point_str += str(max(0, indexes[i])) + ' '
        end_point_str += str(end_indexes[i]) + ' ' 
    return withspan_str, nospan_str, action_str, concept_str, morph_tag_str, point_str, end_point_str


  def dmrs_arceager_str(self, i, right_relation='', include_lexemes=True, include_align=True):
    if include_lexemes:
      concept = strip_unknown_concept(self.nodes[i].concept)
    else:  
      concept = strip_lexeme(self.nodes[i].concept)
    if self.nodes[i].constant:
      # Views constant as a property, to be recovered from span.
      concept += '_CARG'

    s = ''
    left_relations = []
    for k in self.nodes[i].span_left_edges:
      assert not self.nodes[i].span_reent[k]
      s += self.dmrs_arceager_str(self.nodes[i].span_edges[k], '',
          include_lexemes, include_align)
      rel = self.nodes[i].span_relations[k]
      if rel.endswith('-of'):
        left_relation = 'RA:' + rel[:-3] + ' ) '  
      elif rel.endswith('/U'):
        left_relation = 'UA:' + rel[:-2] + ' ) '
      else:
        left_relation = 'LA:' + rel + ' ) '  

      if include_align:
        left_relation += str(
          self.nodes[self.nodes[i].span_edges[k]].alignment_end) + '> '
      left_relations.append(left_relation)

    s += ''.join(left_relations[::-1])
    if right_relation <> '':
      if right_relation.endswith('-of'):
        s += 'LA:' + right_relation[:-3] + ' '
      elif right_relation.endswith('/U'):
        s += 'UA:' + right_relation[:-2] + ' '
      else:
        s += 'RA:' + right_relation + ' '
    if include_align:
      s += '<' + str(self.nodes[i].alignment) + ' '
    s += concept + ' '

    for k in self.nodes[i].span_right_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = '<' + str(
            self.nodes[self.nodes[i].span_edges[k]].alignment) + ' '
        re_span_end = str(
            self.nodes[self.nodes[i].span_edges[k]].alignment_end) + '> '
        if include_lexemes:
          re_concept = strip_unknown_concept(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        else:
          re_concept = strip_lexeme(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          re_concept += '_CARG'
        if include_align:
          s += ('RA*:' + self.nodes[i].span_relations[k] + ' ' 
              + re_span_start + re_concept + ' ) ' + re_span_end)
        else:
          s += ('RA*:' + self.nodes[i].span_relations[k] + ' ' 
              + re_concept + ' ) ')
      else:
        inner_s = self.dmrs_arceager_str(self.nodes[i].span_edges[k],
            self.nodes[i].span_relations[k], include_lexemes, include_align)
        s += inner_s + ') '
        if include_align:
          s += str(self.nodes[self.nodes[i].span_edges[k]].alignment_end) + '> '

    return s


  def dmrs_arceager_point_str(self, i, is_right_dependent):
    s = ''
    left_relations = []
    for k in self.nodes[i].span_left_edges:
      assert not self.nodes[i].span_reent[k]
      s += self.dmrs_arceager_point_str(self.nodes[i].span_edges[k], False)
      span_end = str(self.nodes[self.nodes[i].span_edges[k]].alignment_end)
      left_relation = span_end + ' ' + span_end + ' '
      left_relations.append(left_relation)

    s += ''.join(left_relations[::-1])
    if is_right_dependent:
      s += str(self.nodes[i].alignment) + ' ' 
    s += str(self.nodes[i].alignment) + ' '

    for k in self.nodes[i].span_right_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = str(
            self.nodes[self.nodes[i].span_edges[k]].alignment) + ' '
        re_span_end = str(
            self.nodes[self.nodes[i].span_edges[k]].alignment_end) + ' '
        s += re_span_start + re_span_start + re_span_end
      else:
        s += self.dmrs_arceager_point_str(self.nodes[i].span_edges[k], True)
        s += str(self.nodes[self.nodes[i].span_edges[k]].alignment_end) + ' '

    return s


  def dmrs_inorder_str(self, i, include_lexemes=True):
    if include_lexemes:
      concept = strip_unknown_concept(self.nodes[i].concept)
    else:  
      concept = strip_lexeme(self.nodes[i].concept)
    if self.nodes[i].constant:
      concept += '_CARG'

    span_start = str(self.nodes[i].alignment)
    span_end = str(self.nodes[i].alignment_end)
    span_str = '<' + span_start + ' ' + span_end + '> '

    s = ''
    ind_s = ''
    for k in self.nodes[i].span_left_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = str(self.nodes[self.nodes[i].span_edges[k]].alignment)
        re_span_end = str(self.nodes[self.nodes[i].span_edges[k]].alignment_end)
        re_span_str = '<' + re_span_start + ' ' + re_span_end + '> '
        if include_lexemes:
          re_concept = strip_unknown_concept(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        else:
          re_concept = strip_lexeme(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          re_concept += '_CARG'
        s += (':' + self.nodes[i].span_relations[k] + '*( ' 
            + re_span_str + re_concept + ' ) ')
        ind_s += '*' + re_span_start + ' '
      else:
        inner_s, inner_ind = self.dmrs_inorder_str(self.nodes[i].span_edges[k], include_lexemes)

        s += ':' + self.nodes[i].span_relations[k] + '( ' + inner_s + ') '
        ind_s += inner_ind
    s += span_str + concept + ' '
    ind_s += span_start + ' '
    for k in self.nodes[i].span_right_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = str(self.nodes[self.nodes[i].span_edges[k]].alignment)
        re_span_end = str(self.nodes[self.nodes[i].span_edges[k]].alignment_end)
        re_span_str = '<' + re_span_start + ' ' + re_span_end + '> '
        if include_lexemes:
          re_concept = strip_unknown_concept(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        else:
          re_concept = strip_lexeme(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          re_concept += '_CARG'
        s += (':' + self.nodes[i].span_relations[k] + '*( ' 
            + re_span_str + re_concept + ' ) ')
        ind_s += '*' + re_span_start + ' '
      else:
        inner_s, inner_ind = self.dmrs_inorder_str(self.nodes[i].span_edges[k], include_lexemes)

        s += ':' + self.nodes[i].span_relations[k] + '( ' + inner_s + ') '
        ind_s += inner_ind

    return s, ind_s
      
  def dmrs_inorder_ha_str(self, i, track_span):
    concept = strip_unknown_concept(self.nodes[i].concept)
    if self.nodes[i].constant:
      concept += '_CARG'

    span_start = str(self.nodes[i].alignment)

    s = ''
    ind_s = ''
    for k in self.nodes[i].span_left_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = str(self.nodes[self.nodes[i].span_edges[k]].alignment)

        re_concept = strip_unknown_concept(
            self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          re_concept += '_CARG'
        s += (':' + self.nodes[i].span_relations[k] + '*( ' 
            + re_concept + ' ) ')
        ind_s += track_span + ' ' + re_span_start + ' ' + re_span_start + ' '
      else:
        inner_s, inner_ind, new_track_span = self.dmrs_inorder_ha_str(
            self.nodes[i].span_edges[k], track_span)

        s += ':' + self.nodes[i].span_relations[k] + '( ' + inner_s + ') '
        ind_s += track_span + ' ' + inner_ind + new_track_span + ' '
        track_span = new_track_span
    s += concept + ' '
    ind_s += span_start + ' '
    track_span = span_start
    for k in self.nodes[i].span_right_edges:
      if self.nodes[i].span_reent[k]: 
        re_span_start = str(self.nodes[self.nodes[i].span_edges[k]].alignment)

        re_concept = strip_unknown_concept(
            self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          re_concept += '_CARG'
        s += (':' + self.nodes[i].span_relations[k] + '*( ' 
            + re_concept + ' ) ')
        ind_s += track_span + ' ' + re_span_start + ' ' + re_span_start + ' '
      else:
        inner_s, inner_ind, new_track_span = self.dmrs_inorder_ha_str(
            self.nodes[i].span_edges[k], track_span)

        s += ':' + self.nodes[i].span_relations[k] + '( ' + inner_s + ') '
        ind_s += track_span + ' ' + inner_ind + new_track_span + ' '
        track_span = new_track_span

    return s, ind_s, track_span

  def dmrs_str(self, i, include_lexemes=True, include_align=True):
    if include_lexemes:
      s = strip_unknown_concept(self.nodes[i].concept)
    else:
      s = strip_lexeme(self.nodes[i].concept)

    if self.nodes[i].constant:
      s += '_CARG'
    for k in xrange(len(self.nodes[i].span_edges)):
      span_start = '<' + str(self.nodes[self.nodes[i].span_edges[k]].alignment)
      span_end = str(self.nodes[self.nodes[i].span_edges[k]].alignment_end) + '>'
      if self.nodes[i].span_reent[k]: 
        # Does not distinguish re-entrancies and colapsed nodes
        # on span and concept identity.
        if include_lexemes:
          concept = strip_unknown_concept(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        else:
          concept = strip_lexeme(
              self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          concept += '_CARG'
        
        s += ' ' + ':' + self.nodes[i].span_relations[k] + '*( ' 
        if include_align:
          s += span_start + ' ' + concept + ' ) ' + span_end
        else:
          s += concept + ' )'
      else:
        s += ' ' + ':' + self.nodes[i].span_relations[k] + '( ' 
        if include_align:
          s += span_start + ' '
        s += (self.dmrs_str(self.nodes[i].span_edges[k],
                                     include_lexemes, include_align) + ' )')
        if include_align:
          s += ' ' + span_end
    return s

  def const_dict_str(self, tokens, sentence_str):
    entries = []

    for node in self.nodes:
      ind_start = int(node.ind.split(':')[0])
      ind_end = int(node.ind.split(':')[1])
      sub_str = sentence_str[ind_start:ind_end]
      sub_str = sub_str.replace(' ', '_')
      token_str = tokens[node.alignment]
      if node.constant:
        entries.append(token_str + ' ' + node.constant[1:-1])
    return '\n'.join(entries)

  def predicate_dict_str(self, tokens, sentence_str):
    entries = []
    dic = ['' for _ in tokens]

    for node in self.nodes: # only lexical predicates
      ind_start = int(node.ind.split(':')[0])
      ind_end = int(node.ind.split(':')[1])
      sub_str = sentence_str[ind_start:ind_end]
      sub_str = sub_str.replace(' ', '_')
      token_str = tokens[node.alignment]
      if node.concept[0] == '_' and '/' not in node.concept:
        concept = node.concept[:node.concept.index('_', 1)]
        if dic[node.alignment]:
          if '-' in dic[node.alignment]: # overwrites if hyphenated
            dic[node.alignment] = concept
        else:
          dic[node.alignment] = concept
    for i, concept in enumerate(dic):
      if concept:
        entries.append(tokens[i] + ' ' + concept)
    return '\n'.join(entries)

  """DMRS linear with no spans."""
  def dmrs_ns_str(self, i):
    s = strip_unknown_concept(self.nodes[i].concept)
    if self.nodes[i].constant:
      s += '_CARG'
    for k in xrange(len(self.nodes[i].span_edges)):
      span_str = '< > '

      if self.nodes[i].span_reent[k]: 
        concept = strip_unknown_concept(
            self.nodes[self.nodes[i].span_edges[k]].concept)
        if self.nodes[self.nodes[i].span_edges[k]].constant:
          concept += '_CARG'
        s += (' ' + ':' + self.nodes[i].span_relations[k] + '*( ' 
            + span_str + concept + ' )')
      else:
        s += (' ' + ':' + self.nodes[i].span_relations[k] + '( ' 
          + span_str + self.dmrs_ns_str(self.nodes[i].span_edges[k]) + ' )')
    return s

  def dmrs_point_str(self, i):
    # Needs to shift sequence one right for output.
    s = str(self.nodes[i].alignment) + ' ' 

    for k in xrange(len(self.nodes[i].span_edges)):
      span_start = str(self.nodes[self.nodes[i].span_edges[k]].alignment)
      span_end = str(self.nodes[self.nodes[i].span_edges[k]].alignment_end)
                
      s += span_start + ' '
      if self.nodes[i].span_reent[k]:
        s += span_start + ' '
      else:
        s += self.dmrs_point_str(self.nodes[i].span_edges[k])
      s += span_end + ' '
    return s


  def parse_preds(self, dmrs_list, i):
    node_index = len(self.nodes)
    node = MrsNode('n' + str(node_index + 1))
    while i < len(dmrs_list) and dmrs_list[i][0] <> '<':
      i += 1
    if i + 1 >= len(dmrs_list):
      return node_index, i
    span_start = int(dmrs_list[i][1:])
    concept = dmrs_list[i+1]
    i += 2
    node.alignment = span_start
    node.alignment_end = span_start
    node.concept = concept
    self.nodes.append(node)
    return self.parse_preds(dmrs_list, i)

  def parse_linear_inorder(self, dmrs_list, i, root_index=False):
    verbatim = False
    node_index = len(self.nodes)
    node = MrsNode('n' + str(node_index + 1))
    self.nodes.append(node)
    while (i < len(dmrs_list) and dmrs_list[i].startswith(':') 
           and dmrs_list[i].endswith('(')):
      relation = dmrs_list[i][1:-1]
      if relation.endswith('*'):
        # re-entrancies
        if i + 3 >= len(dmrs_list):
          i = len(dmrs_list)
          continue
        re_relation = relation[:-1]
        if dmrs_list[i+1].startswith('<') and dmrs_list[i+2].endswith('>'):
          child_span_start = int(dmrs_list[i+1][1:])
          child_span_end = int(dmrs_list[i+2][:-1])
          child_concept = dmrs_list[i+3]
          i += 4
        else:
          child_span_start = 0
          child_span_end = 0
          child_concept = dmrs_list[i+1]
          i += 1
          while i < len(dmrs_list) and dmrs_list[i] <> ')':
            i += 1
        child_index = -1
        # Finds re-entrancy node if possible.
        inds = filter(lambda x: self.nodes[x].concept == child_concept, 
                      range(len(self.nodes)-1, -1, -1))
        for ind in inds:
          if (child_span_start == self.nodes[ind].alignment 
              and child_span_end == self.nodes[ind].alignment_end):
            child_index = ind
        if child_index == -1 and inds:
          child_index = inds[0]        
        if child_index == -1:
          child_index = len(self.nodes)
          self.nodes.append(MrsNode('n' + str(child_index + 1), child_concept,
              child_span_start, child_span_end))
        if child_index not in self.nodes[node_index].edges:
          self.append_edge(node_index, child_index, re_relation)  
      else:
        child_index, i = self.parse_linear_inorder(dmrs_list, i+1)
        if relation.endswith('-of'):
          relation = relation[:relation.index('-of')]
          self.append_edge(child_index, node_index, relation)  
        else:
          self.append_edge(node_index, child_index, relation)  
      while i < len(dmrs_list) and dmrs_list[i] == ')':
        i += 1
      if (i + 1 < len(dmrs_list) and dmrs_list[i+1].startswith(':')
          and dmrs_list[i+1].endswith('(')):
        i += 1    

    self.nodes[node_index].concept = 'unknown_rel'
    while (i < len(dmrs_list) and not 
        (dmrs_list[i].startswith('<') or dmrs_list[i].endswith('>'))):
      if not (dmrs_list[i].startswith(':') and dmrs_list[i].endswith('(')):
        self.nodes[node_index].concept = dmrs_list[i]
      if verbatim:
        print dmrs_list[i]
      i += 1  

    if i + 2 >= len(dmrs_list):
      if i < len(dmrs_list) and verbatim:
        print dmrs_list[i:]
      return node_index, i

    if dmrs_list[i].startswith('<') and dmrs_list[i+1].endswith('>'):
      span_start = int(dmrs_list[i][1:])
      span_end = int(dmrs_list[i+1][:-1])
    elif dmrs_list[i].startswith('<'): 
      span_start = int(dmrs_list[i][1:])
      span_end = span_start
      i -= 1
    elif dmrs_list[i].endswith('>'):
      span_start = int(dmrs_list[i][:-1])
      span_end = span_start
      i -= 1
    else:
      print i, dmrs_list[i]
      print dmrs_list
      assert False
    concept = dmrs_list[i+2]
    self.nodes[node_index].alignment = span_start
    self.nodes[node_index].alignment_end = span_end
    self.nodes[node_index].concept = concept
    i += 3
    while (i < len(dmrs_list) and not (dmrs_list[i].startswith(':') 
        and dmrs_list[i].endswith('(')) and not dmrs_list[i] == ')'):
      i += 1
    while (i < len(dmrs_list) and dmrs_list[i].startswith(':') 
           and dmrs_list[i].endswith('(')):
      relation = dmrs_list[i][1:-1]
      if relation.endswith('*'):
        if i + 3 >= len(dmrs_list):
          i = len(dmrs_list)
          continue
        # Reentrancies.
        re_relation = relation[:-1]
        if dmrs_list[i+1].startswith('<') and dmrs_list[i+2].endswith('>'):
          child_span_start = int(dmrs_list[i+1][1:])
          child_span_end = int(dmrs_list[i+2][:-1])
          child_concept = dmrs_list[i+3]
          i += 4
        else:
          child_span_start = 0
          child_span_end = 0
          child_concept = dmrs_list[i+1]
          i += 1
          while i < len(dmrs_list) and dmrs_list[i] <> ')':
            i += 1
        child_index = -1
        # Finds re-entrancy node if possible.
        inds = filter(lambda x: self.nodes[x].concept == child_concept, 
                      range(len(self.nodes)-1, -1, -1))
        for ind in inds:
          if (child_span_start == self.nodes[ind].alignment 
              and child_span_end == self.nodes[ind].alignment_end):
            child_index = ind
        if child_index == -1 and inds:
          child_index = inds[0]        
        if child_index == -1:
          child_index = len(self.nodes)
          self.nodes.append(MrsNode('n' + str(child_index + 1), child_concept,
              child_span_start, child_span_end))
        if child_index not in self.nodes[node_index].edges:
          self.append_edge(node_index, child_index, re_relation)  
      else:
        child_index, i = self.parse_linear_inorder(dmrs_list, i+1)
        if relation.endswith('-of'):
          relation = relation[:relation.index('-of')]
          self.append_edge(child_index, node_index, relation)  
        else:
          self.append_edge(node_index, child_index, relation)  
      if i < len(dmrs_list) and dmrs_list[i] == ')':
        i += 1
      # Lookahead if next item is :ARG or <.
      j = i
      while j < len(dmrs_list) and dmrs_list[j] == ')':
        j += 1
      if (j < len(dmrs_list) and dmrs_list[j].startswith(':') and 
          dmrs_list[j].endswith('(')):
        i = j 
    if root_index and i < len(dmrs_list):
      # Adds to root if not done with parsing.
      while i < len(dmrs_list) and dmrs_list[i] == ')':
        i += 1
      relation = 'ARG/NEQ'
      if i < len(dmrs_list):
        child_index, i = self.parse_linear_inorder(dmrs_list, i, True)
        self.append_edge(node_index, child_index, relation)  
    return node_index, i

  """ End span after close bracket. """
  def parse_linear_new(self, dmrs_list, i, root_index=False):
    verbose = False
    
    while i < len(dmrs_list) and not dmrs_list[i].startswith('<'):
      i += 1
    if i + 1 >= len(dmrs_list):
      return -1, len(dmrs_list)

    span_start = int(dmrs_list[i][1:])
    concept = dmrs_list[i+1]
    if concept.startswith(':') and concept.endswith('('):
      # Missing concept.
      concept = 'unknown_rel'
      i = i - 1
    elif concept == ')' or concept.startswith('<') or concept.endswith('>'): 
      concept = 'unknown_rel'
      if verbose:
        print 'Invalid concept', dmrs_list[i]

    node_index = len(self.nodes)
    node = MrsNode('n' + str(node_index), concept, span_start)
    self.nodes.append(node)
    i = i + 2
    
    while i < len(dmrs_list) and (dmrs_list[i] <> ')' or root_index):
      if dmrs_list[i] == ')': # for root_index=True
        i = i + 1
        continue
      if dmrs_list[i].startswith(':') and dmrs_list[i].endswith('('):
        # Well-formed.
        relation = dmrs_list[i][1:-1]
      else: 
        # Missing relation.
        relation = 'ARG/NEQ'
        i = i - 1
      if (relation.endswith('*') and len(dmrs_list) > i+4 
          and dmrs_list[i+1].startswith('<') and dmrs_list[i+3] == ')' 
          and dmrs_list[i+4].endswith('>')):
        # Reentrancies.
        relation = relation[:-1]
        span_start = int(dmrs_list[i+1][1:])
        child_concept = dmrs_list[i+2]
        span_end = int(dmrs_list[i+4][:-1])
        child_index = -1
        # Finds re-entrancy node if possible.
        inds = filter(lambda x: self.nodes[x].concept == child_concept, range(len(self.nodes)-1, -1, -1))
        align_inds = filter(lambda x: self.nodes[x].concept <> 'CONST' and self.nodes[x].alignment == span_start, range(len(self.nodes)-1, -1, -1))
        match_inds = set(inds).intersection(set(align_inds))
        if match_inds:
          child_index = list(match_inds)[0]
        elif inds:
          child_index = inds[0]
        elif align_inds:
          child_index = align_inds[0]
        else:
          child_index = len(self.nodes)
          self.nodes.append(MrsNode('n' + str(child_index), child_concept,
            span_start, span_end))
        # Re-entrancies not inverted.
        if child_index not in self.nodes[node_index].edges:
          self.append_edge(node_index, child_index, relation)  
        i += 5
      else:
        if relation.endswith('*'):
          relation = relation[:-1]
        child_index, i = self.parse_linear_new(dmrs_list, i+1) # recursive call
        if i+1 < len(dmrs_list) and dmrs_list[i] == ')' and dmrs_list[i+1].endswith('>'):
          span_end = int(dmrs_list[i+1][:-1])
          i += 2
        elif i < len(dmrs_list) and dmrs_list[i].endswith('>'):
          span_end = int(dmrs_list[i][:-1])
          i += 1
        else:
          span_end = -1

        if child_index >= 0:
          self.nodes[child_index].alignment_end = max(self.nodes[child_index].alignment, span_end)
          if relation.endswith('-of'):
            relation = relation[:relation.index('-of')]
            self.append_edge(child_index, node_index, relation)  
          else:
            self.append_edge(node_index, child_index, relation)  
    return node_index, i

  """Find oracle transition sequence."""
  def arceager_oracle(self, node_order, include_lexemes=True):
    actions = []
    concepts = []
    morph_tags = []
    indexes = []
    end_indexes = []
    recorded_edges = [[] for _ in self.nodes]
    all_undirected_deps = self.get_undirected_dependencies()

    stack = []
    buffer_node_counter = 0
    # First action generates first word on buffer, doesn't actually shift.
    actions.append('I')
    buffer_next = node_order[buffer_node_counter]
    concept = self.nodes[buffer_next].concept
    morph_tag = self.nodes[buffer_next].morph_tag
    if include_lexemes:
      concept = strip_unknown_concept(concept)
    else:  
      concept = strip_lexeme(concept)
    if self.nodes[buffer_next].constant:
      concept += '_CARG'
    concepts.append(concept)
    morph_tags.append(morph_tag)
    indexes.append(self.nodes[buffer_next].alignment)
    end_indexes.append(0)
    
    while buffer_node_counter < len(node_order):
      # LA or RA if exist, execute; then SH or RE.
      if stack and stack[-1] in self.nodes[buffer_next].edges:
        edge_ind = self.nodes[buffer_next].edges.index(stack[-1])
        relation = self.nodes[buffer_next].relations[edge_ind]
        if relation.endswith('/U'):
          action = 'UA:' + relation[:-2]
        else:
          action = 'LA:' + relation
        recorded_edges[buffer_next].append(stack[-1])
        actions.append(action)
        concepts.append('')
        morph_tags.append('')
        indexes.append(self.nodes[buffer_next].alignment)
        end_indexes.append(self.nodes[stack[-1]].alignment_end)
      elif stack and buffer_next in self.nodes[stack[-1]].edges:
        edge_ind = self.nodes[stack[-1]].edges.index(buffer_next)  
        relation = self.nodes[stack[-1]].relations[edge_ind]
        if relation.endswith('/U'):
          action = 'UA:' + relation[:-2]
        else:
          action = 'RA:' + relation
        recorded_edges[stack[-1]].append(buffer_next)
        actions.append(action)
        concepts.append('')
        morph_tags.append('')
        indexes.append(self.nodes[buffer_next].alignment) 
        end_indexes.append(self.nodes[stack[-1]].alignment_end)
      action = 'SH'
      # To determine reduce:
      if stack:
        # Weak condition: Blocks arcs to be added.
        stack_dependents_set = all_undirected_deps[buffer_next].intersection(
                           set(stack[:-1]))
        weak_reduce = bool(stack_dependents_set)
        # Strong condition: Has all its dependents that can be added
        # -> no dependency to node on buffer.
        if buffer_node_counter == len(node_order) - 1:
          buffer_set = set()
        else:
          buffer_set = set(node_order[buffer_node_counter+1:])
        strong_reduce = not bool (all_undirected_deps[stack[-1]].intersection(
                                  buffer_set))
        
        # For end spans: Delay if not weak condition and end span >=  buffer_next.
        end_span_covered = (self.nodes[stack[-1]].alignment_end
                            < self.nodes[buffer_next].alignment)
        # Doesn't reduce for crossing arcs from buffer.
        if ((strong_reduce and (weak_reduce or end_span_covered)) or
            buffer_node_counter == len(node_order) - 1):
          action = 'RE'
        elif weak_reduce:
          # There is an arc(s) between buffer_next and symbol on stack that
          # will be missed: Add as `re-entrancy'.
          stack_deps = filter(lambda x: x in stack_dependents_set, stack)
          stack_deps.reverse()
          for dep in stack_deps:
            stack_ind = len(stack) - stack.index(dep)
            actions.append('STACK*' + str(stack_ind))
            concepts.append('')
            morph_tags.append('')
            indexes.append(self.nodes[buffer_next].alignment) 
            end_indexes.append(self.nodes[stack[-1]].alignment_end)
            if dep in self.nodes[buffer_next].edges:
              edge_ind = self.nodes[buffer_next].edges.index(dep)
              relation = self.nodes[buffer_next].relations[edge_ind]
              if relation.endswith('/U'):
                action = 'UA:' + relation[:-2]
              else:
                action = 'LA:' + relation
              recorded_edges[buffer_next].append(dep)
              actions.append(action)
              concepts.append('')
              morph_tags.append('')
              indexes.append(self.nodes[buffer_next].alignment)
              end_indexes.append(self.nodes[stack[-1]].alignment_end)
            elif buffer_next in self.nodes[dep].edges:
              edge_ind = self.nodes[dep].edges.index(buffer_next)  
              relation = self.nodes[dep].relations[edge_ind]
              if relation.endswith('/U'):
                action = 'UA:' + relation[:-2]
              else:
                action = 'RA:' + relation
              recorded_edges[dep].append(buffer_next)
              actions.append(action)
              concepts.append('')
              morph_tags.append('')
              indexes.append(self.nodes[buffer_next].alignment) 
              end_indexes.append(self.nodes[stack[-1]].alignment_end)
          action = 'SH'
       
      # Root transition.
      if action == 'SH' and self.root_index == buffer_next:
        actions.append('ROOT')
        concepts.append('')
        morph_tags.append('')
        indexes.append(self.nodes[buffer_next].alignment)
        if stack:
          end_indexes.append(self.nodes[stack[-1]].alignment_end)
        else:
          end_indexes.append(0)

      actions.append(action)
      if action == 'SH':
        if stack:
          current_stack_top = stack[-1]
        else:
          current_stack_top = -1
        stack.append(buffer_next)
        buffer_node_counter += 1
        if buffer_node_counter == len(node_order):
          concepts.append('</s>')
          morph_tags.append('') 
          indexes.append(-1)
          if stack:
            end_indexes.append(self.nodes[current_stack_top].alignment_end)
          else:
            end_indexes.append(0)
        else:    
          buffer_next = node_order[buffer_node_counter]
          concept = self.nodes[buffer_next].concept
          morph_tag = self.nodes[buffer_next].morph_tag
          if include_lexemes:
            concept = strip_unknown_concept(concept)
          else:  
            concept = strip_lexeme(concept)
          if self.nodes[buffer_next].constant:
            concept += '_CARG'
          concepts.append(concept)
          morph_tags.append(morph_tag)
          indexes.append(self.nodes[buffer_next].alignment)
          if stack:
            end_indexes.append(self.nodes[current_stack_top].alignment_end)
          else:
            end_indexes.append(0)
      elif action == 'RE':
        concepts.append('')
        morph_tags.append('')
        indexes.append(self.nodes[buffer_next].alignment)
        end_indexes.append(self.nodes[stack[-1]].alignment_end)
        stack.pop()
    return actions, concepts, morph_tags, indexes, end_indexes

  """New configuration, shift generates next word on the buffer. """
  def parse_arceager_buffer_shift(self, dmrs_list):
    stack = []
    buffer_next = -1
    re_relation = ''
    span_start = -1
    reduce_ind = -1
    stack_point_dep = -1
    ra_state = False
    shift_state = False
    reduce_state = False
    stack_point_state = False
    for i, item in enumerate(dmrs_list):
      if item == 'RE': # reduce
        if stack:
          reduce_state = True
          reduce_ind = stack[-1]
          stack.pop()
      elif item.startswith('LA:'): # left arc
        if not stack:
          continue
        relation = item[3:]
        assert buffer_next >= 0
        if buffer_next >= 0:
          if stack_point_state:
            if stack_point_dep >= 0:
              self.append_edge(buffer_next, stack_point_dep, relation)
            stack_point_state = False
          else:
            self.append_edge(buffer_next, stack[-1], relation)
      elif item == 'ROOT':
        self.root_index = buffer_next
      elif item.startswith('RA:') or item.startswith('UA:'): # right arc
        relation = item[3:]
        if item.startswith('UA:'):
          relation += '/U'
        if buffer_next >= 0 and stack:
          if stack_point_state:
            if stack_point_dep >= 0:
              self.append_edge(stack_point_dep, buffer_next, relation)
            stack_point_state = False
          else:
            self.append_edge(stack[-1], buffer_next, relation)
      elif item.startswith('STACK*'):
        stack_point_state = True
        ind = int(item[6:])
        if stack and ind <= len(stack):
          stack_point_dep = stack[-ind]
        else:
          stack_point_dep = -1
      elif item.startswith('<'):
        span_start = int(item[1:])
        shift_state = True
      elif item.endswith('>'):
        span_end = int(item[:-1])
        if reduce_state:
          self.nodes[reduce_ind].alignment_end = span_end
          reduce_state = False
      else: # shift
        if buffer_next >= 0: # not for the first concept
          stack.append(buffer_next)
        # Generates next buffer node.
        ind = len(self.nodes)
        if shift_state:
          node = MrsNode('n' + str(ind), item, span_start)
        else:
          node = MrsNode('n' + str(ind), item)
        self.nodes.append(node)
        buffer_next = ind
        shift_state = False


  """Needs transition system to parse."""
  def parse_arceager(self, dmrs_list, shift_to_buffer=False):
    stack = []
    buffer_next = -1
    re_relation = ''
    span_start = -1
    reduce_ind = -1
    stack_point_dep = -1
    ra_state = False
    shift_state = False
    reduce_state = False
    stack_point_state = False
    for i, item in enumerate(dmrs_list):
      if item == ')': # reduce
        if stack:
          reduce_state = True
          reduce_ind = stack[-1]
          stack.pop()
      elif item.startswith('LA:'): # left arc
        if not stack:
          continue
        relation = item[3:]
        if buffer_next == -1:
          buffer_next = len(self.nodes)
          node = MrsNode('n' + str(buffer_next), '')
          self.nodes.append(node)
        if stack_point_state:
          self.append_edge(buffer_next, stack_point_dep, relation)
          stack_point_state = False
        else:
          self.append_edge(buffer_next, stack[-1], relation)
      elif item.startswith('RA*:'): # re-entrancy
        ra_state = True
        re_relation = item[4:]    
      elif item.startswith('RA:') or item.startswith('UA:'): # right arc
        ra_state = True
        relation = item[3:]
        if not (stack or relation == '/H'):
          continue
        if item.startswith('UA:'):
          relation += '/U'
        re_relation = ''
        if buffer_next == -1:
          buffer_next = len(self.nodes)
          node = MrsNode('n' + str(buffer_next), '')
          self.nodes.append(node)
        if relation == '/H':
          self.root_index = buffer_next
        elif stack:
          if stack_point_state:
            self.append_edge(stack_point_dep, buffer_next, relation)
            stack_point_state = False
          else:
            self.append_edge(stack[-1], buffer_next, relation)
      elif item.startswith('STACK*'):
        stack_point_state = True
        ind = int(item[6:])
        stack_point_dep = stack[-ind]
      elif item.startswith('<'):
        span_start = int(item[1:])
        shift_state = True
      elif item.endswith('>'):
        span_end = int(item[:-1])
        if reduce_state:
          self.nodes[reduce_ind].alignment_end = span_end
          reduce_state = False
      else: # shift
        if ra_state and re_relation <> '' and stack:
          inds = filter(lambda x: self.nodes[x].concept == item, 
                        range(len(self.nodes)-1, -1, -1))
          align_inds = filter(lambda x: self.nodes[x].alignment == span_start, range(len(self.nodes)-1, -1, -1))
          match_inds = set(inds).intersection(set(align_inds))
          if match_inds:
            child_index = list(match_inds)[0]
          elif inds:
            child_index = inds[0]
          elif align_inds:
            child_index = align_inds[0]
          else:
            child_index = len(self.nodes)
            self.nodes.append(MrsNode('n' + str(child_index), 
                                      item))
          if child_index not in self.nodes[stack[-1]].edges:
            self.append_edge(stack[-1], child_index, re_relation)  
          stack.append(child_index)
          re_relation = ''
        else:
          if buffer_next >= 0:
            self.nodes[buffer_next].concept = item
            if shift_state:
              self.nodes[buffer_next].alignment = span_start
            stack.append(buffer_next)
            buffer_next = -1
          else:
            ind = len(self.nodes)
            if shift_state:
              node = MrsNode('n' + str(ind), item, span_start)
            else:
              node = MrsNode('n' + str(ind), item)
            self.nodes.append(node)
            stack.append(ind)
        ra_state = False
        shift_state = False


  """Need transition system to parse."""
  def parse_arceager_nospan(self, dmrs_list):
    stack = []
    buffer_next = -1
    re_relation = ''
    ra_state = False
    for i, item in enumerate(dmrs_list):
      if item == ')': # reduce
        if stack:
          stack.pop()
      elif item.startswith('LA:'): # left arc
        if not stack:
          continue
        relation = item[3:]
        if buffer_next == -1:
          buffer_next = len(self.nodes)
          node = MrsNode('n' + str(buffer_next), '')
          self.nodes.append(node)
        self.append_edge(buffer_next, stack[-1], relation)
      elif item.startswith('RA*:'): # re-entrancy
        ra_state = True
        re_relation = item[4:]    
      elif item.startswith('RA:') or item.startswith('UA:'): # right arc
        ra_state = True
        relation = item[3:]
        if item.startswith('UA:'):
          relation += '/U'
        re_relation = ''
        if not (stack or relation == '/H'):
          continue
        if buffer_next == -1:
          buffer_next = len(self.nodes)
          node = MrsNode('n' + str(buffer_next), '')
          self.nodes.append(node)
        if relation == '/H':
          self.root_index = buffer_next
        elif stack:
          self.append_edge(stack[-1], buffer_next, relation)
      else: # shift
        if ra_state and re_relation <> '' and stack:
          inds = filter(lambda x: self.nodes[x].concept == item, 
                        range(len(self.nodes)-1, -1, -1))
          if inds:
            child_index = inds[0]
          else:
            child_index = len(self.nodes)
            self.nodes.append(MrsNode('n' + str(child_index + 1), 
                                      item))
          if child_index not in self.nodes[stack[-1]].edges:
            self.append_edge(stack[-1], child_index, re_relation)  
          stack.append(child_index)
          re_relation = ''
        else:
          if buffer_next >= 0:
            self.nodes[buffer_next].concept = item
            stack.append(buffer_next)
            buffer_next = -1
          else:
            ind = len(self.nodes)
            node = MrsNode('n' + str(ind), item)
            self.nodes.append(node)
            stack.append(ind)
        ra_state = False


  def parse_linear_nospan(self, dmrs_list, i, root_index=False):
    verbose = False
    
    while i < len(dmrs_list) and dmrs_list[i] == ')':
      i += 1
    if i + 1 >= len(dmrs_list):
      return -1, len(dmrs_list)

    concept = dmrs_list[i]
    if concept.startswith(':') and concept.endswith('('):
      # Missing concept   
      concept = 'unknown_rel'
      i = i - 1
    elif concept == ')': 
      concept = 'unknown_rel'
      if verbose:
        print 'Invalid concept', dmrs_list[i]

    node_index = len(self.nodes)
    node = MrsNode('n' + str(node_index + 1), concept)
    self.nodes.append(node)
    i = i + 1
    
    while i < len(dmrs_list) and (dmrs_list[i] <> ')' or root_index):
      if dmrs_list[i] == ')': # for root_index=True
        i = i + 1
        continue
      if dmrs_list[i].startswith(':') and dmrs_list[i].endswith('('):
        # Well-formed.
        relation = dmrs_list[i][1:-1]
      else: 
        # Missing relation.
        relation = 'ARG/NEQ'
        i = i - 1
      if relation.endswith('*') and len(dmrs_list) > i+2 and (len(dmrs_list) == i + 2 or dmrs_list[i+2] == ')'):
        # Re-entrancies.
        relation = relation[:-1]
        child_concept = dmrs_list[i+1]
        child_index = -1
        # Finds re-entrancy node if possible.
        inds = filter(lambda x: self.nodes[x].concept == child_concept, range(len(self.nodes)-1, -1, -1))
        if inds:
          child_index = inds[0]
        else:
          child_index = len(self.nodes)
          self.nodes.append(MrsNode('n' + str(child_index + 1), child_concept))
        # Re-entrancies not inverted.
        if child_index not in self.nodes[node_index].edges:
          self.append_edge(node_index, child_index, relation)  
        i += 3
      else:
        if relation.endswith('*'):
          relation = relation[:-1]
        child_index, i = self.parse_linear_nospan(dmrs_list, i+1) # recursive call

        if child_index >= 0:
          if relation.endswith('-of'):
            relation = relation[:relation.index('-of')]
            self.append_edge(child_index, node_index, relation)  
          else:
            self.append_edge(node_index, child_index, relation)  
    return node_index, i + 1
 

  def parse_linear(self, dmrs_list, i, root_index=False):
    verbose = False
    if len(dmrs_list) < i + 3:
      return -1, len(dmrs_list)
    if not (dmrs_list[i].startswith('<') and dmrs_list[i+1].endswith('>')):
      if verbose:
        print 'Skip ', dmrs_list[i]
      # Start is not well-formed.
      return self.parse_linear(dmrs_list, i+1, root_index)
    
    span_start = int(dmrs_list[i][1:])
    span_end = int(dmrs_list[i+1][:-1])

    if dmrs_list[i+2].startswith(':') and dmrs_list[i+2].endswith('('):
      # Missing concept.
      concept = 'unknown_rel'
      i = i - 1
    else:
      concept = dmrs_list[i+2]
      if not (concept <> ')' and not concept.startswith('<') 
        and not concept.endswith('>')):
        concept = 'unknown_rel'
        if verbose:
          print 'Invalid concept', dmrs_list[i+2]

    node_index = len(self.nodes)
    node = MrsNode('n' + str(node_index + 1), concept, span_start, span_end)
    self.nodes.append(node)
    i = i + 3
    
    while i < len(dmrs_list) and (dmrs_list[i] <> ')' or root_index):
      if dmrs_list[i] == ')':
        i = i + 1
        continue
      if (i + 2 < len(dmrs_list) and
          dmrs_list[i].startswith(':') and dmrs_list[i].endswith('(')
          and dmrs_list[i+1].startswith('<') and dmrs_list[i+2].endswith('>')):
        # Well-formed.
        relation = dmrs_list[i][1:-1]
      elif (i + 2 < len(dmrs_list) and 
          dmrs_list[i].startswith('<') and dmrs_list[i+1].endswith('>')
          and dmrs_list[i+2] <> ')'):
        # Missing relation.
        relation = 'ARG/NEQ'
        i = i - 1
      else:
        # Skip symbol.
        if verbose:
          print 'Skip', dmrs_list[i]
        i = i + 1
        continue

      if relation.endswith('*') and len(dmrs_list) > i+3 and (len(dmrs_list) == i + 4 or dmrs_list[i+4] == ')'):
        # Re-entrancies.
        relation = relation[:-1]
        span_start = int(dmrs_list[i+1][1:])
        span_end = int(dmrs_list[i+2][:-1])
        child_concept = dmrs_list[i+3]
        child_index = -1
        # Find re-entrancy node if possible.
        inds = filter(lambda x: self.nodes[x].concept == child_concept, range(len(self.nodes)-1, -1, -1))
        for ind in inds:
          if span_start == self.nodes[ind].alignment and span_end == self.nodes[ind].alignment_end:
            child_index = ind
        if child_index == -1 and inds:
          child_index = inds[0]        
        if child_index == -1:
          child_index = len(self.nodes)
          self.nodes.append(MrsNode('n' + str(child_index + 1), child_concept, 
              span_start, span_end))
        # Re-entrancies not inverted.
        if child_index not in self.nodes[node_index].edges:
          self.append_edge(node_index, child_index, relation)  
        i += 5
      else:
        if relation.endswith('*'):
          relation = relation[:-1]
        child_index, i = self.parse_linear(dmrs_list, i+1) # recursion

        if child_index >= 0:
          if relation.endswith('-of'):
            relation = relation[:relation.index('-of')]
            self.append_edge(child_index, node_index, relation)  
          else:
            self.append_edge(node_index, child_index, relation)  
    return node_index, i + 1  
 
  def amr_remove_duplicates_str(self, i):
    # Assumes is_amr.
    # Pre-order traversal of graph.
    concept = self.nodes[i].concept
    if self.nodes[i].constant: 
      graph_str = self.nodes[i].constant
      return graph_str

    graph_str = '( ' + concept
    child_graph_strs = set()
    deleted_children_inds = set()
    for k in xrange(len(self.nodes[i].span_edges)):
      relation = self.nodes[i].span_relations[k]
      relation = re.sub(r'\/', '|', relation)  
      child_graph_str = ' :' + relation + ' '
      child_index = self.nodes[i].span_edges[k]
      if self.nodes[i].span_reent[k]:
        child_graph_str += self.nodes[child_index].name
        graph_str += child_graph_str
      else:
        child_graph_str += self.amr_remove_duplicates_str(child_index)
        if child_graph_str in child_graph_strs:
          self.nodes[child_index].edges = []
          self.nodes[child_index].relations = []
          deleted_children_inds.add(k)
        else:
          child_graph_strs.add(child_graph_str)
          graph_str += child_graph_str

    included_indexes = filter(lambda x: x not in deleted_children_inds,
                              range(len(self.nodes[i].span_edges)))
    self.nodes[i].span_edges = [self.nodes[i].span_edges[k] 
                                for k in included_indexes]
    self.nodes[i].span_relations = [self.nodes[i].span_relations[k] 
                                for k in included_indexes]
    return graph_str + ')'


  def amr_graph_str(self, i, indent_level, is_amr=False):
    indent_size = 4
    # Pre-order traversal of graph.
    concept = self.nodes[i].concept
    # For now, always add _rel for consistent evaluation.
    if not concept.endswith('_rel') and not is_amr:
      concept = concept + '_rel'
    if is_amr and self.nodes[i].constant: 
      if indent_level == 1:
        print 'constant', self.nodes[i].constant
      graph_str = self.nodes[i].constant
      return graph_str

    graph_str = '(' + self.nodes[i].name + ' / ' + concept
    if self.nodes[i].constant:   
      graph_str += '\n' + ' '*indent_size*indent_level + ':CARG '  
      graph_str += self.nodes[i].constant         
    for k in xrange(len(self.nodes[i].span_edges)):
      graph_str += '\n' + ' '*indent_size*indent_level
      relation = self.nodes[i].span_relations[k]
      relation = re.sub(r'\/', '|', relation)  
      graph_str += ':' + relation + ' '
      child_index = self.nodes[i].span_edges[k]
      if self.nodes[i].span_reent[k]:
        graph_str += self.nodes[child_index].name
      else:
        graph_str += self.amr_graph_str(child_index, indent_level + 1, is_amr)
    return graph_str + ')'

  def linear_amr_str(self, i):
    s = self.nodes[i].concept
    if self.nodes[i].constant:
      s += ' :CARG() ' + self.nodes[i].constant    
    for k in xrange(len(self.nodes[i].span_edges)):
      if self.nodes[i].span_reent[k]:
        s += (' ' + ':' + self.nodes[i].span_relations[k] + '(*) ' 
            + self.nodes[self.nodes[i].span_edges[k]].concept)
      else:
        s += (' ' + ':' + self.nodes[i].span_relations[k] + '( ' 
            + self.linear_amr_str(self.nodes[i].span_edges[k])) + ' )'
    return s

  def amr_point_str(self, i):
    s = self.nodes[i].concept
    if self.nodes[i].constant:
      assert self.nodes[i].is_aligned >= 0
      s += ' :CARG$() ' + self.nodes[i].constant
    for k in xrange(len(self.nodes[i].span_edges)):
      if self.nodes[i].span_reent[k]:
        if self.nodes[self.nodes[i].span_edges[k]].is_aligned:
          s += (' ' + ':' + self.nodes[i].span_relations[k] + '$( ' 
              + self.nodes[self.nodes[i].span_edges[k]].concept + ' )') 
        else:
          s += (' ' + ':' + self.nodes[i].span_relations[k] + '(*) ' 
              + self.nodes[self.nodes[i].span_edges[k]].concept) # does happen
      else:
        if self.nodes[self.nodes[i].span_edges[k]].is_aligned:
          s += ' ' + ':' + self.nodes[i].span_relations[k] + '$( ' 
        else:
          s += ' ' + ':' + self.nodes[i].span_relations[k] + '( ' 
        s += self.amr_point_str(self.nodes[i].span_edges[k]) + ' )'
    return s
 
  def point_str(self, i):
    if self.nodes[i].alignment >= 0:
      s = str(self.nodes[i].alignment)
    else:
      s = '-1'
    if self.nodes[i].constant:
      assert self.nodes[i].is_aligned
      s += ' -1 ' + str(self.nodes[i].alignment)
    for k in xrange(len(self.nodes[i].span_edges)):
      if self.nodes[i].span_reent[k]:
        if self.nodes[self.nodes[i].span_edges[k]].is_aligned:
          s += ' -1 ' + str(self.nodes[self.nodes[i].span_edges[k]].alignment) + ' -1'
        else:
          s += ' -1 -1'
      else:
        if self.nodes[self.nodes[i].span_edges[k]].is_aligned:
          s += ' -1 '
        else:
          s += ' -1 '
        s += self.point_str(self.nodes[i].span_edges[k]) + ' -1'
    return s

  def edm_str(self):
    include_features = False 
    s = ''
    # Converts to EDM triples for evaluation.
    for i, node in enumerate(self.nodes):
      predicate_triple = node.alignment_str() + ' NAME ' + node.concept
      s += predicate_triple + ' ; '
    if self_root_index >= 0:
      head_triple = '-1:-1 /H ' + self.nodes[self.root_index].alignment_str()
      s += head_triple + ' ; '
    for i, node in enumerate(self.nodes):
      if node.constant:
        constant_triple = node.alignment_str() + ' CARG ' + node.constant
        s += constant_triple + ' ; '
    for i, node in enumerate(self.nodes):
      for k, child_index in enumerate(node.edges): 
        node_alignment = node.alignment_str()
        child_alignment = self.nodes[child_index].alignment_str()
        if (node.relations[k].endswith('/U') 
            and (node.alignment > self.nodes[child_index].alignment)):
          node_alignment, child_alignment = child_alignment, node_alignment
        argument_triple = (node_alignment + ' ' + node.relations[k] 
            + ' ' + child_alignment)
        argument_concept_triple = (node.concept + ' ' + node.relations[k] 
            + ' ' + self.nodes[child_index].concept)
        s += argument_triple + ' ; '
    if include_features:
      for i, node in enumerate(self.nodes):
        for feature in node.features:
          attribute, value = feature.split('=')[0], feature.split('=')[1]  
          property_triple = node.alignment_str() + ' ' + attribute + ' ' + value
          s += property_triple + ' ; '
    return s

  def edm_ch_str(self, labeled=True):
    include_features = False 
    s = ''
    # Converts to EDM triples for evaluation.
    for i, node in enumerate(self.nodes):
      concept = node.concept
      # For now, always add _rel for consistent evaluation.
      if not concept.endswith('_rel'):
        concept = concept + '_rel'
      predicate_triple = node.ind + ' NAME ' + concept
      s += predicate_triple + ' ; '
    if self.root_index >= 0:
      head_triple = '-1:-1 /H ' + self.nodes[self.root_index].ind
      s += head_triple + ' ; '
    for i, node in enumerate(self.nodes):
      if node.constant:
        constant_triple = node.ind + ' CARG ' + node.constant
        s += constant_triple + ' ; '
    for i, node in enumerate(self.nodes):
      for k, child_index in enumerate(node.edges): 
        node_ind = node.ind
        child_ind = self.nodes[child_index].ind
        if (node.relations[k].endswith('/U') 
            and (int(node_ind.split(':')[0]) > int(child_ind.split(':')[1]))):
          node_ind, child_ind = child_ind, node_ind
        if labeled:
          argument_triple = (node_ind + ' ' + node.relations[k] + ' ' + child_ind)
        else:
          argument_triple = (node_ind + ' REL '  + child_ind) 
        argument_concept_triple = (node.concept + ' ' + node.relations[k] 
            + ' ' + self.nodes[child_index].concept)
        s += argument_triple + ' ; '
    if include_features:
      for i, node in enumerate(self.nodes):
        for feature in node.features:
          attribute, value = feature.split('=')[0], feature.split('=')[1]  
          property_triple = node.ind + ' ' + attribute + ' ' + value
          s += property_triple + ' ; '
    return s


  def verify_undirected_edges(self, graph_id):
    und_edges = {}
    und_heads = [-1 for _ in self.nodes] # don't need to use this
    for i, node in enumerate(self.nodes):
      for k, child_index in enumerate(node.edges): 
        label = node.relations[k]
        if label == "/EQ/U": # verified as always together
          # reverse if child has more (directed) heads than parent
          # nodes should have at most one undirected head
          if len(node.heads) < (len(self.nodes[child_index].heads) - 1):
            if und_heads[i] != -1: # asserted
              print("multiple undirected heads")
            print("reversed")
            und_heads[i] = child_index # head
            if und_edges.has_key(child_index):
              und_edges[child_index].append(i) 
            else:
              und_edges[child_index] = [i] 
          else:
            if und_heads[child_index] != -1:
              print("multiple undirected heads")
            und_heads[child_index] = i # head
            if und_edges.has_key(i):
              und_edges[i].append(child_index) 
            else:
              und_edges[i] = [child_index]
    if len(und_edges) > 0:
      print und_edges


  def json_parse_str(self, graph_id, include_features=True):
    s = {"id": graph_id+1}
    node_list = []
    for i, node in enumerate(self.nodes):
      #if not self.spanned[i]: # unconnected nodes included for now
        #continue
      concept = node.concept
      if concept.endswith('_rel'):
        concept = concept[:-4]
      node_s = {"id": i+1}
      node_s["start"] = node.alignment + 1 # start indexing at 1
      node_s["end"] = node.alignment_end + 1
      if i == self.root_index:
        node_s["top"] = True
      prop_s = {}
      if concept[0] == '_':
        # Properties for surface predicates.
        if '/' in concept[1:] and 'u_unknown' in concept[1:]:
          # u and unknown verified to be paired
          split_ind = concept.index('/', 1)
          cross_ind = concept.index('_', 1)
          #prop_s["type"] = "surface"
          prop_s["predicate"] =  "u_unknown"
        elif '_' in concept[1:]:  
          split_ind = concept.index('_', 1)
          sense = concept[split_ind+1:]
          #prop_s["type"] = "surface"
          prop_s["predicate"] = sense
        else:
          #prop_s["type"] = "surface" # but no lemma
          prop_s["predicate"] = concept[1:]
          print("no lemma: %s" % concept)
      else:
        prop_s["predicate"] = concept
        #if node.constant: # Deterministic in post-processing
        #  prop_s["type"] = "constant"
        #prop_s["type"] = "abstract"
        prop_s["abstract"] = True

      if include_features:
        prop_s["type"] = node.pred_type
        feature_str = ''
        for feature in node.features:
          #attribute, value = feature.split('=')[0], feature.split('=')[1]
          #prop_s[attribute] = value
          feature_str += feature + '|'
        if feature_str != '':
          prop_s["features"] = feature_str[:-1]

      node_s["properties"] = prop_s

      edge_list = []
      for k, child_index in enumerate(node.edges): 
        label = node.relations[k]
        if label == "/EQ/U":
          label = "/EQ"
        edge_s = {"label": label}
        edge_s["target"] = child_index + 1
        edge_list.append(edge_s)
      if edge_list:
        node_s["edges"] = edge_list

      node_list.append(node_s)
    
    s["nodes"] = node_list
    return json.dumps(s, sort_keys=True)


  def json_orig_parse_str(self, graph_id, include_features=True):
    s = {"id": graph_id+1}
    node_list = []
    for i, node in enumerate(self.nodes):
      concept = node.concept
      if concept.endswith('_rel'):
        concept = concept[:-4]
      node_s = {"id": i+1}
      start_ind = int(node.ind.split(':')[0])
      end_ind = int(node.ind.split(':')[1])

      node_s["start"] = start_ind
      node_s["end"] = end_ind
      if i == self.root_index:
        node_s["top"] = True
      prop_s = {}
      prop_s["predicate"] = concept

      if node.constant:  
        if node.constant[0] == '"' and node.constant[-1] == '"':
          const = node.constant[1:-1]
        else:
          const = node.constant
        prop_s["constant"] = const

      if include_features:
        prop_s["type"] = node.pred_type
        feature_str = ''
        for feature in node.features:
          #attribute, value = feature.split('=')[0], feature.split('=')[1]
          #prop_s[attribute] = value
          feature_str += feature + '|'
        if feature_str != '':
          prop_s["features"] = feature_str[:-1]

      node_s["properties"] = prop_s

      edge_list = []
      for k, child_index in enumerate(node.edges): 
        label = node.relations[k]
        if label == "/EQ/U":
          label = "/EQ"
        edge_s = {"label": label}
        edge_s["target"] = child_index + 1
        edge_list.append(edge_s)
      if edge_list:
        node_s["edges"] = edge_list

      node_list.append(node_s)
    
    s["nodes"] = node_list
    return json.dumps(s, sort_keys=True)


  def json_str(self, graph_id, surface, offset=0):
    include_features = False
    s = {"id": graph_id+1}
    node_list = []
    for i, node in enumerate(self.nodes):
      concept = node.concept
      if concept.endswith('_rel'):
        concept = concept[:-4]
      node_s = {"id": i+1}
      start_ind = int(node.ind.split(':')[0])
      end_ind = int(node.ind.split(':')[1])

      node_s["form"] = surface[start_ind:end_ind]
      node_s["start"] = start_ind + offset
      node_s["end"] = end_ind + offset
      if i == self.root_index:
        node_s["top"] = True
      prop_s = {}
      if concept[0] == '_':
        # Properties for surface predicates.
        if '/' in concept[1:] and 'unknown' in concept[1:]:
          split_ind = concept.index('/', 1)
          cross_ind = concept.index('_', 1)
          word = concept[1:split_ind]
          pos = concept[split_ind+1:cross_ind]
          if '_unknown' in concept:
            sense = concept[cross_ind+1:concept.index('_unknown')]
          else:
            sense = concept[cross_ind+1]
          prop_s["type"] = "unknown"
          prop_s["predicate"] = sense
          prop_s["lemma"] = word
        elif '_' in concept[1:]:  
          split_ind = concept.index('_', 1)
          lemma = concept[1:split_ind]
          sense = concept[split_ind+1:]
          prop_s["type"] = "surface"
          prop_s["predicate"] = sense
          prop_s["lemma"] = lemma
        else:
          prop_s["type"] = "surface" # but no lemma
          prop_s["predicate"] = concept[1:]
        if node.pos is not None:
          pos = node.pos[1:] if node.pos[0] == '_' else node.pos
          prop_s["pos"] = pos
        if node.ne is not None:
          ne = node.ne[:node.ne.index('_C')] if '_C' in node.ne else node.ne
          if ne <> 'O':
            prop_s["ner"] = ne
      else:
        prop_s["predicate"] = concept
        if node.constant:  
          prop_s["type"] = "constant"
          if node.constant[0] == '"' and node.constant[-1] == '"':
            const = node.constant
          else:  
            const = '"' + node.constant + '"'
          prop_s["lemma"] = const
          if node.ne is not None:
            ne = node.ne[:node.ne.index('_C')] if '_C' in node.ne else node.ne
            prop_s["ner"] = ne
        else:
          prop_s["type"] = "abstract"

      if include_features:
        for feature in node.features:
          attribute, value = feature.split('=')[0], feature.split('=')[1]
          prop_s[attribute] = value

      node_s["properties"] = prop_s

      edge_list = []
      for k, child_index in enumerate(node.edges): 
        edge_s = {"label": node.relations[k]}
        edge_s["target"] = child_index + 1
        edge_list.append(edge_s)
      if edge_list:
        node_s["edges"] = edge_list

      node_list.append(node_s)
    
    s["nodes"] = node_list
    return json.dumps(s, sort_keys=True)


  def epe_str(self, graph_id, surface, offset=0):
    include_features = False
    s = '{"id": ' + str(graph_id+1) + ', '
    node_strs = []
    for i, node in enumerate(self.nodes):
      concept = node.concept
      if concept.endswith('_rel'):
        concept = concept[:-4]
      node_s = '{"id": ' + str(i+1) + ', '
      start_ind = int(node.ind.split(':')[0])
      end_ind = int(node.ind.split(':')[1])

      node_s += '"form": "' + surface[start_ind:end_ind] + '", '
      node_s += '"start": ' + str(start_ind + offset) + ', '
      node_s += '"end": ' + str(end_ind + offset) + ', '
      if i == self.root_index:
        node_s += '"top": true, '
      prop_strs = []
      if concept[0] == '_':
        # Properties for surface predicates.
        if '/' in concept[1:] and 'unknown' in concept[1:]:
          split_ind = concept.index('/', 1)
          cross_ind = concept.index('_', 1)
          word = concept[1:split_ind]
          pos = concept[split_ind+1:cross_ind]
          if '_unknown' in concept:
            sense = concept[cross_ind+1:concept.index('_unknown')]
          else:
            sense = concept[cross_ind+1]
          prop_strs.append('"type": "unknown"')
          prop_strs.append('"predicate": "' + sense + '"')
          prop_strs.append('"lemma": "' + word + '"')
        elif '_' in concept[1:]:  
          split_ind = concept.index('_', 1)
          lemma = concept[1:split_ind]
          sense = concept[split_ind+1:]
          prop_strs.append('"type": "surface"')
          prop_strs.append('"predicate": "' + sense + '"')
          prop_strs.append('"lemma": "' + lemma + '"')
        else:
          prop_strs.append('"type": "surface"') # but no lemma
          prop_strs.append('"predicate": "' + concept[1:] + '"')
        pos = node.pos[1:] if node.pos[0] == '_' else node.pos
        prop_strs.append('"pos": "' + pos + '"')
        ne = node.ne[:node.ne.index('_C')] if '_C' in node.ne else node.ne
        if ne <> 'O':
          prop_strs.append('"ner": "' + ne + '"')
      else:
        prop_strs.append('"predicate": "' + concept + '"')
        if node.constant:  
          prop_strs.append('"type": "constant"')
          if node.constant[0] == '"' and node.constant[-1] == '"':
            const = node.constant
          else:  
            const = '"' + node.constant + '"'
          prop_strs.append('"lemma": ' + const)
          ne = node.ne[:node.ne.index('_C')] if '_C' in node.ne else node.ne
          prop_strs.append('"ner": "' + ne + '"')
        else:
          prop_strs.append('"type": "abstract"')

      if include_features:
        for feature in node.features:
          attribute, value = feature.split('=')[0], feature.split('=')[1]
          prop_strs.append('"' + attribute + '": "' + value + '"')

      node_s += '"properties": {' + ', '.join(prop_strs) + '}'  

      edge_strs = []
      for k, child_index in enumerate(node.edges): 
        edge_s = '{"label": "' + node.relations[k] + '", '
        edge_s += '"target": ' + str(child_index+1) + '}'
        edge_strs.append(edge_s)
      if edge_strs:
        node_s += ', "edges": [' + ', '.join(edge_strs) + ']'

      node_s += '}'
      node_strs.append(node_s)
    
    s += '"nodes": [' + ', '.join(node_strs) + ']'
    s += '}'
    return s


def read_preds_only_graphs(dmrs_file_name):
  dmrs_file = open(dmrs_file_name, 'r')
  graphs = []

  for line in dmrs_file:
    dmrs = MrsGraph([])
    dmrs_list = line.strip().split()
    if dmrs_list and dmrs_list[-1] == '_EOS':
      dmrs_list = dmrs_list[:-1]
      
    _, ind = dmrs.parse_preds(dmrs_list, 0)
    dmrs.root_index = -1
    graphs.append(dmrs)
  return graphs
    

def read_linear_dmrs_graphs(dmrs_file_name, is_exact=True, is_inorder=False, 
                            is_arceager=False, is_arceager_buffer_shift=False, 
                            is_no_span=False, tokens=None):
  dmrs_file = open(dmrs_file_name)
  graphs = []

  for line in dmrs_file:
    dmrs = MrsGraph([])
    dmrs_list = line.strip().split()
    if dmrs_list and dmrs_list[-1] == '_EOS':
      dmrs_list = dmrs_list[:-1]
    if not dmrs_list:
      dmrs_list = [':/H', '<0', '>0', '_UNK']
    start_ind = 1 if (dmrs_list[0].startswith(':/H') or
                      dmrs_list[0].startswith(':focus')) else 0

    if is_arceager_buffer_shift:
      dmrs.parse_arceager_buffer_shift(dmrs_list)
    elif is_arceager and is_no_span:
      dmrs.parse_arceager_nospan(dmrs_list)
    elif is_arceager:
      dmrs.parse_arceager(dmrs_list)
    else:
      dmrs.root_index = 0
      if is_inorder:
        _, ind = dmrs.parse_linear_inorder(dmrs_list, start_ind, True)
      elif is_no_span:
        _, ind = dmrs.parse_linear_nospan(dmrs_list, start_ind, True)
      else:
        _, ind = dmrs.parse_linear_new(dmrs_list, start_ind, True)
    if dmrs.root_index == -1:
      dmrs.root_index = 0

    graphs.append(dmrs)
  return graphs


def parse_eds(eds_str, token_inds=None, token_start=None, 
    token_end=None, sentence_str=None):
  root_node_name = ''
  root_index = -1
  nodes = []
  node_names = {}

  eds_list = eds_str.split(' ; ')
  root_node_name = eds_list[0].strip()
  eds_list = eds_list[1:]

  for line in eds_list:
    if line[0] == '|':
      line = line[1:]
    if line[0] <> '|': # unconnected nodes
      nodename = line[:line.index(':')]
      node = MrsNode(nodename)
      node_names[nodename] = len(nodes)
      node.concept = line[line.index(':')+1:line.index('<')]
      if '/' in node.concept: # lowercase POS
        s_ind = node.concept.index('/')
        node.concept = node.concept[:s_ind] + node.concept[s_ind:].lower()

      if '(' in line and ')' in line:
        node.constant = line[line.index('(')+1:line.index(')')]
      ind = line[line.index('<')+1:line.index('>')]

      # Records all alignments.
      if ind == '':
        ind_start, ind_end = 0, 0
        node.ind = '0:0'
      else:
        ind_start, ind_end = int(ind.split(':')[0]), int(ind.split(':')[1])
        ind_start = max(0, ind_start)
        ind_end = max(ind_start, ind_end)
        node.ind = ind
     
      if token_start is not None:
        if (sentence_str[ind_end-1] in string.punctuation 
            and sentence_str[ind_end-1] <> '-'
            and (ind_end == len(sentence_str) 
                 or sentence_str[ind_end].isspace())
            and ind_end-1 in token_end):
          ind_end = ind_end - 1
        if len(sentence_str) > ind_end and sentence_str[ind_end] == '-':
          ind_end += 1

        if ind_start not in token_start:
          if ind_start == 0 or ind_start-1 not in token_start:
            print ('Tokenization conversion warning on token: ' 
                + sentence_str[ind_start:ind_end] + ': Start index not found.' )
          while ind_start > 0 and ind_start not in token_start:
            ind_start -= 1
        start_ind = token_start.index(ind_start)
        node.alignment = start_ind
       
        if ind_end in token_end:
          end_ind = token_end.index(ind_end)
        else:
          if ind_end in token_start[start_ind+1:]:
            end_ind = token_start.index(ind_end, start_ind + 1) - 1
          else:
            if ind_end == len(sentence_str) or ind_end+1 not in token_end:
              print ('Tokenization conversion warning on token: ' 
                + sentence_str[ind_start:ind_end] + ': End index not found.')
            while ind_end < len(sentence_str) and ind_end not in token_end:
              ind_end += 1
            end_ind = token_end.index(ind_end)
        node.alignment_end = end_ind

      if '{' in line and '}' in line:
        node.pred_type = line[line.index('{')+1]
        features = line[line.index('{')+3:line.index('}')].split(', ')
        node.features = [feat.replace(' ', ':') for feat in features]

      arg_items = line[line.index('[')+1:line.index(']')].split(', ')
      for item in arg_items:
        if item and item.split(' ')[1][0] <> '_':
          node.relations.append(item.split(' ')[0])
          node.edge_names.append(item.split(' ')[1])
      nodes.append(node)

  graph = MrsGraph(nodes)

  # Node names to indexes.
  for i in xrange(len(graph.nodes)):
    for name in graph.nodes[i].edge_names:
      assert node_names.has_key(name), "Node name not found: %s: %s" % (filename, name)
      graph.nodes[i].edges.append(node_names[name])
  
  assert node_names.has_key(root_node_name)
  graph.root_index = node_names[root_node_name]

  # Finds heads.
  for i in xrange(len(graph.nodes)):
    for j in graph.nodes[i].edges:
      graph.nodes[j].heads.append(i)      

  return graph

  
def parse_dmrs(simple_dmrs_str, token_inds=None, token_start=None, 
    token_end=None, sentence_str=None):
  if token_inds is not None:
    assert (token_start is not None and token_end is not None 
        and sentence_str is not None)
  root_index = -1
  nodes = []
  node_names = {}

  # Tokenize.
  split_chars = '[];}' 
  dmrs_list = []
  item = ''
  for char in simple_dmrs_str:
    if not char or char.isspace():
      if item:
        dmrs_list.append(item)
        item = ''
    elif char in split_chars:
      if item:
        dmrs_list.append(item)
      dmrs_list.append(str(char))
      item = ''
    else:
      item += char    

  line = []
  # Order by line and type.
  for item in dmrs_list[2:]:
    if item == ';':
      if '->' in line or '--' in line:
        if len(line) != 3:
          print line
        assert len(line) == 3
        head = line[0]
        if head == '0:/H':
          head_nodename = line[2]
          root_index = node_names[head_nodename]
        else:
          head_nodename = head[:head.index(':')]
          head_node = node_names[head_nodename]
          relation = head[head.index(':')+1:]      
          if line[1] == '--':
            relation += '/U'
          child_nodename = line[2]
          child_node = node_names[child_nodename]
          if child_node in nodes[head_node].edges:
            k = nodes[head_node].edges.index(child_node)
            nodes[head_node].relations[k] += '\\' + relation
          else:
            nodes[head_node].edges.append(child_node)
            nodes[head_node].relations.append(relation)
            nodes[child_node].heads.append(head_node)
      else:
        assert len(line) >= 3, line
        nodename = line[0]
        node = MrsNode(nodename)   
        node_names[nodename] = len(nodes)

        pred = line[2]
        ind = pred[pred.rfind('<')+1:pred.rfind('>')]

        # Records all alignments.
        if (ind == '' or ':' not in ind or not ind.split(':')[0].isdigit() 
            or not ind.split(':')[1].isdigit()):
          ind_start, ind_end = 0, 0
          node.ind = '0:0'
        else:
          ind_start, ind_end = int(ind.split(':')[0]), int(ind.split(':')[1])
          ind_start = max(0, ind_start)
          ind_end = max(ind_start, ind_end)
          node.ind = ind
       
        if token_start is not None:
          if (sentence_str[ind_end-1] in string.punctuation 
              and sentence_str[ind_end-1] <> '-'
              and (ind_end == len(sentence_str) 
                   or sentence_str[ind_end].isspace())
              and ind_end-1 in token_end):
            ind_end = ind_end - 1
          if len(sentence_str) > ind_end and sentence_str[ind_end] == '-':
            ind_end += 1

          if ind_start not in token_start:
            if ind_start == 0 or ind_start-1 not in token_start:
              print ('Tokenization conversion warning on token: ' 
                  + sentence_str[ind_start:ind_end] + ': Start index not found.' )
            while ind_start > 0 and ind_start not in token_start:
              ind_start -= 1
          start_ind = token_start.index(ind_start)
          node.alignment = start_ind
         
          if ind_end in token_end:
            end_ind = token_end.index(ind_end)
          else:
            if ind_end in token_start[start_ind+1:]:
              end_ind = token_start.index(ind_end, start_ind + 1) - 1
            else:
              if ind_end == len(sentence_str) or ind_end+1 not in token_end:
                print ('Tokenization conversion warning on token: ' 
                  + sentence_str[ind_start:ind_end] + ': End index not found.')
              while ind_end < len(sentence_str) and ind_end not in token_end:
                ind_end += 1
              if ind_end in token_end:
                end_ind = token_end.index(ind_end)
              else:
                end_ind = len(token_end) - 1
          node.alignment_end = end_ind
         
        if '<' in pred:
          node.concept = pred[:pred.index('<')] # assumes occuring exactly once
        else:
          node.concept = pred
        if '/' in node.concept: # lowercase POS
          s_ind = node.concept.index('/')
          node.concept = node.concept[:s_ind] + node.concept[s_ind:].lower()
        if len(line) >= 4:
          node.pred_type = line[3]
        else:
          node.pred_type = 'x' # default
        if len(line) > 5:
          node.features = [x for x in line[4:-1]]
          node.morph_tag = '_'.join([feat for feat in node.features])
           
        nodes.append(node)
      line = []
    else:
      line.append(item)
 
  graph = MrsGraph(nodes)
  graph.root_index = 0 if root_index == -1 else root_index
  return graph

