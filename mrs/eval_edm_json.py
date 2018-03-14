"""Computes EDM F1 scores."""

import argparse
import json

def is_const_predicate(concept):
  const_set = set(["named", "card", "named_n", "mofy", "yofc", "ord", 
      "dofw", "dofm", "fraction", "season", "excl", "year_range", 
      "numbered_hour", "holiday", "timezone_p", "polite"])
  return concept in const_set

def list_predicate_spans(triples):
  spans = []
  for triple in triples:
    if len(triple.split(' ')) > 1 and triple.split(' ')[1] == "NAME":
      spans.append(triple.split(' ')[0])  
  return spans

def list_spans(triples):
  spans = set()
  for triple in triples:
    if (len(triple.split(' ')) > 1 and (triple.split(' ')[1] == "NAME" 
        or triple.split(' ')[1] == "CARG")):
      spans.add(triple.split(' ')[0])
    elif len(triple.split(' ')) > 2:
      spans.add(triple.split(' ')[0])
      spans.add(triple.split(' ')[2])
  return list(spans)

def inc_end_spans(spans):
  new_spans = [span.split(':')[0] + ":" + str(int(span.split(':')[1])+1)
               for span in spans]
  return new_spans

def dec_end_spans(spans):
  new_spans = [span.split(':')[0] + ":" + str(int(span.split(':')[1])-1)
               for span in spans]
  return new_spans


class Token():
  def __init__(self, lemma, word, pos, is_ne, ne_tag, 
               char_start=-1, char_end=-1):
    self.lemma = lemma
    self.word = word
    self.pos = pos
    self.is_ne = is_ne
    self.ne_tag = ne_tag
    self.char_start = char_start
    self.char_end = char_end
    self.const_lexeme = word 
    self.is_const = False
    self.is_pred = False
 

class Sentence():
  def __init__(self, sentence, sent_id=0):
    self.sentence = sentence # tokens
    self.sent_id = sent_id

  def word_at(self, i):
    return self.sentence[i].word

  @classmethod
  def parse_json_line(cls, json_line):
    toks = json.loads(json_line)
    tokens = []
    tokens_index = {}
    sent_id = toks["id"]
    
    # Construct tokens.
    for tok in toks["tokens"]:
      assert (tok["id"] - 1) == len(tokens)
      props = tok["properties"]
      
      is_ne = "NE" in props 
      ne_tag = props["NE"] if is_ne else ''
      token = Token(props["lemma"], props["word"], props["POS"],   
                    is_ne, ne_tag, 
                    char_start=tok["start"], char_end=tok["end"])
      if "constant" in props:
        token.is_const = True
        token.const_lexeme = props["constant"]
      elif props["word"].endswith("."):
        token.const_lexeme = props["word"][:-1]
      if "erg_predicate" in props and props["erg_predicate"]:
        token.is_pred = True 

      tokens.append(token)
    return cls(tokens, sent_id)


class MrsNode():
  def __init__(self, name, concept, start=-1, end=-1):
    self.name = name
    self.concept = concept
    self.start = start # character start
    self.end = end
    self.alignment = -1 # token start
    self.alignment_end = -1
    self.constant = ""
    self.top = False
    self.edges = []
    self.relations = []

  def span_str(self, include_end=True):
    if include_end:
      return str(self.start) + ":" + str(self.end)
    else:
      return str(self.start) + ":" + str(self.start)

  def append_edge(self, child_index, relation):
    self.edges.append(child_index)
    self.relations.append(relation)


class MrsGraph():
  def __init__(self, nodes, parse_id=-1):
    self.nodes = nodes
    self.parse_id = parse_id

  @classmethod
  def parse_orig_json_line(cls, json_line):
    mrs = json.loads(json_line)
    parse_id = mrs["id"] 
    nodes = []
    nodes_index = {}

    # First parse nodes.
    for node in mrs["nodes"]:
      node_id = node["id"] - 1
      nodes_index[node_id] = len(nodes)
      props = node["properties"] 
      concept = props["predicate"]
      start = node["start"]
      end = node["end"]
      graph_node = MrsNode(str(node_id), concept, start, end)

      if "constant" in props:
        const = props["constant"]
        if const[0] != '"':
          const = '"' + const
        if const[-1] != '"':
          const = const + '"' 
        graph_node.constant = const
      if "top" in node and node["top"]:
        graph_node.top = True

      # ignore features for now
      nodes.append(graph_node)

    # Then add edges.
    for node in mrs["nodes"]:
      parent_ind = nodes_index[node["id"] - 1]
      if "edges" in node:
        for edge in node["edges"]:
          child_ind = nodes_index[edge["target"] - 1]
          label = edge["label"]
          nodes[parent_ind].append_edge(child_ind, label)

    return cls(nodes, parse_id)

  @classmethod
  def parse_json_line(cls, json_line):
    mrs = json.loads(json_line)
    parse_id = mrs["id"] 
    nodes = []
    nodes_index = {}

    # First parse nodes
    for node in mrs["nodes"]:
      node_id = node["id"] - 1
      props = node["properties"] 
      if "abstract" in props and props["abstract"]:
        concept = props["predicate"]
      else:
        concept = '_' + props["predicate"]

      graph_node = MrsNode(str(node_id), concept)
      graph_node.alignment = node["start"] - 1
      graph_node.alignment_end = node["end"] - 1
      if "top" in node and node["top"]:
        graph_node.top = True
      nodes_index[node_id] = len(nodes)
      nodes.append(graph_node)

    # Then add edges 
    for node in mrs["nodes"]:
      parent_ind = nodes_index[node["id"] - 1]
      if "edges" in node:
        for edge in node["edges"]:
          child_ind = nodes_index[edge["target"] - 1]
          label = edge["label"]
          nodes[parent_ind].append_edge(child_ind, label)

    return cls(nodes, parse_id)

  def predicate_bag(self, include_constants=True):
    bag = []
    for i, node in enumerate(self.nodes):
      bag.append(node.concept)
      if include_constants and node.constant:
        bag.append(node.constant)
    return bag

  def predicate_triples(self, include_constants=True, include_span_ends=True):
    triples = []
    for i, node in enumerate(self.nodes):
      triples.append(node.span_str(include_span_ends) + " NAME " + node.concept)
      if include_constants and node.constant:
        triples.append(node.span_str(include_span_ends) + " CARG " + node.constant)

    return triples

  def relation_triples(self, include_span_ends=True, labeled=True):
    triples = []

    for i, node in enumerate(self.nodes):
      if node.top:
        node_ind = node.span_str(include_span_ends)
        triples.append("-1:-1 /H " + node_ind)

    for i, node in enumerate(self.nodes):
      node_ind = node.span_str(include_span_ends)
      for k, child_index in enumerate(node.edges): 
        child_node = self.nodes[child_index]
        child_ind = child_node.span_str(include_span_ends)
        if (node.relations[k] == "/EQ" and (node.start > child_node.start
            or (node.start == child_node.start and node.end > child_node.end))):
          node_ind, child_ind = child_ind, node_ind
        rel = node.relations[k] if labeled else "REL"
        rel = "/EQ/U" if rel == "/EQ" else rel
        triples.append(node_ind + ' ' + rel + ' ' + child_ind)

    return triples

  def eds_triples(self, score_predicates, score_relations,
    include_span_ends=True, include_constants=True, labeled=True,
    include_predicate_spans=True):
    triples = []

    if score_predicates:
      if include_predicate_spans:
        triples.extend(self.predicate_triples(include_constants,
          include_span_ends))
      else:
        triples.extend(self.predicate_bag(include_constants))
    if score_relations:
      triples.extend(self.relation_triples(include_span_ends, labeled))

    return triples


def compute_f1(gold_graphs, predicted_graphs, score_predicates, 
    score_relations, include_constants=True, labeled=True, 
    include_span_ends=True, include_predicate_spans=True, 
    score_nones=True, verbose=False):
  total_gold = 0
  total_predicted = 0
  total_correct = 0
  none_count = 0 
  
  for parse_id, gold_g in gold_graphs.items(): 
    gold_triples = gold_g.eds_triples(score_predicates, score_relations, 
        include_span_ends, include_constants, labeled, include_predicate_spans)
    if parse_id in predicted_graphs:
      predicted_triples = predicted_graphs[parse_id].eds_triples(
        score_predicates, score_relations, include_span_ends, include_constants,
        labeled, include_predicate_spans)
    else:
      predicted_triples = []
      none_count += 1
      if not score_nones:
        gold_triples = [] 

    # Magic to replace end spans off by 1.
    gold_spans = set(list_spans(gold_triples))
    predicted_spans = list_spans(predicted_triples)

    def replace_new_spans(new_spans):
      for i, new_span in enumerate(new_spans):
        old_span = predicted_spans[i]
        if old_span not in gold_spans and new_span in gold_spans:
          for j, triple in enumerate(predicted_triples):
            # string replacement
            predicted_triples[j] = triple.replace(old_span, new_span) 
                  
    replace_new_spans(inc_end_spans(predicted_spans))
    replace_new_spans(dec_end_spans(predicted_spans))

    gold_triples = set(gold_triples)
    predicted_triples = set(predicted_triples)

    correct_triples = gold_triples.intersection(predicted_triples)
    incorrect_predicted = predicted_triples - correct_triples
    missed_predicted = gold_triples - correct_triples

    if verbose and not include_predicate_spans:
      if incorrect_predicted:
        print("Incorrect: %s" % incorrect_predicted)
      if missed_predicted:
        print("Missed: %s" % missed_predicted)

    total_gold += len(gold_triples)
    total_predicted += len(predicted_triples)
    total_correct += len(correct_triples)

  assert total_predicted > 0 and total_gold > 0, "No correct predictions"

  precision = total_correct/total_predicted
  recall = total_correct/total_gold
  f1 = 2*precision*recall/(precision+recall)

  if verbose:
    print("Precision: {:.2%}".format(precision))
    print("Recall: {:.2%}".format(recall))
  print("F1-score: {:.2%}".format(f1))
 

if __name__=='__main__': 
  parser = argparse.ArgumentParser()
  parser.add_argument("-g", "--gold", help="gold dmrs.orig.json file",
          required=True)
  parser.add_argument("-s", "--system", help="system dmrs json file",
          required=True)
  parser.add_argument("-t", "--toks", help="tokenization and preprocessing .tok.json file")
  parser.add_argument("-x", "--text", help="tokenization and preprocessing .tok.json file")
  parser.add_argument("--orig", help="score orig system file",
          action="store_true")

  parser.add_argument("--exclude_constants", help="do not score constant arguments",
          action="store_true")
  parser.add_argument("--unlabeled", help="do not score constant arguments",
          action="store_true")
  parser.add_argument("--exclude_missing_graphs", help="do not score constant arguments",
          action="store_true")

  parser.add_argument("--detailed", help="Detailed F1 scores",
          action="store_true")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  if not args.orig:
    assert args.toks and args.text

  include_constants = not args.exclude_constants
  labeled = not args.unlabeled
  score_nones = not args.exclude_missing_graphs

  with open(args.gold, 'r') as fg:
    gold_graphs = {}
    for line in fg:
      graph = MrsGraph.parse_orig_json_line(line.strip()) 
      gold_graphs[graph.parse_id] = graph

  if args.orig:
    with open(args.system, 'r') as fs:
      predicted_graphs = {}
      for line in fs:
        graph = MrsGraph.parse_orig_json_line(line.strip()) 
        predicted_graphs[graph.parse_id] = graph
  else:
    with open(args.text, 'r') as fx:
      sentences_raw = [line.strip() for line in fx]

    sentences = {}
    with open(args.toks, 'r') as ft:
      for line in ft:
        sent = Sentence.parse_json_line(line.strip())
        sentences[sent.sent_id] = sent

    with open(args.system, 'r') as fs:
      predicted_graphs = {}
      for line in fs:
        graph = MrsGraph.parse_json_line(line.strip()) 
        sent = sentences[graph.parse_id]
        raw_str = sentences_raw[sent.sent_id-1]
        for node in graph.nodes:
          token = sent.sentence[node.alignment]
          # Convert to char-based spans.
          node.start = token.char_start
          node.end = sent.sentence[node.alignment_end].char_end
          # Process constants.
          if is_const_predicate(node.concept):           
            if token.is_const:
              constant = token.const_lexeme
            else:  
              constant = raw_str[node.start:node.end]
              if ' ' in constant:
                constant = constant[:constant.index(' ')]
            if constant[0] != '"':
              constant = '"' + constant
            if constant[-1] == '"':
              constant = constant[:-1]
            if len(constant) > 1 and constant[-1] in '.,':
              constant = constant[:-1]
            node.constant = constant + '"'
          elif node.concept.startswith("_"): 
            if token.is_pred:
              node.concept = '_' + token.lemma + node.concept
            else:  
              node.concept = ('_' + token.word.lower() + '/' 
                              + token.pos.lower() + '_u_unknown')
        predicted_graphs[graph.parse_id] = graph

  print("Full EDM")
  compute_f1(gold_graphs, predicted_graphs, True, True,
        include_constants, labeled,
        score_nones=score_nones, verbose=args.verbose)

  print("Predicate EDM")
  compute_f1(gold_graphs, predicted_graphs, True, False,
        include_constants, 
        score_nones=score_nones, verbose=args.verbose)

  print("Relation EDM")
  compute_f1(gold_graphs, predicted_graphs, False, True,
          include_constants, labeled, 
          score_nones=score_nones, verbose=args.verbose)

  if args.detailed:
    print("Full EDM, start spans only")
    compute_f1(gold_graphs, predicted_graphs, True, True,
            include_constants, labeled, include_span_ends=False,
            score_nones=score_nones, verbose=args.verbose)

    print("Predicate EDM, start spans only")
    compute_f1(gold_graphs, predicted_graphs, True, False,
            include_constants, include_span_ends=False,
            score_nones=score_nones, verbose=args.verbose)

    print("Predicate EDM, ignoring spans")
    compute_f1(gold_graphs, predicted_graphs, True, False,
            include_constants, include_predicate_spans=False,
            score_nones=score_nones, verbose=args.verbose)

    print("Relation EDM, start spans only")
    compute_f1(gold_graphs, predicted_graphs, False, True,
            include_constants, labeled, include_span_ends=False,
            score_nones=score_nones, verbose=args.verbose)

