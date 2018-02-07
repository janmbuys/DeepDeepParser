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
import util
import json
#from nltk.corpus import wordnet as wn #TODO temp disable

import util as mrs_util

class Token():
  def __init__(self, word, original_word, pos, constant_label, is_ne, is_timex,
      ne_tag, normalized_ne_tag, edge = -1, relation = '', timex_attr = dict(),
      char_start=-1, char_end=-1):
    self.word = word
    self.lemma = word
    self.original_word = original_word
    if ne_tag == '':
      self.pred_lexeme = original_word.lower() + u'/' + pos.lower()
    else:
      self.pred_lexeme = original_word + u'/' + pos.lower()
    self.const_lexeme = original_word 
    self.wiki_lexeme = original_word 
    self.nom_lexeme = original_word 
    self.verb_lexeme = '' 
    self.noun_lexeme = '' 
    self.pos = pos
    self.is_const = False
    self.is_pred = False
    self.is_nom = False
    self.is_wiki = False
    self.is_ne = is_ne
    self.is_timex = is_timex
    self.ne_tag = ne_tag
    self.normalized_ne_tag = normalized_ne_tag
    self.constant_label = constant_label
    self.timex_attr = timex_attr
    self.edge = edge
    self.relation = relation
    self.char_start = char_start
    self.char_end = char_end
    self.pred_char_start = char_start
    self.pred_char_end = char_end
    self.const_char_start = char_start
    self.const_char_end = char_end

    self.children = []

  '''Format:Text, CharacterOffsetBegin, CharacterOffsetEnd, PartOfSpeech, Lemma, NamedEntityTag, '''
  @classmethod
  def parse_stanford_line(cls, line, name_normalize_dict):
    items = []
    timex_attr = dict()
    is_timex = False
    attr_re = re.compile(r'(\w\w*)=\"(\w\w*)\"')
    for item in line.split(' '):
      if is_timex:
        m = attr_re.match(item)
        if m:
          timex_attr[m.group(1)] = m.group(2)
        else:
          break
      elif len(item) >= 3 and '=' in item[1:-1]:
        if 'Timex=' in item:
          is_timex = True
        else:
          items.append(item[item.index('=') + 1:])
      else:
        items[-1] += '_' + item # no spaces in item
    word = items[0]
    char_start = int(items[1])
    char_end = int(items[2])
    pos = items[3]
    lemma = items[4]
    if name_normalize_dict.has_key(lemma):
      lemma = name_normalize_dict[lemma]

    is_ne = False
    ne_tag = ''
    normalized_ne_tag = ''
    if len(items) > 5 and items[5] <> 'O':
      is_ne = True
      ne_tag = items[5]
      if len(items) > 6:
        normalized_ne_tag = ' '.join(items[6].split('_'))
    return cls(lemma, word, pos, '', is_ne, is_timex, ne_tag, normalized_ne_tag,
        timex_attr=timex_attr, char_start=char_start, char_end=char_end)

  '''Format: Index, Word, Pos, constant_label, is_ne, is_timex, ne_tag, normalized_ne_tag.'''
  @classmethod
  def parse_conll_line(cls, conll_line):
    word = conll_line[1]
    original_word = conll_line[2]
    pos = conll_line[3]
    constant_label = '' if conll_line[4] == '_' else conll_line[4]
    is_ne = conll_line[5] == '1'
    is_timex = conll_line[6] == '1'
    ne_tag = '' if conll_line[7]=='_' else conll_line[7]
    normalized_ne_tag = '' if conll_line[8] == '_' else conll_line[8]
    return cls(word, original_word, pos, constant_label, is_ne, is_timex, 
               ne_tag, normalized_ne_tag)

  def conll_line_str(self, i):
    conll_str = str(i+1) + '\t' + self.word + '\t' + self.original_word + '\t'
    conll_str += self.pos + '\t'
    conll_str += self.constant_label + '\t' if self.constant_label else '_\t'
    conll_str += '1\t' if self.is_ne else '0\t'
    conll_str += '1\t' if self.is_timex else '0\t'
    conll_str += self.ne_tag + '\t'  if self.ne_tag else '_\t'
    if self.ne_tag and self.normalized_ne_tag:
      conll_str += self.normalized_ne_tag
    else:
      conll_str += '_'
    return conll_str + '\n'

  def col_line_str(self, i):
    col_str = str(i+1) + '\t' 
    word = self.word.lower()
    if self.original_word == '_':
      col_str += '_\t' + self.word
    else:
      col_str += word + '\t' + self.pos
    return col_str + '\n'
    
  '''Format: Index, Word, Lemma, pos, pos, _, head_index, relation, _, _.'''
  @classmethod
  def parse_conll_dep_line(cls, conll_line):
    word = conll_line[2]
    original_word = conll_line[1]
    pos = conll_line[3]
    if conll_line[6] == '0':
      head_index = -1
      relation = ''
    else:
      head_index = int(conll_line[6]) - 1
      relation = conll_line[7]

    return cls(word, original_word, pos, '', False, False, '', '', head_index,
        relation)

  def conll_dep_line_str(self, i):
    # Assumes no dependencies.
    conll_str = str(i+1) + '\t' + self.word + '\t' 
    conll_str += self.original_word if self.original_word else '_'
    conll_str += '\t' + self.pos + '\t' + self.pos + '\t_\t' 
    if self.edge >= -1:
      conll_str += str(self.edge+1) + '\t'
    else:
      conll_str += '_\t'
    conll_str += self.relation if self.relation else '_'  
    conll_str += '\t_\t_'
    return conll_str + '\n'

  def reset_char_spans(self, char_start, char_end):
    self.char_start = char_start
    self.char_end = char_end
    self.pred_char_start = char_start
    self.pred_char_end = char_end
    self.const_char_start = char_start
    self.const_char_end = char_end

  def find_wordnet_lemmas(self):
    ptb_tag_preffixes = ['J', 'R', 'V', 'N']
    wordnet_tags = ['a', 'r', 'v', 'n']
    wordnet_tag = [] 
    for i, prefix in enumerate(ptb_tag_preffixes):
      if self.pos.startswith(prefix):
        wordnet_tag.append(wordnet_tags[i])
        if prefix == 'J':
          wordnet_tag.append('s')
    derived_nouns = set()
    derived_verbs = set()
    for tag in wordnet_tag:
      for synset in wn.synsets(self.word, pos=tag):
        for wn_lemma in synset.lemmas():
          for form in wn_lemma.derivationally_related_forms():
            word = form.name()
            pos = form.synset().pos()
            if pos == 'n':
              derived_nouns.add(word)
            elif pos == 'v':
              derived_verbs.add(word)
    if derived_nouns and not self.pos.startswith('N'):
      derived_list = list(derived_nouns)
      derived_match = [mrs_util.prefix_sim_long(self.word.lower(), deriv)
                       for deriv in derived_nouns]
      self.noun_lexeme = derived_list[derived_match.index(max(derived_match))]
    if derived_verbs and not self.pos.startswith('V'):
      derived_list = list(derived_verbs)
      derived_match = [mrs_util.prefix_sim_long(self.word.lower(), deriv)
                       for deriv in derived_verbs]
      self.verb_lexeme = derived_list[derived_match.index(max(derived_match))]


class Sentence():
  def __init__(self, sentence, index_map=[], sent_ind=0):
    self.sentence = sentence # tokens
    self.const_vars = util.ConstantVars()
    self.sent_ind = sent_ind
    self.root_index = -1
    self.index_map = index_map

  def word_at(self, i):
    return self.sentence[i].word

  @classmethod
  def parse_conll(cls, sent_conll):
    sentence = []
    for token_line in sent_conll:
      token = Token.parse_conll_line(token_line)
      sentence.append(token)
    return cls(sentence)

  @classmethod
  def parse_conll_dep(cls, sent_conll):
    sentence = []
    for token_line in sent_conll:
      token = Token.parse_conll_dep_line(token_line)
      sentence.append(token)
    return cls(sentence)

  @classmethod
  def parse_json_line(cls, json_line):
    toks = json.loads(json_line)
    tokens = []
    tokens_index = {}
    sent_id = toks["id"]
    
    # Construct tokens
    for tok in toks["tokens"]:
      assert (tok["id"] - 1) == len(tokens)
      props = tok["properties"]
      
      ne_tag = props["NE"] if props.has_key("NE") else ''
      token = Token(props["lemma"], props["word"], props["POS"], '',  
                    props.has_key("NE"), False, '', '', char_start=tok["start"],
                    char_end=tok["end"])
      if tok.has_key("predicate_end"):
        token.pred_char_end = tok["predicate_end"]
      if tok.has_key("constant_end"):
        token.const_char_end = tok["constant_end"]
      if props.has_key("constant"):
        token.is_const = True
        token.const_lexeme = props["constant"]
      elif props["word"].endswith("."):
        token.const_lexeme = props["word"][:-1]

      if props.has_key("erg_predicate") and props["erg_predicate"]:
        token.is_pred = True 

      tokens.append(token)
    return cls(tokens, sent_ind=sent_id)

  def original_sentence_str(self):
    words = [token.original_word for token in self.sentence]
    return ' '.join(words) + '\n'

  def pred_lexeme_str(self):
    words = [token.pred_lexeme for token in self.sentence]
    return ' '.join(words) + '\n'

  def pred_verb_lexeme_str(self):
    words = []
    for token in self.sentence:
      if token.is_pred:
        word = token.pred_lexeme 
      elif token.verb_lexeme:
        word = '_' + token.verb_lexeme
      else:
        word = token.pred_lexeme 
      words.append(word)
    return ' '.join(words) + '\n'

  def nom_lexeme_str(self):
    words = []
    for token in self.sentence:
      if token.is_nom:
        word = token.nom_lexeme 
      elif token.noun_lexeme:
        word = '_' + token.noun_lexeme
      else:
        word = token.nom_lexeme 
      words.append(word)
    return ' '.join(words) + '\n'

  def const_lexeme_str(self):
    conc = u''
    for token in self.sentence:
      try:
        conc += u' ' + token.const_lexeme
      except UnicodeDecodeError:
        print 'Cannot write: ' + token.const_lexeme
        conc += u' X'
    return conc[1:] + u'\n'

  def wiki_lexeme_str(self):
    conc = u''
    for token in self.sentence:
      #if token.is_wiki:
      #  print token.wiki_lexeme
      try:
        conc += u' ' + token.wiki_lexeme
      except UnicodeDecodeError:
        print 'Cannot write: ' + token.wiki_lexeme
        conc += u' X'
    return conc[1:] + u'\n'


  def ch_span_str(self):
    words = [str(token.char_start) + ':' + str(token.char_end)
             for token in self.sentence]
    return ' '.join(words) + '\n'

  def pred_ch_span_str(self):
    words = [str(token.pred_char_start) + ':' + str(token.pred_char_end)
             for token in self.sentence]
    return ' '.join(words) + '\n'

  def const_ch_span_str(self):
    words = [str(token.const_char_start) + ':' + str(token.const_char_end)
             for token in self.sentence]
    return ' '.join(words) + '\n'

  def pos_str(self):
    words = [token.pos for token in self.sentence]
    words = ['_' + word if token.is_pred else word
             for word, token in zip(words, self.sentence)]
    return ' '.join(words) + '\n'

  def ne_tag_str(self):
    words = ['O' if token.ne_tag == '' else token.ne_tag 
             for token in self.sentence]
    words = [word + '_C' if token.is_const else word 
             for word, token in zip(words, self.sentence)]
    return ' '.join(words) + '\n'

  def raw_sentence_str(self, to_lower):
    if to_lower:
      words = [token.word.lower() for token in self.sentence]
    else:
      words = [token.word for token in self.sentence]
    return ' '.join(words) + '\n'


  def json_sentence_str(self, sent_id):
    s = {"id": sent_id+1}
    token_list = []
    for i, token in enumerate(self.sentence):
      token_s = {"id": i+1}
      token_s["start"] = token.char_start
      token_s["end"] = token.char_end # TODO pred/const spans
      token_props = {}

      token_props["word"] = token.original_word
      token_props["lemma"] = token.lemma
      token_props["POS"] = token.pos
      if token.is_pred:
        token_props["erg_predicate"] = True 
      if token.pred_char_end != token.char_end:
        token_s["predicate_end"] = token.pred_char_end

      if token.ne_tag != "":
        token_props["NE"] = token.ne_tag
        if (token.const_lexeme != token.original_word 
            and (token.const_lexeme + ".") != token.original_word):
          token_props["constant"] = token.const_lexeme
        if token.const_char_end != token.char_end:
          # by default use predicate end
          if token.const_char_end != token.pred_char_end:
            token_s["constant_end"] = token.const_char_end

      token_s["properties"] = token_props
      token_list.append(token_s)
    s["tokens"] = token_list
    return json.dumps(s, sort_keys=True)


  def const_variable_sentence(self, varnames=[]):
    var_sent = []
    const_vars = util.ConstantVars() 
    index_map = [] 

    constant_state = False
    constant_type = ''
    var_value = ''
    for i, token in enumerate(self.sentence):
      if token.constant_label:
        if constant_state and token.constant_label == 'I':
          if constant_type == 'name':
            const_vars.names[-1].append(token.word)
          if constant_type == 'number':
            const_vars.numbers[-1].append(token.word)
          var_value += '_' + token.word
        else: 
          if constant_state:
            var_sent[-1].normalized_ne_tag = var_value
          constant_state = True
          constant_type = token.constant_label
          var_name = ''
          var_value = '' 
          if constant_type == 'name':
            const_vars.names.append([token.word])
            var_name = 'name' + str(len(const_vars.names))
          elif constant_type == 'number':
            const_vars.numbers.append([token.word])
            var_name = 'number' + str(len(const_vars.numbers))
          if var_name and varnames and varnames[i]:
            var_name = varnames[i]
          if var_name:
            index_map.append(i)
            var_sent.append(Token(var_name, '_', token.pos, constant_type,
              False, False, token.ne_tag, ''))
            var_value = token.word
          else:
            print 'unknown constant', constant_type
      else:
        if constant_state:
          constant_state = False
          constant_type = ''
          var_sent[-1].normalized_ne_tag = var_value
          var_value = '' 
        index_map.append(i)
        var_sent.append(Token(token.word, token.original_word, token.pos, '',
            False, False, '', ''))
    return Sentence(var_sent, index_map)

  def record_children(self):
    total_children = 0
    for i, token in enumerate(self.sentence):
      self.sentence[i].children = []
    for i, token in enumerate(self.sentence):
      head_index = token.edge
      if head_index == -1:
        self.root_index = i
      else:
        self.sentence[head_index].children.append(i)
        total_children += 1
    for i, token in enumerate(self.sentence):
      self.sentence[i].children.sort() 

  def sentence_str(self, remove_punct):
    words = [token.word for token in self.sentence]
    s = ''
    indexes = []
    if remove_punct:
      for i, w in enumerate(words):
        if not util.is_punct(w):
          s += w + ' '
          if w == 'NULL':
            i = -1
          indexes.append(i)
      s = s[:-1] + '\n'    
    else:
      s = ' '.join(words) + '\n'
      indexes = range(len(words))
    s_ind = ' '.join(map(str, indexes)) + '\n'
    return s.lower(), s_ind

  def linear_dep_str(self, i):
    # Pre-order traversal of graph.
    # No constants.
    graph_str = '( ' + self.sentence[i].word
    for child_index in self.sentence[i].children:
      graph_str += ' :' + self.sentence[child_index].relation + ' ' \
              + self.linear_dep_str(child_index) 
    return graph_str + ' )'

def read_conll_sentences(sent_file_name):
  sent_file = open(sent_file_name, 'r')
  sent_conll_in = [[line.split('\t') for line in sent.split('\n')] 
                  for sent in sent_file.read().split('\n\n')[:-1]]
  sentences = []
  for sent_conll in sent_conll_in:
    sentences.append(Sentence.parse_conll(sent_conll))      
  return sentences

def read_conll_dep_sentences(sent_file_name):
  sent_file = open(sent_file_name, 'r')
  sent_conll_in = [[line.split('\t') for line in sent.split('\n')] 
                  for sent in sent_file.read().split('\n\n')[:-1]]
  sentences = []
  for sent_conll in sent_conll_in:
    sentences.append(Sentence.parse_conll_dep(sent_conll))      
  return sentences

def read_raw_sentences(sent_file_name):
  sent_file = open(sent_file_name, 'r')
  sent_in = [sent.split(' ') for sent in sent_file.read().split('\n')[:-1]]
  return sent_in

