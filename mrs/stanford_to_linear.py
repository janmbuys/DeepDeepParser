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
import codecs
import string
import sys

import sentence as asent
import util as mrs_util

def convert_number(num_str, has_unit):
  try:
    unit_ind = 1 if num_str[0] in ['>', '<', '~'] else 0
    if len(num_str) > 1 and num_str[1] == '=':
      unit_ind = 2
    if has_unit:
      num_str = num_str[unit_ind+1:]
    else:
      num_str = num_str[unit_ind:]

    # For now, ignore first value in ranges.
    if '-' in num_str:
      num_str = num_str[(num_str.index('-') + 1):]
    value = float(num_str)
    try:
      value = int(value)
    except ValueError:
      value = value
    return str(value)
  except ValueError:
    print 'Cannot parse:', num_str
    return num_str

def convert_period(period_str):
  rate_re = re.compile(r'P(\d\d*)([A-Z])')
  time_re = re.compile(r'PT(\d\d*)([A-Z])')
  period_m = rate_re.match(period_str)
  time_m = time_re.match(period_str)
  if period_m:
    period = int(period_m.group(1))
    unit = period_m.group(2)
    if unit == 'Y':
      unit = 'year'
    elif unit == 'M':
      unit = 'month'
    elif unit == 'W':
      unit = 'week'
    elif unit == 'D':
      unit = 'day'
    return period, unit
  elif time_m:
    period = int(time_m.group(1))
    unit = time_m.group(2)
    if unit == 'H':
      unit = 'hour'
    elif unit == 'M':
      unit = 'minute'
    elif unit == 'S':
      unit = 'second'
    return period, unit
  else:
    return 0, ''

def read_sentences_normalize_ne(stanford_file_name): 
  stanford_file = codecs.open(stanford_file_name, 'r', 'utf-8')

  sentences = []
  tokens = []

  token_alignments = []
  text_line = ''

  state = False
  ne_state = False
  money_state = False
  percent_state = False
  number_state = False
  ordinal_state = False
  time_state = False
  date_state = False
  duration_state = False
  set_state = False
  last_ne_tag = ''
  token_counter = 0

  date_re = re.compile(r'^(\d\d\d\d|XXXX)-(\d\d|XX)-(\d\d|XX)$')
  date2_re = re.compile(r'^(\d\d\d\d|XXXX)-(\d\d|XX)$')
  date3_re = re.compile(r'^(\d\d\d\d|XXXX)$')

  for line in stanford_file:
    if line.startswith('Sentence #'):
      if state:
        sentences.append(asent.Sentence(tokens, token_alignments))
        tokens = []
        token_alignments = []
        state = False
        ne_state = False
        money_state = False
        percent_state = False
        number_state = False
        ordinal_state = False
        time_state = False
        date_state = False
        duration_state = False
        set_state = False
        last_ne_tag = ''
        token_counter = 0
    elif line.startswith('[Text=') and line[-2]==']':
      token = asent.Token.parse_stanford_line(line[1:-2], {})
      #For LOCATION, PERSON, ORGANIZATION, MISC.
      if ne_state and not (token.is_ne and token.ne_tag == last_ne_tag):
        ne_state = False
      if not ne_state and token.is_ne and token.ne_tag in \
          ['LOCATION', 'PERSON', 'ORGANIZATION', 'MISC']:
        ne_state = True
        # Appends to the front.
        last_ne_tag = token.ne_tag
        token.constant_label = 'name'
        token.const_lexeme = token.word
      # For MONEY:
      if money_state and not (token.is_ne and token.ne_tag == 'MONEY'):
        money_state = False
      elif not money_state and token.is_ne and token.ne_tag == 'MONEY':
          money_state = True
          money_str = token.normalized_ne_tag
          if len(money_str) == 0:
            # Not treated as money.
            token.is_ne = False
            token.ne_tag = ''
            money_state = False
          elif len(money_str) > 1: # length 1 is for units 
            unit_ind = 1 if money_str[0] in ['>', '<', '~'] else 0
            if money_str[1] == '=':
              unit_ind = 2
            token.const_lexeme = convert_number(money_str, True)
      # Percentage.
      if percent_state and not (token.is_ne and token.ne_tag == 'PERCENT'):
        percent_state = False
      elif not percent_state and token.is_ne and token.ne_tag == 'PERCENT':
        percent_state = True
        percent_str = token.normalized_ne_tag
        if len(percent_str) > 1:
          token.normalized_ne_tag = convert_number(percent_str, True)
      if number_state and not (token.is_ne and token.ne_tag == 'NUMBER'):
        number_state = False
      elif not number_state and token.is_ne and token.ne_tag == 'NUMBER':
        number_state = True
        number_str = token.normalized_ne_tag
        if len(number_str) == 0:
          number_state = False
          token.is_ne = False
          token.ne_tag = ''
        else:
          token.const_lexeme = convert_number(number_str, False)
      if ordinal_state and not (token.is_ne and token.ne_tag == 'ORDINAL'):
        ordinal_state = False
      elif not ordinal_state and token.is_ne and token.ne_tag == 'ORDINAL':
        ordinal_state = True
        number_str = token.normalized_ne_tag
        if len(number_str) == 0:
          number_state = False
          token.is_ne = False
          token.ne_tag = ''
        else:
          token.const_lexeme = convert_number(number_str, False)
      if time_state and not (token.is_timex
          and token.ne_tag in ['DATE', 'TIME']):
        time_state = False
      elif not time_state and (token.is_timex
          and token.ne_tag in ['DATE', 'TIME']):
        # The same date and time expression and contain both DATE and TIME.
        time_state = True
      if time_state and not date_state and token.ne_tag == 'DATE':
        # Only match pure date expressions
        # - cannot convert compound expressions cleanly enough.
        date_str = token.normalized_ne_tag
        if len(date_str.split()) == 1:
          # Strip time from string.
          if 'T' in date_str:
            date_str = date_str[:date_str.index('T')]
          if re.match(r'^\d\d\dX$', date_str):
            date_str = date_str[:3] + '0'
          if re.match(r'^\d\dXX$', date_str):
            date_str = date_str[:2] + '00'
          m = date_re.match(date_str)
          m2 = date2_re.match(date_str)
          m3 = date3_re.match(date_str)
          if m or m2 or m3:
            date_state = True
            if m:
              date_list = list(m.groups())
            elif m2:
              date_list = list(m2.groups())
            elif m3:
              date_list = list(m3.groups())
            date_list = filter(lambda d: 'X' not in d, date_list)
            date_list = [convert_number(date, False) for date in date_list]
            if date_list:
              token.const_lexeme = date_list[0]
          #else don't handle as a date.
      if date_state and token.ne_tag <> 'DATE':
        date_state = False
      # For Duration:
      if duration_state and not (token.is_timex and token.ne_tag == 'DURATION'):
        duration_state = False
      elif not duration_state and token.is_timex and token.ne_tag == 'DURATION':
        duration_state = True
        time_str = token.normalized_ne_tag
        period, unit = convert_period(time_str)
        if period == 0:
          duration_state = False
        else:
          token.const_lexeme = str(period)
          token.ne_tag += '_' + unit
      # For SET:
      if set_state and not (token.is_timex and token.ne_tag == 'SET'):
        set_state = False
      elif not set_state and token.is_timex and token.ne_tag == 'SET':
        set_state = True
        freq = 1
        period = 0
        unit = ''
        if token.timex_attr.has_key('freq'):
          rate_re = re.compile(r'P(\d\d*)([A-Z])')
          freq_m = rate_re.match(token.timex_attr['freq'])
          freq = int(freq_m.group(1))
        if token.timex_attr.has_key('periodicity'):
          period, unit = convert_period(token.timex_attr['periodicity'])
        if period == 0:
          set_state = False
          token.ne_tag = ''
        else:
          if freq > 1:
            token_ne_tag += '_rate'
          token.const_lexeme = str(period)
          token.ne_tag += '_temporal_' + unit
      # Identify numbers:
      if re.match(r'^[+-]?\d+(\.\d+)?$', token.word):
        if token.const_lexeme == '':
          token.const_lexeme = convert_number(token.word, False)
        token.constant_label = 'number'
      token.pred_lexeme = token.word
      tokens.append(token)
      state = True
  if state:
    sentences.append(asent.Sentence(tokens))
  return sentences


def read_sentences(stanford_file_name, file_id):
  stanford_file = codecs.open(stanford_file_name, 'r', 'utf-8')

  sentences = []
  raw_sentences = []
  tokens = []

  text_line = ''
  state_line = ''
  sent_offset = 0
  state = False
  state1 = False

  for line in stanford_file:
    if line.startswith('Sentence #'):
      if state:
        sentences.append(asent.Sentence(tokens))
        sentences[-1].offset = sent_offset
        sentences[-1].raw_txt = text_line
        sentences[-1].file_id = file_id
        text_line = ''
        state_line = ''
        tokens = []
        state = False
        state1 = False
    elif len(line) > 1 and line[-2]==']' and (state or line.startswith('[Text=')):
      if state_line:
        token = asent.Token.parse_stanford_line(state_line + ' ' + line[:-2], {})
      else:
        token = asent.Token.parse_stanford_line(line[1:-2], {})
      if not state1:
        sent_offset = token.char_start
      ind_start = token.char_start - sent_offset 
      ind_end = token.char_end - sent_offset 
      token.reset_char_spans(ind_start, ind_end)

      word = token.original_word
      word = word.replace(u"\u00A0", "_") 
      if '_' in word:
        split_word = word.split('_')
        split_inds = filter(lambda x: word[x] == '_', 
                            range(len(word)))
        first_word = word[:split_inds[0]]
        token.original_word = first_word
        token.word = first_word
        if normalize_ne:
          token.pred_lexeme = first_word.lower()
        else:
          token.pred_lexeme = first_word.lower() + u'/' + token.pos.lower()
        token.const_lexeme = first_word
        token.char_end = token.char_start + split_inds[0]
        tokens.append(token)
        for j, w in enumerate(split_word[1:]):
          char_start = token.char_start + split_inds[j] + 1
          if j + 1 < len(split_inds):
            char_end = token.char_start + split_inds[j+1]
          else:
            char_end = token.char_start + len(word)
          new_token = asent.Token(w, w, token.pos, token.constant_label, 
              token.is_ne, token.is_timex, token.ne_tag,
              token.normalized_ne_tag, char_start=char_start, char_end=char_end)
          tokens.append(new_token)
      else:  
        tokens.append(token)
      state = True
      state1 = True
    elif line.startswith('[Text='):
      state_line = line[1:].strip()
      state = True
    else: #if line.strip():
      if state:
        state_line += ' ' + line.strip()
      else:    
        text_line += line.replace('\n', ' ')
  if state:
    sentences.append(asent.Sentence(tokens))
    sentences[-1].offset = sent_offset
    sentences[-1].raw_txt = text_line
    sentences[-1].file_id = file_id
  return sentences


def process_stanford(input_dir, working_dir, erg_dir, set_name, 
    use_pred_lexicon=True, use_const_lexicon=True, normalize_ne=False,
    read_epe=False):
  nom_map = {}
  wiki_map = {}
  if use_pred_lexicon:
    pred_map = mrs_util.read_lexicon(erg_dir + 'predicates.lexicon')
    if normalize_ne:
      nom_map = mrs_util.read_lexicon(erg_dir + 'nominals.lexicon')
  else:
    pred_map = {}
  if use_const_lexicon:
    const_map = mrs_util.read_lexicon(erg_dir + 'constants.lexicon')
    if normalize_ne:
      wiki_map = mrs_util.read_lexicon(erg_dir + 'wiki.lexicon')
  else:
    const_map = {}

  if read_epe:
    file_ids = []
    in_type = input_dir[4:-1]
    file_list = open(in_type + '.' + set_name + '.list', 'r').read().split('\n')[:-1]
    file_ids = [name[name.rindex('/')+1:] for name in file_list]
    sentences = []
    for file_id in file_ids:
      sentences.extend(read_sentences(
        (working_dir + '/raw-' + set_name + '/' + file_id + '.out'), 
         file_id))
  else:
    suffix = '.raw'
    if normalize_ne:
      sentences = read_sentences_normalize_ne((working_dir + set_name + suffix + '.out'))
    else:
      sentences = read_sentences((working_dir + set_name + suffix + '.out'), '0')

  max_token_span_length = 5
  for i, sent in enumerate(sentences):
    for j, token in enumerate(sent.sentence):
      if normalize_ne:
        sentences[i].sentence[j].find_wordnet_lemmas()

      # Matches lexemes.
      lexeme = ''
      if token.original_word in const_map:
        lexeme = const_map[token.original_word]
      elif token.original_word.lower() in const_map:
        lexeme = const_map[token.original_word.lower()]
      elif token.word in const_map:
        lexeme = const_map[token.word]
      if lexeme <> '':
        sentences[i].sentence[j].const_lexeme = lexeme
        sentences[i].sentence[j].is_const = True

      lexeme = ''
      if token.original_word in pred_map:
        lexeme = pred_map[token.original_word]
      elif token.original_word.lower() in pred_map:
        lexeme = pred_map[token.original_word.lower()]
      elif token.word in pred_map: # lemma
        lexeme = pred_map[token.word]
      if normalize_ne:
        nom_lexeme = ''
        if token.original_word in nom_map:
          nom_lexeme = nom_map[token.original_word]
        elif token.original_word.lower() in nom_map:
          nom_lexeme = nom_map[token.original_word.lower()]
        elif token.word in nom_map: # lemma
          nom_lexeme = nom_map[token.word]
        if nom_lexeme == '':
          sentences[i].sentence[j].nom_lexeme = '_' + token.word
        else:
          sentences[i].sentence[j].nom_lexeme = nom_lexeme
          sentences[i].sentence[j].is_nom = True

      if not normalize_ne:
        if len(lexeme) > 2 and '+' in lexeme[:-1]:
          lexeme = lexeme[:lexeme.index('+')]
        elif len(lexeme) > 2 and '-' in lexeme[:-1]:
          lexeme = lexeme[:lexeme.index('-')]

      if lexeme <> '':
        sentences[i].sentence[j].is_pred = True
      if normalize_ne and lexeme == '': # for AMR
        lexeme = '_' + token.word # lemma
      if lexeme <> '':
        sentences[i].sentence[j].pred_lexeme = lexeme

      # Matches multi-token expressions.
      orth = token.original_word
      for k in range(j+1, min(j+max_token_span_length-1, len(sent.sentence))):
        orth += ' ' + sent.sentence[k].original_word
        if orth in const_map:
          sentences[i].sentence[j].const_lexeme = const_map[orth]
          sentences[i].sentence[j].const_char_end = sentences[i].sentence[k].char_end
          sentences[i].sentence[j].is_const = True
        if orth in pred_map:
          if normalize_ne:
            first_pred = pred_map[orth]
          elif len(pred_map[orth]) > 2 and '+' in pred_map[orth][:-1]:
            first_pred = pred_map[orth][:pred_map[orth].index('+')] 
          elif len(pred_map[orth]) > 2 and '-' in pred_map[orth][:-1]:
            first_pred = pred_map[orth][:pred_map[orth].index('-')] 
          else:
            first_pred = pred_map[orth]
          sentences[i].sentence[j].pred_lexeme = first_pred
          sentences[i].sentence[j].pred_char_end = sentences[i].sentence[k].pred_char_end
          sentences[i].sentence[j].is_pred = True

      if normalize_ne:
        wiki_lexeme = ''
        if token.original_word in wiki_map:
          wiki_lexeme = wiki_map[token.original_word]
        elif token.original_word.lower() in wiki_map:
          wiki_lexeme = wiki_map[token.original_word.lower()]
        elif token.word in wiki_map: # lemma
          wiki_lexeme = wiki_map[token.word]
        elif token.word.lower() in wiki_map: 
          wiki_lexeme = wiki_map[token.word.lower()]
        if wiki_lexeme == '':
          sentences[i].sentence[j].wiki_lexeme = token.const_lexeme
        else:
          sentences[i].sentence[j].wiki_lexeme = wiki_lexeme
          sentences[i].sentence[j].is_wiki = True

  return sentences

'''
Processing performed: Tokenize, lemmize, normalize numbers and time
expressions, insert variable tokens for named entities etc.
'''
if __name__=='__main__':
  assert len(sys.argv) >= 4
  input_dir = sys.argv[1] + '/'
  working_dir = sys.argv[2] + '/'
  erg_dir = sys.argv[3] + '/'

  read_epe = len(sys.argv) > 4 and '-epe' in sys.argv[4:]

  set_list = ['train', 'dev', 'test']
  normalize_ne = len(sys.argv) > 4 and '-n' in sys.argv[4:]

  use_pred_lexicon = True
  use_const_lexicon = True

  for set_name in set_list:
    sentences = process_stanford(input_dir, working_dir, erg_dir, set_name,
        use_pred_lexicon, use_const_lexicon, normalize_ne, read_epe)
 
    sent_output_file = open(working_dir + set_name + '.en', 'w')
    sent_offsets_file = open(working_dir + set_name + '.off', 'w')
    sent_ids_file = open(working_dir + set_name + '.ids', 'w')
    sent_txt_file = open(working_dir + set_name + '.txt', 'w')
    pred_output_file = open(working_dir + set_name + '.lex.pred', 'w')
    const_output_file = open(working_dir + set_name + '.lex.const', 'w')
    wiki_output_file = open(working_dir + set_name + '.lex.wiki', 'w')
    pos_output_file = open(working_dir + set_name + '.pos', 'w')
    ne_output_file = open(working_dir + set_name + '.ne', 'w')
    span_output_file = open(working_dir + set_name + '.span', 'w')
    pred_span_output_file = open(working_dir + set_name + '.span.pred', 'w')
    const_span_output_file = open(working_dir + set_name + '.span.const', 'w')
    if normalize_ne:
      nom_output_file = open(working_dir + set_name + '.lex.nom', 'w')

    for sent in sentences:
      out_str = sent.original_sentence_str()
      sent_output_file.write(out_str.encode('utf-8', 'replace'))
      if normalize_ne:
        lex_str = sent.pred_verb_lexeme_str()
        pred_output_file.write(lex_str.encode('utf-8', 'replace'))
        lex_str = sent.nom_lexeme_str()
        nom_output_file.write(lex_str.encode('utf-8', 'replace'))
      else: 
        lex_str = sent.pred_lexeme_str()
        pred_output_file.write(lex_str.encode('utf-8', 'replace'))
      lex_str = sent.const_lexeme_str()
      lex_enc = lex_str.encode('utf-8', 'replace')
      const_output_file.write(lex_enc)
      lex_str = sent.wiki_lexeme_str()
      lex_enc = lex_str.encode('utf-8', 'replace')
      sent_offsets_file.write(str(sent.offset) + '\n')
      sent_ids_file.write(str(sent.file_id) + '\n')
      txt_enc = sent.raw_txt.encode('utf-8', 'replace')
      sent_txt_file.write(txt_enc + '\n')
      wiki_output_file.write(lex_enc)
      pos_output_file.write(sent.pos_str())
      ne_output_file.write(sent.ne_tag_str())
      span_output_file.write(sent.ch_span_str())
      const_span_output_file.write(sent.const_ch_span_str())
      pred_span_output_file.write(sent.pred_ch_span_str())

