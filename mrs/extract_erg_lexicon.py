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
import sys

if __name__ == '__main__':
  assert len(sys.argv) >= 3
  lexicon_dir = sys.argv[1] + '/'
  lexicon_file = open(lexicon_dir + '/lexicon.tdl', 'r')
  out_dir = sys.argv[2] + '/'

  state = False
  entry = []
  pred_dict = {}
  const_dict = {}

  for line in lexicon_file:
    line = line.strip()
    if state:
      if line <> '':
        entry += line.split(' ')
      elif entry:
        # Parses entry.
        assert 'ORTH' in entry, entry
        orth_start = entry.index('ORTH') + 2
        assert '>,' in entry[orth_start:] or '>' in entry[orth_start:], entry
        if '>' in entry[orth_start:]:
          orth_end = entry.index('>', orth_start)
        else:
          orth_end = entry.index('>,', orth_start)
        orth = ''
        for i in xrange(orth_start, orth_end):
          if entry[i][-1] == ',':
            if entry[i][-3] == '-':
              orth += entry[i][1:-3] + ' '
            else:
              orth += entry[i][1:-2] + ' '
          else:
            orth += entry[i][1:-1] + ' '
        orth = orth[:-1]
        pred = ''
        const = ''

        const_keys = filter(lambda x: 'KEYREL.CARG' in x or x == 'CARG', entry)
        if const_keys:
          assert len(const_keys) == 1
          const = entry[entry.index(const_keys[0]) + 1]
          assert const[0] == '"' 
          if const[-1] == ',':
            assert const[-2] == '"'
            const = const[1:-2]
          elif const[-1] == '"':
            const = const[1:-1]
          else:
            const = const[1:]
        pred_keys = filter(lambda x: 'KEYREL.PRED' in x or x == 'PRED', entry)
        if pred_keys:
          pred = entry[entry.index(pred_keys[0]) + 1] 
          start_ind = 1 if pred[0] == '"' else 0
          if pred[start_ind] == '_':
            pred = pred[start_ind:pred.index('_', 2)]
          else:
            pred = ''           

        if pred:
          if orth in pred_dict:
            if pred <> pred_dict[orth]:
              if pred[1:] == orth or (pred_dict[orth][1:] <> orth and
                  abs(len(orth) - len(pred)) < abs(len(orth) - len(pred_dict[orth])-1)):
                pred_dict[orth] = pred
          else:
            pred_dict[orth] = pred
        if const:
          const_dict[orth] = const  
        state = False
    else:
      if line <> '':
        entry = line.split(' ')
        state = True

  pred_out_file = open(out_dir + 'predicates.erg.lexicon', 'w')
  for orth, pred in pred_dict.iteritems():
    pred_out_file.write(orth + '\n' + pred + '\n')
  pred_out_file.close()

  const_out_file = open(out_dir + 'constants.erg.lexicon', 'w')
  for orth, const in const_dict.iteritems():
    const_out_file.write(orth + '\n' + const + '\n')
  const_out_file.close()

