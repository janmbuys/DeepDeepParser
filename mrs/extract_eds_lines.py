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


"""Reads in deepbank's export format and converts it to various outputs."""

import gzip
import sys

if __name__=='__main__':
  assert len(sys.argv) == 4, 'Invalid number of arguments (3 required).'
  output_dir = sys.argv[1] + '/'

  filename = sys.argv[2]
  set_name = sys.argv[3]
  in_file = gzip.open(filename, 'r')

  sentence_str = ''
  eds_str = ''

  tokens = []
  token_inds = {}
  token_start = []
  token_end = []

  state = 0
  for line in in_file:
    if state == 0 and line[0] == '[' and '`' in line and '\'' in line:
      sentence_str = line[line.index('`')+1:line.rindex('\'')]
      state = 1
    elif state == 1 and line[0] == '<':
      state = 2
    elif state == 2:
      if line[0] == '>':
        state = 3
      else:
        items = line.strip().split(', ')
        tokens.append(items[5].strip()[1:-1])
        ind = items[3].strip()[1:-1] 
        token_inds[ind] = len(token_inds)
        token_start.append(int(ind.split(':')[0]))
        token_end.append(int(ind.split(':')[1]))
    elif state == 3 and line[0] == '{' and ':' in line:
      eds_str = line[1:line.index(':')]
      state = 4
    elif state == 4:
      if line[0] == '}':
        state = 5
      else:
        line = line.strip()
        eds_str += ' ; ' + line
 
  sent_out_file = open(output_dir + set_name + '.raw', 'a')
  sent_out_file.write(sentence_str + '\n')
  sent_out_file.close()

  lin_out_file = open(output_dir + set_name + '.eds', 'a')
  lin_out_file.write(eds_str + '\n')
  lin_out_file.close()

