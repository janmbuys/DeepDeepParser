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

"""Reads in deepbank's export format and extracts EDS."""

import gzip
import sys

if __name__=='__main__':
  assert len(sys.argv) == 4, 'Invalid number of arguments (3 required).'
  output_dir = sys.argv[1] + '/'
  filename = sys.argv[2]
  set_name = sys.argv[3]

  # Reads text file.
  assert filename.endswith('eds.gz'), filename
  file_prefix = filename[:-len('eds.gz')]
  txt_filename = file_prefix + 'txt.gz'
  sentence_str = gzip.open(txt_filename, 'r').read().strip()
  
  in_file = gzip.open(filename, 'r')
  eds_str = ''

  tokens = []
  token_inds = {}
  token_start = []
  token_end = []

  state = 0
  for line in in_file:
    if state == 0 and line[0] == '{' and ':' in line:
      eds_str = line[1:line.index(':')]
      state = 1
    elif state == 1:
      if line[0] == '}':
        state = 2
      else:
        line = line.strip()
        eds_str += ' ; ' + line
 
  sent_out_file = open(output_dir + set_name + '.raw', 'a')
  sent_out_file.write(sentence_str + '\n')
  sent_out_file.close()

  lin_out_file = open(output_dir + set_name + '.eds', 'a')
  lin_out_file.write(eds_str + '\n')
  lin_out_file.close()

