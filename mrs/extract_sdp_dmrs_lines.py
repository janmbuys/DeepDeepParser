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

"""Reads in deepbank's export format and converts it to DMRS."""

import gzip
import sys

import delphin.mrs as mrs
import delphin.mrs.simplemrs as simplemrs
import delphin.mrs.simpledmrs as simpledmrs

if __name__=='__main__':
  assert len(sys.argv) == 4, 'Invalid number of arguments (3 required).'
  output_dir = sys.argv[1] + '/'
  filename = sys.argv[2]
  set_name = sys.argv[3]

  # Reads text file.
  assert filename.endswith('mrs.gz'), filename
  file_prefix = filename[:-len('mrs.gz')]
  txt_filename = file_prefix + 'txt.gz'
  sentence_str = gzip.open(txt_filename, 'r').read().strip()
  
  in_file = gzip.open(filename, 'r')
  simple_mrs_str = ''

  tokens = []
  token_inds = {}
  token_start = []
  token_end = []

  state = 0
  for line in in_file:
    if state == 0 and line.strip().startswith('[ LTOP:'): 
      # State 4: MRS 
      simple_mrs_str = line.strip() + ' '
      state = 4
    elif state == 4:
      if line.strip() == '':
        state = 5
      else:
        simple_mrs_str += line.strip() + ' '

  simple_mrs_code = simple_mrs_str.decode('utf-8', 'replace')
  simple_mrs_str = simple_mrs_code.encode('ascii', 'replace')

  dmrs_xml_str = mrs.convert(simple_mrs_str, 'simplemrs', 'dmrx')
  dmrs_object = mrs.dmrx.loads(dmrs_xml_str)
  simple_dmrs_str = simpledmrs.dumps(dmrs_object) #TODO this can give error
  mrs_object = simplemrs.loads_one(simple_mrs_str)

  dmrs_const_str = ''

  # Adds constants. 
  for ep in mrs_object.eps():
    if ep.args.has_key('CARG'):
      dmrs_const_str += (str(ep.lnk)[1:-1] + ' ' + str(ep.pred) + ' ' 
                         + ep.args['CARG'] + ' ')

  hyphen_sentence_str = sentence_str
  # Removes in-word hyphens in sentence.
  for i in xrange(1, len(sentence_str)-1):
    if (sentence_str[i] == '-' and sentence_str[i-1] <> ' ' and
        sentence_str[i+1] <> ' '):
      sentence_str = sentence_str[:i] + ' ' + sentence_str[i+1:]

  sent_out_file = open(output_dir + set_name + '.raw', 'a')
  sent_out_file.write(sentence_str + '\n')
  sent_out_file.close()

  lin_out_file = open(output_dir + set_name + '.sdmrs', 'a')
  lin_out_file.write(simple_dmrs_str + '\n')
  lin_out_file.close()

  lin_out_file = open(output_dir + set_name + '.carg', 'a')
  lin_out_file.write(dmrs_const_str + '\n')
  lin_out_file.close()

