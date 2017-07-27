import sys

if __name__=='__main__':
  lines1 = [sent.split() for sent in open(sys.argv[1], 'r').read().split('\n')[:-1]]
  lines2 = [sent.split() for sent in open(sys.argv[2], 'r').read().split('\n')[:-1]]
  assert len(lines1) == len(lines2)
  max_length1 = 200
  max_length2 = 400
  lengths = [[0 for _ in xrange(max_length2)] for _ in xrange(max_length1)]
  extras = []
  for line1, line2 in zip(lines1, lines2):
    if len(line1) < max_length1 and len(line2) < max_length2:
      lengths[len(line1)][len(line2)] += 1
    else:
      extras.append((len(line1), len(line2)))
  col_totals = [[0 for _ in xrange(max_length2+1)]]
  for i, leng in enumerate(lengths):
    col_totals.append([])
    for j, l in enumerate(leng):
      col_totals[-1].append(col_totals[i][j]+l)

  acc_lengths = [[0 for _ in xrange(max_length2+1)]]
  for i, leng in enumerate(lengths):
    acc_lengths.append([0])
    for j, l in enumerate(leng):
      acc_lengths[-1].append(acc_lengths[-1][-1] + col_totals[i][j])

  thresholds = [0.4, 0.8, 0.98]
  brackets = []
  # find smallest i+j greater than threshold
  for a in thresholds:
    b_i, b_j = max_length1, max_length2
    for i in xrange(max_length1+1):
      for j in xrange(max_length2+1):
        if acc_lengths[i][j]/(len(lines1)+0.0) > a and i + j < b_i + b_j:
          b_i, b_j = i, j
    brackets.append((b_i, b_j))
  for pair in brackets:
    print str(pair[0]) + ' ' + str(pair[1])

