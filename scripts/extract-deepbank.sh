#!/bin/bash

# Exports DeepBank 1.1 (WSJ section 00-21 with HPSG/MRS annotations).
bash $HOME/DeepDeepParser/scripts/export-deepbank.sh
DB_DIR=$ERG_DIR

# Extracts DMRS and EDS (DMRS is first converted from MRS).
# Sentences are extracted one by one from their individual files to form the train/dev/test split.

for TYPE in dmrs eds; do
  MRS_DIR="deepbank-${TYPE}" 
  mkdir -p $MRS_DIR
  EXTRACT_LINES="${HOME}/DeepDeepParser/mrs/extract_${TYPE}_lines.py"

  for dir in wsj00a wsj00b wsj00c wsj00d wsj01a wsj01b wsj01c wsj01d \
    wsj02a wsj02b wsj02c wsj02d wsj03a wsj03b wsj03c \
    wsj04a wsj04b wsj04c wsj04d wsj04e wsj05a wsj05b wsj05c wsj05d wsj05e \
    wsj06a wsj06b wsj06c wsj06d wsj07a wsj07b wsj07c wsj07d wsj07e \
    wsj08a wsj09a wsj09b wsj09c wsj09d wsj10a wsj10b wsj10c wsj10d \
    wsj11a wsj11b wsj11c wsj11d wsj11e wsj12a wsj12b wsj12c wsj12d \
    wsj13a wsj13b wsj13c wsj13d wsj13e wsj14a wsj14b wsj14c wsj14d wsj14e \
    wsj15a wsj15b wsj15c wsj15d wsj15e \
    wsj16a wsj16b wsj16c wsj16d wsj16e wsj16f wsj17a wsj17b wsj17c wsj17d \
    wsj18a wsj18b wsj18c wsj18d wsj18e wsj19a wsj19b wsj19c wsj19d; do
    for file in $DB_DIR/export/$dir/*; do 
      python $EXTRACT_LINES $MRS_DIR $file train
    done   
  done

  for dir in wsj20a wsj20b wsj20c wsj20d; do
    for file in $DB_DIR/export/$dir/*; do 
      python $EXTRACT_LINES $MRS_DIR $file dev
    done   
  done

  for dir in wsj21a wsj21b wsj21c wsj21d; do
    for file in $DB_DIR/export/$dir/*; do 
      python $EXTRACT_LINES $MRS_DIR $file test
    done   
  done

done

