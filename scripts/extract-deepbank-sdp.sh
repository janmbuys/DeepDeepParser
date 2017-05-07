#!/bin/bash

# Exports the SDP version of DeepBank 1.1.

# Downloads and extracts graphbank.
#wget http://sdp.delph-in.net/osdp-12.tgz
#tar -xvzf osdp-12.tgz # extracts to sdp/
DB_DIR="sdp/2015/eds"

# Extracts DMRS and EDS (DMRS is first converted from MRS).
# Sentences are extracted one by one from their individual files to form the train/dev/test split.

for TYPE in dmrs eds; do
  MRS_DIR="deepbank-sdp-${TYPE}" 
  mkdir -p $MRS_DIR
  EXTRACT_LINES="${HOME}/DeepDeepParser/mrs/extract_sdp_${TYPE}_lines.py"

  if [ $TYPE == "eds" ]; then
    ext="eds"
  else
    ext="mrs"
  fi  

  for file in $DB_DIR/20*.${ext}.gz $DB_DIR/21*.${ext}.gz; do
    python $EXTRACT_LINES $MRS_DIR $file train
  done

  for file in $DB_DIR/220*.${ext}.gz; do
    python $EXTRACT_LINES $MRS_DIR $file dev
  done

  for file in $DB_DIR/221*.${ext}.gz; do
    python $EXTRACT_LINES $MRS_DIR $file test
  done
done

