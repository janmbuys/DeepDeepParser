#!/bin/bash

for TYPE in dmrs eds; do

  MRS_DIR="deepbank-${TYPE}" 
  MRS_WDIR=${MRS_DIR}-working 
  mkdir -p $MRS_WDIR

  # Construct lexicon.
  python $HOME/DeepDeepParser/mrs/extract_erg_lexicon.py $ERG_DIR $MRS_WDIR
  python $HOME/DeepDeepParser/mrs/extract_data_lexicon.py $MRS_DIR $MRS_WDIR

  # Runs Stanford NLP tools over input.

  printf "$MRS_DIR/train.raw\n$MRS_DIR/dev.raw\n$MRS_DIR/test.raw\n" > FILELIST
  $JAVA -cp "$STANFORD_NLP/*" -Xmx16g \
      edu.stanford.nlp.pipeline.StanfordCoreNLP \
      -annotators tokenize,ssplit,pos,lemma,ner \
      -ssplit.eolonly \
      -filelist FILELIST \
      -outputFormat text -outputDirectory $MRS_WDIR \
      -tokenize.options "normalizeCurrency=False,normalizeFractions=False"\
          "normalizeParentheses=False,normalizeOtherBrackets=False,"\
          "latexQuotes=False,unicodeQuotes=True,"\
          "ptb3Ellipsis=False,unicodeEllipsis=True,"\
          "escapeForwardSlashAsterisk=False"
  rm FILELIST

  # Processes Stanford NLP output.
  python $HOME/DeepDeepParser/mrs/stanford_to_linear.py $MRS_DIR $MRS_WDIR $MRS_WDIR

  # Converts MRS graphs to multiple linearizations.
  python $HOME/DeepDeepParser/mrs/read_mrs.py $MRS_DIR $MRS_WDIR $TYPE 

done

