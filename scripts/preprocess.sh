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

  # Copies data for parser training.

  LIN_DIR=${TYPE}-parse-data-deepbank
  mkdir -p $LIN_DIR
  ORACLE=dmrs.ae.ao # Arc-eager parser, alignment-ordered oracle

  for SET in train dev test; do
    cp $MRS_WDIR/${SET}.en $MRS_WDIR/${SET}.pos $MRS_WDIR/${SET}.ne $LIN_DIR/
    cp $MRS_WDIR/${SET}.${ORACLE}.nospan.unlex.lin $LIN_DIR/${SET}.parse
    cp $MRS_WDIR/${SET}.${ORACLE}.point.lin $LIN_DIR/${SET}.att
    cp $MRS_WDIR/${SET}.${ORACLE}.endpoint.lin $LIN_DIR/${SET}.endatt
  done

  python $HOME/DeepDeepParser/scripts/find_bucket_sizes.py $LIN_DIR/train.en $LIN_DIR/train.parse > $LIN_DIR/buckets

done

