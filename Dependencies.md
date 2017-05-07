This file specifies the external software required for running this code and obtaining and processing the data.
Certain environment variables have to be set.

## Python 2.7.

## Tensorflow 0.11 or 0.12.
    https://www.tensorflow.org/
Earlier or later versions may not be compatible.

## Java 1.8. 
    JAVA=java

## Stanford CoreNLP 3.5.2. 
    http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip. 
Extract to
    STANFORD_NLP=stanford-corenlp-full-2015-04-20

## ERG 1214
Includes Redwoods and DeepBank treebanks.
    ERG_DIR=erg1214
    svn checkout http://svn.delph-in.net/erg/tags/1214 $ERG_DIR

## LOGON 
Used to extract graph representations from the ERG treebanks.
    LOGONROOT=logon
    svn checkout http://svn.emmtee.net/trunk $LOGONROOT

Include in your .bashrc:
    $LOGONROOT=logon
    if [ -f ${LOGONROOT}/dot.bashrc ]; then
      . ${LOGONROOT}/dot.bashrc
    fi

## Smatch
    https://github.com/snowblink14/smatch 

## PyDelphin 
    https://github.com/delph-in/pydelphin

## ACE (ERG parser) 
    http://sweaglesw.org/linguistics/ace/

Download ACE:
    http://sweaglesw.org/linguistics/ace/download/ace-0.9.25-x86-64.tar.gz

Download ERG 1214 grammar image, unzip and place in $ERG_DIR:
    http://sweaglesw.org/linguistics/ace/download/erg-1214-x86-64-0.9.25.dat.bz2  

