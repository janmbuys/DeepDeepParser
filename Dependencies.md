This file specifies the external software required for running this code and obtaining and processing the data.
Note that environment variables have to be set.
The implementation is in Python 2.7.

## Tensorflow 0.11 or 0.12.
Earlier or later versions may not be compatible.
https://www.tensorflow.org/

## Stanford CoreNLP 3.5.2. 
Requires Java 1.8.
http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip. 
Set

    JAVA=java
    STANFORD_NLP=stanford-corenlp-full-2015-04-20

## ERG 1214
Includes Redwoods and DeepBank treebanks.

    ERG_DIR=erg1214
    svn checkout http://svn.delph-in.net/erg/tags/1214 $ERG_DIR

## LOGON 
Contains code to extract graph representations from the ERG treebanks.

    LOGONROOT=logon
    svn checkout http://svn.emmtee.net/trunk $LOGONROOT

Include in your .bashrc:

    $LOGONROOT=logon
    if [ -f ${LOGONROOT}/dot.bashrc ]; then
      . ${LOGONROOT}/dot.bashrc
    fi

## PyDelphin 
(D)MRS conversion tools.
https://github.com/delph-in/pydelphin

## ACE 
ERG parser.
http://sweaglesw.org/linguistics/ace/

Download ACE: 
http://sweaglesw.org/linguistics/ace/download/ace-0.9.25-x86-64.tar.gz

as well as the ERG 1214 grammar image (unzip and place in $ERG_DIR): 
http://sweaglesw.org/linguistics/ace/download/erg-1214-x86-64-0.9.25.dat.bz2  

## Smatch
Graph parser evaluation.
https://github.com/snowblink14/smatch 

