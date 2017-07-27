# DeepDeepParser

Code and data preparation scripts for the paper [Robust Incremental Neural Semantic Graph Parsing](https://arxiv.org/abs/1704.07092), Jan Buys and Phil Blunsom, ACL 2017.

## Prerequisites
See Dependencies.md

## Data preparation

To extract DMRS and EDS graphs from DeepBank: scripts/extract-deepbank.sh (requires the LOGON environment and full original data.)

To extract DMRS and EDS graphs from the SDP release of DeepBank: scripts/extract-deepbank-sdp.sh (does not require the LOGON environment.)

Pre-processing (constructs lexicon, runs Stanford CoreNLP, constructs graph linearizations/oracle transition sequences): 
    scripts/preprocess.sh

## Training

Train the transition-based parser:

    python rnn/parser.py --decode_dev --decode_train --use_hard_attention_arc_eager_decoder --predict_span_end_pointers  --data_dir [data_dir] --train_dir [working_dir] --embedding_vectors [embedding_file] --train_name train --dev_name dev --singleton_keep_prob 0.5 --size 256 --input_embedding_size 256 --output_embedding_size 128 --tag_embedding_size 32 --use_encoder_tags --input_drop_prob 0.3 --output_drop_prob 0.3 --initialize_word_vectors 

where `data_dir` contains the pre-processed files for training.

Word embeddings are initialized with pre-trained structured skip-gram embeddings: [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing)

## Decoding

A pre-trained EDS model is available [here](https://drive.google.com/open?id=0BzlDJzogHw4fdGMtazJqb1RHWmc)

Decode with the parser (transition-based model):

    python rnn/parser.py --decode --decode_dev --use_hard_attention_arc_eager_decoder --predict_span_end_pointers --data_dir [data_dir] --train_dir [working_dir] --dev_name [filename] --size 256 --input_embedding_size 256 --output_embedding_size 128 --tag_embedding_size 32 --use_encoder_tags --input_drop_prob 0.3 --output_drop_prob 0.3 --checkpoint_file model.ckpt

where `data_dir` contains the pre-processed files for decoding (`filename.en`, `filename.ne`, `filename.pos`) as well as a `buckets` file, and `working_dir` contains the model (checkpoint) file.

Suggested `buckets`:
    24 77
    37 133
    52 201

### Post-processing

Restore lemmas and constants and convert to output graph formats.

    python mrs/linear_to_mrs.py [data_dir] [filename] [working_dir] output -arceagerbuffershift -unlex -withendspan 

