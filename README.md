# AcademicRoBERTa

We pretrained a RoBERTa-based Japanese masked language model on paper abstracts from the academic database CiNii Articles.

## Download
```

```
## Requirements
Python >= 3.8
fairseq
sentencepiece
tensorboardX (optional)

## Preprocess
```
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/train.src-tgt.src --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TRAIN_TGT $DATASET_DIR/train.src-tgt.tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/valid.src-tgt.src --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/valid.src-tgt.tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/test.src-tgt.src --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/test.src-tgt.tgt --bpe_model $SENTENCEPIECE_MODEL
```
