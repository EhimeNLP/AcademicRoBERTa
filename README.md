# AcademicRoBERTa

We pretrained a RoBERTa-based Japanese masked language model on paper abstracts from the academic database CiNii Articles.  
A Japanese Masked Language Model for Academic Domain(https://aclanthology.org/2022.sdp-1.16/)
```
@inproceedings{yamauchi-etal-2022-japanese,
    title = "A {J}apanese Masked Language Model for Academic Domain",
    author = "Yamauchi, Hiroki  and
      Kajiwara, Tomoyuki  and
      Katsurai, Marie  and
      Ohmukai, Ikki  and
      Ninomiya, Takashi",
    booktitle = "Proceedings of the Third Workshop on Scholarly Document Processing",
    year = "2022",
    url = "https://aclanthology.org/2022.sdp-1.16",
    pages = "152--157",
}

```

## Download
They include a pretrained roberta model (700000.pt), a sentencepiece model (sp.model) and a dictionary (dict.txt), Code for applying sentencepiece (apply_sp.py) .
```
wget aiweb.cs.ehime-u.ac.jp/~yamauchi/academic_model/Academic_RoBERTa_base.tar.gz
```
## Requirements
Python >= 3.8 <br>
fairseq <br>
sentencepiece <br>
tensorboardX (optional) <br>

## Preprocess
The data format assumes a tab delimiter between text and label.

```
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/train.src-tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $VALID_SRC $DATASET_DIR/valid.src-tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TEST_SRC $DATASET_DIR/test.src-tgt --bpe_model $SENTENCEPIECE_MODEL
```
```
fairseq-preprocess \
    --source-lang "src" \
    --target-lang "tgt" \
    --trainpref "${DATASET_DIR}/train.src-tgt" \
    --validpref "${DATASET_DIR}/valid.src-tgt" \
    --testpref "${DATASET_DIR}/test.src-tgt" \
    --destdir "data-bin/" \
    --workers 60 \
    --srcdict ${DICT} \
    --tgtdict ${DICT}
```
## Finetune
This work was supported by papers in Japanese.
```
fairseq-train data-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --max-epoch 999 \
    --patience 10 \
    --no-epoch-checkpoints --seed 88 --log-format simple --log-interval $LOG_INTERVAL --save-interval-updates $SAVE_INTERVAL \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters
```
