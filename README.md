# AcademicRoBERTa

We pretrained a RoBERTa-based Japanese masked language model on paper abstracts from the academic database CiNii Articles.  
[A Japanese Masked Language Model for Academic Domain](https://aclanthology.org/2022.sdp-1.16/)
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
They include a pretrained roberta model (700000_model.pt), a sentencepiece model (sp.model) , a dictionary (dict.txt) and code for applying sentencepiece (apply-sp.py) .
```
wget http://aiweb.cs.ehime-u.ac.jp/~yamauchi/academic_model/Academic_RoBERTa_base.tar.gz
```
## Requirements
Python >= 3.8 <br>
[fairseq](https://github.com/facebookresearch/fairseq) == 0.12.2 (In working order)<br>
[sentencepiece](https://github.com/google/sentencepiece) <br>
tensorboardX (optional) <br>

## Preprocess
We applied SentencePiece for subword segmentation. <br>
Prepare datasets ($TRAIN_SRC, ...), which format assumes a tab delimiter between text and label.

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
This work was supported by papers in Japanese. <br>
The procedure for sentence classification using AcademicRoBERTa is as follows.
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
## Reference
- Hiroki Yamauchi, Tomoyuki Kajiwara, Marie Katsurai, Ikki Ohmukai, Takashi Ninomiya. [A Japanese Masked Language Model for Academic Domain](https://aclanthology.org/2022.sdp-1.16). In Proceedings of the Third Workshop on Scholarly Document Processing (SDP), pp.152-157, 2022.
- 山内洋輝, 梶原智之, 桂井麻里衣, 大向一輝, 二宮崇. 学術ドメインに特化した日本語事前訓練モデルの構築. 言語処理学会第29回年次大会, pp.2842-2846, 2023.
