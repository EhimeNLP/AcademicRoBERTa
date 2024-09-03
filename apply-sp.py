import sentencepiece as spm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('load_file', type=str)
parser.add_argument('save_file', type=str)
parser.add_argument('-bpe_model', type=str, default='sp.model', help='sentencepiace_model')
opt = parser.parse_args()

def apply_sp():
    # モデルの作成
    sp = spm.SentencePieceProcessor()
    sp.Load(opt.bpe_model)

    #load file
    with open(opt.load_file, mode='r') as f:
        lines = f.readlines()

    #apply sentencepiece
    tokenized_lines = []
    label_lines = []
    for l in lines:
        l_list = l.split("\t")
        tokenized_sentence = sp.EncodeAsPieces(l_list[0])
        tokenized_sentence = ' '.join(tokenized_sentence)
        tokenized_lines.append(tokenized_sentence)
        tokenized_target = sp.EncodeAsPieces(l_list[1])
        tokenized_target = " ".join(tokenized_target)
        label_lines.append(tokenized_target)

    del lines

    with open(opt.save_file + '.src' , mode='w') as f:
        f.write('\n'.join(tokenized_lines))
    with open(opt.save_file + '.tgt' , mode='w') as f:
        f.write('\n'.join(label_lines))

def main():
    apply_sp()

if __name__ == "__main__":
    main()
