from flair.data import Sentence, Corpus
from flair.datasets import ColumnCorpus, DataLoader
from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.trainers import ModelTrainer
from torch.optim import AdamW
from torch.utils.data import RandomSampler
from typing import List, Tuple
import torch.nn.functional as F
import json
import os
import codecs 
import torch
import tqdm
import gc
import flair

class FlairTokenizerClass:
    def __init__(self, vocab=None):
        self.sep_token = ''
        self.vocab = vocab or {self.sep_token:0}
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.mask_token_id = 0
        self.corpus = None
        self.dict_corpus = {}
    
    def make_corpus(self, args, field_separator=' '):
        input_dir = os.path.join('..', 'data', args.dataset_name)
        train_all=[]
        with open(os.path.join(input_dir, 'train.txt'), 'r') as f:
            train = f.read()
            train=train.split("\n")
            train=[x.split("\t",4) for x in train]
            caja=[]
            for x in train:
                if x==['']:
                    train_all.append(caja)
                    caja=[]

                else:
                    y = x[0].split(field_separator)
                    caja.append([y[0], y[1].split('-',1)[-1]])
            train_all.append(caja)
            
        test_all=[]

        with open(os.path.join(input_dir, 'test.txt'), 'r') as f:
            test = f.read()
            test=test.split("\n")
            test=[x.split("\t",4) for x in test]
            caja=[]
            for x in test:
                if x==['']:
                    test_all.append(caja)
                    caja=[]

                else:
                    y = x[0].split(field_separator)
                    caja.append([y[0], y[1].split('-',1)[-1]])
            test_all.append(caja)
            

        val_all=[]

        with open(os.path.join(input_dir, 'valid.txt'), 'r') as f:
            val = f.read()
            val=val.split("\n")
            val=[x.split("\t",4) for x in val]
            caja=[]
            for x in val:
                if x==['']:
                    val_all.append(caja)
                    caja=[]

                else:
                    y = x[0].split(field_separator)
                    caja.append([y[0], y[1].split('-',1)[-1]])
            val_all.append(caja)
            

        train_all=[[y if y[0]!="" else ["-"]+y[1:] for y in x] for x in train_all]
        test_all=[[y if y[0]!="" else ["-"]+y[1:] for y in x] for x in test_all]
        val_all=[[y if y[0]!="" else ["-"]+y[1:] for y in x] for x in val_all]


        txt_files=train_all+test_all+val_all
        a = txt_files


        train_sents=a[:len(train_all)]
        test_sents=a[len(train_all):len(train_all)+len(test_all)]
        val_sents=a[len(train_all)+len(test_all):]


        os.makedirs(args.work_dir, exist_ok=True)
        with codecs.open(os.path.join(args.work_dir, 'train_elec.txt'), 'w', encoding="utf-8") as f:
            f.write("\n\n".join("\n".join(["\t".join(z) for z in y]) for y in train_sents))

        with codecs.open(os.path.join(args.work_dir, 'val_elec.txt'), 'w', encoding="utf-8") as f:
            f.write("\n\n".join("\n".join(["\t".join(z) for z in y]) for y in val_sents))


        with codecs.open(os.path.join(args.work_dir, 'test_elec.txt'), 'w', encoding="utf-8") as f:
            f.write("\n\n".join("\n".join(["\t".join(z) for z in y]) for y in test_sents))

        # define columns to construct our model on flair data
        columns = {0 : 'text', 1 : 'ner'} #to train on PERS label

        # initializing the corpus: you must load the training sets in your enviromment
        self.corpus: Corpus = ColumnCorpus(args.work_dir, columns,
                                    train_file = 'train_elec.txt',
                                    test_file = 'test_elec.txt',
                                    dev_file = 'val_elec.txt')

        for split in ['train', 'test', 'val']:
            os.unlink(os.path.join(args.work_dir, f'{split}_elec.txt'))
        
        corpus_len = 0
        self.dict_corpus = {}
        for split in [self.corpus.train, self.corpus.dev, self.corpus.test]:
            for i,s in enumerate(split.sentences, start=corpus_len):
                text = ' '.join(t.text for t in s.tokens if t)
                if text in self.dict_corpus:
                    continue
                self.dict_corpus[text] = s
                for t in s.tokens:
                    t = t.text
                    if t not in self.vocab:
                        l = len(self.vocab)
                        self.inv_vocab[l] = t
                        self.vocab[t] = l
            corpus_len += len(split.sentences)

        # make tag dictionary from the corpus
        tag_type = 'ner'
        self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=tag_type)

        return self.corpus

    def get_vocab(self):
        return self.vocab
    
    def get_inv_vocab(self):
        return self.inv_vocab

    def get_token_id(self, token:str):
        id_ = self.vocab.get(token)
        if id_ is None:
            id_ = len(self.vocab)
            self.vocab[token] = id_
            self.inv_vocab[id_] = token
        return id_

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.vocab, f)
    
    def convert_tokens_to_ids(self, tokens):
        return [self.get_token_id(t) for t in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i.item(), '') for i in ids]
    
    def encode_plus(self, sentence, max_length, *args, **kwargs):
        sentence = [t.text for t in self.dict_corpus[sentence]]
        l = len(sentence)
        pad = max_length - l
        sentence = sentence + [self.sep_token]*pad
        input_ids = self.convert_tokens_to_ids(sentence)
        mask = [1]*l + [0]*pad
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.int64),
            'attention_mask': torch.tensor([mask], dtype=torch.int64)
        }
    
    def __call__(self, text, max_length, *args, **kwargs):
        self.encode_plus(text, max_length, *args, **kwargs)

flairTokenizer = FlairTokenizerClass()

def FlairTokenizer(args=None, *vargs, **kwargs):
    if args is not None and flairTokenizer.corpus is None:
        flairTokenizer.make_corpus(args)
    return flairTokenizer



class FlairModel(SequenceTagger):
    def __init__(self, args, risk=None, **kwargs):
        self.init(args, **kwargs)
        flair_forward_embedding = FlairEmbeddings(
            os.path.join(args.embeddings_dir, 'multi_19M_forward.pt')
        )
        flair_backward_embedding = FlairEmbeddings(
            os.path.join(args.embeddings_dir, 'multi_19M_backward.pt')
        )
        # now create the StackedEmbedding object that combines all embeddings
        stacked_embeddings = StackedEmbeddings(
            embeddings=[flair_forward_embedding, flair_backward_embedding]
        )
        super().__init__(
            hidden_size=256,
            embeddings=stacked_embeddings,
            tag_dictionary=self.tokenizer.tag_dictionary,
            tag_type=self.tag_type,
            use_crf=True,
            **kwargs
        )
    
    def init(self, args, risk=None, **kwargs):
        self.tokenizer = FlairTokenizer()
        self.curriculum = False
        self.training = False
        self.device='cpu'
        self.args = args
        self.loss_sentence_idx = 0

        self.corpus = self.tokenizer.corpus

        # tag to predict
        self.tag_type = 'ner'
        self.num_labels = len(self.tokenizer.tag_dictionary)-2

    def mlm_pred(self, input_ids, attention_mask=None, valid_pos=None):
        return self.forward(input_ids, attention_mask, valid_pos)
    
    def forward(self, input_ids, attention_mask=None, valid_pos=None):
        sentences = [
            self.tokenizer.dict_corpus[' '.join(t for t in self.tokenizer.convert_ids_to_tokens(ids) if t)]
            for ids in input_ids
        ]
        num_labels = len(self.tag_dictionary)
        # flair forward pass
        features = super().forward(sentences)  # list of Sentence-level outputs
        for x in sentences:
            x.clear_embeddings()
        features = features.view(-1, num_labels)
        valid_pos = valid_pos.reshape(-1)
        features = features[valid_pos>0,:-2]
        features = F.softmax(features, dim=-1)
        return features

    def __call__(self, sentences, attention_mask=None, valid_pos=None):
        return self.forward(sentences, attention_mask, valid_pos)
    
    def softmax(self, features):
        features[...,-2:] = torch.tensor(0).log()
        features = F.softmax(features, dim=-1)
        return features

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save(os.path.join(save_dir, 'FlairModel.pt'))
    
    @staticmethod
    def from_pretrained(save_dir, args):
        model = FlairModel(args)
        model.load(os.path.join(save_dir, 'FlairModel.pt'))
        return model

if __name__ == "__main__":
    args = parse_args()
    main(args)
