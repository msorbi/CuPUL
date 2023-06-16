import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset)
from transformers import (AdamW, RobertaTokenizer, get_linear_schedule_with_warmup)
from tqdm import tqdm
from seqeval.metrics import classification_report
# from utils import RoSTERUtils
# from model import RoSTERModel
# from loss import GCELoss
from data_processor import DataProcessor
from NERModel import NERModel
from risk import Risk

loss_type = {
    "CoNLL2003_KB": {
        "voter": "MPN",
        "curriculum": "Conf-MPU"
    }, 
    "Twitter": {
        "voter": "MPN",
        "curriculum": "Conf-MPU"
    }
}

class NERClassifier(object):
    def __init__(self, args):
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.output_dir = args.output_dir
        self.temp_dir = args.temp_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.dir_path = "../data"
        # self.dataset_name = "CoNLL2003_KB"
        self.dataset_name = args.dataset_name
        self.pretrained_model = args.pretrained_model
        # self.label_list = ["O", "PER", "LOC", "ORG", "MISC"]
        self.max_seq_length = args.max_seq_length
        self.train_batch_size = args.train_batch_size
        self.gradient_accumulation_steps = 1
        self.train_lr = args.train_lr 
        self.train_epochs = args.train_epochs
        self.curriculum_train_sub_epochs = args.curriculum_train_sub_epochs
        self.curriculum_train_lr = args.curriculum_train_lr
        self.curriculum_train_epochs = args.curriculum_train_epochs
        self.self_train_lr = args.self_train_lr
        self.self_train_epochs = args.self_train_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion

        self.self_train_update_interval = args.self_train_update_interval

        self.voter_num = args.num_models
        self.entity_threshold = args.entity_threshold
        self.ratio = args.ratio

        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False, cache_dir="/work/LAS/qli-lab/yuepei/bert_model")
        self.processor = DataProcessor(self.dir_path, self.dataset_name, self.tokenizer, args.seed)
        self.label_map, self.inv_label_map = self.processor.get_label_map()
        self.num_labels = len(self.inv_label_map) - 1

        self.risk = Risk(loss_type[args.dataset_name]["voter"], args.m, 0.5, self.num_labels, args.priors)


        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.mask_id = self.tokenizer.mask_token_id
        # print(self.mask_id)
        # exit()
        # setup model
        self.model = NERModel.from_pretrained(args.pretrained_model, num_labels=self.num_labels,
                                                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                                                 cache_dir="/work/LAS/qli-lab/yuepei/bert_model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.setup_dataset(args)

    def setup_dataset(self, args):
        # setup data
        if args.do_train:
            tensor_data = self.processor.get_tensor(data_name="train", max_seq_length=self.max_seq_length, drop_o_ratio=args.drop_other, drop_e_ratio=args.drop_entity)
            
            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.token_weight = torch.zeros_like(all_input_ids).float()
            #self.truth_labels = self.processor.get_train_truth_tensor(max_seq_length=self.max_seq_length)
            self.truth_labels = None

            self.gce_type_weight = torch.ones_like(all_input_ids).float()

            self.tensor_data = tensor_data
            self.train_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)

            print("***** Training stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.train_batch_size}")

        if args.do_eval:
            tensor_data = self.processor.get_tensor(data_name=self.args.eval_on, max_seq_length=self.max_seq_length)
            self.eval_tensor_data = tensor_data

            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.y_true = tensor_data["raw_labels"]

            eval_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)
            self.eval_data = eval_data
            eval_sampler = SequentialSampler(eval_data)
            self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            print("***** Evaluation stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.eval_batch_size}")


    # prepare model, optimizer and scheduler for training
    def prepare_train(self, lr, epochs=100000000000000000):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        model = self.model.to(self.device)
        # if self.multi_gpu:
        #     model = nn.DataParallel(model)
        num_train_steps = int(len(self.train_data)/self.train_batch_size/self.gradient_accumulation_steps) * epochs
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        warmup_steps = int(self.warmup_proportion*num_train_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
        model.train()
        return model, optimizer, scheduler

    def train(self, model_idx=0):
        if os.path.exists(os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt")) and False:
            print(f"\n\n******* Model {model_idx} predictions found; skip training *******\n\n")
            return
        else:
            print(f"\n\n******* Training model {model_idx} *******\n\n")
        model, optimizer, scheduler = self.prepare_train(lr=self.train_lr, epochs=self.train_epochs)
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        # self.noise_train_update_interval = 200

        batch_idx = 0
        rs = []
        loss_sum = []
        for epoch in range(self.train_epochs):
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
                max_len = attention_mask.sum(-1).max().item()
                # input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights = tuple(t[:, :max_len] for t in \
                #         (input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights))
                input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))

                type_logits = model(input_ids, attention_mask, valid_pos)
                labels = labels[valid_pos > 0]
                loss = self.risk.compute_risk(type_logits, labels)

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss_sum.append(loss.item())
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                # i += 1
                batch_idx += 1

            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Model Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)
                # self.predict_data()

        # np.save(os.path.join(self.temp_dir, f"{self.dataset_name}_voter_train_loss"), np.array(loss_sum))

        eval_sampler = SequentialSampler(self.train_data)
        eval_dataloader = DataLoader(self.train_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        # eval_sampler = SequentialSampler(self.eval_data)
        # eval_dataloader = DataLoader(self.eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        y_pred, pred_probs = self.eval(model, eval_dataloader)
        torch.save({"pred_probs": pred_probs}, os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt"))


    def log_split(self, scores, num):
        bins = [2**i for i in range(num)]
        bin_size = len(scores) / sum(bins)
        c_bins = [sum(bins[:i]) for i in range(1, num)]
        c_bins.insert(0, 0)
        sorted_scores = [s for s in enumerate(sorted(scores, key=lambda x: x, reverse=True))]
        res = [sorted_scores[int(b * bin_size)][-1] for b in c_bins]
        res.reverse()
        return res

    def token_hardness_score(self, file_dir):
        print(os.path.exists(os.path.join(self.temp_dir, f"hardness_score.np.npy")))
        if os.path.exists(os.path.join(self.temp_dir, f"hardness_score.np.npy")) and False:
            print(f"***************** Load hardness score *******************")
            self.tokens_score_matrix = np.load(os.path.join(self.temp_dir, f"hardness_score.np.npy"))
            self.tokens_score = [[i for i in line if i != 100] for line in self.tokens_score_matrix]
            return 
        
        print(f"***************** Calculate hardness score *******************")
        def KL_Div(P, Q):
            return P * (P / Q).log()
        pred_prob_list = []
        for f in os.listdir(file_dir):
            if f.startswith('y_pred'):
                pred = torch.load(os.path.join(file_dir, f))
                pred_prob_list.append(pred["pred_probs"])
        # print(pred_prob_list[0][0].shape)
        # print(len(pred_prob_list[0]))
        # loss_fct = nn.KLDivLoss(reduction="sum")
        if len(pred_prob_list) <= 1:
            self.tokens_score_matrix = np.ones(self.token_weight.shape)
            return
        self.tokens_score = []
        self.max_hardness_score = 0
        # random.shuffle(pred_prob_list)
        # print(pred_prob_list.shape)
        # zip list of list
        for sentences in zip(*pred_prob_list):
            scores = []
            for i in range(self.voter_num):
                for j in range(i+1, self.voter_num):
                    P, Q = sentences[i], sentences[j]
                    score = KL_Div(P, Q) + KL_Div(Q, P)
                    score = score.mean(dim=1)
                    scores.append(score.tolist())
            sent_score = np.array(scores).mean(axis=0)
            assert sent_score.size == len(sentences[0])
            if max(sent_score) > self.max_hardness_score:
                self.max_hardness_score = max(sent_score)
            self.tokens_score.append(list(sent_score))
        self.tokens_score_matrix = np.array([xi+[1000]*(self.max_seq_length-len(xi)) for xi in self.tokens_score], dtype=float)
        np.save(os.path.join(self.temp_dir, f"hardness_score.np"), self.tokens_score_matrix)
        print(f"Got max score {self.max_hardness_score}")

    # compute ensembled predictions
    def ensemble_pred(self, fild_dir):
        print(f"\n****** ensemble pred  ******\n")
        pred_prob_list = []
        for f in os.listdir(fild_dir):
            if f.startswith('y_pred'):
                pred = torch.load(os.path.join(fild_dir, f))
                pred_prob_list.append(pred["pred_probs"])
        # pred_prob_list = [pred_prob_list[0]]
        # print(len(pred_prob_list))
        # exit()
        ensemble_probs = []
        for i in range(len(pred_prob_list[0])):
            ensemble_prob_sent = []
            for j in range(len(pred_prob_list[0][i])):
                all_pred_probs = torch.cat([pred_prob_list[k][i][j].unsqueeze(0) for k in range(len(pred_prob_list))], dim=0)
                ensemble_prob_sent.append(torch.mean(all_pred_probs, dim=0, keepdim=True))
            ensemble_probs.append(torch.cat(ensemble_prob_sent, dim=0))
        # # print(ensemble_probs[0])
        # probs = []
        # # print(self.num_labels)
        # for prob, label in zip(ensemble_probs, self.tensor_data["all_labels"]):
        #     label = label[label>=0]
        #     label = F.one_hot(label, num_classes = self.num_labels)
        #     # print(prob, label)
        #     # probs.append((0.2*prob + 0.8*label))
        #     # probs.append((0.0*prob + 1.0*label))
        #     probs.append((1.0*prob + 0.0*label))
        #     # probs.append((0.5*prob + 0.5*label))
        #     # probs.append((0.8*prob + 0.2*label))
        # # torch.concat(probs)
        # # # print()
        # ensemble_probs = probs
        print(ensemble_probs[0])
        # # exit()
        ensemble_preds = []
        for pred_prob in ensemble_probs:
            preds = pred_prob.argmax(dim=-1)
            ensemble_preds.append([self.inv_label_map[pred.item()] for pred in preds])
        all_valid_pos = self.tensor_data["all_valid_pos"]
        # all_valid_pos = self.eval_tensor_data["all_valid_pos"]
        ensemble_label = -100 * torch.ones(all_valid_pos.size(0), all_valid_pos.size(1), self.num_labels)
        ensemble_label[all_valid_pos > 0] = torch.cat(ensemble_probs, dim=0)
        self.ensemble_label = ensemble_label

    def update_token_weight(self, threshold):        
        self.token_weight = torch.zeros_like(self.token_weight).float()
        self.token_weight[self.tokens_score_matrix < threshold] = 1
        # self.token_weight[np.logical_and(self.lower_bound < self.tokens_score_matrix, self.tokens_score_matrix < threshold)] = 1

    def curriculum_train(self):
        print("\n\n******* Training curriculum model *******\n\n")
        #self.load_model(f"{self.output_dir}/cl_model_0")
        model, optimizer, scheduler = self.prepare_train(lr=self.curriculum_train_lr, epochs=self.curriculum_train_epochs*self.curriculum_train_sub_epochs)

        all_idx = self.tensor_data["all_idx"]
        all_input_ids = self.tensor_data["all_input_ids"]
        all_attention_mask = self.tensor_data["all_attention_mask"]
        all_valid_pos = self.tensor_data["all_valid_pos"]
        all_labels = self.tensor_data["all_labels"]
        # all_soft_labels = self.tensor_data["all_labels"]
        all_soft_labels = self.ensemble_label
        self.token_weight = torch.ones_like(all_input_ids).float()
        # self.fusion_labels = self.ensemble_label.to(self.device)

        curriculum_train_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels, all_soft_labels)
        train_sampler = RandomSampler(curriculum_train_data)
        train_dataloader = DataLoader(curriculum_train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        # thresholds = [(self.max_hardness_score/self.curriculum_train_epochs) * (i+1) for i in range(self.curriculum_train_epochs)]
        hardness_scores = self.tokens_score_matrix
        scores = hardness_scores[hardness_scores!=1000]
        thresholds = self.log_split(scores, self.curriculum_train_epochs)
        # self.lower_bound = thresholds[0] / 4
        self.lower_bound = 0

        # batch_idx = 0
        loss_sum = []
        for epoch in range(self.curriculum_train_epochs):
            batch_idx = 0
            model.train()
            print(f"Threshold: {thresholds[epoch]}")
            self.update_token_weight(thresholds[epoch])
            # self.update_token_weight(10000)
            for _ in range(self.curriculum_train_sub_epochs):
                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):

                    idx, input_ids, attention_mask, valid_pos, labels, soft_labels = tuple(t.to(self.device) for t in batch)
                    # print(labels.shape)

                    # if batch_idx % 200==0:
                    #    self.update_token_weight(thresholds[batch_idx//200])
                    batch_idx += 1
                    token_weights = self.token_weight[idx].to(self.device)
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask, valid_pos, labels, soft_labels, token_weights = tuple(t[:, :max_len] for t in \
                            (input_ids, attention_mask, valid_pos, labels, soft_labels, token_weights))

                    type_logits = model(input_ids, attention_mask, valid_pos)
                    # print(token_weights.shape)
                    # # print(labels.shape)
                    soft_labels = soft_labels[valid_pos > 0]
                    labels = labels[valid_pos > 0]
                    # # print(labels.shape)
                    # # exit()
                    token_weights = token_weights[valid_pos > 0]
                    curriculum_pos = token_weights==1
                    # # print(token_weights==1)
                    # # print(token_weights==0)
                    # # exit()
                    soft_labels = soft_labels[curriculum_pos]
                    type_logits = type_logits[curriculum_pos]
                    labels = labels[curriculum_pos]
                    if len(labels) == 0:
                        continue
                    # # print(labels.shape)
                    # # print(type_logits.shape)
                    # # exit()
                    # labels = labels[token_weights!=0]
                    # type_logits = type_logits[token_weights!=0]
                    # labels[token_weights==0] = -100
                    # print(token_weights)
                    # exit()
                    # print(labels)
                    # print(token_weights)
                    # exit()

                    #if batch_idx < (len(self.train_data)/self.train_batch_size/self.gradient_accumulation_steps) \
                    #                * self.curriculum_train_epochs * self.curriculum_train_sub_epochs * self.ratio:
                    #    self.curriculum_risk_type = "Conf-MPU-CE"
                    #else:
                    #    self.curriculum_risk_type = "Conf-MPU"
                    # self.curriculum_risk_type = "Conf-MPU-CE"
                    # self.curriculum_risk_type = "MPN"
                    # priors = [0.0314966102568, 0.0376880632424, 0.0354240324761, 0.015502139428]
                    # # # priors = [0.0314966102568, , 0.0354240324761, 0.015502139428]
                    # risk = Risk("MPN", 25, 0.5, self.num_labels, priors)
                    # loss = self.risk.compute_risk(type_logits, labels)
                    # loss = self.risk.compute_risk(type_logits, labels, risk_type="MPN")
                    # loss = self.risk.compute_risk(type_logits, labels, risk_type="MPU")
                    # loss = self.risk.compute_risk(type_logits, labels, risk_type="MPN-CE2")
                    # print(type_logits.shape, labels.shape)
                    # loss = self.risk.compute_risk(type_logits, labels, risk_type="Conf-MPU", probs=1-soft_labels[:,0])
                    # loss = self.risk.compute_risk(type_logits, labels, risk_type="Conf-MPU-CE", probs=1-soft_labels[:,0])
                    loss = self.risk.compute_risk(type_logits, labels, risk_type=self.curriculum_risk_type, probs=1-soft_labels[:,0])
                    # loss = self.risk.compute_risk2(type_logits, soft_labels, labels)
                    if loss.item() == 0:
                        continue
                    # exit()
                    loss_sum.append(loss.item())
                    # mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(self.device) for i in range(self.num_labels)]
                    # type_logits_set = [torch.index_select(type_logits, dim=0, index=mask[i]) for i in range(self.num_labels)]
                    # soft_labels_set = [torch.index_select(soft_labels, dim=0, index=mask[i]) for i in range(self.num_labels)]
                    # loss_fct = nn.KLDivLoss(reduction='sum')
                    # ft_loss = []
                    # for i in range(self.num_labels):
                    #     # print(i)
                    #     # print(type_logits_set[i].shape)
                    #     # print(soft_labels_set[i].shape)
                    #     loss = loss_fct(type_logits_set[i], soft_labels_set[i])
                    #     print(loss)
                    #     loss = loss / type_logits_set[i].size(0)
                    #     ft_loss.append(loss)
                    # print(ft_loss)
                    # loss = sum(loss)

                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if (step+1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    # if batch_idx > 500:
                    #    break

            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)

            os.makedirs(f"{self.output_dir}/cl_model_{epoch}", exist_ok=True)
            self.save_model(model, f"cl_model_{epoch}.pt", f"{self.output_dir}/cl_model_{epoch}")
        np.save(os.path.join(self.temp_dir, f"{self.dataset_name}_curr_train_loss"), np.array(loss_sum))

    def eval(self, model, eval_dataloader):
        model = model.to(self.device)
        model.eval()
        y_pred = []
        pred_probs = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            _, input_ids, attention_mask, valid_pos, _ = tuple(t.to(self.device) for t in batch)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos))

            with torch.no_grad():
                logits = model(input_ids, attention_mask, valid_pos)
                # entity_prob = torch.sigmoid(bin_logits)
                # type_prob = F.softmax(logits, dim=-1) * entity_prob
                # non_type_prob = 1 - entity_prob
                # type_prob = torch.cat([non_type_prob, type_prob], dim=-1)
                
                # preds = torch.argmax(type_prob, dim=-1)
                probs = logits
                preds = torch.argmax(logits, dim=-1)
                preds = preds.cpu().numpy()
                pred_prob = probs.cpu()

            num_valid_tokens = valid_pos.sum(dim=-1)
            i = 0
            for j in range(len(num_valid_tokens)):
                pred_probs.append(pred_prob[i:i+num_valid_tokens[j]])
                y_pred.append([self.inv_label_map[pred] for pred in preds[i:i+num_valid_tokens[j]]])
                i += num_valid_tokens[j]

        return y_pred, pred_probs

    # print out ner performance given ground truth and model prediction
    def performance_report(self, y_true, y_pred):
        for i in range(len(y_true)):
            if len(y_true[i]) > len(y_pred[i]):
                print(f"Warning: Sequence {i} is truncated for eval! ({len(y_pred[i])}/{len(y_true[i])})")
                y_pred[i] = y_pred[i] + ['O'] * (len(y_true[i])-len(y_pred[i]))
        # print(y_true[0:2])
        # print(y_pred[0:2])
        report = classification_report(y_true, y_pred, digits=4)
        print(report)
        return report 

    # save model, tokenizer, and configs to directory
    def save_model(self, model, model_name, save_dir):
        print(f"Saving {model_name} ...")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        model_to_save.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # torch.save(model_to_save.state_dict(), os.path.join(save_dir, model_name))
        # self.tokenizer.save_pretrained(save_dir)
        # model_config = {"max_seq_length": self.max_seq_length, 
        #                 "num_labels": self.num_labels, 
        #                 "label_map": self.label_map}
        # json.dump(model_config, open(os.path.join(save_dir, "model_config.json"), "w"))

    def load_model(self, model_dir):
        self.model = NERModel.from_pretrained(model_dir)
