import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_risk = []

class Risk(object):
    def __init__(self, risk_type, m, eta, num_class, priors=None):
        self.risk_type = risk_type
        self.m = m
        self.eta = eta
        self.num_class = num_class
        self.priors = priors
        self.class_weights = [1 - sum(self.priors)]
        self.class_weights.extend(self.priors)

    def compute_risk(self, logits, labels, probs=None, risk_type=None):
        risk = 0
        if risk_type:
            self.risk_type = risk_type

        if self.risk_type == 'MPN':
            # print(labels.shape)
            # print(logits.shape)
            mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            logits_set = [torch.index_select(logits, dim=0, index=mask[i]) for i in range(self.num_class)]
            # mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]
            # logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
            #               for i in range(self.num_class)]
            # # print(logits_set)

            neg_prior = 1 - sum(self.priors)

            # risk1 = P(+)_risk
            risk1_list = [self.MAE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class)]
            risk1 = sum([self.priors[i - 1] * risk1_list[i - 1]
                         for i in range(1, self.num_class)])  # index of "O" is 1, and remove [CLS] and [SEP]

            # risk2 = N(-)_risk
            risk2 = neg_prior * self.MAE(logits_set[0], np.eye(self.num_class)[0])

            # batch_risk.append([i.item() for i in risk1_list] + [risk2.item()])
            risk = risk1 * self.m + risk2

        if self.risk_type == 'MPN-CE':
            # print(labels.shape)
            # print(logits.shape)
            mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            logits_set = [torch.index_select(logits, dim=0, index=mask[i]) for i in range(self.num_class)]
            # mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]
            # logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
            #               for i in range(self.num_class)]
            # # print(logits_set)

            neg_prior = 1 - sum(self.priors)

            # risk1 = P(+)_risk
            risk1_list = [self.CE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class)]
            risk1 = sum([self.priors[i - 1] * risk1_list[i - 1]
                         for i in range(1, self.num_class)])  # index of "O" is 1, and remove [CLS] and [SEP]

            # risk2 = N(-)_risk
            risk2 = neg_prior * self.CE(logits_set[0], np.eye(self.num_class)[0])

            # batch_risk.append([i.item() for i in risk1_list] + [risk2.item()])
            # print(self.m)
            risk = risk1 * self.m + risk2



        elif self.risk_type == 'MPU':
            # print(labels.shape)
            # print(logits.shape)
            mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            logits_set = [torch.index_select(logits, dim=0, index=mask[i]) for i in range(self.num_class)]

            # risk1 = P(+)_risk
            risk1_list = [self.MAE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class)]
            risk1 = sum([self.priors[i - 1] * risk1_list[i - 1] for i in range(1, self.num_class)])

            # risk2 = U(-)_risk
            risk2 = (self.MAE(logits_set[0], np.eye(self.num_class)[0]) -
                     sum([self.priors[i - 1] * self.MAE(logits_set[i], np.eye(self.num_class)[0])
                          for i in range(1, self.num_class)]))

            # batch_risk.append([i.item() for i in risk1_list] + [risk2.item()])
            risk = risk1 * self.m + risk2
            if risk2 < 0:
                risk = - risk2

        elif self.risk_type == 'MPU-CE':
            # print(labels.shape)
            # print(logits.shape)
            mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            logits_set = [torch.index_select(logits, dim=0, index=mask[i]) for i in range(self.num_class)]
            # print(logits_set)

            # risk1 = P(+)_risk
            risk1_list = [self.CE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class)]
            risk1 = sum([self.priors[i - 1] * risk1_list[i - 1] for i in range(1, self.num_class)])

            # risk2 = U(-)_risk
            risk2 = (self.CE(logits_set[0], np.eye(self.num_class)[0]) -
                     sum([self.priors[i - 1] * self.CE(logits_set[i], np.eye(self.num_class)[0])
                          for i in range(1, self.num_class)]))
            
            risk = risk1 * self.m + risk2
            if risk2 < 0:
                risk = - risk2

        elif self.risk_type == 'Conf-MPU':
            # print(probs)
            # l_mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            # p_mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            # exit()
            l_mask = []
            p_mask = []
            # print(probs)
            labels = labels.unsqueeze(0)
            logits = logits.unsqueeze(0)
            # probs = (probs**2).unsqueeze(0)
            probs = probs.unsqueeze(0)
            for i in range(self.num_class):
                mask1, mask2 = self.mask_of_label_prob(self.eta, labels, probs, i)
                l_mask.append(mask1)
                p_mask.append(mask2)

            mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]
            logits_set = [logits.masked_select(torch.from_numpy(l_mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                          for i in range(self.num_class)]

            logits_set2 = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                           for i in range(self.num_class)]

            prob_set = [sum(ele, []) for ele in p_mask]
            prob_set = [torch.tensor(ele).to(device) for ele in prob_set]

            # U'(-)
            risk1 = self.MAE(logits_set[0], np.eye(self.num_class)[0])
            # P'(-)
            risk2 = sum([self.priors[i - 1] * self.conf_MAE(logits_set[i], np.eye(self.num_class)[0], prob_set[i])
                         for i in range(1, self.num_class)])
            # P(-)
            risk3 = sum([self.priors[i - 1] * self.MAE(logits_set2[i], np.eye(self.num_class)[0])
                         for i in range(1, self.num_class)])
            # P(+)
            risk4 = sum([self.priors[i - 1] * self.MAE(logits_set2[i], np.eye(self.num_class)[i])
                         for i in range(1, self.num_class)])

            negative_risk = risk1
            positive_risk = risk2 - risk3 + risk4

            # print(negative_risk, positive_risk)
            risk = positive_risk * self.m + negative_risk
            
            if positive_risk < 0:
                risk = negative_risk

        elif self.risk_type == 'Conf-MPU-CE':
            # print(probs)
            # l_mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            # p_mask = [(labels == i).nonzero(as_tuple=False).squeeze().to(device) for i in range(self.num_class)]
            # exit()
            l_mask = []
            p_mask = []
            # print(probs)
            labels = labels.unsqueeze(0)
            logits = logits.unsqueeze(0)
            # probs = (probs**2).unsqueeze(0)
            probs = probs.unsqueeze(0)
            for i in range(self.num_class):
                mask1, mask2 = self.mask_of_label_prob(self.eta, labels, probs, i)
                l_mask.append(mask1)
                p_mask.append(mask2)

            mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]

            logits_set = [logits.masked_select(torch.from_numpy(l_mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                          for i in range(self.num_class)]

            logits_set2 = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                           for i in range(self.num_class)]

            prob_set = [sum(ele, []) for ele in p_mask]
            prob_set = [torch.tensor(ele).to(device) for ele in prob_set]

            # U'(-)
            risk1 = self.CE(logits_set[0], np.eye(self.num_class)[0])
            # P'(-)
            risk2 = sum([self.priors[i - 1] * self.conf_CE(logits_set[i], np.eye(self.num_class)[0], prob_set[i])
                         for i in range(1, self.num_class)])
            # P(-)
            risk3 = sum([self.priors[i - 1] * self.CE(logits_set2[i], np.eye(self.num_class)[0])
                         for i in range(1, self.num_class)])
            # P(+)
            risk4 = sum([self.priors[i - 1] * self.CE(logits_set2[i], np.eye(self.num_class)[i])
                         for i in range(1, self.num_class)])

            negative_risk = risk1
            positive_risk = risk2 - risk3 + risk4

            # print(negative_risk, positive_risk)
            risk = positive_risk * self.m + negative_risk
            if positive_risk < 0:
                risk = negative_risk

        return risk

    @staticmethod
    def conf_MAE(yPred, yTrue, prob):
        # print(prob)
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        # prob = prob.float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean((temp * 1 / prob).sum(dim=1) / yTrue.shape[0])
        return loss

    @staticmethod
    def conf_CE(yPred, yTrue, prob):
        # print(prob)
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        # prob = prob.float().cuda()
        # temp = torch.FloatTensor.abs(y - yPred)
        temp = -torch.log(yPred)
        loss = torch.mean((y * (temp / prob)).sum(dim=1))
        return loss

    @staticmethod
    def MAE(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean(temp.sum(dim=1) / yTrue.shape[0])
        return loss

    @staticmethod
    def CE(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        temp = -torch.log(yPred)
        loss = torch.mean((y * temp).sum(dim=1))
        return loss

    def mask_of_label(self, labels, class_elem):
        masks = []
        for s in labels:
            s_mask = []
            for w in s:
                if w == class_elem:
                    s_mask.append([1] * self.num_class)  # [1,1,1,1,1] if class_num = 5
                else:
                    s_mask.append([0] * self.num_class)
            masks.append(s_mask)
        return np.array(masks)

    def mask_of_label_prob(self, eta, labels, probs, class_elem):
        l_masks = []
        p_masks = []
        for s_l, s_p in zip(labels, probs):
            s_mask_l = []
            s_mask_p = []
            for w_l, w_p in zip(s_l, s_p):
                if w_l == class_elem and 0 < w_l < self.num_class and w_p > eta:
                    s_mask_l.append([1] * self.num_class)  # [1,1,1,1,1] if class_num = 5
                    s_mask_p.append([w_p])
                elif w_l == class_elem and w_l == 0 and w_p <= eta:
                    s_mask_l.append([1] * self.num_class)
                    # s_mask_p.append([w_p])
                else:
                    s_mask_l.append([0] * self.num_class)
                    # s_mask_p.append([w_p])

            l_masks.append(s_mask_l)
            p_masks.append(s_mask_p)
        return np.array(l_masks), p_masks

    def risk_on_val(self, logits, labels):
        mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]

        logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                      for i in range(self.num_class)]

        risk = sum([self.MAE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class - 2)])

        return risk
