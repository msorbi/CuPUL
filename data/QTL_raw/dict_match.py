import sys
sys.path.append("..")
import numpy as np
from collections import defaultdict
from utils.data_process import DataProcess
import copy
from tqdm import tqdm
import multiprocessing as mp


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    if s in [ordinal(n) for n in range(1,999)]:
        return True
    if s in ["first", "second", "third", "forth", "fifth", "eigth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth" "eighteenth", "nineteenth", "twentieth"]:
        return True
 
    return False


def convert(words_):
    words = []
    for w in words_:
        if is_number(w):
            words.append("DIGIT")
        elif w in ["percent", "%", "year", "years", "month", "Mos", "mos", "mo", "months", 
                    "week", "weeks", "wks", "wk", "day", "days", "d", "hour", "hours", "hr", "hrs",
                    "minute", "minutes", "mi", "kg"]:
            words.append("UNIT")
        else:
            words.append(w)
    return words


def lookup_in_Dic(tag2idx, dicFile, sentences, tag, windowSize):
    tagIdx = tag2idx[tag]
    dic = []
    labeled_word = set()
    count = 0
    mistake = defaultdict(int)
    true = defaultdict(int)
    with open(dicFile, "r", encoding='utf-8') as fw:
        for line in fw:
            line = line.strip()
            if len(line) > 0:
                dic.append(line)
    if len(dic) == 0:
        return sentences, 0, 0
    for i, sentence in enumerate(tqdm(sentences, desc= "looking up: ")):
        wordList = [word for word, label, dicFlags in sentence]
        wordList = convert(wordList)
        trueLabelList = [label for word, label, dicFlags in sentence]
        isFlag = np.zeros(len(trueLabelList))
        j = 0
        while j < len(wordList):
            Len = min(windowSize, len(wordList) - j)
            k = Len
            while k >= 1:
                words = wordList[j:j + k]
                words_ = " ".join([w for w in words])

                # 3.14 -> DIGIT, year -> UNIT
                # words_ = convert(words_)
                if words_ in dic:
                    isFlag[j:j + k] = 1
                    j = j + k
                    break
                k -= 1
            j += 1

        for m, flag in enumerate(isFlag):
            if flag == 1:
                count += 1
                labeled_word.add(sentence[m][0])
                sentence[m][2][tagIdx] = 1
                # print(sentence)

    return sentences, len(labeled_word), count


def readFile(trainFile, classNum):
    with open(trainFile, "r", encoding='utf-8') as fw:
        sentences = []
        sentence = []
        for line in fw:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            else:
                splits = line.split(' ')
                sentence.append([splits[0].strip(), splits[1].strip(), np.zeros(classNum)])

        if len(sentence) > 0:
            sentences.append(sentence)

        return sentences


def writeFile(fileName, sentences):
    with open(fileName, 'w', encoding='utf-8') as fw:
        for sentence in sentences:
            for word, truth, labels in sentence:
                if labels.sum() == 1:
                    label = labels.argmax() + 1
                    fw.write(word + " " + truth + " " + str(label) + "\n")
                else:
                    label = 0
                    fw.write(word + " " + truth + " " + str(label) + "\n")
            fw.write("\n")


def getLabelsAndPreds(sentences):
    labels = []
    preds = []
    for sent in sentences:
        for word, label, pred in sent:
            if len(label.split('-')) > 1:
                label = label.split('-')[-1]
            else:
                label = label
            labels.append(label)
            preds.append(pred)
    return labels, preds


def generate_single_from_all(tag, dataset, tag2idx):
    with open("../data/" + dataset + "/train.ALL.txt", 'r', encoding='utf-8') as ALL, \
            open("../data/" + dataset + "/train." + tag + ".txt", 'w', encoding='utf-8') as SIN:
        for line in ALL.readlines():
            if len(line.strip()) > 0:
                token = line.strip('\n').split(' ')
                if int(token[2]) == tag2idx[tag]:
                    SIN.writelines(token[0] + ' ' + token[1] + ' ' + '1\n')
                else:
                    SIN.writelines(token[0] + ' ' + token[1] + ' ' + '0\n')
            else:
                SIN.writelines(line)


def dict_match(dictNames, file, tag2idx, dataset, suffix=''):
    tag2idx_copy = copy.deepcopy(tag2idx)
    tag2idx_copy.pop('O')
    new_tag2idx = {}
    idx = 0
    for tag in tag2idx_copy.keys():
        new_tag2idx[tag] = idx
        idx += 1

    classNum = len(new_tag2idx)
    sentences = readFile(file, classNum)
    maxLen = 10
    if suffix == '':
        for tag in new_tag2idx:
            sentences, num, count = lookup_in_Dic(new_tag2idx, "../dictionaries/" + dataset + "/" + dictNames[tag], sentences, tag, maxLen)
    else:
        for tag in new_tag2idx:
            sentences, num, count = lookup_in_Dic(new_tag2idx, "../dictionaries/" + dataset + "/" + dictNames[tag] + '.' + suffix, sentences, tag,
                                                  maxLen)

    if 'train' in file:
        writeFile("../data/" + dataset + "/train.ALL.txt", sentences)
    else:
        writeFile("../data/" + dataset + "/test_distant_labeling.txt", sentences)


def entityIDGeneration(sentences):
    sent_id = 0
    type = "#"
    flag = -1

    label_start_id = 0
    pred_start_id = 0

    true_entities = []
    pred_entities = []
    for sentence in sentences:
        # print("sentence")
        # print(sentence)
        pre_label = "O"
        sent_true_entities = []
        sent_pred_entities = []
        for i, (word, label, pred) in enumerate(sentence):
            if label == "O":
                if not pre_label == "O":
                    label_end_id = i - 1
                    # print("entity label: ", sent_id, label_start_id, label_end_id, type)
                    sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type]))
            else:
                # O B I B I B
                if "B-" in label:
                    label = label.split("-")[-1]
                    if not pre_label == "O":
                        label_end_id = i - 1
                        sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type]))
                    label_start_id = i
                    type = label
                else:
                    continue
            pre_label = label
        if not pre_label == "O":
            label_end_id = len(sentence) - 1
            # print("entity label: ", sent_id, label_start_id, label_end_id, type)
            sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type]))

        pre_pred = 0
        for i, (word, label, pred) in enumerate(sentence):
            if pred == 0:
                if not pre_pred == 0:
                    pred_end_id = i - 1
                    # print("entity pred: ", sent_id, pred_start_id, pred_end_id, flag)
                    sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
            else:
                if not pre_pred == pred:
                    if not pre_pred == 0:
                        pred_end_id = i - 1
                        sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
                    pred_start_id = i
                    flag = pred
                else:
                    continue
            pre_pred = pred

        if not pre_pred == 0:
            pred_end_id = len(sentence) - 1
            # print("entity pred: ", sent_id, pred_start_id, pred_end_id, flag)
            sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))

        sent_id += 1
        true_entities.append(sent_true_entities)
        pred_entities.append(sent_pred_entities)
    return true_entities, pred_entities


def compute_overall_precision_recall_f1(tag2Idx, true_entities, pred_entities):
    tp = 0
    np_ = len(sum(true_entities, []))
    pp = len(sum(pred_entities, []))
    temp = ' '

    assert len(true_entities) == len(pred_entities)
    for i in range(len(true_entities)):
        sent_true = true_entities[i]
        sent_pred = pred_entities[i]
        for e in sent_true:
            for flag in tag2Idx:
                if flag in e:
                    temp = e.replace(flag, str(tag2Idx[flag]))
            if temp in sent_pred:
                tp += 1
    if pp == 0:
        p = 0
    else:
        p = float(tp) / float(pp)
    if np_ == 0:
        r = 0
    else:
        r = float(tp) / float(np_)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2 * p * r) / float((p + r))
    return p, r, f1


def matching_f1(File, tag2idx):
    with open(File, 'r', encoding='utf-8') as T:
        sentences = []
        sentence = []
        for line in T.readlines():
            if len(line.strip()) != 0:
                line = line.strip().split(' ')
                token = [line[0], line[1], int(line[2])]
                sentence.append(token)
            else:
                sentences.append(sentence)
                sentence = []
        if len(sentence) > 0:
            sentences.append(sentence)

    trueEntityID, predEntityID = entityIDGeneration(sentences)

    overall_precision, overall_recall, overall_f1 = compute_overall_precision_recall_f1(tag2idx, trueEntityID, predEntityID)

    print("OVERALL: Precision: {}, Recall: {}, F1: {}".format(overall_precision, overall_recall, overall_f1))


def generate_entity_data(dataset):
    with open('../data/' + dataset + '/train.ALL.txt', 'r', encoding='utf-8') as ALL, open('../data/' + dataset + '/train.Entity.txt', 'w',
                                                                                           encoding='utf-8') as Entity:
        for line in ALL.readlines():
            if len(line.strip()) > 0:
                line_info = line.strip().split(' ')
                if int(line_info[2]) > 0:
                    Entity.writelines(line_info[0] + ' ' + line_info[1] + ' ' + '1' + '\n')
                else:
                    Entity.writelines(line)
            else:
                Entity.writelines(line)


def main():

    # datasets = ['BC5CDR_Dict_0.2', 'BC5CDR_Dict_0.2_2', 'BC5CDR_Dict_0.2_3', 'BC5CDR_Dict_0.2_4', 'BC5CDR_Dict_0.2_5']
    # for dataset in datasets:
    #     tag2idx = {"O": 0, "Chemical": 1, "Disease": 2}
    #     path = './data/' + dataset + '/train.ALL.txt'
    #     matching_f1(path, tag2idx)

    # dictionary matching
    dataset = 'QTL'
    # tag2idx = {"O": 0, "Trait": 1, "Gene": 2}
    tag2idx = {"O": 0, "Trait": 1}
    dict_names = {"Trait": "Trait.txt"}
    # trainFile = './data/' + dataset + '/train.txt'
    trainFile = './data/' + dataset + '/test.txt'
    dict_match(dict_names, trainFile, tag2idx, dataset)

    # # generate_entity_data
    # # dataset = 'QTL'
    # # generate_entity_data(dataset)

    # # generate single from all
    # dataset = 'QTL'
    # tag2idx = {"O": 0, "Trait": 1, "Gene": 2}
    # tagList = tag2idx.keys() - ['O']
    # for tag in tagList:
    #     generate_single_from_all(tag, dataset, tag2idx)

    # dictionary matching
    # dataset = 'QTL'
    # tag2idx = {"O": 0, "Trait": 1, "Gene": 2}
    # dict_names = {"Trait": "Trait.txt", "Gene": "Gene.txt"}
    # trainFile = './data/' + dataset + '/test.txt'
    # dict_match(dict_names, trainFile, tag2idx, dataset)

    # f1
    # dataset = 'QTL'
    # tag2idx = {"O": 0, "Trait": 1, "Gene": 2}
    # path = './data/' + dataset + '/test_distant_labeling.txt'
    # matching_f1(path, tag2idx)


if __name__ == '__main__':
    main()
