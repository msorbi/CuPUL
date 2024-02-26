

def read_type(fn):
    types = ["O"]
    with open(fn) as fp:
        for line in fp:
            types.append(line.strip("\n"))
    return types 

def read_file(fn):
    res = []
    truth, preds = [], []
    with open(fn) as fp:
        for line in fp:
            splits = line.strip("\n").split(" ")
            if len(splits) < 3:
                res.append([truth, preds])
                truth, preds = [], []
                continue
            truth.append(splits[1])
            preds.append(int(splits[2]))

    return res 


def check_equal(entity1, entity2, exact=True):
    if exact:
        return entity1 == entity2
    else:
        t1, s1, e1 = entity1.split("_")
        t2, s2, e2 = entity2.split("_")
        s1, s2, e1, e2 = [int(i) for i in [s1, s2, e1, e2]]
        # print(t1, t2, s1, s2, e1, e2)
        if t1 != t2:
            return False 
        if s1 > e2 or s2 > e1:
            return False 
        return True 


def get_entity(sentences):
    res = []
    tmp = []
    pre_type = "O"
    for sentence in sentences:
        for index, type in enumerate(sentence+["O"]):
            type = type.split("-")[-1]
            if pre_type == "O":
                if type != "O":
                    entity_start = index 
            else:
                if type != pre_type:
                    entity_end = index-1
                    tmp.append("_".join([str(i) for i in [pre_type, entity_start, entity_end]]))
                    entity_start = index 
            pre_type = type 
        res.append(tmp)
        tmp = []
    return res 



def relax_eval(truth, preds):
    print(truth, preds)
    truth_entity = get_entity(truth)
    preds_entity = get_entity(preds)

    p_tp = 0
    t_tp = 0
    for tru, pre in zip(truth_entity, preds_entity):
        # print(tru, pre)
        p_tmp, t_tmp = [], []
        for t in tru:
            for p in pre:
                if check_equal(t, p, exact=True):
                    # print(t, p)
                    p_tmp.append(p)
                    t_tmp.append(t)
        p_tp += len(set(p_tmp))
        t_tp += len(set(t_tmp))
                    
    
    pe_num = sum([len(i) for i in preds_entity])
    te_num = sum([len(i) for i in truth_entity])
    # print(tp, pe_num, te_num)

    precision = p_tp / pe_num
    recall = t_tp / te_num
    f1 = 2*precision*recall / (precision + recall)

    return precision, recall, f1



def main():
    # data = read_file("/work/LAS/qli-lab/yuepei/DS_NER_ACTIVE/data/QTL_Final/pred_test.txt")
    data = read_file("/work/LAS/qli-lab/yuepei/Conf-MPU-Trait/Conf-MPU-Trait/data/QTL/test_distant_labeling.txt")
    types = read_type("/work/LAS/qli-lab/yuepei/DS_NER_ACTIVE/data/QTL_Final/types.txt")
    # data = read_file("../data/CoNLL2003_KB/train.txt")
    # types = read_type("../data/CoNLL2003_KB/types.txt")
    id2type = {k: v for k, v in enumerate(types)}
    type2id = {v: k for k, v in enumerate(types)}
    truth = [d[0] for d in data]
    preds = [d[1] for d in data]
    preds = [["I-" + id2type[p] if p != 0 else "O" for p in  sent] for sent in preds]

    # relax_f1 = relax_eval(truth[:2], preds[:2])
    # relax_f1 = relax_eval(truth, preds)
    relax_f1 = relax_eval(truth, preds)
    print(relax_f1)

    # r = get_entity([['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'MISC', 'MISC']])
    # print(r)

if __name__ == "__main__":
    main()