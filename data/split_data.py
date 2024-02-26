

def read_file(fn):
    res = []
    words, truth, preds = [], [], []
    with open(fn) as fp:
        for line in fp:
            splits = line.strip("\n").split(" ")
            if len(splits) < 3:
                res.append([words, truth, preds])
                words, truth, preds = [], [], []
                continue
            words.append(splits[0])
            truth.append(splits[1])
            preds.append(splits[2])

    return res 

def write_file(fn, data):
    with open(fn, "w") as fp:
        for words, truth, preds in data:
            for w, t, p in zip(words, truth, preds):
                fp.writelines(" ".join([str(i) for i in [w, t, p]])+"\n")
            fp.writelines("\n")

def main():
    path = "CoNLL2003_KB"
    data = read_file(path + "/train.txt")
    length = len(data)
    for i in [0.01, 0.1]:
        write_file(path + "/train_" + str(i) + ".txt", data[:int(i*length)])
    # data = read_file("/work/LAS/qli-lab/yuepei/Conf-MPU-Trait/Conf-MPU-Trait/data/QTL/test_distant_labeling.txt")
    # types = read_type("/work/LAS/qli-lab/yuepei/DS_NER_ACTIVE/data/QTL_Final/types.txt")
    # data = read_file("../data/CoNLL2003_KB/train.txt")
    # types = read_type("../data/CoNLL2003_KB/types.txt")
    # id2type = {k: v for k, v in enumerate(types)}
    # type2id = {v: k for k, v in enumerate(types)}
    # truth = [d[0] for d in data]
    # preds = [d[1] for d in data]
    # preds = [["I-" + id2type[p] if p != 0 else "O" for p in  sent] for sent in preds]

    # relax_f1 = relax_eval(truth[:2], preds[:2])
    # relax_f1 = relax_eval(truth, preds)
    # print(relax_f1)

    # r = get_entity([['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'MISC', 'MISC']])
    # print(r)

if __name__ == "__main__":
    main()