import matplotlib.pyplot as plt

def read(fn):
    res = []
    rst = []
    with open(fn) as fp:
        for line in fp:
            line = line.strip("\n")
            if "micro avg" in line:
                res.append(list(filter(lambda c: c != "", line.split(" "))))
            if "=== train on" in line:
                print(sorted(res, key=lambda r: r[4], reverse=True)[0])
                rst.append(sorted(res, key=lambda r: r[4], reverse=True)[0])
                res = []
    return rst

def plot(rst):
    cur = [rst[i] for i in range(0, len(rst), 2)]
    mpu = [rst[i] for i in range(1, len(rst), 2)]

    cur_f = [float(i[4]) for i in cur]
    mpu_f = [float(i[4]) for i in mpu]
    x = range(len(cur_f))

    plt.plot(x, cur_f, label ='cur')
    plt.plot(x, mpu_f, '-.', label ='mpu')

    plt.savefig("tmp12.png")



if __name__ == "__main__":
    rst = read("log12.txt")
    plot(rst)

