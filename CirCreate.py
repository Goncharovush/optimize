import unicodedata

span = {"min_R": 0, "max_R": 1, "min_C": 0, "max_C": 10 ** -0}

s = {"R1": "0", "C1": "0",
     "R_C1": "0", "R_D1": "0", "R2": "0",
     "C2": "0", "R_C2": "0",
     "R3": "0", "C3": "0",
     "R_C3": "0", "R_D3": "0",
     "D1": "0", "D3": "0"}


def new_set(name, f, s):
    for j in s:
        f[j] = str(f[j])
    file = open(str(name) + ".cir", "wb")
    file.write("* cir file corresponding to the equivalent circuit.\n".encode('utf-8'))
    file.write("* Цепь 1\n".encode('utf-8'))
    file.write("R1 _net1 Input ".encode('utf-8') + f["R1"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("C1 _net0 _net1 ".encode('utf-8') + f["C1"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("R_C1 _net0 _net1 ".encode('utf-8') + f["R_C1"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("D1 _net0 0 DMOD_D1 AREA=1.0 Temp=26.85\n".encode('utf-8'))
    file.write("R_D1 0 _net0 ".encode('utf-8') + f["R_D1"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("* Цепь 2\n".encode('utf-8'))
    file.write("R2 _net4 Input ".encode('utf-8') + f["R2"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("C2 0 _net4 ".encode('utf-8') + f["C2"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("R_C2 0 _net4 ".encode('utf-8') + f["R_C2"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("* Цепь 3\n".encode('utf-8'))
    file.write("R3 _net3 Input ".encode('utf-8') + f["R3"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("C3 _net2 _net3 ".encode('utf-8') + f["C3"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("R_C3 _net2 _net3 ".encode('utf-8') + f["R_C3"].encode('utf-8') + "\n".encode('utf-8'))
    file.write("D3 0 _net2 DMOD_D3 AREA=1.0 Temp=26.85\n".encode('utf-8'))
    file.write("R_D3 0 _net2 ".encode('utf-8') + f["R_D3"].encode('utf-8') + "\n".encode('utf-8'))
    file.write(".MODEL DMOD_D1 D (Is=".encode('utf-8') + f["D1"].encode('utf-8') + " N=1.65 "
               "Cj0=4e-12 M=0.333 Vj=0.7 Fc=0.5 Rs=0.0686 "
               "Tt=5.76e-09 Ikf=0 Kf=0 Af=1 Bv=75 Ibv=1e-06 "
               "Xti=3 Eg=1.11 Tcv=0 Trs=0 Ttt1=0 Ttt2=0 Tm1=0 "
               "Tm2=0 Tnom=26.85 )\n".encode('utf-8'))
    file.write(".MODEL DMOD_D3 D (Is=".encode('utf-8') + f["D3"].encode('utf-8') + " N=1.65 "
               "Cj0=4e-12 M=0.333 Vj=0.7 Fc=0.5 Rs=0.0686 "
               "Tt=5.76e-09 Ikf=0 Kf=0 Af=1 Bv=75 Ibv=1e-06 "
               "Xti=3 Eg=1.11 Tcv=0 Trs=0 Ttt1=0 Ttt2=0 Tm1=0 "
               "Tm2=0 Tnom=26.85 )\n".encode('utf-8'))
    file.write(".END\n".encode('utf-8'))
    file.close()


def create_cir(name, f):
    """
    This function create cir-file with electric circuit
    :param name: name of file
    :param f: parameters
    :return:
    """
    global s
    list_keys = list(s.keys())
    list_keys.sort()
    # for i in range(5, 13):
    #     f[i] = 0 if f[i] < 50 else f[i]
    new_dict = dict(zip(list_keys, list(f)))
    new_set(name, new_dict, list_keys)


if __name__ == "__main__":
    num = 0
    list_keys = list(s.keys())
    list_keys.sort()
    print(list_keys)
    # f = [p for p in itertools.product(range(2), repeat=11)]
    # f = list(itertools.permutations(f1, 2))
    # for k in range(len(f)):
    #     new_dict = dict(zip(list_keys, list(f[k])))
    #     new_set(k, new_dict, list_keys)
    # print(len(list(itertools.product(list_keys, range(10)))))
    # print(len(f))
    # for num in range(len(f)):
    #     create_cir(num, f[num])
