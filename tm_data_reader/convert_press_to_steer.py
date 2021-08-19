#from pygbx import Gbx, GbxType
#from pygbx.headers import ControlEntry, CGameCtnGhost
from numpy import int32
import matplotlib.pyplot as plt
import sys
import os
import copy

#primeste un set cu stringuri ce se doresc citite.
#returneaza un htable cu valorile dorite.
#ex: get_hyperparams({"CUTOFF_TIME", "HUMAN_TIME"}) => {"CUTOFF_TIME": "10000", "HUMAN_TIME": "9350"}
#!! setul (unordered_set) se initializeaza cu set()
def get_hyperparams(hyperparams: set):
    hyperparams = copy.deepcopy(hyperparams)
    fin = open("../Configs/_CURRENT_MAP_NAME.txt")
    if fin.closed:
        print("../Configs/_CURRENT_MAP_NAME.txt needed!")
        assert(False)

    iname = fin.readline()
    fin.close()

    fin = open("../Configs/" + iname)
    if fin.closed:
        print(f"Couldn't find {iname} in ../Configs/.")
        assert(False)

    print(f"Reading hyperparams from {iname}!")

    config = {}
    cnt_hparams_read, cnt_hparams_need = 0, len(hyperparams)
    for line in fin.readlines():
        name, val = line.split()
        assert(name.find(' ') == -1 and val.find(' ') == -1)
        if name in hyperparams:
            cnt_hparams_read += 1
            config[name] = val
            hyperparams.remove(name)

    if cnt_hparams_read != cnt_hparams_need:
        print(f"(get_hyperparams) Error, wanted to read {cnt_hparams_need} vars, but only read {cnt_hparams_read}.")
        print(f"Couldn't find the following: {hyperparams}.")
        assert(False)

    fin.close()
    return config

GAP_TIME, CUTOFF_TIME = 10, None #ms

def preprocess_hyperparams():
    global CUTOFF_TIME, LEFT_SHIFTS, RIGHT_SHIFTS
    config = get_hyperparams({"CUTOFF_TIME"})
    CUTOFF_TIME = int(config["CUTOFF_TIME"])
    print("tm_data_reader, convert_press_to_steer.py, preprocess_hyperparams:")
    print(f"CUTOFF_TIME = {CUTOFF_TIME}")

def process_inputs (fin, write_to):
    global GAP_TIME, CUTOFF_TIME

    fout = open(write_to + "converted_to_steer.txt", "w")

    input_count = CUTOFF_TIME // GAP_TIME
    steer = [0 for _ in range(input_count)]

    for line in fin.readlines():
        s = line.strip('\n').split(' ')
        if s[2] in ("up", "down"):
            fout.write(line)
        else:
            l, r = map(int, s[0].split('-'))
            l //= GAP_TIME
            r //= GAP_TIME
            steer[l:r] = [65536] * (r-l) if s[2] == "right" else [-65536] * (r-l)

    for i in range(len(steer)):
        fout.write(f"{i * GAP_TIME} steer {steer[i]}\n")

    fout.close()
            

#ia un fisier care are numai comenzi tip "press" si lasa unul cu "steer" in acelasi folder
def main():
    if len(sys.argv) < 2: #"python" nu se numara
        print("Example usage: " + "python convert_press_to_steer.py Converted-txt/tas/")
        quit()

    preprocess_hyperparams()

    path = sys.argv[1]
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".txt"):
                    with open(path + filename, "r") as fin:
                        process_inputs(fin, sys.argv[1])

    plt.show()

if __name__ == '__main__':
    main()
