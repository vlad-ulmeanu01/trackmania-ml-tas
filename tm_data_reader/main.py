#from pygbx import Gbx, GbxType
#from pygbx.headers import ControlEntry, CGameCtnGhost
from numpy import int32
import matplotlib.pyplot as plt
import sys
import os

GAP_TIME, CUTOFF_TIME = 10, 19000 #ms
LEFT_SHIFTS, RIGHT_SHIFTS = 15, 15

def process_inputs (fin, write_to):
    global GAP_TIME, CUTOFF_TIME, LEFT_SHIFTS, RIGHT_SHIFTS

    input_count = CUTOFF_TIME // GAP_TIME
    steer_times, steer_values = [], []
    hard_turns = {}
    press_up = [0 for _ in range(input_count)]
    press_down = [0 for _ in range(input_count)]
    steer = [0 for _ in range(input_count)]
    cnt = 0

    for line in fin.readlines():
        s = line.strip('\n').split(' ')
        if s[1] == "steer":
            steer_times.append(int(s[0]))
            steer_values.append(int(s[2]))
        elif s[2] in ("up", "down", "left", "right"):
            l, r = map(int, s[0].split('-'))
            if s[2] == "up":
                press_up[l//GAP_TIME : r//GAP_TIME] = [1] * (r//GAP_TIME - l//GAP_TIME)
            elif s[2] == "down":
                press_down[l//GAP_TIME : r//GAP_TIME] = [1] * (r//GAP_TIME - l//GAP_TIME)
            else:
                hard_turns[l] = s[2]
                #inputul de 002372 are si press left care suprascrie (????) steer!!!!!!
        cnt += 1

    for i in range(len(steer_times)):
        ub = input_count #!! input_count depinde de CUTOFF_TIME
        if i < len(steer_times) - 1:
            ub = steer_times[i+1] // GAP_TIME
        
        if steer_times[i] in hard_turns:
            steer_values[i] = -65536 if hard_turns[steer_times[i]] == "left" else 65536

        steer[steer_times[i]//GAP_TIME : ub] = [steer_values[i]] * (ub - steer_times[i]//GAP_TIME)

    for shift in range(-LEFT_SHIFTS, RIGHT_SHIFTS + 1):
        fout = open(write_to + "input_" + str(shift) + ".txt", "w")
        for v in (steer, press_up, press_down):
            if shift <= 0:
                fout.write(str(v[-shift:] + [0] * (-shift)).strip('[').strip(']').replace(',', '') + '\n')
            else:
                fout.write(str([0] * shift + v[:len(v) - shift]).strip('[').strip(']').replace(',', '') + '\n')
        fout.close()

    #plt.plot(steer_times, steer_values)
    plt.plot(steer)
    plt.plot(press_up)
    plt.plot(press_down)

def main():
    if len(sys.argv) < 3: #"python" nu se numara
        print("No provided path or writeoff zone!")
        print("Example usage: " + "python main.py Converted-txt/tas/ Processed-inputs/")
        quit()

    path = sys.argv[1]
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".txt"):
                    with open(path + filename, "r") as fin:
                        process_inputs(fin, sys.argv[2])

    plt.show()

if __name__ == '__main__':
    main()
