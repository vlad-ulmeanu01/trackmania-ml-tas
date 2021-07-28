from tminterface.interface import TMInterface
from tminterface.client import Client
import sys
import signal
import time

def read_processed_input(f: str):
    fin = open(f)
    if fin.closed:
        print("Bad input filename given for read_processed_input:" + f)
        return

    sol = []
    for _ in range(3):
        sol.append(list(map(int, fin.readline().split())))

    fin.close()
    return sol #[split, push_up, push_down]

#scrie date procesate intr-un fisier astfel incat sa se poata citi de tminterface
def write_processed_output(g, sol, GAP_TIME):
    fout = open(g, "w")
    if fout.closed:
        print("Couldn't create file with name: " + g)
        return

    steer, push_up, push_down = sol
    assert(len(steer) == len(push_up) and len(push_up) == len(push_down))
    n = len(steer)

    for push, dir in [(push_up, "up"), (push_down, "down")]:
        i = 0
        while i < n:
            while i < n and push[i] == 0:
                i += 1
            if i < n:
                j = i
                while j < n and push[j] == 1:
                    j += 1
                fout.write(str(i * GAP_TIME) + "-" + str(j * GAP_TIME) + " press " + dir + "\n")
                i = j

    for i in range(n):
        fout.write(str(i * GAP_TIME) + " steer " + str(steer[i]) + "\n")

    fout.close()

class MainClient(Client):
    def __init__(self):
        self.GAP_TIME, self.CUTOFF_TIME = 10, 25000 #ms
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = 7, 7
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2
        
        self.processed_input_dir = "../tm_data_reader/Processed-inputs/input_"
        #presupunand ca fisierul din care se ruleaza e TMInterfaceClientPython-master/
        self.processed_output_dir = "Processed-outputs/output_"
        
        self.last_time_in_run_step = 0
        self.time_to_remember_state, self.remembered_state = (-1100, -500), None
        self.did_race_finish = False

        self.input_arrays = {}
        for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
            self.input_arrays[i] = read_processed_input(self.processed_input_dir + str(i) + ".txt")

        pass

    def on_registered(self, iface: TMInterface):
        print(f"Registered to {iface.server_name}")

    def on_deregistered(self, iface):
        pass

    def on_shutdown(self, iface):
        pass

    def on_run_step(self, iface: TMInterface, time: int):
        self.last_time_in_run_step = time

        if self.did_race_finish:
            self.did_race_finish = False
            iface.rewind_to_state(self.remembered_state)
            return

        if time >= self.time_to_remember_state[0] and time <= self.time_to_remember_state[1] and self.remembered_state == None:
            self.remembered_state = iface.get_simulation_state()
            print(f"Remembered state at {time}!")

        tmp_before = 0 #[0, self.GAP_TIME) ok
        if time < -tmp_before or time >= self.CUTOFF_TIME:
            return

        prepared_act_index = (time + tmp_before) // self.GAP_TIME

        if prepared_act_index >= self.CUTOFF_TIME // self.GAP_TIME:
            return

        iface.set_input_state(up = self.input_arrays[0][self.IND_PUSH_UP][prepared_act_index],
                              down = self.input_arrays[0][self.IND_PUSH_DOWN][prepared_act_index],
                              steer = self.input_arrays[0][self.IND_STEER][prepared_act_index])
        pass

    def on_simulation_begin(self, iface):
        pass

    def on_simulation_step(self, iface, time: int):
        pass
    
    def on_simulation_end(self, iface, result: int):
        pass

    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        if current < target:
            return

        self.did_race_finish = True
        iface.prevent_simulation_finish()

        oname = self.processed_output_dir + str(int(time.time())) + "_" + str(self.last_time_in_run_step) + ".txt"
        write_processed_output(oname, self.input_arrays[0], self.GAP_TIME)

        print("wrote to " + oname + "!")

        #deocamdata e self.input_arrays[0] ca sa vad ca merge inapoi
        #cu ce dau eu playback am 23s73, cu tminterface ar trebui sa fie 23s72

        pass

    def on_laps_count_changed(self, iface, current: int):
        pass

def main():
    server_name = 'TMInterface0'
    if len(sys.argv) > 1:
        server_name = 'TMInterface' + str(sys.argv[1])

    print(f'Connecting to {server_name}...')

    iface = TMInterface(server_name)
    def handler(signum, frame):
        iface.close()
        sys.exit(0)

    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    client = MainClient()
    iface.register(client)

    while iface.running:
        time.sleep(0)

if __name__ == '__main__':
    main()
