from tminterface.interface import TMInterface
from tminterface.client import Client
import sys
import signal
import time
import copy

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
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2

        self.processed_output_dir = "Processed-outputs/output_"
        
        self.last_time_in_run_step = 0
        self.time_to_remember_state, self.remembered_state = (-1100, -500), None
        self.did_race_finish = False

        #acest client este un fel de cutie neagra care trebuie sa primeasca niste inputuri, sa le ruleze
        #si sa returneze scorurile pentru inputuri
        #cand nu are nimic de rulat (i.e. self.should_client_work == False), clientul nu face nimic
        #cand primeste >= 1 inputuri, da reset la cursa, calculeaza pana le face pe toate etc
        self.should_client_work = False #variabila care indica ralanti pentru simulari
        self.input_stack = [] #stiva cu [inputuri ce se doresc a fi rulate, scorul inputului dupa rulare]
        self.input_stack_index = -1 #indica care input trebuie rulat acum.

        pass

    def on_registered(self, iface: TMInterface):
        print(f"Registered to {iface.server_name}")

    def on_deregistered(self, iface):
        pass

    def on_shutdown(self, iface):
        pass

    def on_run_step(self, iface: TMInterface, time: int):
        if not self.should_client_work:
            return

        self.last_time_in_run_step = time

        if self.did_race_finish or time >= self.CUTOFF_TIME:
            self.did_race_finish = False

            if time >= self.CUTOFF_TIME:
                self.process_input_stack(iface)

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


        iface.set_input_state(up = self.input_stack[self.input_stack_index][0][self.IND_PUSH_UP][prepared_act_index],
                              down = self.input_stack[self.input_stack_index][0][self.IND_PUSH_DOWN][prepared_act_index],
                              steer = self.input_stack[self.input_stack_index][0][self.IND_STEER][prepared_act_index])

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
        write_processed_output(oname, self.input_stack[self.input_stack_index][0], self.GAP_TIME)

        print("wrote to " + oname + "!")

        self.process_input_stack(iface)

        pass

    def on_laps_count_changed(self, iface, current: int):
        pass

    def fitness_function (self, iface):
        state = iface.get_simulation_state()
        speed = state.get_display_speed()

        score = 0
        score += 250 * (23720 - self.last_time_in_run_step)
        score += speed

        return score

    def add_input_array_to_stack(self, input):
        self.input_stack.append([copy.deepcopy(input), None])
        self.input_stack_index += 1
        if not self.should_client_work:
            self.should_client_work = True

    #se apeleaza la sfarsitul unei simulari
    def process_input_stack (self, iface):
        self.input_stack[self.input_stack_index][1] = self.fitness_function(iface)
        self.input_stack_index -= 1
        if self.input_stack_index < 0:
            self.should_client_work = False

    def empty_stack (self):
        self.input_stack = []
        self.input_stack_index = -1
        pass


class ML():
    def __init__(self):
        self.GAP_TIME, self.CUTOFF_TIME = 10, 25000
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = 7, 7
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2

        self.processed_input_dir = "../tm_data_reader/Processed-inputs/input_"
        #presupunand ca fisierul din care se ruleaza e TMInterfaceClientPython-master/

        self.input_arrays = [[] for _ in range(self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1)]
        for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
            self.input_arrays[i + self.LEFT_SHIFTS] = read_processed_input(self.processed_input_dir + str(i) + ".txt")

        self.have_current_run = False
        self.current_run_score = 0

        #pentru fiecare interval [l, r] ai o combinatie de coeficienti
        #self.intervals[i][0] = (l, r)
        #self.intervals[i][1] = un vector cu coeficientii corespunzatori de lungime
        #self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1
        #TODO fa posibil splitul unui interval in 2 cu o probabilitate aleatoare

        self.intervals = self.make_intervals(50)
        self.interval_bounds = (28, 50) #se lucreaza pe intervalele [.., ..)
        self.curr_itv = self.interval_bounds[0] #indexul intervalului pe care se lucreaza momentan
        self.percentage_increase = 0.01 #dc procentajul este 0.4 se intra in calcul cu el 0.41
        self.kept_change = 0.15 #cat din schimbare chiar este facuta
        self.changed_percentages = [] #tine minte procentele schimbate bagate in stiva, folosite mai
        #tarziu la calculul gradientilor

        self.epoch_count = 1

        pass

    def make_intervals(self, coef):
        itv_starts = [x for x in range(0, self.CUTOFF_TIME // self.GAP_TIME, coef)]
        intervals = []
        for i in range(len(itv_starts) - 1):
            intervals.append([(itv_starts[i], itv_starts[i+1] - 1),
                             [0.0] * self.LEFT_SHIFTS + [1.0] + [0.0] * self.RIGHT_SHIFTS])
        return intervals

    def normalize_percentages(self, percentages):
        assert(len(percentages) == self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1)
        for i in range(len(percentages)):
            percentages[i] = max(0, percentages[i])
        x = sum(percentages)
        assert(x != 0.0)
        x = 1 / x
        for i in range(len(percentages)):
            percentages[i] *= x
        return percentages

    #inputs este un vector gen [[steer -> [], push_up -> [], push_down -> []], ... ]
    #len(inputs) == LEFT_SHIFTS + RIGHT_SHIFTS + 1
    #len(percentages) == in cate itv ai vrut tu sa spargi linia de timp
    #percentages[k] = un vector cu LEFT_SHIFTS + RIGHT_SHIFTS + 1 procente
    #de obicei se apeleaza pentru reuniune de intervale
    def combine_inputs(self, inputs, percentages):
        n = len(inputs[0][self.IND_STEER])
        sol = [[0] * n, [0] * n, [0] * n]
        for i in range(len(inputs)):
            for j in range(3):
                assert(n == len(inputs[i][j]))
                k = 0
                for z in range(n):
                    while k + 1 < len(percentages) and self.intervals[k][0][1] < z:
                        k += 1
                    sol[j][z] += percentages[k][i] * inputs[i][j][z]

        for z in range(n):
            sol[self.IND_STEER][z] = max(-65536, min(65536, int(sol[self.IND_STEER][z])))

        for j in (self.IND_PUSH_UP, self.IND_PUSH_DOWN):
            for z in range(n):
                sol[j][z] = round(sol[j][z])
                assert(sol[j][z] in (0, 1))

        return sol

    def main_loop(self, client: MainClient):
        if not self.have_current_run:
            if len(client.input_stack) == 0: #trebuie dat la client inputul actual
                #client.add_input_array_to_stack(self.input_arrays[0 + self.LEFT_SHIFTS])
                #trb sa bagi si aici self.combine_inputs VV
                percentages = []
                for j in range(len(self.intervals)):
                    percentages.append(copy.deepcopy(self.intervals[j][1]))

                client.add_input_array_to_stack(self.combine_inputs(self.input_arrays, percentages))
                pass
            elif client.input_stack[0][1] != None: #a venit rezultatul
                self.current_run_score = client.input_stack[0][1]
                self.have_current_run = True
                client.empty_stack()
            return

        if not client.should_client_work:
            if len(client.input_stack) == 0:
                #trebuie dat de munca clientului.
                #trebuie calculate noile procentaje si bagate toate inputurile noi in acelasi timp
                #in client.input_stack

                #procentajele ce trebuie manipulate sunt self.intervals[self.curr_itv][1]
                self.changed_percentages = []
                for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
                    local_perc = copy.deepcopy(self.intervals[self.curr_itv][1])
                    local_perc[i + self.LEFT_SHIFTS] += self.percentage_increase
                    local_perc = self.normalize_percentages(local_perc)
                    
                    self.changed_percentages.append(local_perc[i + self.LEFT_SHIFTS]) #folosit in else

                    percentages = []
                    for j in range(self.curr_itv):
                        percentages.append(copy.deepcopy(self.intervals[j][1]))
                    percentages.append(local_perc)
                    for j in range(self.curr_itv + 1, len(self.intervals)):
                        percentages.append(copy.deepcopy(self.intervals[j][1]))

                    client.add_input_array_to_stack(self.combine_inputs(self.input_arrays, percentages))
            else:
                #daca sunt aici inseamna ca clientul a terminat munca pe care i-am dat-o, trebuie analizata,
                #trebuie calculate procentajele noi si trebuie golita stiva
                scores = []
                for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
                    scores.append(client.input_stack[i + self.LEFT_SHIFTS][1])

                local_perc = copy.deepcopy(self.intervals[self.curr_itv][1])
                gradients = []
                assert(len(self.changed_percentages) == self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1)

                for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
                    diff = abs(local_perc[i + self.LEFT_SHIFTS] - self.changed_percentages[i + self.LEFT_SHIFTS])
                    if diff == 0:
                        gradients.append(0.0)
                    else:
                        gradients.append((scores[i + self.LEFT_SHIFTS] - self.current_run_score) / diff)
                    gradients[-1] = max(-self.percentage_increase, min(self.percentage_increase, gradients[-1]))

                for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS):
                    local_perc[i + self.LEFT_SHIFTS] += self.kept_change * gradients[i + self.LEFT_SHIFTS]

                local_perc = self.normalize_percentages(local_perc)
                self.intervals[self.curr_itv][1] = copy.deepcopy(local_perc)
                #!!! doar procentajele trebuie schimbate la sfarsitul reprizei
                #input_arrays ramane LA FEL intotdeauna

                print(f"Finished epoch no. {self.epoch_count} on interval {self.curr_itv}!")

                client.empty_stack()
                self.have_current_run = False #trebuie sa recalculez scorul curent
                self.epoch_count += 1
                self.curr_itv = self.curr_itv+1 if self.curr_itv+1 < self.interval_bounds[1] else self.interval_bounds[0]

                pass
        else:
            #lasa clientul sa lucreze ce i-ai dat.
            pass

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

    ml = ML()
    while iface.running:
        ml.main_loop(client)
        time.sleep(0)

if __name__ == '__main__':
    main()
