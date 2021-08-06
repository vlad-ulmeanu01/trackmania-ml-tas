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
        self.GAP_TIME = 10 #ms
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2
        
        '''
        #A01
        self.CUTOFF_TIME = 25000
        self.HUMAN_TIME = 23720
        #A01
        '''
        '''
        #TAS_Training_Map_1
        self.CUTOFF_TIME = 10000
        self.HUMAN_TIME = 9310
        #TAS_Training_Map_1
        '''
        #StarStadiumA5
        self.CUTOFF_TIME = 19000
        self.HUMAN_TIME = 18090
        #StarStadiumA5

        self.processed_output_dir = "Processed-outputs/output_"
        
        self.last_time_in_run_step = 0
        self.last_speed_in_run_step = 0
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
        self.last_speed_in_run_step = iface.get_simulation_state().get_display_speed()

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
        score = 0
        score += 250 * (self.HUMAN_TIME - self.last_time_in_run_step)
        score += self.last_speed_in_run_step

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
        self.GAP_TIME = 10
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2

        '''
        #A01
        self.CUTOFF_TIME = 25000
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = 7, 7 + 15 #processed_inputs e o combinatie de doesnt_end + does_end; +1 pt lag dubios; input_0 e 2372, input_15 e tasbad care nu termina; input -7..-1 & 1 .. 7 sunt de la tasbad, restul de la 2372
        self.intervals = self.make_intervals(50)
        self.interval_bounds = (28, 41) #se lucreaza pe intervalele [.., ..)
        #A01
        '''
        '''
        #TAS_Training_Map_1
        self.CUTOFF_TIME = 10000
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = 15, 15
        self.intervals = self.make_intervals(10)
        self.interval_bounds = (0, 10) #se lucreaza pe intervalele [.., ..)
        #TAS_Training_Map_1
        '''
        #StarStadiumA5
        self.CUTOFF_TIME = 19000
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = 15, 15
        self.intervals = self.make_intervals(20)
        self.interval_bounds = (4, 20) #se lucreaza pe intervalele [.., ..)
        #StarStadiumA5

        #pentru fiecare interval [l, r] ai o combinatie de coeficienti
        #self.intervals[i][0] = (l, r)
        #self.intervals[i][1] = un vector cu coeficientii corespunzatori de lungime
        #self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1
        #TODO fa posibil splitul unui interval in 2 cu o probabilitate aleatoare

        self.processed_input_dir = "../tm_data_reader/Processed-inputs/input_"
        #presupunand ca fisierul din care se ruleaza e TMInterfaceClientPython-master/

        self.input_arrays = [[] for _ in range(self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1)]
        for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
            self.input_arrays[i + self.LEFT_SHIFTS] = read_processed_input(self.processed_input_dir + str(i) + ".txt")

        self.have_current_run = False
        self.current_run_score = 0

        self.curr_itv = self.interval_bounds[0] #indexul intervalului pe care se lucreaza momentan
        self.percentage_increase = 0.3 #dc procentajul este 0.4 se intra in calcul cu el 0.7
        self.percentage_increase_per_fix = 0.1 #se aduna la self.percentage_increase dupa ?? reprize fara +
        self.kept_change = 0.15 #cat din schimbare chiar este facuta (doar strat A)

        self.changed_percentages = [] #tine minte procentele schimbate bagate in stiva, folosite mai
        #tarziu la calculul gradientilor

        self.epoch_count = 1
        self.epochs_since_last_improvement = 0
        self.max_epochs_no_improvement = self.interval_bounds[1] - self.interval_bounds[0]
        #trb sa fac ceva daca am ?? reprize fara imbunatariri
        self.strat = "B"

        pass

    def make_intervals(self, num_itv):
        LG = self.CUTOFF_TIME // self.GAP_TIME

        if LG % num_itv != 0:
            print(f"(ML.make_intervals) Warning, {LG} is not divisible with {num_itv}.")

        itv_starts = [x for x in range(0, LG, LG // num_itv)]
        intervals = []
        for i in range(len(itv_starts) - 1):
            intervals.append([(itv_starts[i], itv_starts[i+1] - 1),
                             [0.0] * self.LEFT_SHIFTS + [1.0] + [0.0] * self.RIGHT_SHIFTS])

        intervals.append([(itv_starts[-1], LG-1), [0.0] * self.LEFT_SHIFTS + [1.0] + [0.0] * self.RIGHT_SHIFTS])

        return intervals

    def normalize_percentages(self, percentages):
        percentages = copy.deepcopy(percentages) #!! lista nu se copiaza la return
        assert(len(percentages) == self.LEFT_SHIFTS + self.RIGHT_SHIFTS + 1)
        for i in range(len(percentages)):
            percentages[i] = max(0, percentages[i])
        x = sum(percentages)
        assert(x != 0.0)
        x = 1 / x
        for i in range(len(percentages)):
            percentages[i] *= x
        return percentages

    #primesti un vector cu numere float in [0, 1]. Trebuie sa le treci pe toate in {0, 1}, dar nu vrei
    #sa le rotunjesti asa. vrei:
    #.. 0 0) 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 (1 1 1 ... => .. 0 0) 0 0 0 1 1 1 1 1 1 1 (1 1 1 ...
    #selective_round apelat doar din combine inputs imediat mai jos
    def selective_round(self, arr: list):
        eps = 0.00001
        parti_non_01 = [] #vector cu info despre perechi cu itv non01, gen[[l, r, vecin_st, vecin_dr]]
        #sus ar fi [[3, 12, 0, 1]].

        for i in range(len(arr)):
            arr[i] = max(0, min(1, arr[i]))

        i = 0
        while i < len(arr):
            while i < len(arr) and (arr[i] < eps or arr[i] > 1 - eps):
                i += 1
            if i < len(arr):
                j = i
                while j < len(arr) and arr[j] >= eps and arr[j] <= 1 - eps:
                    j += 1
                parti_non_01.append([i, j-1, None, None])
                if i > 0:
                    parti_non_01[-1][2] = round(arr[i-1])
                if j < len(arr):
                    parti_non_01[-1][3] = round(arr[j])
                i = j

        for l, r, lneigh, rneigh in parti_non_01:
            if lneigh == rneigh:
                for i in range(l, r+1):
                    arr[i] = round(arr[i])
                continue
            elif lneigh == None and rneigh != None:
                lneigh = 1 - rneigh
            elif rneigh == None and lneigh != None:
                rneigh = 1 - lneigh

            avg = sum(arr[l:r+1]) / (r-l+1)
            amt1 = max(0, min(r-l+1, round((r-l+1) * avg)))

            if rneigh == 1:
                for i in range(r, r-amt1, -1):
                    arr[i] = 1
                for i in range(r-amt1, l-1, -1):
                    arr[i] = 0
            else:
                for i in range(l, l+amt1):
                    arr[i] = 1
                for i in range(l+amt1, r+1):
                    arr[i] = 0

        for i in range(len(arr)):
            arr[i] = round(arr[i])

        return arr

    #inputs este un vector gen [[steer -> [], push_up -> [], push_down -> []], ... ]
    #len(inputs) == LEFT_SHIFTS + RIGHT_SHIFTS + 1
    #len(percentages) == in cate itv ai vrut tu sa spargi linia de timp
    #percentages = un vector de vectori cu cum te-ai decis sa imparti procentele in fiecare interval acum
    #percentages[k] = un vector cu LEFT_SHIFTS + RIGHT_SHIFTS + 1 procente
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
            sol[j] = self.selective_round(sol[j])

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
                    percentages.append(local_perc) #?? daca nu merge da deepcopy si aici
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
                    if self.strat == "A":
                        diff = abs(local_perc[i + self.LEFT_SHIFTS] - self.changed_percentages[i + self.LEFT_SHIFTS])
                        if diff == 0:
                            gradients.append(0.0)
                        else:
                            gradients.append((scores[i + self.LEFT_SHIFTS] - self.current_run_score) / diff)

                        gradients[-1] = max(-self.percentage_increase, min(self.percentage_increase, gradients[-1]))
                    elif self.strat == "B":
                        gradients.append(scores[i + self.LEFT_SHIFTS] - self.current_run_score)

                if self.strat == "A":
                    for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS):
                        local_perc[i + self.LEFT_SHIFTS] += self.kept_change * gradients[i + self.LEFT_SHIFTS]
                elif self.strat == "B":
                    best_improvement = (None, 0.0)
                    for i in range(-self.LEFT_SHIFTS, self.RIGHT_SHIFTS + 1):
                        if gradients[i + self.LEFT_SHIFTS] > best_improvement[1]:
                            best_improvement = (i, gradients[i + self.LEFT_SHIFTS])

                    if best_improvement[0] == None:
                        self.epochs_since_last_improvement += 1
                        print(f"Currently {self.epochs_since_last_improvement} epochs with no improvement.")
                        if self.epochs_since_last_improvement % self.max_epochs_no_improvement == 0:
                            if self.percentage_increase < 1:
                                self.percentage_increase += self.percentage_increase_per_fix
                                print(f"Changed percentage increase to {self.percentage_increase}.")
                            else:
                                self.strat = "A"
                                self.percentage_increase = 2 * self.percentage_increase_per_fix
                                print("Changed to strat A.")
                    else:
                        local_perc[best_improvement[0] + self.LEFT_SHIFTS] += self.percentage_increase
                        print(f"Improved interval no. {self.curr_itv} on pos. {best_improvement[0]} with grad. {best_improvement[1]} (score {scores[best_improvement[0] + self.LEFT_SHIFTS]}, current run score {self.current_run_score}); unnormalized new perc. {local_perc[best_improvement[0] + self.LEFT_SHIFTS]}!")
                        print(f"Now local_perc is {local_perc}.")

                        self.current_run_score = scores[best_improvement[0] + self.LEFT_SHIFTS]
                        #actualizez aici scorul
                        self.epochs_since_last_improvement = 0
                    pass
                else:
                    print(f"No known strat named {self.strat}!")
                    assert(False)

                local_perc = self.normalize_percentages(local_perc)
                self.intervals[self.curr_itv][1] = copy.deepcopy(local_perc)
                #!!! doar procentajele trebuie schimbate la sfarsitul reprizei
                #input_arrays ramane LA FEL intotdeauna

                print(f"Finished epoch no. {self.epoch_count} on interval {self.curr_itv}!")

                client.empty_stack()
                self.epoch_count += 1
                self.curr_itv = self.curr_itv+1 if self.curr_itv+1 < self.interval_bounds[1] else self.interval_bounds[0]

                if self.strat == "A": #trebuie sa recalculez scorul curent
                    self.have_current_run = False
                elif self.strat == "B":
                    pass #deja am actualizat current_run_score in else mai sus

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
