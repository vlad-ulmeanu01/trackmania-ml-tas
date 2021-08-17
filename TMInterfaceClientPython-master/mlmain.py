from tminterface.structs import Event
from tminterface.interface import TMInterface
from tminterface.client import Client
import sys
import signal
import time
import copy
import decimal
import operator

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


class MainClient(Client):
    def __init__(self):
        self.GAP_TIME = 10 #ms
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2

        config = get_hyperparams({"CUTOFF_TIME", "HUMAN_TIME"})
        self.CUTOFF_TIME = int(config["CUTOFF_TIME"])
        self.HUMAN_TIME = int(config["HUMAN_TIME"])
        print(f"MainClient __init__: CUTOFF_TIME {self.CUTOFF_TIME}, HUMAN_TIME {self.HUMAN_TIME}")

        self.processed_output_dir = "Processed-outputs/output_"

        self.last_time_in_sim_step = 0
        self.remembered_state = None
        self.did_race_finish = False
        self.best_fitness_score, self.current_fitness_score = None, None

        #acest client este un fel de cutie neagra care trebuie sa primeasca niste inputuri, sa le ruleze
        #si sa returneze scorurile pentru inputuri
        #cand nu are nimic de rulat (i.e. self.is_client_redoing == True), clientul nu face nimic
        #cand primeste >= 1 inputuri, da reset la cursa, calculeaza pana le face pe toate etc
        self.should_client_work = False #variabila care indica DACA AR trebui sa ruleze clientul (ie e ceva in stiva)
        self.is_client_redoing = True #variabila care indica ralanti pentru simulari
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
        pass

    def on_simulation_begin(self, iface: TMInterface):
        iface.remove_state_validation()
        self.did_race_finish = False

    def on_simulation_step(self, iface: TMInterface, time: int):
        self.last_time_in_sim_step = time - 2610

        if self.last_time_in_sim_step == -10:
            self.remembered_state = iface.get_simulation_state()
            #print("(on_simulation_step) remembered state!")

        if self.last_time_in_sim_step >= self.CUTOFF_TIME:
            self.on_checkpoint_count_changed(iface, -1, -1)
            #trebuie sa fortez terminarea cursei.. jocul se asteapta sa valideze ceva ce a terminat cursa

        if self.did_race_finish:
            #intotdeauna repet ultimul EventBufferData pana cand este modificat in on_checkpoint_count_changed
            #daca am ajuns aici sigur am trecut prin on_checkpoint_count_changed si am facut modificarile
            #la buffer daca trebuiau facute.
            iface.rewind_to_state(self.remembered_state)
            self.did_race_finish = False

        pass

    def on_simulation_end(self, iface: TMInterface, result: int):
        print("All simulations finished?? You weren't supposed to see this you know")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current < target:
            return

        self.did_race_finish = True
        iface.prevent_simulation_finish()

        if not self.is_client_redoing:
            self.current_fitness_score = self.fitness_function(iface)

            if self.best_fitness_score == None or self.best_fitness_score < self.current_fitness_score:
                oname = self.processed_output_dir + str(time.time()).replace('.', '_') + "_" + str(self.last_time_in_sim_step) + ".txt"
                write_processed_output(oname, self.input_stack[self.input_stack_index][0], self.GAP_TIME)

                print(f"wrote to {oname}! (new score {self.current_fitness_score} vs old score {self.best_fitness_score})")
                self.best_fitness_score = self.current_fitness_score

            self.process_input_stack(iface)

        #daca nu am intrat in iful de mai sus, nu am avut ce sa rulez pentru un timp, asa ca am rulat ultima
        #chestie din nou.
        #(self.is_client_redoing, self.should_client_work) in ((False, True), (True, False), (True, True))

        #self.is_client_redoing, self.should_client_work sunt actualizate in self.process_input_stack mai sus
        #daca e cazul
        if self.should_client_work:
            #inseamna ca (optional mai) am ceva in stiva
            self.is_client_redoing = False
            self.write_input_array_to_EventBufferData(iface, self.input_stack[self.input_stack_index][0])
            pass

        pass

    def on_laps_count_changed(self, iface, current: int):
        pass

    def compute_speed_for_sim(self, iface: TMInterface):
        vx, vy, vz = iface.get_simulation_state().get_velocity()
        decimal.getcontext().prec = 3
        return round(float(decimal.Decimal(vx * vx + vy * vy + vz * vz).sqrt()) * 3.6, 3)

    def fitness_function (self, iface: TMInterface):
        score = 0
        score += 250 * (self.HUMAN_TIME - self.last_time_in_sim_step)
        score += self.compute_speed_for_sim(iface)

        return score

    def add_input_array_to_stack(self, input):
        self.input_stack.append([copy.deepcopy(input), None])
        self.input_stack_index += 1
        if not self.should_client_work:
            self.should_client_work = True

    #se apeleaza la sfarsitul unei simulari
    #(la sfarsitul on_checkpoint_count_changed, trb actualizat self.current_fitness_score)
    #!! aici trb sa apelezi write_input_array_to_EventBufferData (daca mai ai ceva in stiva)
    def process_input_stack (self, iface: TMInterface):
        self.input_stack[self.input_stack_index][1] = self.current_fitness_score
        self.input_stack_index -= 1
        if self.input_stack_index < 0:
            self.should_client_work = False
            self.is_client_redoing = True
            #print("(process_input_stack) should_client_work = False, is_client_redoing = True!")

    def empty_stack (self):
        self.input_stack = []
        self.input_stack_index = -1
        pass

    def debug_print_EventBufferData(self, iface: TMInterface):
        print(f"Printing Event Buffer Data")

        ebd = iface.get_event_buffer()

        print(f"ebd.events_duration: {ebd.events_duration}")
        print(f"ebd.control_names: {ebd.control_names}")
        print(f"len(ebd.events): {len(ebd.events)}")
        print(f"type(ebd.events[-1]): {type(ebd.events[-1])}")

        ebd.control_names[0] = "buffer_end"
        for ev in ebd.events:
            print(f"time {ev.time - 100010} opcode {ev.name_index} ({ebd.control_names[ev.name_index]}) value {ev.analog_value} bvalue {ev.binary_value}")

        pass

    def write_input_array_to_EventBufferData(self, iface: TMInterface, input_array: list):
        ebd = copy.deepcopy(iface.get_event_buffer())
        ebd.events_duration = self.CUTOFF_TIME
        ebd.control_names[0] = "buffer_end"

        def make_event(event_type: int, time: int, value_type: str, value: int) -> Event:
            ev = Event(100010 + time, 0)
            ev.name_index = event_type
            if value_type == "binary":
                ev.binary_value = value
            elif value_type == "analog":
                ev.analog_value = value
            else:
                print(f"(make_event) no such value_type as {value_type}!")
                assert(False)
            return ev

        ebd.events = []
        ebd.events.append(make_event(5, -2590, "binary", 0)) #"Respawn"
        ebd.events.append(make_event(4, -10, "binary", 1)) #"_FakeIsRaceRunning"
        ebd.events.append(make_event(0, self.CUTOFF_TIME, "binary", 1)) #"buffer_end"

        #steer, push_up, push_down events
        for arr, event_type, value_type in ((input_array[self.IND_STEER], 1, "analog"),
                                            (input_array[self.IND_PUSH_UP], 2, "binary"),
                                            (input_array[self.IND_PUSH_DOWN], 3, "binary")):
            ebd.events.append(make_event(event_type, 0, value_type, arr[0]))
            for i in range(1, len(arr)):
                if arr[i] != arr[i-1]:
                    ebd.events.append(make_event(event_type, i * self.GAP_TIME, value_type, arr[i]))

        ebd.events.sort(key = operator.attrgetter("time"), reverse = True)
        iface.set_event_buffer(ebd)


class ML():
    def __init__(self):
        self.GAP_TIME = 10
        self.IND_STEER, self.IND_PUSH_UP, self.IND_PUSH_DOWN = 0, 1, 2

        config = get_hyperparams({"CUTOFF_TIME", "LEFT_SHIFTS", "RIGHT_SHIFTS", "INTERVAL_COUNT",
                                  "INTERVAL_BOUND_LO", "INTERVAL_BOUND_HI", "PERCENTAGE_INCREASE",
                                  "PERCENTAGE_INCREASE_PER_FIX"})

        self.CUTOFF_TIME = int(config["CUTOFF_TIME"])
        self.LEFT_SHIFTS, self.RIGHT_SHIFTS = int(config["LEFT_SHIFTS"]), int(config["RIGHT_SHIFTS"])

        self.interval_count = int(config["INTERVAL_COUNT"])
        self.intervals = self.make_intervals(self.interval_count)
        self.interval_bounds = (int(config["INTERVAL_BOUND_LO"]), int(config["INTERVAL_BOUND_HI"]))
        #se lucreaza pe intervalele [.., ..)

        self.curr_itv = self.interval_bounds[0] #indexul intervalului pe care se lucreaza momentan
        self.percentage_increase = float(config["PERCENTAGE_INCREASE"])
        self.percentage_increase_per_fix = float(config["PERCENTAGE_INCREASE_PER_FIX"])

        print(f"ML __init__: CUTOFF_TIME {self.CUTOFF_TIME}, LEFT_SHIFTS {self.LEFT_SHIFTS}, RIGHT_SHIFTS {self.RIGHT_SHIFTS}, INTERVAL_COUNT {self.interval_count}, INTERVAL_BOUND_LO {self.interval_bounds[0]}, INTERVAL_BOUND_HI {self.interval_bounds[1]}, PERCENTAGE_INCREASE {self.percentage_increase}, PERCENTAGE_INCREASE_PER_FIX {self.percentage_increase_per_fix}")

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
                            if self.percentage_increase + self.percentage_increase_per_fix > 0 and self.percentage_increase + self.percentage_increase_per_fix < 1:
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
