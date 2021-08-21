This file describes how to get the program to run on a map of your choice.

* the code runs with `TMI 1.0.6`, I don't know if newer versions will run. There appears to be a bug linked to older processors (in the simulation step if the processor doesn't compute the next batch of simulations fast enough, the game has to redo the last simulation to let the processor catch up; the bug involves the processor thinking it computed the next batch, but not really finishing it. `6th` gen laptop `i5` works ok, but `1st` gen doesn't)

* you also need the `pygbx` python package (https://github.com/donadigo/pygbx)

* get your `".Replay"` file that is compatible with `TMN/TMUF`

* if your game version doesn't have the map the replay is supposed to work on, also get the `".Challenge"` map file.

* put both the Replay and the map file (if needed) in their places in the `"Trackmania/Tracks/Challenges"` and `"Replays"` folders.

* go to the `"Configs/"` folder. Copy `"TMUF_D1.txt"` in the same directory and change its name to `"B.txt"` for example. Set `HUMAN_TIME` to your replay's time in ms. `CUTOFF_TIME` should be `HUMAN_TIME` rounded up to the next second. A good value for `INTERVAL_COUNT` is `CUTOFF_TIME / 1000`. `INTERVAL_BOUND_LO` should be 0, and `INTERVAL_BOUND_HI` should be equal to `INTERVAL_COUNT`. Each left shift allows the program to consider pressing inputs earlier by `10` ms, each right shift allows the program to consider pressing inputs later by `10` ms. `15` is a good value for both. More intervals allow for more distinct opportunities to left/right shift the inputs. One interval per second is fine for a start.

* modify the contents of the `"_CURRENT_MAP_NAME.txt"` so that it only contains `"B.txt"` or what you named your file.

* you now need to convert the `".Replay"` to a `TMI` script format. Go to `"tm_data_reader/Replays/selected"` and move out the existing replays somewhere else (`"tm_data_reader/Replays"`)

* leave your replay by itself in the `"selected"` folder.

* while in the `"tm_data_reader"` folder, run `"python generate_input_file.py Replays/selected"`

* a `.txt` file should appear in the `"tm_data_reader"` folder. Check if it's not empty; also check if you can load it into `TMI` and see the controls ingame.

* move the `.txt` file into `"tm_data_reader/Converted-txt/tas"`. Leave it there by itself, move everything else in `"Converted-txt"`.

* if your replay was done with keyboard inputs, you need to convert them to analog inputs. While in the `"tm_data_reader"` folder, run `"python convert_press_to_steer.py Converted-txt/tas/"`. A converted file named `"converted_to_steer.txt"` should appear in `"tm_data_reader/Converted-txt/tas"`. Get rid of the old one and name the new one to your liking.

* you now need to convert the `TMI` script to a format that is understandable by the program. While in the `"tm_data_reader"` folder, make sure that the `"Processed-inputs"` folder EXISTS and is empty.

* While in the `"tm_data_reader"` folder, run `"python main.py Converted-txt/tas/ Processed-inputs/"`. The `"Processed_inputs"` folder should have `LEFT_SHIFTS` + `RIGHT_SHIFTS` + 1 files in it (values from the `.txt` in `"Configs/"`).

* Now start the game. Go to the `"TMInterfaceClientPython-master"` folder and run the program: `"python mlmain.py"`. If everything is ok, you should get some messages stating the in-game connection and the read `"Configs/"` parameters. You should leave the game as a small window to be able to check the console messages as well.

* Go to `"Edit Replays"`, find your replay, `"Launch"`, then click `"Validate"`.

* You will find all improvements in the `"TMInterfaceClientPython-master/Processed-outputs/"` folder. The first improvement is just the original replay. Have fun!

* You can mess with the learning rate by modifying the `PERCENTAGE_INCREASE` and `PERCENTAGE_INCREASE_PER_FIX` parameters in the `"Configs/"` folder.