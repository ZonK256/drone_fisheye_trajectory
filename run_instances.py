import os
import datetime

instances = 20
trajecotries_per_instance = 100_000
secs_to_sleep = 1400
trajectory = "squiggle"

command = "screen -dmS {} python3 main.py --max_trajectories {} --additional_seed {} --trajectory {}"

if __name__ == "__main__":
    while True:
        print(f"Starting new batch at {datetime.datetime.now()}")

        for i in range(instances):
            seed = i
            print(f" Running instance {i}")
            screen_name = f"instance_{i}"
            os.system(command.format(screen_name, trajecotries_per_instance, seed))
        os.system(f"sleep {secs_to_sleep}")
