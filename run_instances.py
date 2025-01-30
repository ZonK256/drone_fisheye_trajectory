import os
import datetime

# vectors = ["-1 -1", "-1 0", "-1 1", "0 -1", "0 1", "1 -1", "1 0", "1 1"]

instances = 3
trajecotries_per_instance = 100_000
secs_to_sleep = 400


command = "screen -dmS {} python3 main.py --max_trajectories {} --additional_seed {}"

if __name__ == "__main__":
    while True:
        print(f"Starting new batch at {datetime.datetime.now()}")

        for i in range(instances):
            seed = i
            print(f" Running instance {i}")
            screen_name = f""
            os.system(command.format(screen_name, trajecotries_per_instance, seed))
        os.system(f"sleep {secs_to_sleep}")
