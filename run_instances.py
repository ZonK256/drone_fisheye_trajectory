import os
import datetime

vectors = ["-1 -1", "-1 0", "-1 1", "0 -1", "0 1", "1 -1", "1 0", "1 1"]

instances_per_vector = 3
trajecotries_per_instance = 20_000
secs_to_sleep = 60


command = "screen -dmS {} python3 main.py --velocity_vector {} --max_trajectories {}  --additional_seed {}"
random_command = (
    "screen -dmS random python3 main.py --max_trajectories {}  --additional_seed {}"
)

if __name__ == "__main__":
    while True:
        print(f"Starting new batch at {datetime.datetime.now()}")

        for instance in range(instances_per_vector):
            # Run random instance
            print(f" Running random instance")
            os.system(random_command.format(trajecotries_per_instance, instance))
            for vector in vectors:
                print(f" Running [{vector}] instance {instance}")
                seed = instance
                screen_name = f"{vector.replace(' ', '_')}_{instance}"
                os.system(
                    command.format(screen_name, vector, trajecotries_per_instance, seed)
                )
            os.system(f"sleep {secs_to_sleep}")
