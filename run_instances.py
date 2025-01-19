import os


vectors = ["-1 -1", "-1 0", "-1 1", "0 -1", "0 1", "1 -1", "1 0", "1 1"]

instances_per_vector = 3


command = "screen -dmS {} python3 main.py --velocity_vector {} --max_trajectories 0 --additional_seed {}"


for vector in vectors:
    # run screen with current vector
    for i in range(instances_per_vector):
        os.system(command.format(f"vector_{vector.replace(' ', '_')}_{i}", vector, i))
