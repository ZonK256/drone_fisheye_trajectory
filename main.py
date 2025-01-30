import numpy as np
import matplotlib.pyplot as plt
import socket
import argparse
import time

start_time = time.time()

# Argument parser
parser = argparse.ArgumentParser(description="Fisheye trajectory simulation")
parser.add_argument(
    "--max_trajectories",
    type=int,
    default=100000,
    help="Number of trajectories to generate, (default: 10000, if 0 then infinite)",
)
parser.add_argument(
    "--additional_seed",
    type=int,
    default=0,
    help="Additional seed for random number generator, for multiple runs with the same velocity vector",
)

### PARAMETRY ###
Rpix = 128  # promień kamery w pikselach

# Parametry bryły
radius = 2500  # Promień cylindra i stożka
height = 2500  # Wysokość cylindra i stożka
# cone_height = 1000  # Wysokość stożka
cone_bottom_radius = 0  # Promień podstawy stożka (na dole)

AUTOSAVE_INTERVAL = 5000  # Co ile trajektorii wypisać informację o czasie

args = parser.parse_args()

wektory2D = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

output_matrix = np.zeros((Rpix * 2 + 1, Rpix * 2 + 1, 9), dtype=int)

max_trajectories = 0 if args.max_trajectories < 0 else args.max_trajectories
file_suffix = 0 if not args.additional_seed else args.additional_seed


# Funkcja do zapisania stanu generatora do pliku (lub pamięci)
def save_rng_state(rng, filepath):
    state = rng.__getstate__()
    np.save(filepath, state)


# Funkcja do wczytania stanu generatora z pliku (lub pamięci)
def load_rng_state(filepath):
    state = np.load(filepath, allow_pickle=True).item()
    rng = np.random.default_rng()
    rng.__setstate__(state)
    return rng


def cart2fish_image(x, y, z) -> np.array:
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    elevation = np.pi / 2 - elevation
    fi_max = np.pi  # fisheye FoV in radians
    r_norm = elevation / (fi_max / 2)
    Px = r_norm * np.cos(azimuth)
    Py = r_norm * np.sin(azimuth)

    pixel_x = int(Px * Rpix + Rpix)
    pixel_y = int(Py * Rpix + Rpix)

    return pixel_x, pixel_y


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def fisheye_trajectory(xyz_trajectory: np.array) -> np.array:
    az, el, r = cart2sph(
        xyz_trajectory[:, 0], xyz_trajectory[:, 1], xyz_trajectory[:, 2]
    )
    elm = np.pi / 2 - el
    fi_max = np.pi  # fisheye FoV in radians
    r_norm = elm / (fi_max / 2)
    Px = r_norm * np.cos(az)
    Py = r_norm * np.sin(az)

    return np.column_stack((Px, Py))


outside_circle = 0
inside_cone_circle = 0
inside_cone = 0
outside_angle = 0


def is_outside_area(xyz) -> bool:
    global outside_circle, inside_cone_circle, inside_cone, outside_angle
    x, y, z = xyz
    distance_from_center = np.sqrt(x**2 + y**2)

    # is_test_mode = args.test_mode

    if distance_from_center > radius:  # poza okręgiem
        outside_circle += 1
        return True
    if (
        distance_from_center < cone_bottom_radius
    ):  # wewnątrz stożka (wewnętrzny walec dla uproszczenia)
        inside_cone_circle += 1
        return True
    if (
        distance_from_center
        < cone_bottom_radius + (radius - cone_bottom_radius) * z / height
    ):  # jeśli jest wewnątrz odwróconego stożka
        inside_cone += 1
        return True
    return False


valid_vectors = 0
invalid_vectors = 0


def simulate_trajectory():
    global output_matrix, invalid_vectors, valid_vectors
    angle = rng.random() * np.pi * 2

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = rng.uniform(0, height)

    x_velocity = rng.uniform(-1, 1)
    y_velocity = rng.uniform(-1, 1)

    new_drone_image_position = cart2fish_image(x, y, z)
    previous_drone_image_position = new_drone_image_position

    # print(f"Starting point: ({x}, {y}, {z}), velocity: ({x_velocity}, {y_velocity})")
    x += x_velocity
    y += y_velocity
    while not is_outside_area((x, y, z)):
        x += x_velocity
        y += y_velocity
        # print(f"Moving to: ({x}, {y}, {z})")
        new_drone_image_position = cart2fish_image(x, y, z)
        x_diff = new_drone_image_position[0] - previous_drone_image_position[0]
        y_diff = new_drone_image_position[1] - previous_drone_image_position[1]

        if (x_diff, y_diff) in wektory2D:
            output_matrix[
                new_drone_image_position[0],
                new_drone_image_position[1],
                wektory2D.index((x_diff, y_diff)),
            ] += 1
            valid_vectors += 1

            previous_drone_image_position = new_drone_image_position

        else:
            print(
                f"Invalid direction, velocity vector out of bounds ({x_diff}, {y_diff})"
            )
            invalid_vectors += 1
            break


def generate_trajectories_chunk(iterations=AUTOSAVE_INTERVAL):
    global output_matrix
    for _ in range(iterations):
        simulate_trajectory()


def autosave_progress():
    # zapisz stan generatora
    save_rng_state(rng, f"rng_{file_suffix}.npy")
    # zapisz macierz wyjściową
    np.save(f"matrix_{file_suffix}.npy", output_matrix)


def generate_trajectories() -> np.array:

    if max_trajectories == 0:
        print("Running infinite mode, press Ctrl+C to stop")
        i = 1
        while True:
            generate_trajectories_chunk()
            autosave_progress()
            print(
                f"Calculated {(i * AUTOSAVE_INTERVAL):_} trajectories in {(time.time() - start_time):.2f} seconds"
            )
            i += 1

    else:
        print(f"Running for {max_trajectories:_} trajectories")
        for i in range(0, max_trajectories, AUTOSAVE_INTERVAL):
            iterations_chunk = min(AUTOSAVE_INTERVAL, max_trajectories - i)
            generate_trajectories_chunk(iterations_chunk)
            autosave_progress()
            print(
                f"Calculated {i + iterations_chunk:_}/{max_trajectories:_} trajectories in {(time.time() - start_time):.2f} seconds"
            )

    print(
        f"trajectories ended due to: \n outside_circle: {outside_circle} \n inside_cone_circle: {inside_cone_circle} \n inside_cone: {inside_cone}"
    )

    print(
        f"calculated {max_trajectories} trajectories in {(time.time() - start_time):.2f} seconds"
    )


def load_or_initialize():
    global output_matrix, rng
    try:
        latest_file = f"matrix_{file_suffix}.npy"
        output_matrix = np.load(latest_file)
        print(f"Loaded output matrix {latest_file}")
    except (FileNotFoundError, ValueError):
        print("No output matrix file found, initializing new output matrix")
        output_matrix = np.zeros((Rpix * 2 + 1, Rpix * 2 + 1, 9), dtype=int)

    try:
        rng = load_rng_state(f"rng_{file_suffix}.npy")
        print("Loaded rng state from file")
    except FileNotFoundError:
        print("No rng state file found, generating new rng state")
        IP_STR = socket.gethostbyname(socket.gethostname())
        rng = np.random.default_rng(
            seed=int(IP_STR.replace(".", "") + str(args.additional_seed))
        )


if __name__ == "__main__":
    load_or_initialize()
    generate_trajectories()

    print(f"Valid vectors: {valid_vectors:_}")
    print(f"Invalid vectors: {invalid_vectors:_}")
