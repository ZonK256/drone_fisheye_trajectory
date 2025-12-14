import numpy as np
import matplotlib.pyplot as plt
import socket
import argparse
import time
import os

start_time = time.time()

# Argument parser
parser = argparse.ArgumentParser(description="Fisheye trajectory simulation")
parser.add_argument(
    "--max_trajectories",
    type=int,
    default=250,
    help="Number of trajectories to generate, (default: 250, if 0 then infinite)",
)
parser.add_argument(
    "--additional_seed",
    type=int,
    default=0,
    help="Additional seed for random number generator, for multiple runs with the same velocity vector",
)

parser.add_argument(
    "--trajectory",
    type=str,
    default="linear",
    choices=["linear", "squiggle", "scatter", "circular"],
    help="Type of trajectory to simulate (default: linear)",
)

parser.add_argument(
    "--test_mode",
    action="store_true",
    help="Enable test mode with trajectory showcase",
)

### PARAMETRY ###
Rpix = 128  # promień kamery w pikselach

# Parametry bryły
radius = 2500  # Promień cylindra i stożka
height = 2500  # Wysokość cylindra i stożka
MIN_HEIGHT = height * 0.05  # Minimalna wysokość drona
# cone_height = 1000  # Wysokość stożka
cone_bottom_radius = 0  # Promień podstawy stożka (na dole)
TRAJECTORY_MAX_SEGMENTS = 25000  # Maksymalna liczba segmentów trajektorii

AUTOSAVE_INTERVAL = 100  # Co ile trajektorii wypisać informację o czasie

WEKTORY2D = [  # możliwe wektory ruchu w pikselach na obrazie fishey, pomija symulację w przypadku przekroczenia dozwolonych wartości
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

### KONIEC PARAMETRÓW ###


args = parser.parse_args()

longest_trajectory = 0


output_matrix = np.zeros((Rpix * 2 + 1, Rpix * 2 + 1, 9), dtype=int)

max_trajectories = 0 if args.max_trajectories < 0 else args.max_trajectories
file_suffix = (
    f"{0 if not args.additional_seed else args.additional_seed}_{args.trajectory}"
)


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


# def cart2sph(x, y, z):
#     azimuth = np.arctan2(y, x)
#     elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
#     r = np.sqrt(x**2 + y**2 + z**2)
#     return azimuth, elevation, r


# def fisheye_trajectory(xyz_trajectory: np.array) -> np.array:
#     az, el, r = cart2sph(
#         xyz_trajectory[:, 0], xyz_trajectory[:, 1], xyz_trajectory[:, 2]
#     )
#     elm = np.pi / 2 - el
#     fi_max = np.pi  # fisheye FoV in radians
#     r_norm = elm / (fi_max / 2)
#     Px = r_norm * np.cos(az)
#     Py = r_norm * np.sin(az)

#     return np.column_stack((Px, Py))


outside_circle = 0
inside_cone_circle = 0
inside_cone = 0
outside_angle = 0
valid_vectors = 0
invalid_vectors = 0


def is_outside_area(xyz) -> bool:
    global outside_circle, inside_cone_circle, inside_cone, outside_angle
    x, y, z = xyz
    distance_from_center = np.sqrt(x**2 + y**2)

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


angle = 0

center_x, center_y, traj_radius = 0, 0, 0


def move_drone(x, y, z, x_velocity, y_velocity, z_velocity):
    drone_move_style = args.trajectory
    x += x_velocity
    y += y_velocity
    z += z_velocity
    z = max(MIN_HEIGHT, min(height, z))  # utrzymuj wysokość w granicach

    if drone_move_style == "linear":  # domyślna
        pass
    elif drone_move_style == "squiggle":  # utrzymuj jednorodną prędkość
        if rng.random() < 0.02:
            velocity_change = np.random.uniform(-0.1, 0.1)
            x_velocity += velocity_change
            y_velocity -= velocity_change
    elif drone_move_style == "scatter":  # losowa zmiana prędkości
        if rng.random() < 0.01:
            x_velocity = rng.uniform(-1, 1)
        if rng.random() < 0.01:
            y_velocity = rng.uniform(-1, 1)
        if rng.random() < 0.01:
            z_velocity = rng.uniform(-1, 1)
    elif drone_move_style == "circular":  # circle around center
        global angle, traj_radius, center_x, center_y
        # Zwiększ kąt i oblicz nowe położenie na okręgu
        delta_angle = 0.005  # im większe, tym szybciej "krąży"
        angle += delta_angle
        x = center_x + traj_radius * np.cos(angle)
        y = center_y + traj_radius * np.sin(angle)

    return x, y, z, x_velocity, y_velocity, z_velocity


def simulate_trajectory():
    global output_matrix, invalid_vectors, valid_vectors, center_x, center_y, traj_radius

    angle = rng.random() * np.pi * 2

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = rng.uniform(0, height)

    x_velocity = rng.uniform(-1, 1)
    y_velocity = rng.uniform(-1, 1)
    z_velocity = 0

    if args.trajectory == "circular":
        traj_radius = rng.uniform(radius * 0.5, radius * 0.9)
        center_angle = rng.uniform(0, 2 * np.pi)
        center_x = (radius - traj_radius) * np.cos(center_angle)
        center_y = (radius - traj_radius) * np.sin(center_angle)
        angle = rng.uniform(0, 2 * np.pi)
        x = center_x + traj_radius * np.cos(angle)
        y = center_y + traj_radius * np.sin(angle)

    # move one time to avoid sharp movements
    x, y, z, x_velocity, y_velocity, z_velocity = move_drone(
        x, y, z, x_velocity, y_velocity, z_velocity
    )

    new_drone_image_position = cart2fish_image(x, y, z)
    previous_drone_image_position = new_drone_image_position

    while not is_outside_area((x, y, z)) and current_segment < TRAJECTORY_MAX_SEGMENTS:
        current_segment += 1

        x, y, z, x_velocity, y_velocity, z_velocity = move_drone(
            x, y, z, x_velocity, y_velocity, z_velocity
        )
        # print(f"Moving to: ({x}, {y}, {z})")
        new_drone_image_position = cart2fish_image(x, y, z)
        x_diff = new_drone_image_position[0] - previous_drone_image_position[0]
        y_diff = new_drone_image_position[1] - previous_drone_image_position[1]

        if (x_diff, y_diff) in WEKTORY2D:
            output_matrix[
                new_drone_image_position[0],
                new_drone_image_position[1],
                WEKTORY2D.index((x_diff, y_diff)),
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

    rng_state_file = f"rng_{file_suffix}.npy"
    if os.path.exists(rng_state_file):
        rng = load_rng_state(rng_state_file)
        print("Loaded rng state from file")
    else:
        print("No rng state file found, generating new rng state")
        IP_STR = socket.gethostbyname(socket.gethostname())
        rng = np.random.default_rng(
            seed=int(IP_STR.replace(".", "") + str(args.additional_seed))
        )


def test_trajectory():
    global center_x, center_y, traj_radius, longest_trajectory

    z = rng.uniform(MIN_HEIGHT, height)
    x_velocity = rng.uniform(-1, 1)
    y_velocity = rng.uniform(-1, 1)
    z_velocity = 0
    angle = rng.uniform(0, 2 * np.pi)

    if args.trajectory == "circular":
        traj_radius = rng.uniform(radius * 0.5, radius * 0.9)
        center_angle = rng.uniform(0, 2 * np.pi)
        center_x = (radius - traj_radius) * np.cos(center_angle)
        center_y = (radius - traj_radius) * np.sin(center_angle)
        x = center_x + (traj_radius - 1e-3) * np.cos(angle)
        y = center_y + (traj_radius - 1e-3) * np.sin(angle)

    else:

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

    # Use move_crone instead of manual velocity assignment
    x, y, z, x_velocity, y_velocity, z_velocity = move_drone(
        x, y, z, x_velocity, y_velocity, z_velocity
    )
    points = [(x, y, z)]
    while not is_outside_area((x, y, z)) and len(points) < TRAJECTORY_MAX_SEGMENTS:
        x, y, z, x_velocity, y_velocity, z_velocity = move_drone(
            x, y, z, x_velocity, y_velocity, z_velocity
        )
        points.append((x, y, z))

    longest_trajectory = max(longest_trajectory, len(points))
    return np.array(points)


if __name__ == "__main__":

    if args.test_mode:
        load_or_initialize()
        trajectories = []

        for i in range(1, args.max_trajectories + 1):
            if i % 100 == 0:
                print(f"Generating test trajectory {i}/{args.max_trajectories}")
            traj = test_trajectory()
            trajectories.append(traj)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.set_title(f"Trajectory: {args.trajectory}")

        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)

        plt.show()
    else:
        load_or_initialize()
        generate_trajectories()
        print(f"Valid vectors: {valid_vectors:_}")
        print(f"Invalid vectors: {invalid_vectors:_}")

    print(f"Longest trajectory segments: {longest_trajectory}")
