import numpy as np
import matplotlib.pyplot as plt
import socket
import argparse
import time

# puścić liczenie
# 9 macierzy markowa 3x3 wyświetlone jako obrazki z gradientem kolorów reprezentującym częstość występowania danego stanu

start_time = time.time()

# Argument parser
parser = argparse.ArgumentParser(description="Fisheye trajectory simulation")
parser.add_argument(
    "--velocity_vector",
    type=int,
    nargs=2,
    default=None,
    help="Specify the velocity vector as two integers, e.g., --velocity_vector 1 2",
)
parser.add_argument(
    "--test_mode",
    action="store_true",
    help="Run in test mode (hold C to generate new trajectory, press Q to quit)",
)
parser.add_argument(
    "--max_trajectories",
    type=int,
    default=10000,
    help="Number of trajectories to generate, (default: 10000, if 0 then infinite)",
)
parser.add_argument(
    "--additional_seed",
    type=int,
    default=0,
    help="Additional seed for random number generator, for multiple runs with the same velocity vector",
)

### PARAMETRY ###
Rpix = 1000  # promień kamery w pikselach
znany_max_pix_velocity = 1  # to jest wyznaczane na podstawie max_pix_velocity
# sqrt(2) => ~1.4.... najwyższa prędkość dla wektora 1,1


MIN_SEGMENTS_PER_TRAJECTORY = (
    5  # Minimalna liczba segmentów na trajektorii, jeżeli mniej to losujemy jeszcze raz
)
MAX_SEGMENTS_PER_TRAJECTORY = 2000  # Maksymalna liczba segmentów na trajektorii
# 2000 wyglądana wystarczająco dla pix_velocity 1

# Parametry bryły
radius = 1000  # Promień cylindra i stożka
height = 1000  # Wysokość cylindra i stożka
# cone_height = 1000  # Wysokość stożka
cone_bottom_radius = 100  # Promień podstawy stożka (na dole)

AUTOSAVE_INTERVAL = 5000  # Co ile trajektorii wypisać informację o czasie

args = parser.parse_args()

if args.velocity_vector:
    wektory2D = [args.velocity_vector]
else:
    # Generowanie wektorów 2D
    wektory2D = []
    for v1 in range(-znany_max_pix_velocity, znany_max_pix_velocity + 1):
        for v2 in range(-znany_max_pix_velocity, znany_max_pix_velocity + 1):
            wektory2D.append([v1, v2])

max_trajectories = 0 if args.max_trajectories < 0 else args.max_trajectories

file_suffix = (
    "random"
    if args.velocity_vector is None
    else f"{args.velocity_vector[0]}.{args.velocity_vector[1]}"
)
if args.additional_seed:
    file_suffix += f"_{args.additional_seed}"


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


# np.random.seed(int(IP_STR.replace(".", "")))  # Seed do losowania trajektorii


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def fisheye_trajectory(xyz_trajectory: np.array) -> np.array:
    # len_xyz = xyz_trajectory.shape[0]
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

    is_test_mode = args.test_mode

    if distance_from_center > radius:  # poza okręgiem
        outside_circle += 1
        if is_test_mode:
            print(
                f"End of trajectory due to outside circle (distance: {distance_from_center}, radius: {radius})"
            )
        return True
    if (
        distance_from_center < cone_bottom_radius
    ):  # wewnątrz stożka (wewnętrzny walec dla uproszczenia)
        inside_cone_circle += 1
        if is_test_mode:
            print(
                f"End of trajectory due to inside cone circle (distance: {distance_from_center}, cone_bottom_radius: {cone_bottom_radius})"
            )
        return True
    if (
        distance_from_center
        < cone_bottom_radius + (radius - cone_bottom_radius) * z / height
    ):  # jeśli jest wewnątrz odwróconego stożka
        inside_cone += 1
        if is_test_mode:
            print(
                f"End of trajectory due to inside cone (distance: {distance_from_center}, cone_bottom_radius: {cone_bottom_radius}, radius: {radius}, height: {height})"
            )
        return True
    return False


def generate_trajectory() -> np.array:
    len_xyz = 0

    while len_xyz < MIN_SEGMENTS_PER_TRAJECTORY:
        xyzT = np.zeros((MAX_SEGMENTS_PER_TRAJECTORY, 3))

        z = (
            rng.random() * height * 0.99
        )  # nie losuj z całej wysokości, dla granicznych przypadków następuje nieprawidłowe zachowanie

        # losuj koordynaty na obwodzie okręgu
        point_angle = rng.random() * np.pi * 2
        xyzT[0] = [
            radius * np.cos(point_angle),
            radius * np.sin(point_angle),
            z,
        ]

        trajectory_x, trajectory_y = wektory2D[rng.integers(0, len(wektory2D))]

        for i in range(1, MAX_SEGMENTS_PER_TRAJECTORY):
            xyzT[i] = [
                xyzT[i - 1][0] + trajectory_x,
                xyzT[i - 1][1] + trajectory_y,
                z,
            ]

            if is_outside_area(xyzT[i]):
                break

        # usuń z listy punkty 0,0,0
        xyzT = xyzT[~np.all(xyzT == 0, axis=1)]
        len_xyz = len(xyzT)

    return xyzT


def generate_trajectories_chunk(iterations=AUTOSAVE_INTERVAL):
    global output_matrix

    for _ in range(iterations):
        xyzT = generate_trajectory()
        # max_trajectory_length = max(max_trajectory_length, len(xyzT))

        # Projekcja do kamery fisheye
        fT = fisheye_trajectory(xyzT)
        plt.plot(fT[:, 0], fT[:, 1])

        ft_int = np.round(fT * Rpix).astype(int) + Rpix
        # Zliczanie punktów na macierzy wyjściowej

        output_matrix[ft_int[:, 0], ft_int[:, 1]] += 1


def autosave_progress():

    # zapisz stan generatora
    save_rng_state(rng, f"rng_{file_suffix}.npy")
    # zapisz macierz wyjściową
    np.save(f"matrix_{file_suffix}.npy", output_matrix)


def generate_trajectories() -> np.array:
    # max_pix_velocity = 0
    # output_matrix = np.zeros((Rpix * 2 + 1, Rpix * 2 + 1), dtype=int)
    max_trajectory_length = 0

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

        # # Obliczenie maksymalnej prędkości w pikselach
        # velocity_vector = np.sqrt(np.sum(np.diff(ft_int, axis=0) ** 2, axis=1))
        # # print(f"current velocity: {velocity_vector.max()}")
        # if len(velocity_vector) > 0:
        #     max_pix_velocity = max(max_pix_velocity, velocity_vector.max())

    # print(f"total_max_pix_velocity: {max_pix_velocity}")
    print(f"max_trajectory_length: {max_trajectory_length}")

    print(
        f"trajectories ended due to: \n outside_circle: {outside_circle} \n inside_cone_circle: {inside_cone_circle} \n inside_cone: {inside_cone}"
    )

    print(
        f"calculated {max_trajectories} trajectories in {(time.time() - start_time):.2f} seconds"
    )


def load_or_initialize():
    global output_matrix, rng
    try:
        output_matrix = np.load(f"matrix_{file_suffix}.npy")
        print("Loaded output matrix from file")
    except FileNotFoundError:
        print("No output matrix file found, initializing new output matrix")
        output_matrix = np.zeros((Rpix * 2 + 1, Rpix * 2 + 1), dtype=int)

    try:
        rng = load_rng_state(f"rng_{file_suffix}.npy")
        print("Loaded rng state from file")
    except FileNotFoundError:
        print("No rng state file found, generating new rng state")
        IP_STR = socket.gethostbyname(socket.gethostname())
        rng = np.random.default_rng(
            seed=int(IP_STR.replace(".", "")) + args.additional_seed
        )


def run_test_mode():
    fig = plt.figure()
    plt.axis("equal")
    plt.grid()
    plt.title(
        "Fisheye trajectory simulation (press C to generate new trajectory, Q to quit)"
    )

    # plot outside circle (radius, 30% opacity)
    circle = plt.Circle((0, 0), 1, color="r", alpha=0.3)
    fig.gca().add_artist(circle)

    running = True

    def on_key(event):
        global running
        if event.key == "q":
            running = False
            plt.close()
        elif event.key == "c":
            xyzT = generate_trajectory()
            ft = fisheye_trajectory(xyzT)
            plt.plot(ft[:, 0], ft[:, 1])
            plt.draw()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial trajectory
    xyzT = generate_trajectory()
    ft = fisheye_trajectory(xyzT)
    plt.plot(ft[:, 0], ft[:, 1])

    plt.show()


if __name__ == "__main__":
    load_or_initialize()

    if args.test_mode:
        run_test_mode()
    else:
        plt.axis("equal")
        plt.grid()
        generate_trajectories()
        plt.show()
