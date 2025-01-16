import numpy as np
import matplotlib.pyplot as plt
import socket

# puścić liczenie
# 9 macierzy markowa 3x3 wyświetlone jako obrazki z gradientem kolorów reprezentującym częstość występowania danego stanu


# Parametry
Rpix = 200  # promień kamery w pikselach
znany_max_pix_velocity = 2  # to jest wyznaczane na podstawie max_pix_velocity

MAX_TRAJECTORIES = 10000  # Maksymalna liczba trajektorii
MAX_SEGMENTS_PER_TRAJECTORY = 50  # Maksymalna liczba segmentów na trajektorii

# Parametry bryły
radius = 1000  # Promień cylindra i stożka
height = 1000  # Wysokość cylindra i stożka
# cone_height = 1000  # Wysokość stożka
cone_bottom_radius = 100  # Promień podstawy stożka (na dole)


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


rng = np.random.default_rng(seed=42)
# sprawdź czy jest zapisany stan generatora
try:
    rng = load_rng_state("rng_state.npy")
    print("Loaded rng state from file")
except FileNotFoundError:
    print("No rng state file found, generating new rng state")
    IP_STR = socket.gethostbyname(socket.gethostname())
    rng = np.random.default_rng(seed=int(IP_STR.replace(".", "")))

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


# Generowanie wektorów 2D
wektory2D = []
for v1 in range(-znany_max_pix_velocity, znany_max_pix_velocity + 1):
    for v2 in range(-znany_max_pix_velocity, znany_max_pix_velocity + 1):
        wektory2D.append([v1, v2])

# Trajektorie
plt.figure(1)
plt.axis("equal")
plt.grid()

max_pix_velocity = 0

outside_circle = 0
inside_cone_circle = 0
inside_cone = 0
outside_angle = 0


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


max_trajectory_length = 0
for _ in range(MAX_TRAJECTORIES):

    xyzT = np.zeros((MAX_SEGMENTS_PER_TRAJECTORY, 3))

    z = (
        rng.random() * height * 0.99
    )  # nie losuj z całej wysokości, dla granicznych przypadków następuje nieprawidłowe zachowanie

    # losuj koordynaty na obwodzie okręgu
    point_angle = rng.random() * np.pi * 0.25  # losuj kąt 0-45*
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
    # print(xyzT)
    max_trajectory_length = max(max_trajectory_length, len(xyzT))

    if len(xyzT) > 400:
        print(f"{len(xyzT)} -> {xyzT[-10:]}\n")

    # Projekcja do kamery fisheye
    fT = fisheye_trajectory(xyzT)
    plt.plot(fT[:, 0], fT[:, 1])

    ft_int = np.round(fT * Rpix).astype(int)

    # Obliczenie maksymalnej prędkości w pikselach
    velocity_vector = np.sqrt(np.sum(np.diff(ft_int, axis=0) ** 2, axis=1))
    # print(f"current velocity: {velocity_vector.max()}")
    max_pix_velocity = max(max_pix_velocity, velocity_vector.max())

print(f"rng_state: {rng.bit_generator.state}")
save_rng_state(rng, "rng_state.npy")

print(f"total_max_pix_velocity: {max_pix_velocity}")
print(f"max_trajectory_length: {max_trajectory_length}")

print(
    f"trajectories ended due to: \n outside_circle: {outside_circle} \n inside_cone_circle: {inside_cone_circle} \n inside_cone: {inside_cone} \n trajectory_segment_limit: {MAX_TRAJECTORIES - outside_circle - inside_cone_circle - inside_cone - outside_angle}"
)

plt.show()
