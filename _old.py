import numpy as np
import matplotlib.pyplot as plt
import socket

##TODO: Dodać losową wysokość Z
##TODO: Dodać różne trajektorie, nie tylko z lewej do prawej
### TODO: Trajektoria wyliczana na podstawie początkowego seeda z IP maszyny i numeru iteracji, żeby była odtwarzalna
### TODO: zapis postępu do pliku co x iteracji (10k np)
##TODO: Naprawić błąd w trajektoriach przy użyciu punktów (docelowo ma być 1/8 pierścienia)


##obecna prędkość to przesunięcie o vx, vy względem poprzedniego punktu
##max_velocity to maksymalna prędkość używana do określenia liczby wektorów 2D


# USE_POINTS = True  # Czy używać punktów z bryły

# Parametry
Rpix = 200  # promień kamery w pikselach
znany_max_pix_velocity = 2  # to jest wyznaczane na podstawie max_pix_velocity

# # Parametry siatki
# X_RESOLUTION = 100
# Y_RESOLUTION = 100
# Z_RESOLUTION = 10
# MID_CUTOFF = 0.2  # Pomiń środek o daną część od środka (0.0 - 1.0)

MAX_TRAJECTORIES = 10000  # Maksymalna liczba trajektorii
MAX_SEGMENTS_PER_TRAJECTORY = 10  # Maksymalna liczba segmentów na trajektorii

IP_STR = socket.gethostbyname(socket.gethostname())
np.random.seed(int(IP_STR.replace(".", "")))  # Seed do losowania trajektorii

# Parametry bryły
radius = 1000  # Promień cylindra i stożka
height = 1000  # Wysokość cylindra i stożka
# cone_height = 1000  # Wysokość stożka
cone_bottom_radius = 20  # Promień podstawy stożka (na dole)


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

# if USE_POINTS:

#     # Tworzenie chmury punktów
#     x = np.linspace(-radius, radius, X_RESOLUTION)
#     y = np.linspace(-radius, radius, Y_RESOLUTION)
#     z = np.linspace(0, height, Z_RESOLUTION)
#     X, Y, Z = np.meshgrid(x, y, z)

#     # Obliczanie odległości w płaszczyźnie XY
#     distance = np.sqrt(X**2 + Y**2)

#     # Maska dla punktów w cylindrze
#     cylinder_mask = (distance <= radius) & (Z >= 0) & (Z <= height)

#     # Obliczanie promienia stożka w danym Z
#     cone_radius = radius - (radius - cone_bottom_radius) * (height - Z) / cone_height
#     cone_mask = (Z >= height - cone_height) & (Z <= height) & (distance <= cone_radius)

#     # Punkt w cylindrze, ale nie w stożku
#     points_mask = cylinder_mask & ~cone_mask

#     # Przypisanie wartości
#     values = np.zeros_like(X, dtype=bool)  # Wypełnienie macierzy wartościami False
#     values[points_mask] = True  # Przypisanie wartości True w miejscach z maską

#     # Filtracja punktów o wartości 1
#     X_filtered = X[points_mask]
#     Y_filtered = Y[points_mask]
#     Z_filtered = Z[points_mask]

#     if False:  # Wizualizacja całej chmury punktów
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection="3d")
#         scatter = ax.scatter(
#             X_filtered, Y_filtered, Z_filtered, c="black", s=5, alpha=0.25
#         )

#         # Ustawienia osi
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_box_aspect([1, 1, 1])  # Proporcje wykresu
#         plt.show()

#     # Filtracja punktów tylko w kącie od 0 do 45 stopni (1/8 okręgu)
#     angle_mask = (np.arctan2(Y_filtered, X_filtered) >= 0) & (
#         np.arctan2(Y_filtered, X_filtered) <= np.pi / 4
#     )
#     X_filtered = X_filtered[angle_mask]
#     Y_filtered = Y_filtered[angle_mask]
#     Z_filtered = Z_filtered[angle_mask]

#     if False:  # Wizualizacja wyciętej chmury punktów
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection="3d")
#         scatter = ax.scatter(
#             X_filtered, Y_filtered, Z_filtered, c="black", s=5, alpha=0.25
#         )

#         # Ustawienia osi
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_box_aspect([1, 1, 1])  # Proporcje wykresu
#         plt.show()

# Trajektorie
plt.figure(1)
plt.axis("equal")
plt.grid()

max_pix_velocity = 0

# lower_bound = -Y_RESOLUTION * MID_CUTOFF
# upper_bound = Y_RESOLUTION * MID_CUTOFF


# for y in range(-Y_RESOLUTION, Y_RESOLUTION):
#     if y > lower_bound and y < upper_bound:
#         print(f"y iteracja: {y+Y_RESOLUTION}/{Y_RESOLUTION*2} - pomijam")
#         continue
#     print(f"y iteracja: {y+Y_RESOLUTION}/{Y_RESOLUTION*2}")

#     xyzT = None
#     if USE_POINTS:
#         # Trajektoria dla punktów w cylindrze
#         xyzT = np.array([[x / 2, y, 1] for x in X_filtered])
#     else:
#         # Trajektoria dla wszystkich punktów
#         xyzT = np.array([[x / 2, y, 1] for x in range(-X_RESOLUTION, X_RESOLUTION)])


# losuj trajektorie z listy wektorów 2D
# poruszaj się po trajektorii do momentu aż nie wyjdziesz poza okrąg albo nie trafisz na obszar stożka albo nie osiągniesz górnego limitu trajektorii


outside_circle = 0
inside_cone_circle = 0
inside_cone = 0


def is_outside_area(xyz) -> bool:
    global outside_circle, inside_cone_circle, inside_cone
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
        > cone_bottom_radius + (radius - cone_bottom_radius) * z / height
    ):  # jeśli jest wewnątrz odwróconego stożka
        inside_cone += 1
        return True
    return False


max_trajectory_length = 0
for _ in range(MAX_TRAJECTORIES):

    xyzT = np.zeros((MAX_SEGMENTS_PER_TRAJECTORY, 3))

    z = np.random.uniform(
        0, height * 0.99
    )  # nie losuj z całej wysokości, dla granicznych przypadków następuje nieprawidłowe zachowanie

    # losuj koordynaty na obwodzie okręgu
    point_angle = np.random.rand() * np.pi * 0.25  # losuj kąt 0-45*
    xyzT[0] = [
        radius * np.cos(point_angle),
        radius * np.sin(point_angle),
        z,
    ]
    trajectory_x, trajectory_y = wektory2D[np.random.randint(len(wektory2D))]

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


print(f"total_max_pix_velocity: {max_pix_velocity}")
print(f"max_trajectory_length: {max_trajectory_length}")

print(
    f"trajectories ended due to: \n outside_circle: {outside_circle} \n inside_cone_circle: {inside_cone_circle} \n inside_cone: {inside_cone}"
)

plt.show()
