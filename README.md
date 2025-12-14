# Drone Fisheye Trajectory

## Opis

Projekt do generowania trajektorii dronów z wykorzystaniem obrazów szerokokątnych (fisheye). Narzędzie umożliwia generowanie i analizę wybranych trajektorii, a następnie konwersję pozycji rzutu kartezjańskiego na współrzędne obrazu szerokokątnego.

## Cechy

- Generowanie trajektorii dronów
- Dobór parametrów symulacji i rozdzielczości obrazu wyjściowego
- 

## Instalacja

```bash
git clone https://github.com/ZonK256/drone_fisheye_trajectory.git
cd fisheye
pip install -r requirements.txt
```

## Użycie

python3 fisheye.py [-h] [--max_trajectories MAX_TRAJECTORIES] [--additional_seed ADDITIONAL_SEED] [--trajectory {linear,squiggle,scatter,circular}] [--test_mode]

Opcje:
    -h, --help            pokaż tę wiadomość pomocy i wyjdź
    --max_trajectories MAX_TRAJECTORIES
                                                Liczba trajektorii do wygenerowania (domyślnie: 250, jeśli 0 to nieskończenie)
    --additional_seed ADDITIONAL_SEED
                                                Dodatkowe ziarno dla generatora liczb losowych, dla wielu uruchomień z tym samym wektorem prędkości
    --trajectory {linear,squiggle,scatter,circular}
                                                Typ trajektorii do symulacji (domyślnie: linear)
    --test_mode           Włącz tryb testowy z prezentacją trajektorii na wykresie (domyślnie: False)

## Dodatki

output_matrix_renderer.py - Skrypt do renderowania macierzy wyjściowej jako obrazu.
run_instances.py - Skrypt do uruchamiania wielu instancji symulacji z różnymi parametrami.

## Wymagania

- Python 3.11+
- Matplotlib
- NumPy
