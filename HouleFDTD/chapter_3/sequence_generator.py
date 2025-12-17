import random

choices = [
    "TM program",
    "TM program with PML added",
    "TM program with plane wave source",
    "TM simulation of a plane wave source impinging on a dielectric cylinder analysis using fourier transform",
]

while len(choices) > 0:
    choice = random.choices(choices)
    choices.remove(choice[0])
    if len(choices) > 0:
        input(f"{choice[0]}")
    else:
        print(choice[0])
        break
