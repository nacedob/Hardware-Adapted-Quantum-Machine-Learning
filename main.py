import subprocess
from icecream import ic

JOBS = 7


common_str = ('python -m src.experiments.final_experiment --n_seeds=5  --layers_max=8 '
              '--layers_min=1 --tuning=True --dataset={dataset} '
              '--n_epochs=30 --load=True --trials=35 --optimizer=rms --n_jobs=7 --n_qubits=2 '
              '--realistic_gates=False --save_qnn=True --eqk=False --starting_seed=0')
comando = [
    common_str.format(dataset='shell'),
    common_str.format(dataset='sinus3d'),
    common_str.format(dataset='digits'),
    common_str.format(dataset='fashion'),
    common_str.format(dataset='corners3d'),
    common_str.format(dataset='helix'),
    common_str.format(dataset='helix'),
    common_str.format(dataset='spiral'),
]


for cmd in comando:
    ic(cmd)
    subprocess.run(cmd, shell=True, text=True, capture_output=False, check=True)
    ic('[INFO CMD] cmd FINISHED')

