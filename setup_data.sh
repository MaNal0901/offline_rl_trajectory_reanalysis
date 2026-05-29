# setup_data.sh — à mettre à la racine du projet
#!/bin/bash
echo "Téléchargement des datasets D4RL..."

mkdir -p data

wget -q --show-progress -P data \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5

wget -q --show-progress -P data \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5

wget -q --show-progress -P data \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5

echo "Datasets prêts dans data/"
ls -lh data/*.hdf5