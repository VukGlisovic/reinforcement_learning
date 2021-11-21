# reinforcement_learning


## Installation

#### Installation using environment.yaml
Install the python environment by executing the following command:
```bash
conda env install -f environment.yaml
```
For visualizations, don't forget to install the following:
```bash
sudo apt-get install freeglut3-dev
```

#### Installation using requirements.txt
Execute the following commands:
```bash
conda create -n rl python=3.7
conda activate rl
pip install -r requirements.txt
```


## Gym rendering
Since I am using WSL, I basically have a headless system. For some environments however,
it is not easy to render videos/animations of episodes. I found a solution by using xvfb
and installing the necessary tools. Afterwards, you can start jupyter by using:
```bash
xvfb-run -s "-screen 0 1400x900x24" jupyter lab
```
This will enable you to play videos in the jupyter notebooks while on a headless system
like I am.
