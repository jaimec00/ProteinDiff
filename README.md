# proteus
### jaime cardenas

my codebase
will have multiple projects, goal is to have a generalized framework in `proteus` directory, and add a `projects` directory where inidividual ideas
will be implemented 

## Setup (for the current project i have, inverse folding)

We use pixi for this repo, so just run

`pixi shell`

to drop into an interactive environment

Download training data (ProteinMPNN dataset):

```shell
DATA_PATH=/PATH/TO/YOUR/DATA && \
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz -P $DATA_PATH && \
tar -xzf $DATA_PATH/pdb_2021aug02.tar.gz -C $DATA_PATH && \
rm $DATA_PATH/pdb_2021aug02.tar.gz
```
