#!/bin/bash
#PBS -l nodes=1:gpus=1:ppn=1

ENV="rl_trade"
DIR="./git_files/RL_System-Trade/my_rl_trader"
EXECUTE_FILE_NAME="main.py"

source activate $ENV
echo "Activate $ENV envirnmnet"

cd $DIR
echo "$EXECUTE_FILE_NAME running"
python $EXECUTE_FILE_NAME

conda deactivate
timestamp=`date +%Y/%m/%d/ %H:%M`
echo "$timestamp"
echo "Deactivate envirnment"
echo "completed!"
