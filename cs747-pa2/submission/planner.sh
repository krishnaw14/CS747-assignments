#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--mdp)
    MDP="$2"
    shift # past argument
    shift # past value
    ;;
    -a|--algorithm)
    ALGORITHM="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

python main.py --mdp ${MDP} --algorithm ${ALGORITHM} 