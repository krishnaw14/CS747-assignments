#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--instance)
    INSTANCE="$2"
    shift # past argument
    shift # past value
    ;;
    -a|--algorithm)
    ALGORITHM="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--randomSeed)
    RANDOMSEED="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--epsilon)
    EPSILON="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--horizon)
    HORIZON="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# echo "INSTANCE  = ${INSTANCE}"
# echo "ALGORITHM    = ${ALGORITHM}"
# echo "RANDOMSEED   = ${RANDOMSEED}"
# echo "EPSILON   = ${EPSILON}"
# echo "HORIZON        = ${HORIZON}"

python main.py --instance ${INSTANCE} --algorithm ${ALGORITHM} --randomSeed ${RANDOMSEED} --horizon ${HORIZON} --epsilon ${EPSILON}
