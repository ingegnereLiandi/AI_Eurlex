#!/bin/bash

MODEL="$1"

echo $MODEL

if ! [ -f /usr/local/bin/ollama ]; then
  curl -fsSL https://ollama.com/install.sh | sh

  ollama serve &
  sleep 10

  echo "--------------------------------"
  echo        Installo model embed
  echo "--------------------------------"
  ollama "BAAI/bge-base-en-v1.5
  sleep 12

  if [ -z "$1" ]; then
    MODEL="llama3.1"
  else
    MODEL="$1"
  fi

  echo "--------------------------------"
  echo     run model  $MODEL
  echo "--------------------------------"
  ollama run $MODEL

else
  echo "--------------------------------"
  echo       Ollama già installato
  echo "--------------------------------"
  ollama ps
fi