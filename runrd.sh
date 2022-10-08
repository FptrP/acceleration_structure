#!/usr/bin/bash

case $1 in
  clean)
    rm captures/*
    exit;;
  run)
    renderdoccmd capture --opt-api-validation -c captures/debug -d . ./build/main
    ;;
  *)
    echo "Unknown parameter"
    exit
    ;;
esac