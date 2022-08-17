#!/bin/bash

[ -d "/dist" ] && rm dist/*
poetry build
pip3 install dist/evops-*-py3-none-any.whl

python3 -m pytest
