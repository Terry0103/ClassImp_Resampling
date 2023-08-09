
#! /usr/bin/bash

$(python -m venv classimpenv)
echo "Python enviroment is built, env name: classimpenv"

$(cd classimpenv/Scripts/)
$(activate.bat)
echo "classimpenv is activated"


$(cd ../../../)
echo $(pwd)
$(pip install -r requirements.txt)
echo "The following python packages are installed"
echo $(cat requirements.txt)
