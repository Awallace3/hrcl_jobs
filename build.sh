#/usr/bin/bash

base_path=`conda info | grep -i 'base environment' | awk '{print $4}'`
echo "Base path: ${base_path}"
source ${base_path}/etc/profile.d/conda.sh
conda activate base
python3 -m build
python3 -m twine upload --skip-existing dist/*
