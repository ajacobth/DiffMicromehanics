#!/bin/bash
# Usage: ./export.sh case_6
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <case_name>  (e.g. $0 case_6)"
    exit 1
fi

python export_model.py --config "configs/$1.py" --workdir . --out ../final/models/thermal
