#example.sh
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run classification training
python3 train.py --task classification

# Run regression training
python3 train.py --task regression 
python3 train.py --task regression 

# Run regression exercise 3 training
python3 train.py --task regression_ex3

# Evaluate classification models
python3 eval.py --task classification

# Evaluate regression models
python3 eval.py --task regression --target_columns "Class2.1,Class2.2"
python3 eval.py --task regression --target_columns "Class7.1,Class7.2,Class7.3"

# Evaluate regression exercise 3 models
python3 eval.py --task regression_ex3
