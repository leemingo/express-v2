cd ..
python datasets.py --data_path /data/MHL/bepro/processed --save_path /data/MHL/exPressV2/wo_cv
python train.py --mode train --root_path wo_cv