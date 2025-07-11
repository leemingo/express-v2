cd ..
python preprocess_bepro.py --data_path /data/MHL/bepro/raw
# python preprocess_dfl.py --data_path /data/MHL/dfl-confidential/raw
# python preprocess_dfl.py --data_path /data/MHL/dfl-spoho/raw
python pressing_intensity.py --source bepro
# python pressing_intensity.py --source dfl-confidential
# python pressing_intensity.py --source dfl-spoho-local