cd ..
# python preprocess_bepro.py --data_path /data/MHL/bepro/raw
python pressing_intensity.py --source bepro --data_path /data/MHL/bepro/processed
# python preprocess_dfl.py --data_path /data/MHL/dfl-confidential/raw
# python pressing_intensity.py --source dfl-confidential --data_path /data/MHL/dfl-confidential/processed
# python preprocess_dfl.py --data_path /data/MHL/dfl-spoho/raw
# python pressing_intensity.py --source dfl-spoho-local --data_path /data/MHL/dfl-spoho/processed