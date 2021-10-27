python3.7 -m pip install --upgrade -r requirement.txt
python3.7 -m pip install pyclipper
python3.7 -m pip install lap

cd models/post_processing/pa/
python3.7 setup.py build_ext --inplace
cd ../../../

CUDA_VISIBLE_DEVICES=0 python3.7 dist_train.py config/pan/pan_r18_ctw_train.py --nprocs 1 --resume checkpoints/pan_r18_ctw_train