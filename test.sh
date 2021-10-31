for i in `seq 510 10 600` ; do

    echo "checkpoint "$i": "

    python3.7 test.py config/pan/pan_r18_ctw_train.py checkpoints/pan_r18_ctw_train/"checkpoint_570ep.pdparams"

    rm -f outputs/submit_ctw/results/.ipynb_checkpoints

    cd eval/ctw && python eval.py && cd ../..

done

echo "checkpoint final :"

python3.7 test.py config/pan/pan_r18_ctw_train.py checkpoints/pan_r18_ctw_train/checkpoint_570ep.pdparams

rm -f outputs/submit_ctw/results/.ipynb_checkpoints

cd eval/ctw && python eval.py && cd ../..

