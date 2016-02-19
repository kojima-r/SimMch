
python sim_tf.py ./sample/microcone_geotf.zip ./sample/test16.wav 90 ./sample/test_geotf00.wav > sample/result_geotf.txt
python sim_tf.py ./sample/microcone_rectf.zip ./sample/test16.wav 90 ./sample/test_rectf00.wav > sample/result_rectf.txt
python sim_td.py ./sample/microcone_geotf.zip ./sample/test.wav ./sample/test_td00.wav 10 90 > sample/result_td.txt

python make_noise.py sample/noise01.wav -c 7
python sim_tf.py ./sample/microcone_geotf.zip ./sample/test16.wav 1 90 1 ./sample/test_geotf00.wav 
python ./wav_mix.py ./sample/noise01.wav ./sample/test_geotf00.wav -o mixed.wav
python const_sep.py -t ./sample/microcone_geotf.zip -d 90,0 mixed.wav 

