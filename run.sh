
python sim_tf.py ./sample/microcone_geotf.zip ./sample/test16.wav 90 ./sample/test_geotf00.wav > sample/result_geotf.txt
python sim_tf.py ./sample/microcone_rectf.zip ./sample/test16.wav 90 ./sample/test_rectf00.wav > sample/result_rectf.txt
python sim_td.py ./sample/microcone_geotf.zip ./sample/test.wav ./sample/test_td00.wav 10 90 > sample/result_td.txt

