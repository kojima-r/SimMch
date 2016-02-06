if [ $# -ne 4 ]; then
	echo "usage: <in: wav> <in: wav> <out: wav> <out:wav>" 1>&2
	exit 1
fi
dir=`pwd`
target1=$dir/$1 #./sample/test16.wav
target2=$dir/$2 #./sample/test16.wav
output1=$dir/$3 #./sample/test16.wav
output2=$dir/$4 #./sample/test16.wav

cd `dirname $0`

tf_gen=./sample/microcone_geotf.zip
tf_sep=./sample/microcone_geotf.zip
ch=0
#python ./make_noise.py temp/noise.wav -c 7
python ./sim_tf.py ${tf_gen} ${target1} ${ch} 0 1 ./temp/sim1.wav 
python ./sim_tf.py ${tf_gen} ${target2} ${ch} 90 1 ./temp/sim2.wav 
python ./wav_mix.py ./temp/sim1.wav ./temp/sim2.wav -o ./temp/mixed.wav
python ./const_sep.py -t ${tf_sep} -d 0,90 ./temp/mixed.wav 
mv ./sep_files/sep_0.wav ${output1}
mv ./sep_files/sep_1.wav ${output2}
cd $prev_dir

