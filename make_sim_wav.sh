if [ $# -lt 4 ]; then
	echo "usage: <in: wav> <in: wav> <out: wav> <out:wav> [<in: dir1> <in:dir2>]" 1>&2
	exit 1
fi
dir=`pwd`
target1=$dir/$1 #./sample/test16.wav
target2=$dir/$2 #./sample/test16.wav
output1=$dir/$3 #./sample/test16.wav
output2=$dir/$4 #./sample/test16.wav
dir1=0
dir2=90

if [ $# -gt 4 ]; then
dir1=$5
dir2=$6
fi

cd `dirname $0`

tf_gen=./sample/microcone_geotf.zip
tf_sep=./sample/microcone_geotf.zip
ch=0
#python ./make_noise.py temp/noise.wav -c 7
python ./sim_tf.py ${tf_gen} ${target1} ${ch} $dir1 1 ./temp/sim1.wav 
python ./sim_tf.py ${tf_gen} ${target2} ${ch} $dir2 1 ./temp/sim2.wav 
python ./wav_mix.py ./temp/sim1.wav ./temp/sim2.wav -o ./temp/mixed.wav
python ./const_sep.py -t ${tf_sep} -d ${dir1},${dir2} ./temp/mixed.wav 
mv ./sep_files/sep_0.wav ${output1}
mv ./sep_files/sep_1.wav ${output2}
cd $prev_dir

