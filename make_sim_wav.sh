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
main_dir=`pwd`

tf_gen=./sample/microcone_geotf.zip
tf_sep=./sample/microcone_geotf.zip
ch=0

temp_dir=./$$
mkdir -p ${temp_dir}
#python ./make_noise.py temp/noise.wav -c 7
python ./sim_tf.py ${tf_gen} ${target1} ${ch} 0 1 ${temp_dir}/sim1.wav 
python ./sim_tf.py ${tf_gen} ${target2} ${ch} 90 1 ${temp_dir}/sim2.wav 
python ./wav_mix.py ${temp_dir}/sim1.wav ${temp_dir}/sim2.wav -o ${temp_dir}/mixed.wav
cd ${temp_dir}
mkdir -p sep_files
python ../const_sep.py -i ${main_dir}/const_sep.n.tmpl -t ${main_dir}/${tf_sep} -d 0,90 ./mixed.wav -s ./const_sep.n
cd ${main_dir}
mv ${temp_dir}/sep_files/sep_0.wav ${output1}
mv ${temp_dir}/sep_files/sep_1.wav ${output2}
rm -rf ${temp_dir}
cd $prev_dir

