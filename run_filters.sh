python wiener_filter.py ./sample/jinsei_noise_mix.wav ./sample/noise.wav --noise aaa
python wav_mix.py ./sample/jinsei_tanosii.wav ./sample/noise.wav -o ./sample/jinsei_noise_mix.wav -N 1,1
python wiener_filter.py ./sample/jinsei_noise_mix.wav ./sample/jinsei_tanosii.wav --tf ./sample/tamago_rectf.zip
python gsc.py ./sample/tamago_rectf.zip ./sample/jinsei_noise_mix.wav
