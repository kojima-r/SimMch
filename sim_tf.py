# -*- coding: utf-8 -*-
# ==================================
#
#    Short Time Fourier Trasform
#
# ==================================
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft# , ifft
from scipy import ifft # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read
import wave
import array
from matplotlib import pylab as pl
import sys
import numpy as np
import numpy.random as npr
import math

from HARK_TF_Parser.read_mat import read_hark_tf
from HARK_TF_Parser.read_param import read_hark_tf_param

#from simuration_random import gen_random_source
# gen_random_source(num,near,far,num_type)
# [{"distance":d,"azimuth":theta,"elevation":0,"type":type_id}]


# ======
#  STFT
# ======
"""
x : 入力信号
win : 窓関数
step : シフト幅
"""
def stft(x, win, step):
    l = len(x) # 入力信号の長さ
    N = len(win) # 窓幅、つまり切り出す幅
    M = int(ceil(float(l - N + step) / step)) # スペクトログラムの時間フレーム数
    
    new_x = zeros(N + ((M - 1) * step), dtype = float64)
    new_x[: l] = x # 信号をいい感じの長さにする
    
    X = zeros([M, N], dtype = complex64) # スペクトログラムの初期化(複素数型)
    for m in xrange(M):
        start = step * m
        X[m, :] = fft(new_x[start : start + N] * win)
    return X

# =======
#  iSTFT
# =======
def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in xrange(M):
        start = step * m
        ### 滑らかな接続
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2 
    pos = (wsum != 0)
    x_pre = x.copy()
    ### 窓分のスケール合わせ
    x[pos] /= wsum[pos]
    return x

def nearest_direction_index(tf_config,theta):
	nearest_theta=math.pi
	nearest_index=None
	for key_index,value in tf_config["tf"].items():
		pos=value["position"]
		th=math.atan2(pos[1],pos[0])# -pi ~ pi
		dtheta=abs(theta-th)
		if dtheta>2*math.pi:
			dtheta-=2*math.pi
		
		if dtheta>math.pi:
			dtheta=2*math.pi-dtheta

		if dtheta<nearest_theta:
			nearest_theta=dtheta
			print nearest_theta
			nearest_index=key_index

	return nearest_index

if __name__ == "__main__":
	# argv check
	if len(sys.argv)<5:
		print >>sys.stderr, "Usage: sim_tf.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav> <in:src theta> <out: dest.wav>"
		quit()
	#
	npr.seed(1234)
	tf_filename=sys.argv[1]
	tf_config=read_hark_tf(tf_filename)
	src_theta=float(sys.argv[3])/180.0*math.pi
	src_index=nearest_direction_index(tf_config,src_theta)
	output_filename=sys.argv[4]
	if not src_index in tf_config["tf"]:
		print >>sys.stderr, "Error: tf index",src_index,"does not exist in TF file"
		quit()
	mic_pos=read_hark_tf_param(tf_filename)
	print "# mic positions:",mic_pos
	wav_filename=sys.argv[2]
	wr = wave.open(wav_filename, "rb")

	# print info
	print "# channel num : ", wr.getnchannels()
	print "# sample size : ", wr.getsampwidth()
	print "# sampling rate : ", wr.getframerate()
	print "# frame num : ", wr.getnframes()
	print "# params : ", wr.getparams()
	print "# sec : ", float(wr.getnframes()) / wr.getframerate()
	
	# reading data
	data = wr.readframes(wr.getnframes())
	nch=wr.getnchannels()
	wavdata = np.frombuffer(data, dtype= "int16")
	fs=wr.getframerate()
	mono_wavdata = wavdata[1::nch]
	wr.close()
	
	data = mono_wavdata

	fftLen = 512
	win = hamming(fftLen) # ハミング窓
	step = fftLen / 4

	### STFT
	spectrogram = stft(data, win, step)
	spec=spectrogram[:, : fftLen / 2 + 1]
	#print spectrogram[7000, :]
	#[4,3,2,1,2*,3*]
	
	### Apply TF
	tf=tf_config["tf"][src_index]
	print "# position:",tf["position"]
	pos=tf["position"]
	th=math.atan2(pos[1],pos[0])# -pi ~ pi
	print "# theta(deg):",th/math.pi*180
	out_wavdata=[]
	for mic_index in xrange(tf["mat"].shape[0]):
		tf_mono=tf["mat"][mic_index]
		print "# src spectrogram:",spec.shape
		print "# tf spectrogram:",tf_mono.shape
		tf_spec=spec*tf_mono
		spec_c=np.conjugate(tf_spec[:,:0:-1])
		out_spec=np.c_[tf_spec,spec_c[:,1:]]
		### iSTFT
		resyn_data = istft(out_spec, win, step)
		out_wavdata.append(resyn_data)
	# concat waves
	mch_wavdata=np.vstack(out_wavdata).transpose()
	# save data
	out_wavdata = mch_wavdata.copy(order='C')
	print "# save data:",out_wavdata.shape
	ww = wave.Wave_write(output_filename)
	ww.setparams(wr.getparams())
	ww.setnchannels(out_wavdata.shape[1])
	ww.setnframes(out_wavdata.shape[0])
	ww.writeframes(array.array('h', out_wavdata.astype("int16").ravel()).tostring())
	ww.close()

