# -*- coding: utf-8 -*-
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

import simmch
from HARK_TF_Parser.read_mat import read_hark_tf
from HARK_TF_Parser.read_param import read_hark_tf_param

#from simuration_random import gen_random_source
# gen_random_source(num,near,far,num_type)
# [{"distance":d,"azimuth":theta,"elevation":0,"type":type_id}]

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
			nearest_index=key_index

	return nearest_index

def make_noise(x):
	rad=(npr.rand()*2*math.pi)
	return (math.cos(rad)+1j*math.sin(rad))

def apply_tf(data,fftLen, step,tf_config,src_index,noise_amp=0):
	win = hamming(fftLen) # ハミング窓
	### STFT
	spectrogram = simmch.stft(data, win, step)
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
		noise_spec=np.zeros_like(out_spec)
		v_make_noise = np.vectorize(make_noise)
		noise_spec=v_make_noise(noise_spec)
		out_spec=out_spec+noise_amp*noise_spec
		### iSTFT
		resyn_data = simmch.istft(out_spec, win, step)
		out_wavdata.append(resyn_data)
	# concat waves
	mch_wavdata=np.vstack(out_wavdata)
	return mch_wavdata

if __name__ == "__main__":
	# argv check
	if len(sys.argv)<5:
		print >>sys.stderr, "Usage: sim_tf.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav> <in:ch> <in:src theta> <in:volume> <out: dest.wav>"
		quit()
	#
	npr.seed(1234)
	tf_filename=sys.argv[1]
	tf_config=read_hark_tf(tf_filename)
	target_ch=int(sys.argv[3])
	src_theta=float(sys.argv[4])/180.0*math.pi
	src_index=nearest_direction_index(tf_config,src_theta)
	src_volume=float(sys.argv[5])
	output_filename=sys.argv[6]
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
	mono_wavdata = wavdata[target_ch::nch]
	wr.close()
	
	data = mono_wavdata

	fftLen = 512
	step = fftLen / 4
	# apply transfer function
	mch_wavdata=apply_tf(data,fftLen, step,tf_config,src_index)
	mch_wavdata=mch_wavdata*src_volume
	# save data
	simmch.save_mch_wave(mch_wavdata,output_filename,params=wr.getparams())


