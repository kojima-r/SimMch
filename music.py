# -*- coding: utf-8 -*-
import sys
import numpy as np
import numpy.random as npr
from scipy import hamming,interpolate
import scipy

import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import simmch
from HARK_TF_Parser.read_mat import read_hark_tf
from HARK_TF_Parser.read_param import read_hark_tf_param

def stft_mch(data,win, step):
	fftLen=len(win)
	out_spec=[]
	### STFT
	for m in xrange(data.shape[0]):
		spectrogram = simmch.stft(data[m,:], win, step)
		spec=spectrogram[:, : fftLen / 2 + 1]
		out_spec.append(spec)
	mch_spec=np.stack(out_spec,axis=0)
	return mch_spec

def istft_mch(data,win, step):
	fftLen=len(win)
	out_wav=[]
	### STFT
	for m in xrange(data.shape[0]):
		spec=data[m,:,:]
		full_spec=simmch.make_full_spectrogram(spec)
		resyn_wav = simmch.istft(full_spec, win, step)
		out_wav.append(resyn_wav)
	mch_wav=np.stack(out_wav,axis=0)
	return mch_wav

def slice_window(x, win_size, step):
	l = x.shape[0]
	N = win_size
	M = int(np.ceil(float(l - N + step) / step))
	out=[]
	for m in xrange(M):
		start = step * m
		if start + N <= l:
			out.append(x[start : start + N])
	o=np.stack(out,axis=0)
	return o

def estimate_spatial_correlation(spec,win_size,step):
	# ch,frame,spec -> frame,spec,ch
	x=np.transpose(spec,(1,2,0))
	# data: frame,block,spec,ch
	data=slice_window(x, win_size, step)
	a=np.transpose(data,(0,2,3,1))
	b=np.transpose(data,(0,2,1,3)).conj()
	print data.shape
	c=np.einsum('ijkl,ijlm->ijkm',a,b)
	# out_corr: frame,spec,ch1,ch2
	out_corr=c*1.0/win_size;
	#print c[0,0]
	return out_corr

def estimate_spatial_correlation2(spec,win_size,step):
	# ch,frame,spec -> frame,spec,ch
	n_ch=spec.shape[0]
	n_frame=spec.shape[1]
	n_bin=spec.shape[2]
	corr=np.zeros((n_frame,n_bin,n_ch,n_ch),dtype=complex)
	
	# out_corr: frame,spec,ch1,ch2
	for i in xrange(n_ch):
		for j in xrange(n_ch):
			corr[:,:,i,j]=spec[i]*spec[j].conj()
	now_frame=0
	out=[]
	while(now_frame+win_size<=n_frame):
		o=np.mean(corr[now_frame:now_frame+win_size],axis=0)
		out.append(o)
		now_frame+=step
	return np.array(out)


f=[10.0,
	12.5,
	16.0,
	20.0,
	31.5,
	63.0,
	125.0,
	250.0,
	500.0,
	1000.0,
	2000.0,
	4000.0,
	8000.0,
	12500.0,
	16000.0,
	20000.0]
w=[np.power(10.0,-70.4/20.0),
	np.power(10.0,-63.4/20.0),
	np.power(10.0,-56.7/20.0),
	np.power(10.0,-50.5/20.0),
	np.power(10.0,-39.4/20.0),
	np.power(10.0,-26.2/20.0),
	np.power(10.0,-16.1/20.0),
	np.power(10.0,-8.6/20.0),
	np.power(10.0,-3.2/20.0),
	np.power(10.0,0.0/20.0),
	np.power(10.0,1.2/20.0),
	np.power(10.0,1.0/20.0),
	np.power(10.0,-1.1/20.0),
	np.power(10.0,-4.3/20.0),
	np.power(10.0,-6.6/20.0),
	np.power(10.0,-9.3/20.0)]
f1=interpolate.interp1d(f,w,kind='cubic')

def A_characteristic(freq):
		return f1(freq)

def compute_music_spec(spec,src_num,tf_config,win_size=50,step=50):
	corr=estimate_spatial_correlation2(spec,win_size,step)
	power=np.zeros((corr.shape[0],corr.shape[1],len(tf_config["tf"])),dtype=complex)
	for frame, freq in np.ndindex((corr.shape[0],corr.shape[1])):
		# normalize correlation
		rxx=corr[frame,freq]
		r=rxx/np.max(np.absolute(rxx))
		#
		#e_val,e_vec = np.linalg.eigh(corr[frame,freq])
		e_val,e_vec = scipy.linalg.eig(r)
		# sort 
		eigen_id = np.argsort(e_val)[::-1]
		e_val = e_val[eigen_id]
		e_vec = e_vec[:,eigen_id]
		e=e_vec[:,src_num:]
		# directions
		for k,v in tf_config["tf"].items():
			a_vec=v["mat"][:,min_freq_bin+freq]
			a_vec=a_vec/np.absolute(a_vec)
			weight=A_characteristic((min_freq_bin+freq)*df)
			#power[frame,freq,k]=weight*np.dot(a_vec.conj(),a_vec)/np.dot(np.dot(a_vec.conj(),ee),a_vec)
			s=np.dot(a_vec.conj(),e)
			power[frame,freq,k]=weight*np.dot(a_vec.conj(),a_vec)/np.dot(s,s.conj())
	return power

if __name__ == "__main__":
	# argv check
	if len(sys.argv)<3:
		print >>sys.stderr, "Usage: music.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav>"
		quit()
	# read tf
	npr.seed(1234)
	tf_filename=sys.argv[1]
	tf_config=read_hark_tf(tf_filename)
	#if not src_index in tf_config["tf"]:
	#	print >>sys.stderr, "Error: tf index",src_index,"does not exist in TF file"
	#	quit()
	mic_pos=read_hark_tf_param(tf_filename)
	print "# mic positions:",mic_pos
	# read wav
	wav_filename=sys.argv[2]
	print "... reading", wav_filename
	wav_data=simmch.read_mch_wave(wav_filename)
	wav=wav_data["wav"]/32767.0
	fs=wav_data["framerate"]
	nch=wav_data["nchannels"]
	# print info
	print "# channel num : ", nch
	print "# sample size : ", wav.shape
	print "# sampling rate : ", fs
	print "# sec : ", wav_data["duration"]
	
	# reading data
	fftLen = 512
	step = 160 #fftLen / 4
	df=fs*1.0/fftLen
	# cutoff bin
	min_freq=300
	max_freq=1000
	min_freq_bin=int(np.ceil(min_freq/df))
	max_freq_bin=int(np.floor(max_freq/df))
	print "# min freq:",min_freq
	print "# max freq:",max_freq
	print "# min fft bin:",min_freq_bin
	print "# max fft bin:",max_freq_bin

	# apply transfer function
	win = hamming(fftLen) # ハミング窓
	spec=stft_mch(wav,win,step)
	spec=spec[:,:,min_freq_bin:max_freq_bin]
	src_num=2
	print "# src_num:",src_num
	# power: frame, freq, direction_id
	power=compute_music_spec(spec,src_num,tf_config,win_size=50,step=50)
	p=np.sum(np.real(power),axis=1)
	m_power=10*np.log10(p+1.0)


	# save
	outfilename="music.npy"
	np.save(outfilename,m_power)
	#np.savetxt("music.csv", m_power, delimiter=",")
	print "[save]",outfilename

	# plot heat map
	ax = sns.heatmap(m_power.transpose(),cbar=False,cmap=cm.Greys)
	sns.plt.axis("off")
	sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
	plt.tight_layout()
	ax.tick_params(labelbottom='off')
	ax.tick_params(labelleft='off')
	outfilename_heat="music.png"
	sns.plt.savefig(outfilename_heat, bbox_inches="tight", pad_inches=0.0)
	print "[save]",outfilename_heat

	outfilename_heat_bar="music_bar.png"
	sns.plt.clf()
	sns.heatmap(m_power,cbar=True,cmap=cm.Greys)
	sns.plt.savefig(outfilename_heat_bar, bbox_inches="tight", pad_inches=0.0)
	sns.plt.clf()
	print "[save]",outfilename_heat_bar
	
	outfilename_fft="fft.png"
	x=(np.absolute(spec[0].T)**2)
	ax = sns.heatmap(x[::-1,:],cbar=False,cmap='coolwarm')
	sns.plt.axis("off")
	sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
	plt.tight_layout()
	ax.tick_params(labelbottom='off')
	ax.tick_params(labelleft='off')
	sns.plt.savefig(outfilename_fft, bbox_inches="tight", pad_inches=0.0)
	print "[save]",outfilename_fft

