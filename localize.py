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
import json

import simmch
from HARK_TF_Parser.read_mat import read_hark_tf
from HARK_TF_Parser.read_param import read_hark_tf_param
import music

from optparse import OptionParser

def detect_island(vec):
	o=[]
	out=[]
	for i,el in enumerate(vec):
		if el>0:
			o.append((i,el))
		else:
			if len(o)>0:
				out.append(o)
				o=[]
	if len(o)>0:
		if vec[0]>0:
			if len(out)>0:
				out[0]=o+out[0]
			else:
				out.append(o)
		else:
			out.append(o)
	return out

def detect_all_peaks(vec):
	islands=detect_island(vec)
	peaks=[]
	for el in islands:
		max_i=None
		max_v=None
		for e in el:
			if e[1]==max_v:
				max_i.append(e[0])
			elif e[1]>max_v:
				max_v=e[1]
				max_i=[e[0]]
		peaks.append((max_i,max_v))
	return peaks

def detect_peak(vec):
	peaks=[]
	all_peaks=detect_all_peaks(vec)
	for el in all_peaks:
		c=len(el[0])/2
		peaks.append(el[0][c])
	return peaks

if __name__ == "__main__":
	usage = 'usage: %s tf [options] <in: src.wav> <out: dest.wav>' % sys.argv[0]
	parser = OptionParser()
	
	parser.add_option(
		"--min-freq", dest="min_freq",
		help="minimum frequency (Hz)",
		default=2000,
		type=float,
		metavar="F")

	parser.add_option(
		"--max-freq", dest="max_freq",
		help="maximum frequency (Hz)",
		default=8000,
		type=float,
		metavar="F")
	
	parser.add_option(
		"--thresh", dest="thresh",
		help="threshold of MUSIC power spectrogram",
		default=None,
		type=float,
		metavar="F")

	parser.add_option(
		"--src-num", dest="src_num",
		help="number of sound source (for MUSIC)",
		default=2,
		type=int,
		metavar="N")
	
	parser.add_option(
		"--stft-win", dest="stft_win",
		help="window size for STFT",
		default=512,
		type=int,
		metavar="W")

	parser.add_option(
		"--stft-adv", dest="stft_adv",
		help="advance step size for STFT",
		default=160,
		type=int,
		metavar="W")

	parser.add_option(
		"--music-win", dest="music_win",
		help="window size for MUSIC",
		default=50,
		type=int,
		metavar="W")

	parser.add_option(
		"--music-adv", dest="music_adv",
		help="advance step size for MUSIC",
		default=50,
		type=int,
		metavar="W")

	parser.add_option(
		"--event-min-size", dest="event_min_size",
		help="minimum event size (MUSIC frame)",
		default=3,
		type=int,
		metavar="W")

	parser.add_option(
		"--out-npy",
		dest="npy_file",
		help="[output] numpy MUSIC spectrogram file (dB :same with HARK)",
		default=None,
		type=str,
		metavar="FILE")

	parser.add_option(
		"--out-full-npy",
		dest="npy_full_file",
		help="[output] numpy MUSIC spectrogram file for each frequency bin (raw spectrogram)",
		default=None,
		type=str,
		metavar="FILE")


	parser.add_option(
		"--out-loc",
		dest="loc_file",
		help="[output] localization file(.json)",
		default=None,
		type=str,
		metavar="FILE")
	

	parser.add_option(
		"--plot-h",
		dest="plot_h_file",
		help="[output] heatmap file",
		default=None,
		type=str,
		metavar="FILE")
	
	parser.add_option(
		"--plot-hb",
		dest="plot_hb_file",
		help="[output] heatmap file with bar",
		default=None,
		type=str,
		metavar="FILE")
	
	parser.add_option(
		"--plot-fft",
		dest="plot_fft_file",
		help="[output] spectrogram",
		default=None,
		type=str,
		metavar="FILE")


	(options, args) = parser.parse_args()
	
	# argv check
	if len(args)<2:
		print >>sys.stderr, "Usage: music.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav>"
		quit()
	#
	# read tf
	npr.seed(1234)
	tf_filename=args[0]
	tf_config=read_hark_tf(tf_filename)
	mic_pos=read_hark_tf_param(tf_filename)
	print "# mic positions:",mic_pos
	# read wav
	wav_filename=args[1]
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
	fftLen = options.stft_win
	step = options.stft_adv #fftLen / 4
	df=fs*1.0/fftLen
	step_ms = fs/step
	# cutoff bin
	min_freq=options.min_freq
	max_freq=options.max_freq
	min_freq_bin=int(np.ceil(min_freq/df))
	max_freq_bin=int(np.floor(max_freq/df))
	print "# min freq:",min_freq
	print "# max freq:",max_freq
	print "# min fft bin:",min_freq_bin
	print "# max fft bin:",max_freq_bin

	# apply transfer function
	win = hamming(fftLen) # ハミング窓
	spec=simmch.stft_mch(wav,win,step)
	spec_m=spec[:,:,min_freq_bin:max_freq_bin]
	src_num=options.src_num
	print "# src_num:",src_num

	# power: frame, freq, direction_id
	music_win=options.music_win
	music_step=options.music_adv
	music_step_ms=music_step*step_ms
	power=music.compute_music_spec(spec_m,src_num,tf_config,df,min_freq_bin,win_size=music_win,step=music_step)
	p=np.sum(np.real(power),axis=1)
	m_power=10*np.log10(p+1.0)

	# save
	if options.npy_file is not None:
		outfilename=options.npy_file
		np.save(outfilename,m_power)
		#np.savetxt("music.csv", m_power, delimiter=",")
		print "[save]",outfilename,m_power.shape

	if options.npy_full_file is not None:
		outfilename=options.npy_full_file
		np.save(outfilename,power)
		#np.savetxt("music.csv", m_power, delimiter=",")
		print "[save]",outfilename,power.shape
	

	if options.plot_h_file is not None:
		# plot heat map
		ax = sns.heatmap(m_power.transpose(),cbar=False,cmap=cm.Greys)
		sns.plt.axis("off")
		sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
		plt.tight_layout()
		ax.tick_params(labelbottom='off')
		ax.tick_params(labelleft='off')
		outfilename_heat=options.plot_h_file
		sns.plt.savefig(outfilename_heat, bbox_inches="tight", pad_inches=0.0)
		print "[save]",outfilename_heat,m_power.shape


	if options.plot_hb_file is not None:
		sns.plt.clf()
		sns.heatmap(m_power,cbar=True,cmap=cm.Greys)
		outfilename_heat_bar=options.plot_hb_file
		sns.plt.savefig(outfilename_heat_bar, bbox_inches="tight", pad_inches=0.0)
		sns.plt.clf()
		print "[save]",outfilename_heat_bar,m_power.shape

	
	if options.plot_fft_file is not None:
		outfilename_fft=options.plot_fft_file
		x=(np.absolute(spec[0].T)**2)
		ax = sns.heatmap(x[::-1,:],cbar=False,cmap='coolwarm')
		sns.plt.axis("off")
		sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
		plt.tight_layout()
		ax.tick_params(labelbottom='off')
		ax.tick_params(labelleft='off')
		sns.plt.savefig(outfilename_fft, bbox_inches="tight", pad_inches=0.0)
		print "[save]",outfilename_fft

	# threshold
	threshold=options.thresh
	def threshold_filter(x):
		if x<threshold:
			return 0
		else:
			return x
	
	f=np.vectorize(threshold_filter)
	m_power_thresh=f(m_power)
	print "# MUSIC power after thresholding"
	print m_power_thresh
	# peak detection
	peak_tl=[]
	for i in xrange(m_power_thresh.shape[0]):
		#m_power_thresh[i]
		peaks=detect_peak(m_power_thresh[i])
		peak_tl.append(peaks)
	
	# tracking
	n_dir= m_power.shape[1]
	def diff_peak(a,b):
		d=abs(a-b)
		if(d<n_dir/2):
			return d
		else:
			return n_dir-d
	
	if len(peak_tl)==1:
		print >> sys.stderr, "[ERROR] too short"
	# detect next frame
	tracking_peaks=[[] for t in xrange(len(peak_tl))]
	for t in xrange(len(peak_tl)-1):
		p1s=peak_tl[t]
		p2s=peak_tl[t+1]
		for i1,p1 in enumerate(p1s):
			nearest_p2=None
			for i2,p2 in enumerate(p2s):
				if nearest_p2==None or diff_peak(p1,p2)<nearest_p2[1]:
					if diff_peak(p1,p2)<6:
						nearest_p2=(i2,p2,abs(p1-p2))
			tracking_peaks[t].append([i1,p1,nearest_p2,None])
	# for last event
	ps=peak_tl[len(peak_tl)-1]
	for i1,p1 in enumerate(ps):
		tracking_peaks[len(peak_tl)-1].append([i1,p1,None,None])
	# make id
	print "# detected peaks and IDs"
	id_cnt=0
	for t in xrange(len(peak_tl)-1):
		evts=tracking_peaks[t]
		evts2=tracking_peaks[t+1]
		for evt in evts:
			p2=evt[2]
			# index,dir,next,count
			if evt[3]==None:
				evt[3]=id_cnt
				id_cnt+=1
			print ">>",evt
			if(p2 is not None):
				i2=p2[0]
				# index,dir,diff
				evts2[i2][3]=evt[3]
	# count id
	id_counter={}
	for t in xrange(len(peak_tl)):
		evts=tracking_peaks[t]
		for evt in evts:
			if evt[3] is not None:
				if not evt[3] in id_counter:
					id_counter[evt[3]]=0
				id_counter[evt[3]]+=1
	print "# event counting( ID => count ):",id_counter
	# cut off & re-set id
	event_min_size=options.event_min_size
	print "# cut off: minimum event size:",event_min_size
	reset_id_counter={}
	tl_objs=[]
	for t in xrange(len(peak_tl)):
		evts=tracking_peaks[t]
		objs=[]
		for evt in evts:
			if evt[3] is not None and id_counter[evt[3]]>=event_min_size:
				poss=tf_config["positions"]
				pos=poss[evt[1]]
				eid=evt[3]
				if not eid in reset_id_counter:
					reset_id_counter[eid]=len(reset_id_counter)
				new_eid=reset_id_counter[eid]
				obj={"id":new_eid,"x":[pos[1],pos[2],pos[3]],"power":1}
				objs.append(obj)
		tl_objs.append(objs)
	
	print "# re-assigned IDs"
	for o in tl_objs:
		print o
	save_obj={
			"interval":music_step_ms,
			"tl":tl_objs
			}
	if options.loc_file is not None:
		filename=options.loc_file
		print "[save]",filename
		with open(filename, 'w') as f:
			json.dump(save_obj, f)
		
