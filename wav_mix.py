# -*- coding: utf-8 -*-

import wave
import array
import sys
import numpy as np
import numpy.random as npr
import math
from optparse import OptionParser
import simmch

if __name__ == "__main__":
	usage = 'usage: %s tf [options] <in: src.wav ...>' % sys.argv[0]
	parser = OptionParser()
	parser.add_option(
		"-o", "--output",
		dest="output_file",
		help="output file",
		default=None,
		type=str,
		metavar="FILE")
	
	parser.add_option(
		"-V", "--volume", dest="volume",
		help="volumes of input sound (0<=v<=1)",
		default=None,
		type=str,
		metavar="VOL")
	
	(options, args) = parser.parse_args()
	
	# argv check
	if len(args)<1:
		quit()
	#
	npr.seed(1234)
	src_volume=options.volume
	output_filename=options.output_file
	data=[]
	print "... reading .wav files"
	for wav_filename in args:
		print wav_filename
		wav_data=simmch.read_mch_wave(wav_filename)
		wav=wav_data["wav"]
		fs=wav_data["framerate"]
		nch=wav_data["nchannels"]
		data.append((wav,fs,nch,wav_filename))
	print "... checking"
	wavdata=data[0][0]
	fs=data[0][1]
	nch=data[0][2]
	wav_length=0
	for wav_info in data:
		if nch!=wav_info[2]:
			print >>sys.stderr,"[ERROR] #channel error:",nch,"!=",wav_info[2],"(",wav_info[3],")"
		if fs!=wav_info[1]:
			print >>sys.stderr,"[ERROR] sampling rate error",fs,"!=",wav_info[1],"(",wav_info[3],")"
		if wav_length<wav_info[0].shape[1]:
			wav_length=wav_info[0].shape[1]
	print "... mixing"
	mix_wavdata=np.zeros((nch,wav_length),dtype="float")
	for wav_info in data:
		l=wav_info[0].shape[1]
		mix_wavdata[:,:l]+=wav_info[0].astype("float")
	# save data
	if output_filename!=None:
		simmch.save_mch_wave(mix_wavdata,output_filename)
	
