# -*- coding: utf-8 -*-

from scipy import ceil, complex64, float64, hamming, zeros
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
from sim_tf import apply_tf
from sim_tf import nearest_direction_index

from optparse import OptionParser

def make_noise(x):
	rad=(npr.rand()*2*math.pi)
	return (math.cos(rad)+1j*math.sin(rad))


if __name__ == "__main__":
	usage = 'usage: %s [options] <in: src.wav> <out: dest.wav>' % sys.argv[0]
	parser = OptionParser(usage)
	parser.add_option(
		"-t", "--tf",
		dest="tf",
		help="tf.zip(HARK2 transfer function file>",
		default=None,
		type=str,
		metavar="TF")
	
	parser.add_option(
		"-d", "--duration", dest="duration",
		help="duration of sound (sec)",
		default=None,
		type=float,
		metavar="D")

	parser.add_option(
		"-s", "--samples", dest="samples",
		help="duration of sound (samples)",
		default=None,
		type=int,
		metavar="D")

	parser.add_option(
		"-f", "--frames", dest="frames",
		help="duration of sound (frames)",
		default=None,
		type=int,
		metavar="D")

	parser.add_option(
		"-r", "--samplingrate", dest="samplingrate",
		help="sampling rate",
		default=16000,
		type=int,
		metavar="R")


	parser.add_option(
		"-c", "--channel", dest="channel",
		help="target channel of input sound (>=0)",
		default=None,
		type=int,
		metavar="CH")
	
	parser.add_option(
		"-p", "--power", dest="power",
		help="power of noise sound",
		default=1,
		type=float,
		metavar="POWER")
	
	(options, args) = parser.parse_args()
	
	# argv check
	if len(args)<1:
		parser.print_help()
		quit()
	#
	npr.seed(1234)
	output_filename=args[0]
	# save data
	nch=options.channel
	fftLen = 512
	step = fftLen
	nsamples=None
	if options.samples!=None:
		nsamples=options.samples
		length=((nsamples-(fftLen-step))-1)/step+1
	elif options.frames!=None:
		length=options.frames
		nsamples=length*step+(fftLen-step)
	elif options.duration!=None:
		nsamples=options.samplingrate*options.duration
		length=((nsamples-(fftLen-step))-1)/step+1
	else:
		print >>sys.stderr,"[ERROR] unknown duration (please indicate --duration, --samples, or --frames)"
		quit()

	# stft length <-> samples
	
	data=np.zeros((nch,length,fftLen/2+1),dtype = complex64)
	v_make_noise = np.vectorize(make_noise)
	data=v_make_noise(data)#*math.sqrt(1.0*fftLen)*options.power
	print "#channels:",nch
	print "#frames:",length
	print "#samples:",nsamples
	print "window size:",fftLen
	#win = hamming(fftLen) # ハミング窓
	win = np.array([1.0]*fftLen)
	out_wavdata=[]
	for mic_index in xrange(data.shape[0]):
		spec=data[mic_index]
		spec_c=np.conjugate(spec[:,:0:-1])
		out_spec=np.c_[spec,spec_c[:,1:]]
		#pow_f=np.mean(np.mean(abs(out_spec)**2,axis=1))
		print "?",out_spec.shape
		spectrum=np.sum(out_spec,axis=0)
		test=np.sum(np.abs(out_spec),axis=0)
		#print spectrum
		#pow_test=np.mean(abs(test)**2)
		#print "[CHECK] power(test):",pow_test
		pow_f=np.mean(abs(spectrum)**2)
		print "[CHECK] power(f):",pow_f
		### iSTFT
		resyn_data = simmch.istft(out_spec, win, step)
		print resyn_data.shape
		pow_w=np.sum(resyn_data**2)
		print "[CHECK] power(x):",pow_w#/length
		print "[CHECK] power:",pow_f/pow_w#/length
		out_wavdata.append(resyn_data)
	# concat waves
	mch_wavdata=np.vstack(out_wavdata)
	simmch.save_mch_wave(mch_wavdata,output_filename,sample_width=2,framerate=options.samplingrate)
	print "[CHECK] power in time domain:",np.mean(mch_wavdata**2,axis=1)
	
