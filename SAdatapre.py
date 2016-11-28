#encoding="utf-8"
#!/usr/bin/env python

import sys
import numpy as np
import jieba
jieba.load_userdict( "./../../Paper/Travel/step1_classfication/config/usedict" )
import re
import random
import unicodedata
import tensorflow as tf
import time

reload( sys )
sys.setdefaultencoding( "utf-8" )

"""
#  parameters the raw line
#  return the clear line that can segment to words
"""

def clearStr( line ) :
	line = line.replace( '=' , '' )
	line = line.replace( '~' , '' )
	line = line.replace( '-' , '' )
	line = line.replace( '#' , '' )
	line = line.replace( '*' , '' )
	return line.strip()

"""
# load the stopwords 
"""
def getstopwords( stopwordsPath ) :
	with open( stopwordsPath , 'r' ) as f :
		return { line.strip().decode( "utf-8" ) for line in f }

"""
# load the word2vec path
"""
def lodaModel( w2vModelPath ) :
	wordModel = dict()
	with open( w2vModelPath , 'r' ) as f :
		for line  in f :
			fields = line.strip().split( " " )
			word = fields[ 0 ]
			vector = map( float , fields[ 1 : ] )
			if len( vector ) != 200 :
				continue
			else :
				wordModel[ word.decode( "utf-8" ) ] = vector
	return wordModel

def wordTovector( segListFilter , wordModel , num_steps ) :
	vector = []

	zeroVec = [0.0] * 200
	step = 0
	for word in segListFilter :
		if wordModel.has_key( word ) :
			step += 1
			Vector = wordModel.get( word )
			vector.extend( Vector )

	if step >= num_steps :
		return vector[:num_steps * 200]
	else :
		newvector = []
		for i in range(num_steps - step) :
			newvector.extend( zeroVec )
		newvector.extend(vector)
		return newvector

"""
#  parameters the file path
#  return the feature vector and his label 
"""
def load_data_label( dataPath , stopwordsPath , w2vModelPath , num_steps) :
	begin = time.clock()

	wordModel = lodaModel( w2vModelPath )
	stopwords = getstopwords( stopwordsPath )
	i = 0
	X = list()
	Y = list()
	with open( dataPath , 'r' ) as f :
		for line in f :
			i += 1
			if i > 3000 :
				break
			line = line.strip().split( "," , 1 )
			label = float( line[ 0 ] )
			content = clearStr( line[ 1 ] )
			seg_list = jieba.cut( content , cut_all = False )
			segListFilter = [ word for word in seg_list if word not in stopwords and not re.match( r'.*(\w)+.*' , unicodedata.normalize('NFKD', word ).encode('ascii','ignore') ) ]
			vector = wordTovector( segListFilter , wordModel , num_steps)
			X.append( vector )
			Y.append( label )

	#change the label to softmax format
	Y = [[1 - label , label] for label in Y]
	end = time.clock()
	print "The time that generate the training and testing data is : {:f}".format(end - begin)
	return X , Y

"""
#  @iterater time
#  @batch_size
#  return generater the data
"""
def batch_iter(data,shuffled=True ) :
	"""
	# we mainly generate a batch of data for training
	"""
	data_size = len( data )
	data_seq = range( data_size )
	if shuffled :
		random.shuffle( data_seq )
	for index in data_seq :
		yield data[index]

def main() :
	dataPath = "./../../Paper/Travel/step1_classfication/data/step1_data"
	stopwordsPath = "./../word2vec+CNN+senClassify/stopwords.txt"
	w2vModelPath = "./../../Paper/Travel/result_min5_iter5.bin"

	num_steps = 40
	X , Y = load_data_label( dataPath , stopwordsPath , w2vModelPath , num_steps)
	max_max_epochs=3
	batches = batch_iter(list(zip(X , Y) ), shuffled=True)
	for batch in batches :

		x ,y = batch[0],batch[1]
		print type(x ) , type(y)
		print len(x ) , len(y)
		# print x

if __name__ == '__main__' :
	main()
