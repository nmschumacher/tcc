import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer


# Returns a dictionary dic where:
#	dic['X'] contains the X train values
#	dic['y'] contains the y train labels
#	dic['m'] contains the total number of examples
#	dic['n'] contains the total number of features
def loadTrainData():
	base_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/'
	return _loadData(base_path + 'octave-spam-X.csv', base_path + 'octave-spam-y.csv')

# Returns a dictionary dic where:
#	dic['Xtest'] contains the X test values
#	dic['ytest'] contains the y test labels
#	dic['mtest'] contains the total number of test examples
#	dic['ntest'] contains the total number of test features
def loadTestData():
	base_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/'
	dic = _loadData(base_path + 'octave-spam-Xtest.csv', base_path + 'octave-spam-ytest.csv')
	dic['Xtest'], dic['ytest'] = dic.pop('X'), dic.pop('y') # Renaming dic keys
	dic['mtest'], dic['ntest'] = dic.pop('m'), dic.pop('n')
	return dic

# Loads Octave data given the X and y paths and returns a dictionary
def _loadData(XFilepath, yFilepath):
	X = pd.read_csv(XFilepath, header=None)
	y = pd.read_csv(yFilepath, header=None)
	m = X.shape[0]
	n = X.shape[1]
	X = np.array(X) # shape: (m, n)
	y = np.array(y) # shape: (m, 1)
	y = y.reshape((m,)) # Reshaping from 2 to 1 dimension
	dic = {}
	dic['X'], dic['y'], dic['m'], dic['n'] = X, y, m, n
	return dic

# Given an array of strings (emails), applies the preprocess step used in Octave version
# without stemming
def preprocess(emails, stemming=False):
	pemails = []
	m = len(emails)
	i = 0
	ps = PorterStemmer()
	while (i < m):
		# Lower case
		email_contents = emails[i].lower()
		# Strip all HTML
		# Looks for any expression that starts with < and ends with > and replace
		# and does not have any < or > in the tag it with a space
		reg = re.compile('<[^<>]+>')
		email_contents = reg.sub(' ', email_contents)
		# Handle Numbers
		# Look for one or more characters between 0-9
		reg = re.compile('[0-9]+')
		email_contents = reg.sub('number', email_contents)
		# Handle URLS
		# Look for strings starting with http:// or https://
		reg = re.compile('(http|https)://[^\s]*')
		email_contents = reg.sub('httpaddr', email_contents)
		# Handle Email Addresses
		# Look for strings with @ in the middle
		reg = re.compile('[^\s]+@[^\s]+')
		email_contents = reg.sub('emailaddr', email_contents)
		# Handle $ sign
		reg = re.compile('[$]+')
		email_contents = reg.sub('dollar', email_contents)
		email_contents = re.split(' |\@|\$|\/|\#|\.|\-|\:|\&|\*|\+|\=|\[|\]|\?|\!|\(|\)|\{|\}|\,|\'|\'|\"|\>|\_|\<|\;|\%', email_contents)
		email_contents = ' '.join(email_contents)
		reg = re.compile('[^a-zA-Z0-9]')
		email_contents = reg.sub(' ', email_contents)
		email_contentsStem = ''
		if stemming:
			for w in email_contents.split():
				w_stem = ps.stem(w)
				if w_stem == 'are': # Some words were stemmed differently than the vocab ones
					w_stem = 'ar'
				elif w_stem == 'money':
					w_stem = 'monei'
				email_contentsStem += w_stem + ' '
			pemails.append(email_contentsStem.strip())
		else:
			pemails.append(email_contents.strip())
		i += 1
	return pemails

