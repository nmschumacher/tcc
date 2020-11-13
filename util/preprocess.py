import numpy as np
import pandas as pd
import util.octave_to_py as op
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def octave_preprocess(X_text):
	base_path = '/home/nicolas/Área de Trabalho/TCC implementacao/'
	vocab = pd.read_csv(base_path + 'octave-vocab.txt', delimiter='\t')
	n = vocab.shape[0]
	# O vocabulário é mapeado para um dicionário invertido (a palavra aponta para o índice)
	vocab_idx_dic = {} # vocab_idx_dic['aa':'zip'] = [1:1899]
	for index, row in vocab.iterrows():
		vocab_idx_dic[row['words']] = row['idx'] - 1 # Subtract 1 to adapt to python's array
	X_ptext = op.preprocess(X_text, stemming=True) # Pré-processamento dos e-emails
	return one_hot(X_ptext, vocab_idx_dic)


# Split dataset randomly into 3 substets:
# train, cross validation and test.
def split_dataset(X, y, train_part=60, test_part=20, randsplit=True):
	randnum = 42
	if randsplit:
		randnum = None
	first_split = 100 - train_part
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=first_split/100.0, random_state=randnum)
	second_split = test_part/first_split
	X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=second_split, random_state=randnum)
	return (X_train, X_cv, X_test, y_train, y_cv, y_test)


'''
	min_df=5, strip_accents='unicode', stop_words='english'
'''
def tf_idf(X_text, max_features=1899, min_df=1, strip_accents=None, stop_words=None):
	vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, strip_accents=strip_accents, \
			stop_words=stop_words, lowercase=True)
	return vectorizer.fit_transform(X_text).toarray()


# Given a list of texts X_text with m examples, returns a matrix Xone_hot 
# having a shape of (m, n) where n is the number os words in the vocabulary, 
# i.e. the number of features.
# vocabIdxDic is a dictionary where the keys are the words and the values
# are the words' indexes of that vocabulary.
# The result is a matrix Xone_hot where each feature is 1 or 0 depending if 
# the j'th word of that i'th example is present or not, respectively.
def one_hot(X_text, vocabIdxDic):
	m = len(X_text)
	n = len(vocabIdxDic)
	Xone_hot = np.zeros((m, n))
	i = 0
	while (i < m):
		for w in X_text[i].split():
			# Add this word to X if it's in the vocabulary
			if vocabIdxDic.get(w) != None:
				Xone_hot[i, vocabIdxDic[w]] = 1
		i += 1
	return Xone_hot



"""
X_text, y = dat.load_spamAssassin_3k() # É esquisito, mas acessa o primeiro elemento da sequência: uma função para carregar o dataset
X = pre.octave_preprocess(X_text)
X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
model = svm.SVC(kernel='linear', C=0.07)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
#calc.metrics.f1_score(y_test, y_test_pred) # Para confirmar o modelo/parâmetro escolhido

i = 0
idx_test_errors = []
while i < len(X_test):
	if y_test_pred[i] != y_test[i]:
		idx_test_errors.append((i, y_test[i]))
	i += 1


# Fazemos o mesmo split, mas agora sem pré processamento para encontrar o texto original
X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X_text, y, train_part=60, test_part=20, randsplit=False)
"""



