import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as NN
from sklearn import metrics


# Given the train and cross-vaidation datasets, returns the errors 
# returns 
def learning_curves(X_train, y_train, X_cv, y_cv, model, model_fit_params=None, step=1):
	m = X_train.shape[0]
	errors = {} # Errors of the training and CV dataset
	errors['train'], errors['cv'] = [], []
	errors['num_examples'] = [] # Number of examples used 
	last_loop = False
	i = step
	while(True):
		model.fit(X_train[0:i], y_train[0:i])#, model_fit_params) # Calling fit multiple times clears the results of previous calls
		y_train_pred = model.predict(X_train[0:i]) # Predict only on the trained examples
		errors['train'].append(1 - metrics.f1_score(y_train[0:i], y_train_pred))
		y_cv_pred = model.predict(X_cv) # Predict over all the X_cv
		errors['cv'].append(1 - metrics.f1_score(y_cv, y_cv_pred))
		errors['num_examples'].append(i)
		i += step
		if i >= m and not last_loop:
			i = m
			last_loop = True
		elif last_loop:
			break
	return errors

'''
Busca o valor de C que maximiza a medida F1-Score do modelo SVM dado uma lista
de possíveis valores de C. Retorna o melhor C encontrado e o valor F1-Score 
correspondente, respectivamente.
	X_cv: conjunto de dados de validação-cruzada
	y_cv: conjunto de labels da validação-cruzada
	Cs: lista de valores de C a serem testados
	inner_reg: se verdadeiro, será realizada uma nova busca de C ao selecionar
		valores vizinhos ao melhor C encontrado na primeira avaliação
	algorithm: valores aceitos: 'SVM', 'SVM sigmoid', 'SVM poly [0-9]+' e 'Naive Bayes'
'''
def regularization(X_train, y_train, X_cv, y_cv, inner_reg=True, algorithm='SVM', Cs=None):
	if Cs == None and algorithm.split()[0] == 'SVM':
		Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
	elif Cs == None and algorithm.split()[0] == 'NN':
		Cs = [1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1, 3, 10]
	elif Cs == None: # Ambos Naive Bayes
		Cs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
	regul, best_model = _regularization(X_train, y_train, X_cv, y_cv, Cs, algorithm)
	if not inner_reg:
		return regul, best_model
	best_C_idx = regul['CV_error'].index(min(regul['CV_error'])) # O melhor C escolhido é o que minimiza o CV_error
	best_C = regul['C'][best_C_idx]
	Cs = np.array(range(-5,6)) * 0.1 * best_C + best_C
	Cs = np.delete(Cs, 5) # Remove o valor central de new_Cs (best_C) para evitar repetição
	regul2, best_model = _regularization(X_train, y_train, X_cv, y_cv, Cs, algorithm)
	regul = _reg_update(regul, regul2)
	# Ordena de forma crescente a partir do parâmetro C
	regul['C'], regul['Train_error'], regul['CV_error'] = zip(*sorted(zip(regul['C'], regul['Train_error'], regul['CV_error'])))
	return regul, best_model

'''
Escolhe o valor de C que maximiza a medida F1 do modelo SVM.
Função privada que pode ser chamada repetidas vezes na função pública.
	algorithm, valores aceitos: 'SVM', 'SVM sigmoid', 'SVM poly [0-9]+' e 'Naive Bayes'
'''
def _regularization(X_train, y_train, X_cv, y_cv, Cs, algorithm):
	regul = {'C':[], 'Train_error':[], 'CV_error':[]}
	model = None
	best_model = None
	idx_best_cv_error = 0
	algorithm = algorithm.split()
	for C in Cs:
		if algorithm[0] == 'SVM':
			if len(algorithm) == 1: # 'SVM'
				model = svm.SVC(kernel='linear', C=C)
			elif algorithm[1] == 'sigmoid': # 'SVM sigmoid'
				model = svm.SVC(kernel='sigmoid', C=C, gamma='scale')
			elif algorithm[1] == 'poly': # 'SVM poly [0-9]+'
				model = svm.SVC(kernel='poly', degree=int(algorithm[2]), C=C, gamma='scale')
		elif algorithm[0] == 'NN':
			num1 = int(algorithm[1][1:-1]) if algorithm[1][-1:] == ',' else int(algorithm[1][1:-2])
			h_layer_sizes = (num1,) if algorithm[1][-1:] == ')' else (num1, int(algorithm[2][:-1]))
			solver = algorithm[3] if len(algorithm) > 3 else algorithm[2]
			model = NN(hidden_layer_sizes=h_layer_sizes, solver=solver, alpha=C)
		elif algorithm[0] == 'Multinomial':
			model = MultinomialNB(alpha=C)
		else:
			model = GaussianNB(var_smoothing=C)
		model.fit(X_train, y_train)
		if best_model == None:
			best_model = model
		y_train_pred = model.predict(X_train)
		train_error = 1 - metrics.f1_score(y_train, y_train_pred)
		y_cv_pred = model.predict(X_cv)
		cv_error = 1 - metrics.f1_score(y_cv, y_cv_pred)
		regul['C'].append(C)
		regul['Train_error'].append(train_error)
		regul['CV_error'].append(cv_error)
		if cv_error < regul['CV_error'][idx_best_cv_error]:
			idx_best_cv_error = len(regul['CV_error']) - 1
			#print('_regularization melhor C antes: %.5g, melhor C agora: %.5g' %(best_model.C, model.C)) # Debug!
			best_model = model
		#print('Train_error:', regul['Train_error'], 'C:', regul['C'], 'CV_error:', regul['CV_error']) # Debug!
	return regul, best_model

'''
Dado o melhor valor de C encontrado na regularização, best_C, retorna um array de 10 valores vizinhos próximos, 5
maiores e 5 menores que best_C a passos distantes de 10 em 10% do valor de best_C.
Exemplo:
	Se best_C = 2
	Cs = array([1, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3])
'''
def C_neighbors(best_C):
	Cs = np.array(range(-5,6)) * 0.1 * best_C + best_C
	Cs = np.delete(Cs, 5) # Remove o valor central de new_Cs (best_C) para evitar repetição
	return Cs

'''
Dados 2 resultados de regularização, concatena-os atualizando o valor do índice
de melhor valor de C
'''
def _reg_update(regul, regul2):
	res = {}
	res['C'] = regul['C'] + regul2['C'] # Concatena os valores de C utilizados
	res['Train_error'] = regul['Train_error'] + regul2['Train_error'] # Concatena os valores de Train_error obtidos
	res['CV_error'] = regul['CV_error'] + regul2['CV_error'] # Concatena os valores de CV_error obtidos
	return res

'''
	Retorna 3 métricas, na seguinte ordem:
	precisão, recall e F1-score.
'''
def all_metrics(model, X, y):
	y_pred = model.predict(X)
	return metrics.precision_score(y, y_pred), metrics.recall_score(y, y_pred), metrics.f1_score(y, y_pred)

def f1_score(model, X, y):
	y_pred = model.predict(X)
	return metrics.f1_score(y, y_pred)

def accuracy(model, X, y):
	y_pred = model.predict(X)
	return metrics.accuracy_score(y, y_pred)


