'''
Para estudar:
https://blog.barracuda.com/2013/10/03/ham-v-spam-whats-the-difference/
https://www.sciencedirect.com/science/article/abs/pii/S1566253518303968
'''

from sklearn.neighbors import KNeighborsClassifier
from util.feature_expander import FeatureExpander
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from pandas import DataFrame
from sklearn.svm import SVC
import time
import numpy as np
import util.preprocess as pre
import util.datasets as dat
import util.calculos as calc
import util.data_visualisation as dv
import util.estimator_selection as es


# Primeiros resultados, modelos simples, parâmetros default, a parte mais 
# trabalhada é a do pré-processamento de texto, sendo utilizada a mesma
# técnica do exercício em Octave
def etapa1():
	load_datasets_functions = [(dat.load_enron, 'Enron 3k'), \
			(dat.load_spamAssassin, 'Spam Assassin 3k'), \
			(dat.load_ling_spam, 'Ling Spam'), \
			(dat.load_TREC, 'TREC 3k')]

	for load_dataset in load_datasets_functions:
		X_text, y = load_dataset[0]() # Acessa o primeiro elemento da sequência: uma função para carregar o dataset
		X = pre.octave_preprocess(X_text)
		X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)

		# SVM section
		model = SVC(kernel='linear', gamma='scale') # Utilizando o valor padrão de C (1.0)
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='sigmoid', gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='poly', degree=2, gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='poly', degree=3, gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='poly', degree=4, gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='poly', degree=5, gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = SVC(kernel='poly', degree=7, gamma='scale')
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])

		# Naive Bayes section
		model = GaussianNB()
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])
		model = MultinomialNB()
		learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])

		# KNN section
		ks = [1, 3, 5, 10]
		for k in ks:
			model = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
			learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])

		for k in ks[1:]:
			model = KNeighborsClassifier(n_neighbors=k, n_jobs=4, weights='distance')
			learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, dataset=load_dataset[1])

		# Neural Network section


def learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, step=55, dataset='TREC 3k', etapa='E1'):
	# Obs.: calc.learning_curves() realiza fit() no modelo e, assim, ele é alterado
	# internamente, não precisa ser retornado. Dessa forma, ao utilizar este mesmo
	# modelo depois de executar calc.learning_curves() ele já estará treinado sobre
	# o conjunto X_train com os mesmos parâmetros que entrou.
	errors = calc.learning_curves(X_train, y_train, X_cv, y_cv, model, step=step)
	dv.save_learning_curves_f1(errors, etapa + ' ' + _get_model_params(model) + dataset)
	return _etapas_common(model, errors, X_test, y_test, dataset, etapa=etapa)
	### REMOVER ESTA PARTE abaixo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#model.fit(X_train, y_train)
	#f1_train = calc.f1_score(model, X_train, y_train)
	#f1_cv = calc.f1_score(model, X_cv, y_cv)
	#prec_test, rec_test, f1_test = calc.all_metrics(model, X_test, y_test)
	#return f1_train, f1_cv, prec_test, rec_test, f1_test



# Plot do gráfico do parâmetro de regularização C (curva de validação)
########### PARA ENTENDER MELHOR A CURVA DE VALIDAÇÃO, OLHAR SEMANA 6 DO COURSERA
def etapa2():
	load_datasets_functions = [(dat.load_enron, 'Enron 3k'), \
			(dat.load_spamAssassin, 'Spam Assassin 3k'), \
			(dat.load_ling_spam, 'Ling Spam'), \
			(dat.load_TREC, 'TREC 3k')]

	for load_dataset in load_datasets_functions:
		dataset = load_dataset[1]
		X_text, y = load_dataset[0]()
		X = pre.octave_preprocess(X_text)
		X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
		# SVM linear, sigmoid e poly
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM sigmoid')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM poly 2')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM poly 3')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM poly 4')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM poly 5')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='SVM poly 7')
		# Naive Bayes Gaussian e Multinomial
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='Naive Bayes')
		_etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset=dataset, algorithm='Multinomial')


'''
	algorithm, valores aceitos: 'SVM', 'SVM sigmoid', 'SVM poly [0-9]+' e 'Naive Bayes'
'''
def _etapa2(X_train, y_train, X_cv, y_cv, inner_reg=True, dataset='TREC 3k', algorithm='SVM'):
	regul, best_model = calc.regularization(X_train, y_train, X_cv, y_cv, inner_reg, algorithm)
	dv.save_validation_curve(regul, 'E2 ' + algorithm + ' ' + load_dataset[1])
	errors = calc.learning_curves(X_train, y_train, X_cv, y_cv, best_model, step=55)
	dv.save_learning_curves_f1(errors, 'E2 ' + _get_model_params(best_model) + load_dataset[1])
	_etapas_common(best_model, errors, X_test, y_test, load_dataset[1], etapa='E2')


'''
	Similar à etapa 2, porém utilizando o conjunto total de dados dos datasets que foram truncados.
'''
def etapa3():
	load_datasets_functions = [(dat.load_enron, 'Enron'), \
			(dat.load_spamAssassin, 'Spam Assassin'), \
			(dat.load_TREC, 'TREC')]

	for load_dataset in load_datasets_functions:
		step = 100 if load_dataset[1] == 'Spam Assassin' else 200 # foi usado 140 antes (para TREC)
		X_text, y = load_dataset[0](truncate=False)
		X = pre.octave_preprocess(X_text)
		X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
		models1, params1, scoring = _etapa3_GS()
		helper1 = es.EstimatorSelectionHelper(models1, params1)
		helper1.fit(np.concatenate((X_train, X_cv), axis=0), np.concatenate((y_train, y_cv), axis=0), scoring=scoring, n_jobs=-1, cv=3)
		helper1.score_summary(sort_by='min_score').to_csv('E3 ' + load_dataset[1] + '.csv')
		helper1.update_params()
		helper1.fit(np.concatenate((X_train, X_cv), axis=0), np.concatenate((y_train, y_cv), axis=0), scoring=scoring, n_jobs=-1, cv=3)
		helper1.score_summary(sort_by='min_score').to_csv('E3 ' + load_dataset[1] + ' regularized.csv')

		# Curvas de aprendizado de cada modelo regularizado
		params = helper1.retrieve_models_best_regparams()
		for model_k in helper1.models.keys():
			best_model = models1[model_k].set_params(**params[model_k])
			learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, best_model, step=step, dataset=load_dataset[1], etapa='E3')

		# Treinamento de KNN
		ks = [1, 3, 5, 10]
		for k in ks:
			model = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
			learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, step=step, dataset=load_dataset[1], etapa='E3')

		for k in ks[1:]:
			model = KNeighborsClassifier(n_neighbors=k, n_jobs=4, weights='distance')
			learning_curves_and_metrics(X_train, y_train, X_cv, y_cv, X_test, y_test, model, step=step, dataset=load_dataset[1], etapa='E3')


def _etapa3_GS():
	models1 = {
		'SVC_l': SVC(gamma='scale'),
		'SVC_s': SVC(gamma='scale'),
		'SVC_p2': SVC(gamma='scale'),
		'SVC_p3': SVC(gamma='scale'),
		'SVC_p4': SVC(gamma='scale'),
		'SVC_p5': SVC(gamma='scale'),
		'SVC_p7': SVC(gamma='scale'),
		'GaussianNB': GaussianNB(),
		'MultinomialNB': MultinomialNB()
	}
	params1 = {
		'SVC_l':  {'kernel': ['linear'],  'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
		'SVC_s':  {'kernel': ['sigmoid'], 'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
		'SVC_p2': {'kernel': ['poly'], 'C': [0.1, 0.3, 1, 3, 10, 30, 100, 300], 'degree':[2]},
		'SVC_p3': {'kernel': ['poly'], 'C': [0.3, 1, 3, 10, 30, 100, 300, 1000], 'degree':[3]},
		'SVC_p4': {'kernel': ['poly'], 'C': [1, 3, 10, 30, 100, 300, 1000, 3000], 'degree':[4]},
		'SVC_p5': {'kernel': ['poly'], 'C': [10, 30, 100, 300, 1000, 3000, 10000], 'degree':[5]},
		'SVC_p7': {'kernel': ['poly'], 'C': [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000], 'degree':[7]},
		'GaussianNB': {'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1]},
		'MultinomialNB': {'alpha': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1]}
		# O valor minimo de alpha em MultinomialNB é 1e-10, segundo o código do scikit-learn em
		# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
	}
	scoring = {'f1'}

	return models1, params1, scoring


'''
	params: deve ser no formato 'SVC_l': {'kernel': 'linear', 'C': 0.01}, isto é, apenas um valor
			por parâmetro de cada modelo, não uma lista de possíveis valores de parâmetros.
'''
def _etapa3_learning_curves(X_train, y_train, X_cv, y_cv, X_test, y_test, models, params, dataset):
	step = 100 if dataset == 'Spam Assassin' else 200 # foi usado 140 antes (para TREC)
	for model_k in models.keys():
		best_model = models[model_k].set_params(**params[model_k])
		errors = calc.learning_curves(X_train, y_train, X_cv, y_cv, best_model, step=step)
		dv.save_learning_curves_f1(errors, 'E3 ' + _get_model_params(best_model) + dataset)
		_etapas_common(best_model, errors, X_test, y_test, dataset, etapa='E3')


'''
	A partir de alguns modelos treinados na etapa 2, este método armazena em txt emails que foram
	incorretamente classificados a fim de realizar a etapa manual da metodologia: analizar padrões
	em tais email e procurar modelar novas features para a etapa 4 do experimento.
'''
def pre_etapa4():
	Cs = [0.021, 0.07, 0.15, 0.024]
	for c in Cs:
		model = SVC(kernel='linear', C=c, gamma='scale')
		wrong_predicted_emails_report(model)


def wrong_predicted_emails_report(model):
	load_datasets_functions = [(dat.load_enron, 'Enron 3k'), \
		(dat.load_spamAssassin, 'Spam Assassin 3k'), \
		(dat.load_ling_spam, 'Ling Spam'), \
		(dat.load_TREC, 'TREC 3k')]

	for n, load_dataset in enumerate(load_datasets_functions):
		print('Preparando relatório para o dataset ' + load_dataset[1] + '...')
		X_text, y = load_dataset[0]()
		X = pre.octave_preprocess(X_text)
		X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
		model.fit(X_train, y_train)
		y_pred = model.predict(X)
		f_name = 'Erro de classificacao ' + load_dataset[1] + _get_model_params(model) + '.txt'

		with open(f_name, 'a') as f:
			i = 0
			while(i < len(y)):
				if y[i] != y_pred[i]:
					f.write('### ### ### (y: '+ str(y[i]) + ', y_pred: ' + str(y_pred[i]) +') ### ### ###\n\n' + X_text[i] + '\n\n')
				i += 1		


def etapa4_truncated():
	load_datasets_functions = [(dat.load_enron, 'Enron 3k'), \
		(dat.load_spamAssassin, 'Spam Assassin 3k'), \
		(dat.load_ling_spam, 'Ling Spam'), \
		(dat.load_TREC, 'TREC 3k')]

	last_new_features_idx = 61
	total_features = range(last_new_features_idx +1)

	for load_dataset in load_datasets_functions:
		print('\nCarregando o dataset ' + load_dataset[1] + '...')
		ds_start_time = time.time()
		X_text, y = load_dataset[0]()
		X = pre.octave_preprocess(X_text)
		dic_etapa4 = {'Added feature': [], 'SVM Linear': [], 'MultinomialNB': [], 'KNN K=1': [], 'KNN K=5 Distance': []}

		for idx in total_features:
			print('\nExtraindo features...')
			start_time = time.time()
			X_f, added_feature = _etapa4_features(X_text, X, idx)
			dic_etapa4['Added feature'].append(added_feature)
			print('Feature ' + added_feature + ' adicionada, ' + _tempo_passado(start_time) + '\n')
			print('Treinando modelos...\n')
			start_time = time.time()
			X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X_f, y, train_part=60, test_part=20, randsplit=False)
			models1, params1, scoring = _etapa4_truncated_GS()
			helper1 = es.EstimatorSelectionHelper(models1, params1)
			helper1.fit(np.concatenate((X_train, X_cv), axis=0), np.concatenate((y_train, y_cv), axis=0), scoring=scoring, n_jobs=-1, cv=3)

			print('\n\nResultados, '+ added_feature+ ' dataset '+ load_dataset[1]+ ' (treinamento em '+ _tempo_passado(start_time)+ '):\n')
			params = helper1.retrieve_models_best_regparams()
			for model_k in helper1.models.keys():
				model = models1[model_k].set_params(**params[model_k])
				model.fit(X_train, y_train)
				f1 = calc.f1_score(model, X_test, y_test)
				model_name = model_k if model_k == 'MultinomialNB' else 'SVM Linear'
				dic_etapa4[model_name].append(str(round(f1, 3)))
				print(_get_model_params(model) + '\nF1 test: %.4f\n' %f1)

			# KNN K=1 e K=5 distance, em geral, foram os modelos de melhor desempenho dentre os KNNs
			# KNN K=1
			model = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
			model.fit(X_train, y_train)
			f1 = calc.f1_score(model, X_test, y_test)
			dic_etapa4[_get_model_params(model).strip()].append(str(round(f1, 3)))
			print(_get_model_params(model) + '\nF1 test: %.4f\n' %f1)

			# KNN K=5 Distance
			model = KNeighborsClassifier(n_neighbors=5, n_jobs=4, weights='distance')
			model.fit(X_train, y_train)
			f1 = calc.f1_score(model, X_test, y_test)
			dic_etapa4[_get_model_params(model).strip()].append(str(round(f1, 3)))
			print(_get_model_params(model) + '\nF1 test: %.4f\n' %f1)

		print('\nTempo total no dataset ' + load_dataset[1] + ' ' + _tempo_passado(ds_start_time))
		df = DataFrame(dic_etapa4)
		df.to_csv('resultados E4 ' + load_dataset[1] + ' truncado.csv', index=False)


def etapa4():
	load_datasets_functions = [(dat.load_enron, 'Enron', 240), \
		(dat.load_spamAssassin, 'Spam Assassin', 120), \
		(dat.load_ling_spam, 'Ling Spam', 55), \
		(dat.load_TREC, 'TREC', 240)]

	dic_etapa4 = {'added_feature': [], 'model_params': [], 'f1_train': [], \
				'f1_cv': [], 'prec_test': [], 'rec_test': [], 'f1_test': []}

	for load_dataset in [load_datasets_functions[3], load_datasets_functions[0]]:
		print('\nCarregando o dataset ' + load_dataset[1] + '...')
		ds_start_time = time.time()
		X_text, y = load_dataset[0](truncate=False)
		print('Tempo de carregamento ' + _tempo_passado(ds_start_time))
		is_first = [True, False] # Para alternar entre os 2 tipos de pré-processamento de cada dataset
		step = load_dataset[2]

		for first in [False]:#is_first: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ALTERAR
			preproc = 'pre1' if first else 'pre2'
			print('\nExtraindo features...')
			start_time = time.time()
			X, added_feature = _etapa4(X_text, is_first=first, dataset=load_dataset[1])
			del X_text ####<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVER!!!!!!
			print('Feature ' + added_feature + ' adicionada, ' + _tempo_passado(start_time))
			print('\nEscolhendo parâmetros...\n')
			start_time = time.time()
			X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
			models1, params1, scoring = _etapa4_GS()
			helper1 = es.EstimatorSelectionHelper(models1, params1)
			helper1.fit(np.concatenate((X_train, X_cv), axis=0), np.concatenate((y_train, y_cv), axis=0), scoring=scoring, n_jobs=-1, cv=3)
			params = helper1.retrieve_models_best_regparams()
			print('\nParâmetros encontrados em ' + _tempo_passado(start_time))

			print('\nTreinando modelos...\n')
			start_time = time.time()
			for model_k in helper1.models.keys():
				model = models1[model_k].set_params(**params[model_k])
				f1_train, f1_cv, prec_test, rec_test, f1_test = learning_curves_and_metrics(X_train, y_train, \
							X_cv, y_cv, X_test, y_test, model, step=step, dataset=load_dataset[1], etapa='E4 ' + preproc)
				model_params = _get_model_params(model)
				for k in dic_etapa4.keys():
					value = eval(k)
					str_or_num = value if type(value) == type('') else round(value, 5)
					dic_etapa4[k].append(str_or_num)

			# Treinamento de KNN
			ks = [1, 3, 5, 10]
			for k in ks:
				model = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
				f1_train, f1_cv, prec_test, rec_test, f1_test = learning_curves_and_metrics(X_train, y_train, \
							X_cv, y_cv, X_test, y_test, model, step=step, dataset=load_dataset[1], etapa='E4 ' + preproc)
				model_params = _get_model_params(model)
				for k in dic_etapa4.keys():
					value = eval(k)
					str_or_num = value if type(value) == type('') else round(value, 5)
					dic_etapa4[k].append(str_or_num)

			for k in ks[1:]:
				model = KNeighborsClassifier(n_neighbors=k, n_jobs=4, weights='distance')
				f1_train, f1_cv, prec_test, rec_test, f1_test = learning_curves_and_metrics(X_train, y_train, \
							X_cv, y_cv, X_test, y_test, model, step=step, dataset=load_dataset[1], etapa='E4 ' + preproc)
				model_params = _get_model_params(model)
				for k in dic_etapa4.keys():
					value = eval(k)
					str_or_num = value if type(value) == type('') else round(value, 5)
					dic_etapa4[k].append(str_or_num)

			df = DataFrame(dic_etapa4)
			df.to_csv('resultados E4 ' + load_dataset[1] + '.csv', mode='a', index=False, header=first)
			print('\nModelos treinados em ' + _tempo_passado(start_time))

		print('\nTempo total no dataset ' + load_dataset[1] + ' ' + _tempo_passado(ds_start_time))


def _etapa4_GS():
	models1, params1, scoring = _etapa3_GS()
	params1['SVC_l']['C'] += [30]
	params1['SVC_s']['C'] += [30]
	params1['SVC_p4']['C'] += [10000, 30000]
	params1['SVC_p5']['C'] = [1, 3] + params1['SVC_p5']['C'] + [30000, 100000, 300000]
	params1['SVC_p7']['C'] = [1, 3] + params1['SVC_p7']['C'] + [300000, 1e6, 3e6, 1e7, 3e7, 1e8]
	params1['GaussianNB']['var_smoothing'] += [3, 10, 30]
	params1['MultinomialNB']['alpha'] += [3, 10, 30, 100, 300]
	return models1, params1, scoring


def _etapa4(X_text, is_first, dataset):

	if dataset == 'Enron':
		if is_first:
			return _etapa4_features(X_text, X=None, idx=8)  # idx 8:  'TF-IDF min_df=5 no_accents stop_words'
		else:
			return _etapa4_features(X_text, X=None, idx=61) # idx 61: TF-IDF + todas as features normalizadas + WSD max 3 sentidos

	elif dataset == 'Spam Assassin':
		if is_first:
			return _etapa4_features(X_text, X=None, idx=6)  # idx 6: 'TF-IDF min_df=5 max_features=3000'
		else:
			return _etapa4_features(X_text, X=None, idx=60) # idx 60: TF-IDF + todas as features normalizadas + WSD max 2 sentido

	elif dataset == 'Ling Spam':
		if is_first:
			return _etapa4_features(X_text, X=None, idx=5)  # idx 5:  'TF-IDF min_df=5 max_features=1000'
		else:
			return _etapa4_features(X_text, X=None, idx=10) # idx 10: 'TF-IDF min_df=5 no_accents stop_words preproc stem'

	elif dataset == 'TREC':
		if is_first:
			return _etapa4_features(X_text, X=None, idx=5)  # idx 5:  'TF-IDF min_df=5 max_features=1000'
		else:
			return _etapa4_features(X_text, X=None, idx=59) # idx 59: TF-IDF + todas as features normalizadas + WSD max 1 sentido


'''
	X_text: lista de strings, que são os emails
	X: X_text pré-processado no formato 1-hot encoding
	idx:
		[0]     Sem adição de novas features
		[1~5]   WSD com max_senses de valores [1, 2, 3, 5, 7]
		[6~10]  adiciona pos_tags sem e com normalização e utilizando a procentagem de cada tag
				com relação ao total de palavras do texto
		[11~16] adiciona largest_uppercase_seq padrão, na forma 1-hot, com normalização e com a
				porcentagem de palavras em maiúsculo com relação ao total de palavras do texto
		[17~21] adiciona count_punctuations padrão, na forma 1-hot, com normalização e 
				limitando o valor das features na normalização
		[22~27] conta tags html, utiliza a contagem padrão, na forma 1-hot, com normalização e 
				utilizando a porcentagem de cada tag com relação ao total de tags do email
'''
def _etapa4_features(X_text, X, idx):
	features_list = ['None'] + 10*['TF-IDF'] + 5*['WSD max_senses=%d'] + \
		5*['pos_tag'] + 6*['upper_case'] + 5*['punctuations'] + \
		6*['seek_tag'] + 6*['seek_URL'] + 7*['upper_case + seek_URL'] + \
		11*['upper_case']

	added_feature = features_list[idx]
	fe = FeatureExpander()

	# [0]: sem adição de novas features, apenas repassa o que recebeu
	if idx == 0:
		return (X, added_feature)
	last_idx = 0

	# [1~10]: tf-idf()
	if idx <= 10:
		X = None
		original_max_features = 1899
		if idx == last_idx + 1:
			min_df = 1
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d' %min_df

		elif idx == last_idx + 2:
			min_df = 5
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d' %min_df

		elif idx == last_idx + 3:
			min_df = 10
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d' %min_df

		elif idx == last_idx + 4:
			min_df = 30
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d' %min_df

		elif idx == last_idx + 5:
			min_df = 5
			max_features = 1000
			X = pre.tf_idf(X_text, max_features=max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d max_features=%d' %(min_df, max_features)

		elif idx == last_idx + 6:
			min_df = 5
			max_features = 3000
			X = pre.tf_idf(X_text, max_features=max_features, min_df=min_df, strip_accents=None, stop_words=None)
			added_feature += ' min_df=%d max_features=%d' %(min_df, max_features)

		elif idx == last_idx + 7:
			min_df = 5
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents='unicode', stop_words=None)
			added_feature += ' min_df=%d no_accents' %min_df

		elif idx == last_idx + 8:
			min_df = 5
			X = pre.tf_idf(X_text, max_features=original_max_features, min_df=min_df, strip_accents='unicode', stop_words='english')
			added_feature += ' min_df=%d no_accents stop_words' %min_df

		elif idx == last_idx + 9:
			min_df = 5
			X_p_text = pre.op.preprocess(X_text, stemming=False)
			X = pre.tf_idf(X_p_text, max_features=original_max_features, min_df=min_df, strip_accents='unicode', stop_words='english')
			added_feature += ' min_df=%d no_accents stop_words preproc' %min_df

		elif idx == last_idx + 10:
			min_df = 5
			X_ps_text = pre.op.preprocess(X_text, stemming=True)
			X = pre.tf_idf(X_ps_text, max_features=original_max_features, min_df=min_df, strip_accents='unicode', stop_words='english')
			added_feature += ' min_df=%d no_accents stop_words preproc stem' %min_df

		return (X, added_feature)
	last_idx += 10

	# [11~15]: wsd()
	if idx <= 5:
		max_senses_list = [1, 2, 3, 5, 7]
		added_feature = added_feature %max_senses_list[idx-1]
		X_wsd = fe.wsd(X_text, max_senses=max_senses_list[idx-1])
		return (pre.octave_preprocess(X_wsd), added_feature)
	last_idx += 5

	# [16~20]: pos_tags()
	if idx <= last_idx + 5:
		dic_pos_tags = fe.pos_tags(X_text)

		if idx == last_idx + 2:
			dic_pos_tags = fe.dic_normalize(dic_pos_tags)
			added_feature += ' norm'

		elif idx == last_idx + 3:
			feature_max_val = 50
			dic_pos_tags = fe.dic_normalize(dic_pos_tags, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		elif idx == last_idx + 4:
			total_words = fe.count_words(X_text)
			dic_pos_tags = fe.dic_normalize_percent(dic_pos_tags, total_words)
			added_feature += ' percent'

		elif idx == last_idx + 5:
			total_words = fe.count_words(X_text)
			dic_pos_tags = fe.dic_normalize_percent(dic_pos_tags, total_words)
			dic_pos_tags = fe.dic_normalize(dic_pos_tags)
			added_feature += ' percent norm'

		return (fe.add_dic_features_to_npmatrix(X, dic_pos_tags), added_feature)
	last_idx += 5

	# [21~26]: largest_uppercase_seq()
	if idx <= last_idx + 6:
		dic_case = fe.largest_uppercase_seq(X_text)

		if idx == last_idx + 2:
			dic_case = fe.dic_features_to_one_hot(dic_case)
			added_feature += ' 1-hot'

		elif idx == last_idx + 3:
			dic_case = fe.dic_normalize(dic_case)
			added_feature += ' norm'

		elif idx == last_idx + 4:
			feature_max_val = 100
			dic_case = fe.dic_normalize(dic_case, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		elif idx == last_idx + 5:
			total_words = fe.count_words(X_text)
			dic_case = fe.dic_normalize_percent(dic_case, total_words)
			added_feature += ' percent'

		elif idx == last_idx + 6:
			total_words = fe.count_words(X_text)
			dic_case = fe.dic_normalize_percent(dic_case, total_words)
			dic_case = fe.dic_normalize(dic_case)
			added_feature += ' percent norm'

		return (fe.add_dic_features_to_npmatrix(X, dic_case), added_feature)
	last_idx += 6

	# [27~31]: count_punctuations()
	if idx <= last_idx + 5:
		dic_punct = fe.count_punctuations(X_text)

		if idx == last_idx + 2:
			dic_punct = fe.dic_features_to_one_hot(dic_punct)
			added_feature += ' 1-hot'

		elif idx == last_idx + 3:
			dic_punct = fe.dic_normalize(dic_punct)
			added_feature += ' norm'

		elif idx == last_idx + 4:
			feature_max_val = 5
			dic_punct = fe.dic_normalize(dic_punct, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		elif idx == last_idx + 5:
			feature_max_val = 3
			dic_punct = fe.dic_normalize(dic_punct, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		return (fe.add_dic_features_to_npmatrix(X, dic_punct), added_feature)
	last_idx += 5

	# [32~37]: seek_tag()
	if idx <= last_idx + 6:
		dic_tags = fe.seek_tag(X_text)

		if idx == last_idx + 2:
			dic_tags = fe.dic_features_to_one_hot(dic_tags)
			added_feature += ' 1-hot'

		elif idx == last_idx + 3:
			dic_tags = fe.dic_normalize(dic_tags)
			added_feature += ' norm'

		elif idx == last_idx + 4:
			total_any_tag = dic_tags.pop('total_any_tag')
			dic_tags = fe.dic_normalize_percent(dic_tags, total_any_tag)
			dic_tags['total_any_tag'] = fe.feature_to_one_hot(total_any_tag)
			added_feature += ' percent'

		elif idx == last_idx + 5:
			total_any_tag = dic_tags.pop('total_any_tag')
			dic_tags = fe.dic_normalize_percent(dic_tags, total_any_tag)
			dic_tags = fe.dic_normalize(dic_tags)
			dic_tags['total_any_tag'] = fe.feature_to_one_hot(total_any_tag)
			added_feature += ' percent norm'

		elif idx == last_idx + 6:
			total_any_tag = dic_tags.pop('total_any_tag')
			dic_tags = fe.dic_normalize_percent(dic_tags, total_any_tag)
			dic_tags = fe.dic_normalize(dic_tags)
			feature_max_val = 30
			dic_tags['total_any_tag'] = fe.normalize(total_any_tag, feature_max_val=feature_max_val)
			added_feature += ' percent norm2'

		return (fe.add_dic_features_to_npmatrix(X, dic_tags), added_feature)
	last_idx += 6

	# [38~43]: seek_URL()
	if idx <= last_idx + 6:
		urls = fe.seek_URL(X_text)

		if idx == last_idx + 2:
			urls = fe.feature_to_one_hot(urls)
			added_feature += ' 1-hot'

		elif idx == last_idx + 3:
			urls = fe.normalize(urls)
			added_feature += ' norm'

		elif idx == last_idx + 4:
			feature_max_val = 7
			urls = fe.normalize(urls, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		elif idx == last_idx + 5:
			feature_max_val = 5
			urls = fe.normalize(urls, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		elif idx == last_idx + 6:
			feature_max_val = 3
			urls = fe.normalize(urls, feature_max_val=feature_max_val)
			added_feature += ' norm f_max_val=%d' %feature_max_val

		return (fe.add_feature_to_npmatrix(X, urls), added_feature)
	last_idx += 6

	# [44~50]: upper_case + seek_URL + punctuations + seek_tag + pos_tag percent...
	if idx <= last_idx + 7:
		# upper_case
		dic_case = fe.largest_uppercase_seq(X_text)
		X_all_features = fe.create_npmatrix_from_dic_features(dic_case)
		# seek_URL
		urls = fe.seek_URL(X_text)
		X_all_features = fe.add_feature_to_npmatrix(X_all_features, urls)

		if idx >= last_idx + 2:
			# punctuations
			dic_punct = fe.count_punctuations(X_text)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_punct)
			added_feature += ' + punctuations'

		if idx >= last_idx + 3:
			# seek_tag
			dic_tags = fe.seek_tag(X_text)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_tags)
			added_feature += ' + seek_tag'

		if idx >= last_idx + 4:
			# pos_tag percent
			dic_pos_tags = fe.pos_tags(X_text)
			total_words = fe.count_words(X_text)
			dic_pos_tags = fe.dic_normalize_percent(dic_pos_tags, total_words)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_pos_tags)
			added_feature += ' + pos_tag percent'

		if idx == last_idx + 5:
			# TF-IDF min_df=5 no_accents + pos_tag percent
			added_feature = 'TF-IDF min_df=5 no_accents + ' + added_feature
			X_tfidf = pre.tf_idf(X_text, max_features=1899, min_df=5, strip_accents='unicode')
			return (fe.add_feature_to_npmatrix(X_tfidf, X_all_features), added_feature)

		if idx == last_idx + 6:
			# WSD max_senses=1
			max_senses = 1
			X_wsd = fe.wsd(X_text, max_senses=max_senses)
			X_wsd = pre.octave_preprocess(X_wsd)
			added_feature += ' + WSD max_senses=%d' %max_senses
			return (fe.add_feature_to_npmatrix(X_wsd, X_all_features), added_feature)

		if idx == last_idx + 7:
			# TF-IDF min_df=5 no_accents + WSD max_senses=1
			max_senses = 1
			X_wsd = fe.wsd(X_text, max_senses=max_senses)
			X_wsd_tfidf = pre.tf_idf(X_wsd, max_features=1899, min_df=5, strip_accents='unicode')
			added_feature = 'TF-IDF min_df=5 no_accents + ' + added_feature
			added_feature += ' + WSD max_senses=%d' %max_senses
			return (fe.add_feature_to_npmatrix(X_wsd_tfidf, X_all_features), added_feature)

		return (fe.add_feature_to_npmatrix(X, X_all_features), added_feature)
	last_idx += 7

	# [51~61]: upper_case + seek_URL + punctuations + seek_tag + pos_tag percent...
	if idx <= last_idx + 11:
		# upper_case norm f_max_val=100
		feature_max_val = 100
		dic_case = fe.largest_uppercase_seq(X_text)
		dic_case = fe.dic_normalize(dic_case, feature_max_val=feature_max_val)
		X_all_features = fe.create_npmatrix_from_dic_features(dic_case)
		added_feature += ' norm f_max_val=%d' %feature_max_val
		# seek_URL 1-hot
		urls = fe.seek_URL(X_text)
		urls = fe.feature_to_one_hot(urls)
		X_all_features = fe.add_feature_to_npmatrix(X_all_features, urls)
		added_feature += ' + seek_URL 1-hot'

		if idx >= last_idx + 2:
			# punctuations 1-hot
			dic_punct = fe.count_punctuations(X_text)
			dic_punct = fe.dic_features_to_one_hot(dic_punct)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_punct)
			added_feature += ' + punctuations 1-hot'

		if idx >= last_idx + 3:
			# seek_tag seek_tag 1-hot
			dic_tags = fe.seek_tag(X_text)
			dic_tags = fe.dic_features_to_one_hot(dic_tags)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_tags)
			added_feature += ' + seek_tag 1-hot'

		if idx >= last_idx + 4:
			# pos_tag norm f_max_val=50
			feature_max_val = 50
			dic_pos_tags = fe.pos_tags(X_text)
			dic_pos_tags = fe.dic_normalize(dic_pos_tags, feature_max_val=feature_max_val)
			X_all_features = fe.add_dic_features_to_npmatrix(X_all_features, dic_pos_tags)
			added_feature += ' + pos_tag norm f_max_val=%d' %feature_max_val

		if idx == last_idx + 5:
			# TF-IDF min_df=5 no_accents + pos_tag norm f_max_val=50
			added_feature = 'TF-IDF min_df=5 no_accents + ' + added_feature
			X_tfidf = pre.tf_idf(X_text, max_features=1899, min_df=5, strip_accents='unicode')
			return (fe.add_feature_to_npmatrix(X_tfidf, X_all_features), added_feature)

		if idx >= last_idx + 6 and idx < last_idx + 9:
			# WSD max_senses
			max_senses_list = [1, 2, 3]
			X_wsd = fe.wsd(X_text, max_senses=max_senses_list[idx-last_idx-6])
			X_wsd = pre.octave_preprocess(X_wsd)
			added_feature += ' + WSD max_senses=%d' %max_senses_list[idx-last_idx-6]
			return (fe.add_feature_to_npmatrix(X_wsd, X_all_features), added_feature)

		if idx >= last_idx + 9:
			# WSD max_senses
			max_senses_list = [1, 2, 3]
			X_wsd = fe.wsd(X_text, max_senses=max_senses_list[idx-last_idx-9])
			added_feature = 'TF-IDF min_df=5 no_accents + ' + added_feature
			added_feature += ' + WSD max_senses=%d' %max_senses_list[idx-last_idx-9]
			X_wsd_tfidf = pre.tf_idf(X_wsd, max_features=1899, min_df=5, strip_accents='unicode')
			return (fe.add_feature_to_npmatrix(X_wsd_tfidf, X_all_features), added_feature)

		return (fe.add_feature_to_npmatrix(X, X_all_features), added_feature)


def _etapa4_truncated_GS():
	models1 = {
		'SVC_l': SVC(gamma='scale'),
		'MultinomialNB': MultinomialNB()
	}
	params1 = {
		'SVC_l':  {'kernel': ['linear'],  'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
		'MultinomialNB': {'alpha': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10, 30, 100, 300, 1000]}
	}
	scoring = {'f1'}
	return models1, params1, scoring


def top_features_etapa4(): #### ALTERAR 123 PARA O VALOR DE C DO SVM LINEAR E 5 PARA O idx UTILIZADO NA ETAPA4 DE CADA DATASET!!!!
	load_datasets_functions = [(dat.load_enron, 'Enron', 5, 123), \
		(dat.load_spamAssassin, 'Spam Assassin', 5, 123), \
		(dat.load_ling_spam, 'Ling Spam', 5, 123), \
		(dat.load_TREC, 'TREC', 5, 123)]

	for load_dataset in load_datasets_functions:
		X_text, y = load_dataset[0](truncate=False)
		X, added_feature = _etapa4_features(X_text, X=None, idx=load_dataset[2])
		del X_text
		X_train, X_cv, X_test, y_train, y_cv, y_test = pre.split_dataset(X, y, train_part=60, test_part=20, randsplit=False)
		model = SVC(kernel='linear', C=load_dataset[3], gamma='score')
		model.fit(X_train, y_train)
		weight_idxs = np.argsort(model.coef_)[0] # argsort retorna os índices que ordenam model.coef_
		for weight_idx in weight_idxs[::-1]: # [::-1] para a ordem decrescente dos pesos
			print('Idx: %d, peso: %.3f' %(weight_idx, model.coef_[weight_idx]))


def _top_features_etapa4(idx, X_text, model):
	vectorizer = None
	if idx == 5:
		vectorizer = pre.TfidfVectorizer(max_features=1899, min_df=5, strip_accents='unicode', \
							stop_words=None, lowercase=True)
	X_tfidf = vectorizer.fit_transform(X_text).toarray()
	# TODO: implementar aqui o acesso à cada feature de acordo com o índice, concatenar 
	features_list = vectorizer.get_feature_names()
	features_list.append('$') # adicionar aqui os nomes de cada feature, em ordem!!
	# talvez tenha q refazer _etapa4_features aqui para adicionar as keywords a uma lista, isso já automatizaria o processo
	return features_list


def _etapas_common(model, errors, X_test, y_test, dataset='TREC 3k', etapa='E1'):
	algorithm = _get_model_params(model)
	prec_test, rec_test, f1_test = calc.all_metrics(model, X_test, y_test)
	f1_train = (1-errors['train'][len(errors['train']) - 1])
	f1_cv = (1-errors['cv'][len(errors['cv']) - 1])
	print(etapa + ' ' + algorithm + dataset)
	print('F1-score treino:    %.5f' %f1_train)
	print('F1-score validação: %.5f' %f1_cv)
	_print_all_metrics(prec_test, rec_test, f1_test)
	return f1_train, f1_cv, prec_test, rec_test, f1_test


def	_print_all_metrics(prec_test, rec_test, f1_test):
	print('Precisão teste:     %.5f' %prec_test)
	print('Recall teste:       %.5f' %rec_test)
	print('F1-score teste:     %.5f' %f1_test + '\n')


def _get_model_params(model):
	algorithm = str(model).split('(')[0]
	alg_dic = {'SVC': 'SVM C=%.5g ', \
			'KNeighborsClassifier': 'KNN K=%d ', \
			'GaussianNB': 'Naive Bayes var_smoothing=%.3e ', \
			'MultinomialNB': 'Multinomial NB alpha=%.3e '}

	if algorithm == 'SVC':
		algorithm = alg_dic[algorithm] %model.C
		if model.kernel == 'poly':
			algorithm += 'poly ' + str(model.degree) + ' '
		elif model.kernel == 'sigmoid':
			algorithm += 'sigmoid '

	elif algorithm == 'KNeighborsClassifier':
		algorithm = alg_dic[algorithm] %model.n_neighbors + ('Distance ' if model.weights == 'distance' else '')

	elif algorithm == 'GaussianNB':
		algorithm = alg_dic[algorithm] %model.var_smoothing

	elif algorithm == 'MultinomialNB':
		algorithm = alg_dic[algorithm] %model.alpha

	return algorithm


def _get_model_name(model):
	algorithm = str(model).split('(')[0]
	if algorithm == 'SVC':
		algorithm = 'SVM'
		if model.kernel == 'poly':
			algorithm += ' poly ' + str(model.degree)
		elif model.kernel == 'sigmoid':
			algorithm += ' sigmoid'

	elif algorithm == 'KNeighborsClassifier':
		algorithm = 'KNN K=%d' %model.n_neighbors + (' Distance ' if model.weights == 'distance' else '')
	# GaussianNB e MultinomialNB não precisam ser editados

	return algorithm


def _tempo_passado(start_time):
	mins = (time.time()-start_time)//60
	hours = mins//60
	mins = mins%60
	secs = (time.time()-start_time)%60
	tempo = '%dh '%hours if hours > 0 else ''
	tempo += '%dmin '%mins if (mins > 0 or hours > 0) else ''
	tempo += '%ds'%secs
	return tempo


