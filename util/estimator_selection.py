from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class EstimatorSelectionHelper:

	def __init__(self, models, params):
		if not set(models.keys()).issubset(set(params.keys())):
			missing_params = list(set(models.keys()) - set(params.keys()))
			raise ValueError("Some estimators are missing parameters: %s" % missing_params)
		self.models = models
		self.params = params
		self.keys = models.keys()
		self.grid_searches = {}

	def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
		for key in self.keys:
			print("Running GridSearchCV for %s." % key)
			model = self.models[key]
			params = self.params[key]
			gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring, refit=refit, return_train_score=False)
			gs.fit(X, y)
			self.grid_searches[key] = gs

	def score_summary(self, sort_by='mean_score'):
		df = None
		for k in self.keys:
			self.grid_searches[k].cv_results_['estimator'] = k
			if df is not None :
				df = pd.concat([df, pd.DataFrame(self.grid_searches[k].cv_results_)])
			else:
				df = pd.DataFrame(self.grid_searches[k].cv_results_)
		return df

	'''
		Atualiza os parâmetros C de SVC e var_smooth de GaussianNB para seus vizinhos mais próximos de
		acordo com o melhor valor test_f1 encontrado em self.grid_searches.
	'''
	def update_params(self):
		for model in self.models.keys():
			if self.params.get(model) is not None:
				new_params = self.param_neighbors(self.best_param(model))
				if model.startswith('SVC'):
					self.params[model]['C'] = new_params
				elif model.startswith('Gaussian'): # GaussianNB
					self.params[model]['var_smoothing'] = new_params
				elif model.startswith('Multinomial'): # MultinomialNB
					self.params[model]['alpha'] = new_params

	def best_param(self, model):
		idx = self.grid_searches[model].cv_results_['rank_test_f1'].tolist().index(1)
		return self.grid_searches[model].cv_results_[self._regul_param_name_GS(model)][idx]

	'''
	Dado o melhor valor do parâmetro encontrado na regularização, param, retorna um array de 10 valores vizinhos próximos, 5
	maiores e 5 menores que param a passos distantes de 10 em 10% do valor de param.
	Exemplo:
		Se param = 2
		então [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
	'''
	def param_neighbors(self, param):
		params = np.array(range(-5,6)) * 0.1 * param + param
		#params = np.delete(Cs, 5) # Remove o valor central de new_Cs (best_C) para evitar repetição
		return params.tolist()

	'''
	Nome do parâmetro de regularização do modelo em questão.
	Exemplo:
		Se model = 'SVC'
		então 'C'
	'''
	def _regul_param_name(self, model):
		if model.startswith('SVC'): return 'C'
		if model.startswith('Gaussian'): return 'var_smoothing'
		if model.startswith('Multinomial'): return 'alpha'

	'''
	Nome do parâmetro de regularização do modelo no Grid Search. Apenas adiciona 'param_' à 
	resposta de _regul_param_name(model).
	Exemplo:
		Se model = 'SVC'
		então 'param_C'
	'''
	def _regul_param_name_GS(self, model): 
		return 'param_' + self._regul_param_name(model)

	'''
	Modelos e seu melhor parâmetros de regularização de acordo com o rank do Grid Search.
	Também retorna parâmetros que sejam valores únicos de uma lista.
	Exemplo:
		params = {
			'SVC_l' : {'kernel': ['linear'], 'C': [0.03, 0.1, 0.3, 1]}, 
			'SVC_p3' : {'kernel': ['poly'], 'C': [1, 3, 10, 30]}, 
			'MultinomialNB' : {'alpha' : [0.003, 0.001, 0.03]}
		}

		Um possível retorno será:

		models_params = {
			'SVC_l' : {'kernel': 'linear', 'C': 0.1}, 
			'SVC_p3' : {'kernel': 'poly', 'C': 3}, 
			'MultinomialNB' : {'alpha' : 0.003}
		}
	'''
	def retrieve_models_best_regparams(self):
		models_params = {}
		for model in self.models.keys():
			models_params[model] = {self._regul_param_name(model) : self.best_param(model)}
			# Para parâmetros com apenas 1 valore em uma lista, estes se tornarão valores únicos,
			# não há necessidade de estarem em listas
			for param in self.params[model].keys():
				if len(self.params[model][param]) == 1:
					models_params[model][param] = self.params[model][param][0]
		return models_params


