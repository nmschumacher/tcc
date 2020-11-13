import matplotlib.pyplot as plt


def plot_learning_curves(errors):
	_learning_curves_common(errors)
	plt.show()
	plt.clf()

# Mostra o último F1-score de ambos os conjuntos de treino e de validação
# Exemplo adaptado do link abaixo
# https://stackoverflow.com/questions/52240633/matplotlib-display-value-
# next-to-each-point-on-chart
def plot_learning_curves_f1(errors):
	_learning_curves_f1(errors)
	plt.show()
	plt.clf()

def save_learning_curves_f1(errors, fname):
	fname = fname if fname.endswith('.png') else fname + '.png'
	_learning_curves_f1(errors)
	plt.savefig(fname)
	plt.clf()

def _learning_curves_f1(errors):
	last_error = len(errors['num_examples']) - 1
	m = errors['num_examples'][last_error]
	f1_train = 1 - errors['train'][last_error]
	f1_cv = 1 - errors['cv'][last_error]
	fig = plt.figure()
	plt.title('Curvas de Aprendizado')
	plt.xlabel('Número de dados utilizados no treino')
	plt.ylabel('Erro')
	ax = fig.add_subplot(111)
	line_cv = plt.plot(errors['num_examples'], errors['cv'], label='Validação')
	line_train = plt.plot(errors['num_examples'], errors['train'], label='Treino')
	plt.legend()
	ax.text(m, 1 - f1_train + .01, "F1 = %.3f" %f1_train, ha="right") # ha="center"
	ax.text(m, 1 - f1_cv + .015, "F1 = %.3f" %f1_cv, ha="right") # ha="center"

def save_learning_curves(errors, fname):
	fname = fname if fname.endswith('.png') else fname + '.png'
	_learning_curves_common(errors)
	plt.savefig(fname)
	plt.clf()

def _learning_curves_common(errors):
	plt.title('Curvas de Aprendizado')
	plt.xlabel('Número de dados utilizados no treino')
	plt.ylabel('Erro')
	line_cv = plt.plot(errors['num_examples'], errors['cv'], label='Validação')
	line_train = plt.plot(errors['num_examples'], errors['train'], label='Treino')
	plt.legend()

def plot_validation_curve(regul):
	_validation_curve_common(regul)
	plt.show()
	plt.clf()

def save_validation_curve(regul, fname):
	fname = fname if fname.endswith('.png') else fname + '.png'
	_validation_curve_common(regul)
	plt.savefig(fname)
	plt.clf()

def _validation_curve_common(regul):
	plt.title('Curva de Validação')
	plt.xlabel('Valor de C')
	plt.ylabel('Erro')
	line_train = plt.plot(regul['C'], regul['Train_error'], 'r', label='Treino')
	line_cv = plt.plot(regul['C'], regul['CV_error'], 'g', label='Validação')
	plt.legend()

