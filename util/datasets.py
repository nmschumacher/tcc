from os import listdir
import numpy as np
import pandas as pd
import util.preprocess as pre
import util.octave_to_py as op

'''
class DatasetRecursiveRetrieve(object):
	def __init__(self, dataset_name, preprocess_method, truncate=True, load_header=False):
		self.dataset_name = dataset_name
		self.preprocess_method = preprocess_method
		self.truncate = truncate
		self.load_header = load_header
	def __iter__(self):
		bare_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/Ling Spam/lingspam_public/bare/'
		# Acessamos a pasta bare, buscando cada um dos 10 subdiretórios
		for bare_part in listdir(bare_path):
			# Percorremos todos os emails da pasta
			for fname in listdir(bare_path + bare_part):
				with open(bare_path + bare_part + '/' + fname, 'r', encoding='ISO-8859-1') as f:
					data = f.read()
					if data.startswith('Subject:'):
						data = data[data.index('\n\n')+2 :]
					X = self.preprocess_method(data)
					y = 1 if fname.startswith('spmsg') else 0
					yield (X, y)

		for r, d, f in os.walk(self.dirname):
			for file in f[:150]:
				if '.txt' in file:
					fp = open(os.path.join(r, file))
					txt = fp.read()
					fp.close()
					txt = stopw.transform([txt])[0]
					yield txt.split()
'''


'''
	Carrega dados do dataset "spam_ham" adquiridos em https://www.kaggle.com/venky73/spam-mails-dataset.
	É um subconjunto de dados de Enron-Spam, especificamente de Enron1 do link http://www2.aueb.gr/users/ion/data/enron-spam/.
	São 5171 e-mails, sendo 29% spam e 71% ham, aproximadamente.
'''
def load_enron1():
	# Estrutura do CSV:
	# csv.columns => Index(['Unnamed: 0', 'label', 'text', 'label_num'], dtype='object')
	# 'text' contém o corpo do email; 'label_num' = 0 (ham) ou 1 (spam);
	data = np.array(pd.read_csv('/home/nicolas/Área de Trabalho/TCC implementacao/datasets/spam_ham_dataset.csv'))
	m = data.shape[0] # Quantidade total de exemplos
	X_text = data[:,2] # Todos os m exemplos da coluna 'text'
	y = data[:,3] # Todos os m exemplos da coluna 'label_num'
	y = y.astype('int')
	return (X_text, y)


def load_enron(truncate=True, load_header=False):
	X_text_ham,  y_ham  = _load_enron(load_ham=True,  truncate=truncate, load_header=load_header)
	X_text_spam, y_spam = _load_enron(load_ham=False, truncate=truncate, load_header=load_header)
	return (X_text_ham + X_text_spam, y_ham + y_spam)


def _load_enron(load_ham=True, truncate=True, load_header=False):
	X_text, y = [], []
	max_emails_trunc = 1500 # 1500 hams e 1500 spams
	max_emails = 10000 # 10000 hams e 10000 spams
	email_count = 0
	base_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/Enron/'
	ham_or_spam_dir = 'ham/' if load_ham else 'spam/'
	ham_or_spam_y = 0 if load_ham else 1
	for receiver in sorted(listdir(base_path + ham_or_spam_dir)):
		for receiver_subdir in listdir(base_path + ham_or_spam_dir + receiver):
			#print('Load de dados de ' + ham_or_spam_dir + receiver + '/' + receiver_subdir + '/' + '...')
			for fname in listdir(base_path + ham_or_spam_dir + receiver + '/' + receiver_subdir):
				f_path = base_path + ham_or_spam_dir + receiver + '/' + receiver_subdir + '/' + fname
				with open(f_path, 'r', encoding='ISO-8859-1') as f: # errors='ignore'
					data = f.read()
				data_split_len = len(data.split('\n\n'))
				# Arquivos que possuem apenas header
				if data_split_len < 2 or (data_split_len > 1 and len(data[data.index('\n\n')+2 :].strip()) == 0):
					if not load_header:
						continue
				# Se possui corpo de email e load_header == False, remover header
				if not load_header:
					data = data[data.index('\n\n')+2 :]
				X_text.append(data)
				y.append(ham_or_spam_y)
				email_count += 1
				if (truncate and email_count == max_emails_trunc) or (not truncate and email_count == max_emails):
					#print('Total de emails carregados ' + str(email_count) + '\n')
					return (X_text, y)
			#print('Total de emails carregados ' + str(email_count) + '\n')
	return (X_text, y)


def load_spamAssassin(truncate=True, load_header=False):
	# Estrutura do SpamAssassin:
	# São diversos datasets separados. Quando truncate=True, este método carrega 2551 hams da 
	# pasta /20021010_easy_ham e 500 spams da pasta /20030228_spam, totalizando
	# 3051 emails, 84%/16% de ham/spam. Cada pasta dessas contém emails no formato txt com header e body.
	# Para todo email com header, após o primeiro '\n\n' começa o corpo do email, antes disso está o conteúdo do header.
	email_folders = ['20021010_easy_ham/easy_ham/', '20030228_spam/spam/', '20030228_easy_ham_2/easy_ham_2/', \
					'20021010_spam/spam/', '20021010_hard_ham/hard_ham/', '20030228_easy_ham/easy_ham/', \
					'20030228_spam_2/spam_2/', '20030228_hard_ham/hard_ham/', '20050311_spam_2/spam_2/']
	if truncate:
		return _load_spamAssassin(email_folders[:2], load_header)
	return _load_spamAssassin(email_folders, load_header)


def _load_spamAssassin(email_folders, load_header=False):
	base_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/SpamAssassin/'
	X_text, y = [], []
	is_spam = 1
	for e_folder in email_folders:
		is_spam = 1 if e_folder.find('spam') >= 0 else 0
		for fname in listdir(base_path + e_folder):
			with open(base_path + e_folder + fname, 'r', encoding='ISO-8859-1') as f:
				data = f.read()
				if not load_header:
					try:
						data = data[data.index('\n\n')+2 :]
					except ValueError:
						print('Erro no arquivo: ' + e_folder + fname)
				X_text.append(data)
				y.append(is_spam)
	return (X_text, y)


def load_ling_spam(load_subject=False):
	# Estrutura do Ling Spam dataset:
	# São 4 datasets, sendo o 'bare' sem aplicação de lematização nem remoção de stop words.
	# Este é o dataset utilizado neste experimento, com 2893 mensagens, sendo 2412 ham e 481 spam,
	# isto é, 83%/17% ham/spam, divididos em outras 10 pastas para a realização, originialmente, 
	# de 10-fold cross-validation. Os emails contém o assunto na primeira linha (subject), 
	# seguido de '\n\n' e o corpo do email na sequência.
	bare_path = '/home/nicolas/Área de Trabalho/TCC implementacao/datasets/Ling Spam/lingspam_public/bare/'
	X_text, y = [], []
	# Acessamos a pasta bare, buscando cada um dos 10 subdiretórios
	for bare_part in listdir(bare_path):
		# Percorremos todos os emails da pasta
		for fname in listdir(bare_path + bare_part):
			with open(bare_path + bare_part + '/' + fname, 'r', encoding='ISO-8859-1') as f:
				data = f.read()
				if not load_subject and data.startswith('Subject:'):
					data = data[data.index('\n\n')+2 :]
				X_text.append(data)
				y.append(1 if fname.startswith('spmsg') else 0)
	return (X_text, y)


### Remover header <<<< implementar essa função!!
def load_TREC(truncate=True, load_header=False):
	# São pastas com 300 e-mails cada, exceto a última, contendo 91 mensagens.
	# A classe (spam ou ham) de cada e-mail é informada em index.txt.
	# Os primeiros 300 emails contendo corpo de email são 537 hams e 2463 spams, isto é, 18%/82% ham/spam.
	# Atenção: há arquivos apenas com informações no header, mas corpo do email vazio!!
	trec_dir = '/home/nicolas/Downloads/coursera (ex6) use SPAM + Kaggle dataset/dataset/trec05p-1/'
	index_path = trec_dir + 'full/index'
	X_text, y = [], []
	max_emails_trunc = 3000
	max_emails = 20000
	email_count = 0
	with open(index_path, 'r') as f:
		index = f.read()
	for line in index.strip().split('\n'):
		file_path = trec_dir + line.split()[1][3:]
		with open(file_path, 'r', encoding='ISO-8859-1') as f: # errors='ignore'
			data = f.read()
		data_split_len = len(data.split('\n\n'))
		# Arquivos que possuem apenas header
		if data_split_len < 2 or (data_split_len > 1 and len(data[data.index('\n\n')+2 :].strip()) == 0):
			if not load_header:
				continue
		if not load_header:
			data = data[data.index('\n\n')+2 :]
		X_text.append(data)
		y.append(0 if line.split()[0] == 'ham' else 1)
		email_count += 1
		if (truncate and email_count == max_emails_trunc) or (not truncate and email_count == max_emails):
			return (X_text, y)
	return (X_text, y)


'''
	field: o campo do qual se deseja obter o texto
'''
def _get_header_field(field):
	fields = ['received', 'recieved', 'subject', 'x-mimeole', 'content-class', 'thread-topic', 'content-transfer-encoding']
	return ''


'''
trec_dir = '/home/nicolas/Downloads/coursera (ex6) use SPAM + Kaggle dataset/dataset/trec05p-1/'
index_path = trec_dir + 'full/index'
index, data = None, None
X_text, y = [], []
with open(index_path, 'r') as f:
	index = f.read()


for line in index.strip().split('\n'):
	#y.append(0 if line.split()[0] == 'ham' else 1)
	file_path = trec_dir + line.split()[1][3:]
	with open(file_path, 'r', encoding='ISO-8859-1') as f: # errors='ignore'
		data = f.read()
	data_split = data.split('\n\n')
	# Arquivos que possuem apenas header
	if len(data_split) == 2 and len(data_split[1].strip()) == 0:
		print('Arquivo apenas com header:' + line.split()[1][3:])
	#if not data[:50].split(':')[0].lower() in fields:
	#	print('-------------------\nArquivo:' + line.split()[1][3:] + '\n' + data[:50].split(':')[0] + '\n')


	if not load_header: # and data.startswith('Received:'):
		data = data[data.index('\n\n')+2 :]
	X_text.append(data)
return (X_text, y)
'''


