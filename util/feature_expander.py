import re
import wrapperWSD
import numpy as np
from nltk import pos_tag
from nltk import word_tokenize
from spellchecker import SpellChecker
from sklearn import preprocessing as sk_pre

class FeatureExpander:

	'''
		Conta verbos, adjetivos e advérbios.
	'''
	def pos_tags(self, texts, is_tokenized=False, use_nltk_tokenizer=True, tokenize=True):
		dic_gram_lists = {'verbos': [], 'adjetivos': [], 'adverbios': []}
		tokenized_txts = self._prepare_texts(texts, is_tokenized, use_nltk_tokenizer, True)
		for txt in tokenized_txts:
			dic_gram = {'VB': 0, 'JJ': 0, 'RB': 0}
			tagged_txt = pos_tag(txt)
			for tagged_word in tagged_txt:
				if len(tagged_word[1]) >= 2 and dic_gram.get(tagged_word[1][:2]) is not None:
					dic_gram[tagged_word[1][:2]] += 1
			dic_gram_lists['verbos'].append([dic_gram['VB']])
			dic_gram_lists['adjetivos'].append([dic_gram['JJ']])
			dic_gram_lists['adverbios'].append([dic_gram['RB']])
		for k in dic_gram_lists.keys():
			dic_gram_lists[k] = np.array(dic_gram_lists[k])
		return dic_gram_lists


	'''
		Utiliza a biblioteca WrapperWSD, presente em https://pypi.org/project/wrapperWSD/, para 
		encontrar o sentido de determinadas palavras do texto.

		max_senses: limita a quantidade de sentidos que serão relacionados a cada termo.
	'''
	# TODO: TESTAR BABELFY!!!! Pelo site parece ser muito bom, talvez melhor que esse wrapperWSD usando NLTK
	def wsd(self, texts, max_senses=5, is_tokenized=False):
		new_texts = []
		texts = self._prepare_texts(texts, is_tokenized, False, False)
		wsd = wrapperWSD.WrapperWSD()
		for t in texts:
			txt = ' '.join(t) if is_tokenized else t # WSD trabalha com texto não tokenizado
			new_text = ''
			wsd_list = wsd.wsdNLTK(txt)
			if len(wsd_list) == 0:
				new_texts.append(txt)
			else:
				last_idx = 0
				for wsd_tuple in wsd_list:
					lemma_names = wsd_tuple[1].lemma_names()[:max_senses]
					senses = self._adjust_wsd_lemmas(lemma_names, wsd_tuple[0])
					new_text += txt[last_idx : wsd_tuple[3]] + senses
					last_idx = wsd_tuple[3]
				new_text += txt[last_idx : ]
				new_texts.append(new_text)
		return new_texts


	'''
		Une todas as palavras da lista lemma_names em uma string, exceto a palavra orinigal, separando-as com 
		espaço (' '), trocando o caractere '_' por ' ' de palavras n-gram (n > 1), isto é, 'text_edition' 
		ficará 'text edition', por exemplo. Além disso, acrescenta espaço no início e no fim desta string.
	'''
	def _adjust_wsd_lemmas(self, lemma_names, original_word):
		lnames = ''
		for w in lemma_names:
			lnames += ' ' + w.replace('_', ' ') if w.lower() != original_word.lower() else ''
		return lnames


	"""
		Retorna o tamanho da maior sequência de palavras em letras maiúsculas, a maior quantidade de letras maiúsculas em sequência e 
		o número total de palavras escritas em maiúsculo. O argumento count_words ativa a contagem da quantidade de palavras de cada
		texto.

		Exemplo:
			>>> texts = ["Epopeia É UMA NARRATIVA QUE apresenta com maior qualidade OS FATOS ORIGINALMENTE contados EM versos."]
			>>> largest_uppercase_seq(texts)
			    {'w_sequences_len': [4], 'l_sequences_len': [20], 'total_upper_w': [8]}

		Assim, a maior sequência de palavras maiúsculas é de 4 palavras ("É UMA NARRATIVA QUE"), mas a maior sequência de 
		letras maiúsculas seguidas é outra e possui 20 letras ("OS FATOS ORIGINALMENTE") e o total de palavras em maiúsuculo 
		é 8 ['É', 'UMA', 'NARRATIVA', 'QUE', 'OS', 'FATOS', 'ORIGINALMENTE', 'EM'].

		Obs.: palavras que não estejam totalmente em letra maiúscula não serão contadas, como 'Epopeia'.
	"""
	def largest_uppercase_seq(self, texts, is_tokenized=False):
		words_seq_len, letters_seq_len, total_upper_words = 0, 0, 0
		curr_w_len, curr_l_len = 0, 0
		dic_features = {'w_sequences_len': [], 'l_sequences_len': [], \
						'total_upper_w': []}
		tokenized_txts = self._prepare_texts(texts, is_tokenized)
		for txt in tokenized_txts:
			for w in txt:
				if w.isupper():
					total_upper_words += 1
					curr_w_len += 1
					curr_l_len += len(w)
					words_seq_len = curr_w_len if words_seq_len < curr_w_len else words_seq_len
					letters_seq_len = curr_l_len if letters_seq_len < curr_l_len else letters_seq_len
				else:
					curr_w_len = 0
					curr_l_len = 0
			dic_features['w_sequences_len'].append([words_seq_len])
			dic_features['l_sequences_len'].append([letters_seq_len])
			dic_features['total_upper_w'].append([total_upper_words])
			curr_w_len, curr_l_len, words_seq_len, letters_seq_len, total_upper_words = 0, 0, 0, 0, 0
		for k in dic_features.keys():
			dic_features[k] = np.array(dic_features[k])
		return dic_features


	'''
		Conta pontuações como !, ?, $, % e retorna um dicionário com a quantidade encontrada de cada uma.
	'''
	def count_punctuations(self, texts, is_tokenized=False, use_nltk_tokenizer=False, tokenize=False):
		exclamacao, interrogacao, cifrao, porcentagem = 0, 0, 0, 0
		dic_pont_aux = {'!': 0, '?': 0, '$': 0, '%': 0}
		dic_pontuacoes = {'!': [], '?': [], '$': [], '%': []}
		reg = re.compile('(\!|\?|\$|\%)')
		prepared_txts = self._prepare_texts(texts, is_tokenized, use_nltk_tokenizer, tokenize)
		for txt in prepared_txts:
			txt_untokenized = ' '.join(txt) if (is_tokenized or use_nltk_tokenizer or tokenize) else txt
			for punctuation in reg.findall(txt_untokenized):
				dic_pont_aux[punctuation] += 1
			for k in dic_pont_aux.keys():
				dic_pontuacoes[k].append([dic_pont_aux[k]])
			dic_pont_aux = {'!': 0, '?': 0, '$': 0, '%': 0}
		for k in dic_pontuacoes.keys():
			dic_pontuacoes[k] = np.array(dic_pontuacoes[k])
		return dic_pontuacoes


	'''
		Procura strings que possuam tags HTML. Cada uma delas é contada e sua soma retornada.
		Também há uma contagem de todas as tags encontradas, independente de quais sejam.
		A contagem de todas as tags encontradas em cada documento é armazenada na chave do 
		dicionário 'total_any_tag'.

		tags: lista de tags que serão contadas. Se tags == None, então a lista padrão será a 
				das seguintes tags: 'img', 'input', 'a', 'object', 'source', 'script', 
				'iframe', 'br' e 'p'.
	'''
	def seek_tag(self, texts, is_tokenized=False, tags=None):

		if tags == None:
			tags = ['img', 'input', 'a', 'object', 'source', 'script', 'iframe', 'br', 'p']

		dic_tags = {'total_any_tag': []}
		dic_tag_aux = {'total_any_tag': 0}
		for tag in tags:
			dic_tags[tag] = []
			dic_tag_aux[tag] = 0

		reg_tag = re.compile('\<[\/]{0,1}[a-z]+[^<>\n\t\r]*\>') # '<[^<>]+>'
		reg_name = re.compile('[a-z0-9]+')
		prepared_txts = self._prepare_texts(texts, is_tokenized, False, False)
		for txt in prepared_txts:
			for whole_tag in reg_tag.findall(txt):
				tag_name = whole_tag.lower().split()[0]
				tag_name = reg_name.findall(tag_name)[0]
				dic_tag_aux['total_any_tag'] += 1
				if dic_tag_aux.get(tag_name) is not None:
					dic_tag_aux[tag_name] += 1
			for k in dic_tag_aux.keys():
				dic_tags[k].append([dic_tag_aux[k]])
				dic_tag_aux[k] = 0

		for k in dic_tags.keys():
			dic_tags[k] = np.array(dic_tags[k])

		return dic_tags


	'''
		Procura trechos que possuam 'http' ou 'https' seguido de '://' e uma string contando-os em
		cada texto.
	'''
	def seek_URL(self, texts, is_tokenized=False, use_nltk_tokenizer=False, tokenize=False):
		urls = []
		# Regex encontrada em https://www.geeksforgeeks.org/python-check-url-string/
		regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
		reg = re.compile(regex) # regex anteriormente utilizada '(http|https)://[^\s]*'
		prepared_txts = self._prepare_texts(texts, is_tokenized, use_nltk_tokenizer, tokenize)

		for txt in prepared_txts:
			urls.append([len(reg.findall(txt))])

		return np.array(urls)


	'''
		Conta todas as palavras de cada texto.
	'''
	def count_words(self, texts, is_tokenized=False, use_nltk_tokenizer=False):

		tokenized_txts = self._prepare_texts(texts, is_tokenized, use_nltk_tokenizer, True)
		counted_words = []

		for txt in tokenized_txts:
			counted_words.append(len(txt))

		return counted_words


	'''
		Contagem de possíveis erros de escrita. Limitações: nomes próprios e emails, por
		exemplo, são considerados erros.
	'''
	def count_misspellings(self, texts, is_tokenized=False):
		errors = []
		spell = SpellChecker()
		prepared_txts = self._prepare_texts(texts, is_tokenized, False, False)
		for txt in prepared_txts:
			tokenized_txt = txt if is_tokenized else spell.split_words(txt)
			errors.append([len(spell.unknown(tokenized_txt))])
		return np.array(errors)


	'''
		Conta quantas palavras são maiores que a segunda maior palavra da língua inglesa,
		composta por 36 letras, Hippopotomonstrosesquippedaliophobia, segundo o site
		https://irisreading.com/10-longest-words-in-the-english-language/
	'''
	def count_longest_words(self, texts, is_tokenized=False):
		# 2a maior palavra do inglês, segundo a fonte:
		# https://irisreading.com/10-longest-words-in-the-english-language/
		second_longest_en_word_sz = len('Hippopotomonstrosesquippedaliophobia')
		longest_words = []
		spell = SpellChecker()
		prepared_txts = self._prepare_texts(texts, is_tokenized, False, False)
		for txt in prepared_txts:
			tokenized_txt = txt if is_tokenized else spell.split_words(txt)
			long_words = 0
			for w in tokenized_txt:
				if len(w) > second_longest_en_word_sz:
					long_words += 1
			longest_words.append([long_words])
		return np.array(longest_words)


	'''
		Conta quantos caracteres considerados estranhos ou incomuns que há 
		em cada texto, são eles: ä, è, ï, ò, æ, ð, ½, ¼, ø, ¢, ¹, ², ·, × e ¬.
	'''
	def count_unusual_chars(self, texts, is_tokenized=False):
		regex = 'ä|è|ï|ò|æ|ð|½|¼|ø|¢|¹|²|·|×|¬'
		reg = re.compile(regex)
		strange_chars = []
		prepared_txts = self._prepare_texts(texts, is_tokenized, False, False)
		for txt in prepared_txts:
			untokenized_txt = ' '.join(txt) if is_tokenized else txt
			strange_chars.append([len(reg.findall(untokenized_txt.lower()))])
		return np.array(strange_chars)


	'''
		Procura pelas palavras: html, head, body, DOCTYPE, title, script, div, &nbsp; e conta-as.
		Algumas mensagens não seguem o padrão de tags, mas estão codificadas. Por exemplo, no lugar
		de <html> está =3Chtml=3E. Essa feature busca contar possíveis palavras reservadas comuns
		em HTML.
	'''
	def count_possible_html(self, texts, is_tokenized=False):
		regex = '(html|head|body|doctype|title|script|div|\&nbsp\;)'
		reg = re.compile(regex)
		html_words = []
		prepared_txts = self._prepare_texts(texts, is_tokenized, False, False)
		for txt in prepared_txts:
			untokenized_txt = ' '.join(txt) if is_tokenized else txt
			html_words.append([len(reg.findall(untokenized_txt.lower()))])
		return np.array(strange_chars)


	def create_npmatrix_from_dic_features(self, np_dic_features):
		result = None
		np_dic_features_cp = np_dic_features.copy()
		for k in np_dic_features_cp:
			result = np_dic_features_cp.pop(k)
			# Precisamos apenas da primeira chave
			break
		return self.add_dic_features_to_npmatrix(result, np_dic_features_cp)


	'''
		Adiciona as features presentes no dicionário np_dic_features à matriz np_matrix de acordo com a ordem dos 
		elementos recuperados de np_dic_features.keys(). Todos os membros devem ser do tipo array numpy seguindo
		a estrutura de lista de listas.
		Exemplo:
			>>> a = array([[0, 0],
						   [1, 1],
						   [2, 2]])

			>>> b = np.array([[3], [3], [3]])
			>>> c = np.array([[4], [4], [4]])
			>>> dic = {'b': b, 'c': c}
			>>> add_dic_features_to_npmatrix(a, dic)

			array([[0, 0, 3, 4],
				   [1, 1, 3, 4],
				   [2, 2, 3, 4]])
	'''
	def add_dic_features_to_npmatrix(self, np_matrix, np_dic_features):
		result = np_matrix
		for np_feature_k in np_dic_features.keys():
			result = self.add_feature_to_npmatrix(result, np_dic_features[np_feature_k])
		return result


	'''
		Adiciona uma ou mais features presentes em np_feature à matriz np_matrix. Todos os membros devem ser do tipo array numpy
		seguindo a estrutura de lista de listas.
		Exemplo:
			>>> a = np.array( [[0, 0],
							   [1, 1],
							   [2, 2]])

			>>> b = np.array([[2], [2], [2]]])
			>>> add_feature_to_npmatrix(a, b)

			array([[0, 0, 2],
				   [1, 1, 2],
				   [2, 2, 2]])
	'''
	def add_feature_to_npmatrix(self, np_matrix, np_feature):
		return np.append(np_matrix, np_feature, axis = 1)


	'''
		feature_max_val: percorre todos os valores e limita-os para o valor máximo estabelecido por meior 
			deste argumento. Isto ocorrerá apenas em casos em que feature_max_val > 0. Dessa forma, se 
			feature_max_val = 100, todos os valores presentes em np_feature serão limitados a 100. Para 
			não utilizar esta função basta deixar o valor padrão feature_max_val = None.
		option: 0 para valores reais [0.0, 1.0].
	'''
	def normalize(self, np_feature, feature_max_val=None, option=0):
		np_feature_cp = np.copy(np_feature)
		if feature_max_val != None:
			i = 0
			while(i < len(np_feature_cp)):
				if np_feature_cp[i][0] > feature_max_val:
					np_feature_cp[i][0] = feature_max_val
				i += 1
		if option == 0:
			min_max_scaler = sk_pre.MinMaxScaler()
			return min_max_scaler.fit_transform(np_feature_cp)
		#elif option == 1:
		#	min_max_scaler = sk_pre.MinMaxScaler()
		#	return min_max_scaler.fit_transform(np_feature)
		return np_feature_cp


	def dic_normalize(self, np_dic_features, feature_max_val=None, option=0):
		new_np_dic = {}
		for k in np_dic_features.keys():
			new_np_dic[k] = self.normalize(np_dic_features[k], feature_max_val, option)
		return new_np_dic


	'''
		Retorna a feature normalizada com base na sua porcentagem relativa ao argumento max_values.
		Assim, cada valor em np_feature é dividido pelo valor de max_values no respectivo índice.
		Todos os valores retornados estarão no [0.0~1.0]. Se algum valor da feature for maior que 
		max_values correspondente ou se houver divisão por zero, o valor resultante será 1.

		max_values: lista de valores.
	'''
	def normalize_percent(self, np_feature, max_values):
		m = np_feature.size
		p_normalized = np_feature.reshape(m, ) / np.array(max_values).reshape(m, )
		aux = []
		for val in p_normalized:
			if val > 1.0 or str(val) == 'inf':
				val = 1.0
			elif str(val) == 'nan':
				val = 0.0
			aux.append([val])
		return np.array(aux)


	'''
		Se max_values for uma lista, todos os elementos em np_dic_features serão normalizados com
		relação a esta lista. Mas se for um dicionário, deve conter as mesmas chaves de 
		np_dic_features para normalizar conforme a lista de cada chave. 
	'''
	def dic_normalize_percent(self, np_dic_features, max_values):
		np_resp = {}
		if type(max_values) != type({}):
			for k in np_dic_features.keys():
				np_resp[k] = self.normalize_percent(np_dic_features[k], max_values)
		else:
			for k in np_dic_features.keys():
				np_resp[k] = self.normalize_percent(np_dic_features[k], max_values[k])
		return np_resp


	def feature_to_one_hot(self, np_feature):
		np_feature_cp = np.copy(np_feature)
		i=0
		while(i < len(np_feature_cp)):
			if np_feature_cp[i][0] > 0:
				np_feature_cp[i][0] = 1
			else:
				np_feature_cp[i][0] = 0
			i += 1
		return np_feature_cp


	def dic_features_to_one_hot(self, np_dic_features):
		new_np_dic = {}
		for k in np_dic_features.keys():
			new_np_dic[k] = self.feature_to_one_hot(np_dic_features[k])
		return new_np_dic


	'''
		Este método busca ajustar o formato de texts para que seu conteúdo siga o padrão de lista de textos tokenizados,
		isto é, [['txt', '1'], ['txt', '2'], ...]. A não ser que a opção tokenize seja alterada para False, então o texto
		não será tokenizado.
	'''
	def _prepare_texts(self, texts, is_tokenized=False, use_nltk_tokenizer=False, tokenize=True):
		# Caso 'txt 1', precisa tokenizar (tokenize == True) e adicionar a uma lista de textos, mesmo sendo único
		if not is_tokenized and type(texts) == type(''):
			if tokenize or use_nltk_tokenizer:
				return [word_tokenize(texts)] if use_nltk_tokenizer else [texts.split()]
			return [texts]
		# Caso ['txt 1', 'txt 2', 'txt 3', ...], precisa tokenizar
		elif not is_tokenized and type(texts) == type([]) and type(texts[0]) == type(''):
			if not tokenize and not use_nltk_tokenizer:
				return texts
			texts_new = []
			for txt in texts:
				if use_nltk_tokenizer:
					texts_new.append(word_tokenize(txt))
				elif tokenize:
					texts_new.append(txt.split())
			return texts_new
		# Caso seja apenas um texto tokeniado ['txt', '1'], precisa adicionar a uma lista de textos
		elif is_tokenized:
			if type(texts) == type([]) and type(texts[0]) == type(''):
				return [texts]
		return texts



