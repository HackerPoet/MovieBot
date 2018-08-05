import csv, re
import numpy as np
import unidecode

MAX_LEN = 256
word_list = []
word_dict = {}

def hasNumbers(w):
	return any(c.isdigit() for c in w)

def shouldSplit(wlist):
	if len(wlist) != 2:
		return False
	if wlist[0] in ['', 'www']:
		return False
	if wlist[1] in ['', 'com', 'org', 'net', 'edu']:
		return False
	num_single = 0
	for w in wlist:
		if hasNumbers(w):
			return False
		if len(w) == 1:
			num_single += 1
	return num_single < 2
	
def splitWords(words, delim):
	final_words = []
	for w in words:
		if len(w) == 0: continue
		wlist = w.split(delim)
		if shouldSplit(wlist):
			final_words.append(wlist[0])
			final_words.append(delim)
			final_words.append(wlist[1])
		else:
			final_words.append(w)
	return final_words

print "Loading Movies..."
movies = []
acronym = re.compile(r"(\w\.\w) (\.)")
with open('imdb5000/movies_metadata.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')

	for row in reader:
		summary = unidecode.unidecode(unicode(row[9] + ' ', 'utf-8')).lower()
		summary = summary.replace('&amp;',' and ')
		summary = summary.replace('\n',' ')
		summary = summary.replace("'"," ' ")
		summary = summary.replace("`"," ' ")
		summary = summary.replace('/',' ')
		summary = summary.replace('-',' ')
		summary = summary.replace('"',' ')
		summary = summary.replace('(',' ')
		summary = summary.replace(')',' ')
		summary = summary.replace(']',' ')
		summary = summary.replace('[',' ')
		summary = summary.replace('...',' . ')
		summary = summary.replace('..',' . ')
		summary = summary.replace('; ',' ; ')
		summary = summary.replace(': ',' ; ')
		summary = summary.replace(', ',' , ')
		summary = summary.replace('. ',' . ')
		summary = summary.replace('?',' ? ')
		summary = summary.replace('!',' ! ')
		summary = re.sub(acronym, r"\1\2", summary)
		words = summary.split(' ')
		
		words = [w for w in words if len(w) > 0]
		words = splitWords(words, '.')
		words = splitWords(words, ',')

		if len(words) <= 6:
			continue

		cur_movie = np.empty((MAX_LEN,), dtype=np.int32)
		for i in xrange(MAX_LEN):
			w = words[i] if i < len(words) else '.'
			assert(not w.isspace())
			if w not in word_dict:
				word_dict[w] = len(word_list)
				word_list.append(w)
			cur_movie[i] = word_dict[w]
		movies.append(cur_movie)

print "Loaded " + str(len(movies)) + " Movies."
print "Dictionary Size: " + str(len(word_list))

print "Saving..."
movies = np.array(movies, dtype=np.int32)
np.save('data/movies.npy', movies)

with open('data/words.txt', 'w') as fout:
	for w in word_list:
		fout.write(w + '\n')



