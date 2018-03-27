import networkx
import itertools
import re
import os
import math
import nltk
from konlpy.tag import Mecab
cwd = os.path.dirname(os.path.realpath(__file__))



def repl(m):
    return ' ' * len(m.group())
# keywords, sentences
class TextRank:

    def preprocess(self, text):

        target_list = ["\t", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", ":phone:", "☏", "※", ":arrow_forward:", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴", "┘"]


        for target in target_list:
            text = text.replace(target, " ")    
        regularExpression1 = "\r?\n|\r|\t"
        regularExpression2 = "[a-z0-9_+]+@([a-z0-9-]+\\.)+[a-z0-9]{2,4}|[a-z0-9_+]+@([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
        regularExpression3 = "(file|gopher|news|nntp|telnet|https?|ftps?|sftp):\\/\\/([a-z0-9-]+\\.)+[a-z0-9]{2,4}|(file|gopher|news|nntp|telnet|https?|ftps?|sftp):\\/\\/([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
        regularExpression4 = "([a-z0-9-]+\\.)+[a-z0-9]{2,4}|([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
        regularExpression5 = "\\(.*?\\)|\\[.*?\\]|【.*?】|<.*?>"
        regularExpression6 = "[!@+=%^;:]"
        regularExpression7 = "[ ]{1,20}"
        regularExpression8 = "[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z]기자|[가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z]기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z]기자" 
        
        part1 = re.compile(regularExpression1)
        part2 = re.compile(regularExpression2)
        part3 = re.compile(regularExpression3)
        part4 = re.compile(regularExpression4)
        part5 = re.compile(regularExpression5)
        part6 = re.compile(regularExpression6)
        part7 = re.compile(regularExpression7)
        part8 = re.compile(regularExpression8)
        text = re.sub(part1, "", text)
        text = re.sub(part2, "", text)
        text = re.sub(part3, "", text)
        text = re.sub(part4, "", text)
        text = re.sub(part5, "", text)
        text = re.sub(part6, " ", text)
        text = re.sub(part7, " ", text)
        text = re.sub(part8, "", text)
        

        trimPoint = text.rfind('다.')
        if trimPoint > -1:
            try:
                text = text[0:trimPoint+2]
            except Exception as e:
                print(e)
        if text:
            textList = re.split('\\. |\\.', text)
            textList.pop()

            stopwordList = self.stopwords
            textList = list(map(lambda x: x + '. ', list(filter(lambda x: x in stopwordList, textList))))
            
            textList = (''.join(textList)).strip()
            textList = textList.replace('.', '. ')
        else:
            textList = ""

        return text

    # quote remover and line splitter  result convert
    def convert_to_original_string(self, strings, original):
        idx = 0
        res = []
        for string in strings:
            new_string = ""
            for i in range((len(string))):
                new_string += original[idx]
                idx += 1
            res.append(new_string)
        return res

    # split content to list of sentences
    def readSentence(self):
        # split content using regular expression
        # 본문에서 따옴표, 쌍따옴표 쌍 안에 들어오는 문장을 전부다 공백으로 치환
        content = re.sub(self.quoteRemover, repl, self.content)
        # 인용구를 공백문자로 전부 바꾼 뒤 문장을 자른다. 
        ch = self.lineSplitter.split(content)
        # 문장 다 잘랐으면 인용구를 복구
        ch = self.convert_to_original_string(ch, self.content)

        # concatenate sentence and delimiter
        # 문장과 구분자를 합쳐서 return
        for s in map(lambda a, b: a+b, ch[::2], ch[1::2]):
            if not s: continue

            # remove left/right space
            yield s.strip()

    # split content to list of keywords
    def readTagger(self):
        original_content = self.preprocess(self.content)
        # split content using regular expression
        content = re.sub(self.quoteRemover, repl, original_content)
        ch = self.lineSplitter.split(content)
        ch = self.convert_to_original_string(ch, original_content)

        # concatenate sentence and delimiter
        for s in map(lambda a, b: a+b, ch[::2], ch[1::2]):
            if not s: continue
            
            # pos tagger apply
            try:
                yield self.numeric(s)
            except:
                []

    def readTaggerForTitle(self):
        return self.numeric(self.title)

    # build additional graph information for applying pagerank algorithm
    def loadTagger(self):
        # construct bigram count dictionary
        def insertPair(a, b):
            if a > b:
                a, b = b, a
            elif a == b:
                return
            self.taggerDictBiCount[a, b] = self.taggerDictBiCount.get((a, b), 0) + 1

        def insertNearPair(a, b):
            self.dictNear[a, b] = self.dictNear.get((a, b), 0) + 1

        taggerIter = self.readTagger()
        # save pos tagging result
        wordFilter = self.taggerTokenizer
        for sent in taggerIter:
            # word filtering
            sent = list(filter(wordFilter, sent))
            # english filtering
            sent = list(filter(self.englishFilter, sent))
            for i, word in enumerate(sent):

                self.taggerDictCount[word] = self.taggerDictCount.get(word, 0) + 1
                self.nTotal += 1

                # create bigram for similar word filtering
                joined1 = '%s %s' % (sent[i-1][0], word[0])
                joined2 = '%s%s' % (sent[i-1][0], word[0])
                # check similar word (ex: iPhone6, iPhone 6)
                # inset near pair. then, similar word is removed
                if i - 1 >= 0 and wordFilter((joined1, 'NNP')) or wordFilter((joined2, 'NNP')):
                    insertNearPair(sent[i-1], word)
                if i+1 < len(sent):
                    # news content based bigram
                    joined3 = '%s %s' % (word[0], sent[i+1][0])
                    joined4 = '%s%s' % (word[0], sent[i+1][0])
                    if wordFilter((joined3, 'NNP')) or wordFilter((joined4, 'NNP')):
                        insertNearPair(word, sent[i+1])

                # construct bigram count
                for j in range(i+1, min(i + self.window + 1, len(sent))):
                        joined1 = '%s %s' % (word[0], sent[j][0])
                        joined2 = '%s%s' % (word[0], sent[j][0])
                        if sent[j] != word and (wordFilter((joined1, 'NNP')) or wordFilter((joined2, 'NNP'))):
                            insertPair(word, sent[j])

        # similar word count concatenate (ex: iPhone6, iPhone 6)
        unigrams = self.taggerDictCount.keys()
        for key in self.taggerDictBiCount.keys():
            joined1 = ('%s %s' % (key[0][0], key[1][0]), 'NNP')
            joined2 = ('%s%s' % (key[0][0], key[1][0]), 'NNP')
            if joined1 in unigrams:
                self.taggerDictCount[joined1] = self.taggerDictCount.get(joined1, 0) + self.taggerDictBiCount.get(key, 0)
                self.taggerDictBiCount[key] = 0
            if joined2 in unigrams:
                self.taggerDictCount[joined1] = self.taggerDictCount.get(joined2, 0) + self.taggerDictBiCount.get(key, 0)
                self.taggerDictBiCount[key] = 0


    # build additional graph information for applying pagerank algorithm
    def loadSentence(self):
        # get similarity of two sentences
        def similarity(a, b):
            n = len(a.intersection(b))
            return n / float(len(a) + len(b) - n) / (math.log(len(a) + 1) * math.log(len(b) + 1))

        sentSet = []
        sentenceIter = self.readSentence()
        for sent in filter(None, sentenceIter):
            if type(sent) == str:
                s = set(filter(None, self.sentenceTokenizer(sent)))
                s = set(filter(self.englishFilter, s))
            if len(s) < 2:
                continue
            self.sentenceDictCount[len(self.sentenceDictCount)] = sent
            sentSet.append(s)

        for i in range(len(self.sentenceDictCount)):
            for j in range(i+1, len(self.sentenceDictCount)):
                s = similarity(sentSet[i], sentSet[j])
                if s < self.threshold:
                    continue
                self.sentenceDictBiCount[i, j] = s

    # build keyword graph
    def loadKeywordGraph(self):
        self.taggerGraph = networkx.Graph()
        self.taggerGraph.add_nodes_from(self.taggerDictCount.keys())

        unigrams = self.taggerDictCount.keys()
        wordFilter = self.taggerTokenizer
        for (a, b), n in self.taggerDictBiCount.items():
            self.taggerGraph.add_edge(a, b, weight=n*n*n*self.coef + (1 - self.coef))

    # build sentence graph
    def loadSentenceGraph(self):
        self.sentenceGraph = networkx.Graph()
        self.sentenceGraph.add_nodes_from(self.sentenceDictCount.keys())

        for (a, b), n in self.sentenceDictBiCount.items():
            self.sentenceGraph.add_edge(a, b, weight=n*n*n*self.coef + (1 - self.coef))

    # pagerank
    def pagerank(self, graph):
        return networkx.pagerank(graph, weight='weight')

    # get Information I(X) = -log(p, X)
    def getI(self, a):
        if a not in self.taggerDictCount:
            return None
        else:
            return math.log(self.nTotal / self.taggerDictCount[a])

    # get Pointwise Mutual Information PMI(X, Y) = log(P(X intersection Y) / (P(X) * P(y)))
    def getPMI(self, a, b):
        co = self.dictNear.get((a, b), 0)
        if not co:
            return None
        else:
            return math.log(float(co) * self.nTotal / self.taggerDictCount[a] /
                    self.taggerDictCount[b])

    # pagerank apply
    def keyword_rank(self):
        self.loadTagger()
        self.loadKeywordGraph()
        self.keyword_ranks = self.pagerank(self.taggerGraph)

    # get keywords 
    # 키워드 추출
    def keywords(self, num=15):
        ranks = self.keyword_ranks
        wordFilter = self.taggerTokenizer
        cand = sorted(ranks, key=ranks.get, reverse=True)
        pairness = { }
        startOf = { }
        tuples = { }
        for k in cand:
            if k[1] != 'VA' and k[1] != 'VV':
                tuples[(k, )] = self.getI(k) * ranks[k]
            for l in cand:
                if k == l:
                    continue
                pmi = self.getPMI(k, l)
                if pmi:
                    pairness[k, l] = pmi
        
        for (k, l) in sorted(pairness, key=pairness.get, reverse=True):
            if k not in startOf:
                startOf[k] = (k, l)

        for (k, l), v in pairness.items():
            pmis = v
            rs = ranks[k] * ranks[l]
            path = (k, l)
            tuples[path] = pmis / (len(path) - 1) * rs ** (1 / len(path)) * len(path)
            last = l
            while last in startOf and len(path) < 7:
                if last in path:
                    break
                pmis += pairness[startOf[last]]
                last = startOf[last][1]
                rs *= ranks[last]
                path += (last, )
                # G(TR) * A(PMI) * Length
                tuples[path] = pmis / (len(path) - 1) * rs ** (1 / len(path)) * len(path)

        used = set()
        both = { }
        for k in sorted(tuples, key=tuples.get, reverse=True):
            if used.intersection(set(k)):
                continue
            both[k] = tuples[k]
            for w in k:
                used.add(w)
                
        unigrams = []
        bigrams = []
        avg = 0.0
        for key in both.keys():
            avg += both[key]
        if len(both.keys()) > 0:
            avg /= len(both.keys())

        titleWords = list(filter(wordFilter, self.readTaggerForTitle()))
        titleWords = list(map(lambda x: x[0], titleWords))
        for key in both.keys():
            if len(key) == 2:
                if self.taggerDictCount[key[0]] <= self.minimum_low_freq or self.taggerDictCount[key[1]] <= self.minimum_low_freq:
                    both[key] -= avg * self.low_freq_word_subtraction_multiplier
                if key[0][1] == 'NNP' or key[1][1] == 'NNP':
                    pass
                    both[key] += avg * self.nnp_addition_multiplier
                if key[0][0] in titleWords or key[1][0] in titleWords:
                    both[key] += avg * self.title_word_addition_multiplier
                joined1 = '%s %s' % (key[0][0], key[1][0])
                joined2 = '%s%s' % (key[0][0], key[1][0])
                if joined1 in self.content and wordFilter((joined1, 'NNP')):
                    bigrams.append((joined1, both[key]))
                elif joined2 in self.content and wordFilter((joined2, 'NNP')):
                    bigrams.append((joined2, both[key]))
                elif key[0][0] in self.content:
                    unigrams.append((key[0][0], both[key]))
                elif key[1][0] in self.content:
                    unigrams.append((key[1][0], both[key]))
            else:
                if key[0][1] == 'NNP':
                    both[key] += avg * self.nnp_addition_multiplier
                if self.taggerDictCount[key[0]] <= self.minimum_low_freq:
                    both[key] -= avg * self.low_freq_word_subtraction_multiplier
                if key[0][0] in titleWords:
                    both[key] += avg * self.title_word_addition_multiplier

                # 수사가 아니면 단일 단어 추가
                if key[0][1] != 'SN':
                    unigrams.append(('%s' % (key[0][0]), both[key]))

        unigrams = sorted(list(filter(lambda x: x[1] > 0.0, unigrams)))[:num*2]
        bigrams = sorted(list(filter(lambda x: x[1] > 0.0, bigrams)))[:num*2]


        res = set([])
        ## 공백없는 복합단어를 합친다. ex) 한양대 대나무/숲/향기 -> 한양대 대나무숲향기
        bigrams, ex_words = self.expand_n_gram_to_match_context(bigrams)
        unigrams = list(filter(lambda x: x[0] not in ex_words, unigrams))
        unigrams, ex_words = self.expand_n_gram_to_match_context(unigrams)
        unigrams = list(filter(lambda x: x[0] not in ex_words, unigrams))


        res = self.merge_expanded_similar_word(bigrams, unigrams)
        res = list(filter(lambda x: x[1] > 0.0, res))
        res = sorted(res, key=lambda x: x[1], reverse=True)[:num]

        return res


    def merge_expanded_similar_word(self, bigrams, unigrams):
        # convert to dict
        # bigram_dict[word] = cost
        cost_dict = { }
        for bigram in bigrams:
            cost_dict[bigram[0]] = bigram[1]
        for unigram in unigrams:
            cost_dict[unigram[0]] = unigram[1]
            


        iters = bigrams + unigrams
        for bigram in iters:
            words = bigram[0].split(' ')
            if len(words) == 1:
                continue
            elif len(words) == 2:
                nonspace_word = words[0] + words[1]
                if nonspace_word in cost_dict:
                    cost_dict[bigram[0]] += cost_dict[nonspace_word]
                    cost_dict.pop(nonspace_word, None)

        res = []
        for key in cost_dict.keys():
            res.append((key, cost_dict[key]))

        return res

    ## 본문에 나온 형태로 띄어쓰기 없는 명사단어를 붙여주는 함수
    def expand_n_gram_to_match_context(self, n_gram):

        res = []
        ex_words = set([])
        for words in n_gram:

            # words 를 본문에서 찾아서 공백이 없을때까지 다 합친다. 그리고 words 에서 나온 단어를 제외하고 붙인다.

            position = self.content.find(words[0])
            left = position - 1
            right = position + len(words[0])

            while left >=0 and self.boundaryChecker(self.content[left]): left -= 1
            while right < len(self.content) and self.boundaryChecker(self.content[right]): right += 1

            morph = self.content[left+1: right+1]
            poses = self.tagger.pos(morph)

            left = -1
            right = len(poses)
            for idx, pos in enumerate(poses):
                if idx == len(poses)-1:
                    break
                # bigram check
                joined1 = poses[idx][0] + poses[idx+1][0]
                joined2 = poses[idx][0] + ' ' + poses[idx+1][0]
                if joined1 == words[0] or joined2 == words[0]:
                    left = idx - 1
                    right = idx + 2
                # unigram check
                elif poses[idx][0] == words[0]:
                    left = idx - 1
                    right = idx + 1

            lower = 0
            upper = len(poses)-1

            merged_bigram = words[0]

            while lower <= left:
                if self.nounChecker(poses[lower][1]):
                    merged_bigram = poses[lower][0] + merged_bigram
                    ex_words.add(poses[lower][0])
                lower += 1

            while right <= upper:
                if self.nounChecker(poses[upper][1]):
                    merged_bigram = merged_bigram + poses[upper][0]
                    ex_words.add(poses[lower][0])
                upper -=1

            res.append((merged_bigram, words[1]))

        return res, ex_words

    # pagerank apply
    def sentence_rank(self):
        self.loadSentence()
        self.loadSentenceGraph()
        self.sentence_ranks = self.pagerank(self.sentenceGraph)

    # get sentences
    def sentences(self, ratio = 0.333):

        r = self.sentence_ranks
        ks = sorted(r, key=r.get, reverse=True)[:int(len(r)*ratio)]

        ks = list(map(lambda k:(self.sentenceDictCount[k], r[k]), sorted(ks)))
        return sorted(ks, key=lambda x: x[1], reverse=True)


    # 숫자 + 명사를 한 단어로 인식하게 하는 함수
    def numeric(self, sent):

        words = self.tagger.pos(sent)
        for idx, word in enumerate(words):
            if idx == 0:
                continue

            prev_word = words[idx-1]
            first_word_used = False
            # number + noun
            if prev_word[1] == 'SN':
                noun = ''
                for idx2 in range(idx, len(words)):
                    if words[idx2][1][0] == 'N':
                        prefix_noun1 = noun + words[idx2][0]
                        prefix_noun2 = noun + ' ' + words[idx2][0]
                        if prefix_noun1 in self.content:
                            first_word_used = True
                            noun = prefix_noun1
                        elif prefix_noun2 in self.content:
                            if first_word_used:
                                break
                            first_word_used = True
                            noun = prefix_noun2
                    else:
                        break

                prefix_noun1 = prev_word[0] + noun
                prefix_noun2 = prev_word[0] + ' ' + noun

                # 그냥 숫자일 경우 넘김
                if not noun:
                    continue
                if prefix_noun1 in self.content:
                    words[idx] = ((prefix_noun1, 'NNP'))
                elif prefix_noun2 in self.content:
                    words[idx] = ((prefix_noun2, 'NNP'))


            

        return words

    def __init__(self, **kwargs):

        self.graph = None
        self.tagger = kwargs.get('tagger', Mecab())
        self.window = kwargs.get('window', 5)
        self.coef = kwargs.get('coef', 1.0)
        self.threshold = kwargs.get('threshold', 0.005)
        self.content = kwargs.get('content', '')
        self.title = kwargs.get('title', '')
        self.stopwords = kwargs.get('stopwords', set([]))
        self.singlewords = kwargs.get('singlewords', set([]))
        self.title_word_addition_multiplier = kwargs.get('title_word_addition_multiplier', 1)
        self.minimum_low_freq = kwargs.get('minimum_low_freq', 1)
        self.low_freq_word_subtraction_multiplier = kwargs.get('low_freq_word_subtraction_multiplier', 0)
        self.nnp_addition_multiplier = kwargs.get('nnp_addition_multiplier', 0)
        self.keyword_ranks = []
        self.sentence_ranks = []

        # line splitter
        self.quoteRemover = '"[^"]*"|\'[^\']*\'|\([^()]*\)|\{[^{}]*\}|\[[^\[\]]*\]|\<[^\<\>]*\>|`[^`]*`'
        self.lineSplitter = re.compile('(다[.!?:](?=[^"\'\`\]\}\)\.\!\?]))')

        # variables for keyword extract
        self.taggerDictCount = { }
        self.taggerDictBiCount = { }
        self.dictNear = { }
        self.nTotal = 0

        # only extract keyword NNG, NNP (일반 명사, 고유 명사, 외국어, 숫자, 한자 제외), 영어일 경우 영어 명사 사전에 등록된 단어인지 검색
        self.nounChecker = lambda x: x in ('NNG', 'NNP', 'SL')
        # 숫자, 영어, 한글이면 일단 단어 경계에서 제외 (단어를 확장할때 쓴다)
        self.boundaryChecker = lambda x: re.compile('[a-zA-Z0-9가-힣]+').match(x)

        self.taggerTokenizer = lambda x: x[0] not in self.stopwords and self.nounChecker(x[1]) and (len(x[0]) > 1 or x[0] in self.singlewords) and re.compile('[一-龥]+').match(x[0]) is None

        self.englishFilter = lambda x: x[1] != 'SL' or nltk.pos_tag([x[0]])[0][1][0] == 'N'

        # variables for sentence summarization
        self.sentenceDictCount = { }
        self.sentenceDictBiCount = { }
        self.sentenceTokenizer = lambda sent: filter(self.taggerTokenizer, self.numeric(sent))
