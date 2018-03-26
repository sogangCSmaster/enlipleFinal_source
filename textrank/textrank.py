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

    #박차장님께서 도와주신 전처리 함수입니다.
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
    # 인용구 지웠던 것을 다시 복구 시키는 함수입니다.
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
        taggerArr = list(map(lambda x: list(x), taggerIter))

        # save pos tagging result
        self.pos = list(itertools.chain(*taggerArr))
        wordFilter = self.taggerTokenizer

        for sent in taggerArr:
            sent = list(filter(self.englishFilter, sent))
            for i, word in enumerate(sent):

                # english filtering
                
                # filtering
                if wordFilter and not wordFilter(word):
                    continue

                self.taggerDictCount[word] = self.taggerDictCount.get(word, 0) + 1
                self.nTotal += 1

                # create bigram for similar word filtering
                joined1 = '%s %s' % (sent[i-1][0], word[0])
                joined2 = '%s%s' % (sent[i-1][0], word[0])
                # check similar word (ex: iPhone6, iPhone 6)
                # inset near pair. then, similar word is removed
                if i - 1 >= 0 and wordFilter(sent[i-1]) and (wordFilter((joined1, 'NNG')) or wordFilter((joined2, 'NNG'))):
                    insertNearPair(sent[i-1], word)
                if i+1 < len(sent):
                    # news content based bigram
                    joined3 = '%s %s' % (word[0], sent[i+1][0])
                    joined4 = '%s%s' % (word[0], sent[i+1][0])
                    if wordFilter(sent[i+1]) and (wordFilter((joined3, 'NNG')) or wordFilter((joined4, 'NNG'))):
                        insertNearPair(word, sent[i+1])

                # construct bigram count
                for j in range(i+1, min(i + self.window + 1, len(sent))):
                        if wordFilter and not wordFilter(sent[j]):
                            continue
                        joined1 = '%s %s' % (word[0], sent[j][0])
                        joined2 = '%s%s' % (word[0], sent[j][0])
                        if sent[j] != word and (wordFilter((joined1, 'NNG')) or wordFilter((joined2, 'NNG'))):
                            insertPair(word, sent[j])

        # similar word count concatenate (ex: iPhone6, iPhone 6)
        unigrams = self.taggerDictCount.keys()
        for key in self.taggerDictBiCount.keys():
            joined1 = ('%s%s' % (key[0][0], key[1][0]), 'NNG')
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
        ranks = self.keyword_ranks  ## 구성한 그래프를 가져온다. (페이지 랭크)
        wordFilter = self.taggerTokenizer   ## 단어 추출 기준 (pos tagging 된 것 중에 어떤게 키워드 추출할 수 있는 단어인지 구분)
        cand = sorted(ranks, key=ranks.get, reverse=True)   ## 정렬
        pairness = { }
        startOf = { }
        tuples = { }

        # unigram 추출 및 PMI 계산
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

        #--------------------------------unigram 추출 및 PMI 계산 여기까지


        # bigram 추출합니다.
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
                # G(TR) * A(PMI) * Length 이 수식으로 bigram 추출 밑에 한 줄 코드가 이 수식. 나눗셈이 나오는건 산술평균 때문
                tuples[path] = pmis / (len(path) - 1) * res ** (1 / len(path)) * len(path)

        used = set()
        both = { }
        for k in sorted(tuples, key=tuples.get, reverse=True):
            if used.intersection(set(k)):
                continue
            both[k] = tuples[k]
            for w in k:
                used.add(w)
                
        unigram = []
        bigram = []
        avg = 0.0
        # cost 평균 계산
        for key in both.keys():
            avg += both[key]
        if len(both.keys()) > 0:
            avg /= len(both.keys())
        
        # 제목 단어 추출 (두줄)
        titleWords = list(filter(wordFilter, self.readTaggerForTitle()))
        titleWords = list(map(lambda x: x[0], titleWords))


        # unigram 및 bigram을 본문에 나오는 형태로 저장 및 제목 가중치 조절 및 빈도 수 가중치 조절이 같이 들어가 있습니다.
        for key in both.keys():
            if len(key) == 2:
                # 특정 빈도수 이하는 가중치 뺄셈
                if self.taggerDictCount[key[0]] <= self.minimum_low_freq or self.taggerDictCount[key[1]] <= self.minimum_low_freq:
                    both[key] -= avg * self.low_freq_word_subtraction_multiplier

                # 제목에 등장한 단어 가중치 추가
                if key[0][0] in titleWords or key[1][0] in titleWords:
                    both[key] += avg * self.title_word_addition_multiplier
                joined1 = '%s %s' % (key[0][0], key[1][0])
                joined2 = '%s%s' % (key[0][0], key[1][0])
                if joined1 in self.content and wordFilter((joined1, 'NNG')):
                    bigram.append((joined1, both[key]))
                elif joined2 in self.content and wordFilter((joined2, 'NNG')):
                    bigram.append((joined2, both[key]))
            else:
                if self.taggerDictCount[key[0]] <= self.minimum_low_freq:
                    both[key] -= avg * self.low_freq_word_subtraction_multiplier
                if key[0][0] in titleWords:
                    both[key] += avg * self.title_word_addition_multiplier

                # 수사가 아니면 단일 단어 추가
                if key[0][1] != 'SN':
                    unigram.append(('%s' % (key[0][0]), both[key]))

        # cost가 음수가 되는 것들 제거
        unigram = list(filter(lambda x: x[1] > 0.0, unigram))
        bigram = list(filter(lambda x: x[1] > 0.0, bigram))


        res = set([])

        # 본문 명사 단어를 추출
        words = list(map(lambda x: (x, 0), self.tagger.nouns(self.preprocess(self.content))))

        ## 공백없는 복합단어를 최대 2개까지 합친다. ex) 한양대 대나무/숲/향기 -> 한양대 대나무숲향기
        ## 현재는 4개
        for _ in range(4):
            bigram, ex_words = self.merge_n_gram_with_unigram(bigram, words)
            unigram = list(filter(lambda x: x[0] not in ex_words, unigram))

        res = set(unigram + bigram)
        res = list(filter(lambda x: x[1] > 0.0, res))

        # 정렬 후 num 개수 만큼 추출 (tr.keywords(num=15) <-- 여기의 num변수)
        res = sorted(res, key=lambda x: x[1], reverse=True)[:num]

        return res


    ## 합친 단어가 전부 명사인지 다시한번 체크해주는 함수 (예: 다이어트 핵심이(X), 다이어트 핵심법(O))
    def check_word_is_noun(self, word):
        words = self.tagger.pos(word)
        return len(list(filter(lambda x: self.nounChecker(x[1]), words))) == len(words)


    ## 본문에 나온 형태로 띄어쓰기 없는 명사단어를 붙여주는 함수
    def merge_n_gram_with_unigram(self, n_gram, nouns):

        res = []
        ex_words = set([])
        for words in n_gram:
            found = False
            for word in nouns:
                ## 본문에는 붙어나오지만 형태소 분석기때문에 잘린 단어를 복구 (예:대나무숲 -> 숲이 제거되고 대나무만 추출 시)
                prefix_joined = word[0] + words[0]
                suffix_joined = words[0] + word[0]

                # 합칠 때 공백을 최대 1개만 허용
                if prefix_joined in self.content and self.check_word_is_noun(prefix_joined):
                    found = True
                    res.append((prefix_joined, word[1] + words[1])) # cost sum
                    ex_words.add(word[0])
                elif suffix_joined in self.content and self.check_word_is_noun(suffix_joined):
                    found = True
                    res.append((suffix_joined, word[1] + words[1])) # cost sum
                    ex_words.add(word[0])
            if not found:
                res.append(words)
                
        # 합친 결과, ex_word(합쳐져서 제외 할 단어) return
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
        # sent -> 한 문장
        # 8500만원 => 8500:SN 만:NR 원:NNBC 명사긴 한데 이상한 명사라서 뒤에 다 합침
        words = self.tagger.pos(sent)

        # 반복문으로 안나올때까지 합침
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
                    words[idx] = ((prefix_noun1, 'NNG'))
                elif prefix_noun2 in self.content:
                    words[idx] = ((prefix_noun2, 'NNG'))


            

        return words
    #Class 객체를 호출하면 맨 처음 실행하는 함수를 오버라이딩
    def __init__(self, **kwargs):
        # 변수 default값으로 초기화 혹은 값을 받으면 그것으로 초기화
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
        self.taggerTokenizer = lambda x: x[0] not in self.stopwords and self.nounChecker(x[1]) and (len(x[0]) > 1 or x[0] in self.singlewords) and re.compile('[一-龥]+').match(x[0]) is None
        # 영어일 경우 영어 명사 사전에 등록된 단어인지 검색
        self.englishFilter = lambda x: x[1] != 'SL' or nltk.pos_tag([x[0]])[0][1][0] == 'N'

        # variables for sentence summarization
        self.sentenceDictCount = { }
        self.sentenceDictBiCount = { }
        self.sentenceTokenizer = lambda sent: filter(self.taggerTokenizer, self.numeric(sent))
