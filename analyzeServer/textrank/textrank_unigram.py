import networkx
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


    ## 키워드용 전처리 함수
    ## 불필요한 문장을 제거
    ## text: string
    ## return: preprocessed string
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
    # 문장 복구 함수
    # 인용 구문을 처리하기 위해 손상시킨 문장을 원래 문장과 비교하고 복구한다
    # strings: Array(string)
    # original: String
    # return: 복구된 Array
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
    # TEXT를 여러개의 문장으로 구분해 주는 함수
    # self.content 에 담긴 내용을 문장으로 잘라서 return
    # return: Array(generator)
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
    # self.content(뉴스 기사 본문) 에서 문장별로 형태소를 추출하는 함수
    # 형태소를 추출하고 마지막에 수사 + 명사를 합치는 numeric 함수를 호출한다.
    # return: Array(tuple(string, pos))
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

    # self.title(뉴스 기사 제목) 에서 형태소를 추출하는 함수
    # return: Array(tuple(string, pos))
    def readTaggerForTitle(self):
        return self.numeric(self.title)

    # build additional graph information for applying pagerank algorithm
    # 키워드를 페이지랭크에 적용하기 위한 작업
    # 페이지랭크 알고리즘에서 필요한 빈도수, 간선 cost 를 구한다.
    def loadTagger(self):
        # construct bigram count dictionary
        # 두 단어의 빈도 수 증가시키는 함수
        # a: tuple(string, pos)
        # b: tuple(string, pos)
        def insertPair(a, b):
            if a > b:
                a, b = b, a
            elif a == b:
                return
            self.taggerDictBiCount[a, b] = self.taggerDictBiCount.get((a, b), 0) + 1

        # 인정한 두 단어의 빈도 수를 증가시키는 함수
        # a: tuple(string, pos)
        # b: tuple(string, pos)
        def insertNearPair(a, b):
            self.dictNear[a, b] = self.dictNear.get((a, b), 0) + 1

        # 형태소를 추출
        taggerIter = self.readTagger()
        # save pos tagging result
        wordFilter = self.taggerTokenizer

        # generator 를 하나씩 받아온다.
        for sent in taggerIter:
            # word filtering
            # 문장에서 명사이거나 영어이거나 불용어가 아니거나 한 글자 단어가 아니거나(한 글자이지만 whitelist 에 있으면 통과)
            # 한자가 아닌 단어만 가져온다.
            sent = list(filter(wordFilter, sent))

            # english filtering
            # 영어일 경우 nltk 의 영어 형태소 분석을 이용하여 명사만 가져온다.
            sent = list(filter(self.englishFilter, sent))
            for i, word in enumerate(sent):

                # word: tuple(string, pos)
                # 빈도수를 1 증가시킨다.
                self.taggerDictCount[word] = self.taggerDictCount.get(word, 0) + 1
                self.nTotal += 1
                # create bigram for similar word filtering
                # tuple(string, NNP) 형태의 bigram 을 강제로 생성하여 wordFilter 를 통과하는지 체크한다.
                joined1 = '%s %s' % (sent[i-1][0], word[0])
                joined2 = '%s%s' % (sent[i-1][0], word[0])

                # check similar word (ex: iPhone6, iPhone 6)
                # inset near pair. then, similar word is removed
                # 불용어 사전을 통과하는지 검사한다.
                # 이때 bigram 형성시 불용어 사전에서 검사한다.
                # 바로 이전 단어와 bigram을 형성한다.
                if i - 1 >= 0 and wordFilter((joined1, 'NNP')) or wordFilter((joined2, 'NNP')):
                    insertNearPair(sent[i-1], word)
                if i+1 < len(sent):
                    # news content based bigram
                    # 바로 다음 단어와 bigram을 형성한다.
                    joined3 = '%s %s' % (word[0], sent[i+1][0])
                    joined4 = '%s%s' % (word[0], sent[i+1][0])
                    if wordFilter((joined3, 'NNP')) or wordFilter((joined4, 'NNP')):
                        insertNearPair(word, sent[i+1])

                # construct bigram count
                # 멀리 떨어져 있는 단어와 bigram을 형성한다.
                # pagerank 그래프 구성 후 가중치 계산에 영향을 미친다.
                for j in range(i+1, min(i + self.window + 1, len(sent))):
                        joined1 = '%s %s' % (word[0], sent[j][0])
                        joined2 = '%s%s' % (word[0], sent[j][0])
                        if sent[j] != word and (wordFilter((joined1, 'NNP')) or wordFilter((joined2, 'NNP'))):
                            insertPair(word, sent[j])

        # similar word count concatenate (ex: iPhone6, iPhone 6)
        # 공백의 유무만 다르고 같은 단어가 사용된 bigram은 빈도수를 합친다.
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
    # 문장을 페이지랭크 알고리즘에 적용하기 위한 사전 작업
    # 빈도수, 간선 cost 를 문장 추출에 유용한 형태로 계산한다.
    def loadSentence(self):
        # get similarity of two sentences
        # 두 문장의 유사도를 측정한다
        # a: string
        # b: string
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
    # 계산된 결과를 가지고 페이지 랭크 그래프 구성
    def loadKeywordGraph(self):
        self.taggerGraph = networkx.Graph()
        self.taggerGraph.add_nodes_from(self.taggerDictCount.keys())

        wordFilter = self.taggerTokenizer
        for (a, b), n in self.taggerDictBiCount.items():
            self.taggerGraph.add_edge(a, b, weight=n*n*n*self.coef + (1 - self.coef))

    # build sentence graph
    # 계산된 결과를 가지고 페이지 랭크 그래프 구성
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

    # pagerank apply
    # 형태소 분석
    # 명사 추출
    # 영어 명사 추출
    # 수사 변형
    # 페이지랭크 그래프 구성
    # 페이지랭크 계산
    def keyword_rank(self):
        self.loadTagger()
        self.loadKeywordGraph()
        self.keyword_ranks = self.pagerank(self.taggerGraph)

    # get keywords 
    # 키워드 추출
    # 계산된 페이지랭크 값을 이용하여 키워드를 추출한다.
    # unigram, bigram 에 수식을 적용하여 cost 를 변형한다.
    # 변형된 cost 의 평균값을 구한다.
    # - 본문에 등장한 단어를 포함하는 unigram, bigram은 평균값 * 일정값 만큼 가중치를 증가시킨다.
    # - 빈도수가 낮은 단어를 포함하는 unigram, bigram은 평균값 * 일정값 만큼 가중치를 감소시킨다.
    # - NNP 단어를 포함하는 unigram, bigram은 가중치를 평균값 * 일정값 만큼 증가시킨다.
    # - NNG 단어를 포함하는 unigram, bigram은 가중치를 평균값 * 일정값 만큼 증가시킨다.
    # 본문에 등장하는 형태로 unigram, bigram을 변형시킨다.
    # 변형된 단어를 비교하여 같은 형태로 변형되었을 경우(공백의 유무만 다른 bigram 일 경우) cost 를 합친다.
    # 재 정렬하여 지정한 개수만큼만 결과를 내보낸다.
    #
    # num: 추출할 키워드의 개수
    # return: Array(tuple(string, cost)) - Reversed by cost
    def keywords(self, num=15):
        ranks = self.keyword_ranks
        wordFilter = self.taggerTokenizer
        cand = sorted(ranks, key=ranks.get, reverse=True)
        pairness = { }
        startOf = { }
        tuples = { }

        for k in cand:
            tuples[(k, )] = self.getI(k) * ranks[k]

        used = set()
        both = { }
        for k in sorted(tuples, key=tuples.get, reverse=True):
            if used.intersection(set(k)):
                continue
            both[k] = tuples[k]
            for w in k:
                used.add(w)
                
        unigrams = []
        avg = 0.0
        for key in both.keys():
            avg += both[key]
        if len(both.keys()) > 0:
            avg /= len(both.keys())

        titleWords = list(filter(wordFilter, self.readTaggerForTitle()))
        titleWords = list(map(lambda x: x[0], titleWords))
        for key in both.keys():
            if key[0][1] == 'NNP':
                both[key] += avg * self.nnp_addition_multiplier
            if key[0][1] == 'NNG':
                both[key] += avg * self.nng_addition_multiplier
            if self.taggerDictCount[key[0]] <= self.minimum_low_freq:
                both[key] -= avg * self.low_freq_word_subtraction_multiplier
            if key[0][0] in titleWords:
                both[key] += avg * self.title_word_addition_multiplier

            # 수사가 아니면 단일 단어 추가
            if key[0][1] != 'SN':
                unigrams.append(('%s' % (key[0][0]), both[key]))

        unigrams = sorted(list(filter(lambda x: x[1] > 0.0, unigrams)), key=lambda x: x[1], reverse=True)[:num*2]

        unigrams = self.expand_n_gram_to_match_context(unigrams, is_bigram=False)

        res = self.merge_expanded_similar_word([], unigrams)
        res = list(filter(lambda x: x[1] > 0.0, res))
        res = sorted(res, key=lambda x: x[1], reverse=True)[:num]

        return res

    # 같은 의미지만 다른 형태를 띈 단어끼리 cost 를 합친다.
    # bigrams: Array(tuple(string, cost))
    # unigrams: Array(tuple(string, cost))
    # return: Array(tuple(string, cost))

    def merge_expanded_similar_word(self, bigrams, unigrams):
        # convert to dict
        # bigram_dict[word] = cost
        cost_dict = { }
        for bigram in bigrams:
            # 이미 중복된 키이면 cost 를 더한다
            if bigram[0] in cost_dict:
                cost_dict[bigram[0]] += bigram[1]
            else:
                cost_dict[bigram[0]] = bigram[1]
        for unigram in unigrams:
            # 이미 중복된 키이면 cost 더함
            if unigram[0] in cost_dict:
                cost_dict[unigram[0]] += unigram[1]
            else:
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
    # bigram 이면 pos tagging 비교가 불가능 하기 때문에(이미 본문에 나온 형태로 변형이 이뤄짐) 곧바로 다음 공백까지 단어를 확장
    # unigram 이면 post tagging 하여 같은 형태의 단어에서 확장을 시작한다.
    # n_gram: Array(tuple(string, cost))
    def expand_n_gram_to_match_context(self, n_gram, is_bigram=True):

        res = []
        for words in n_gram:

            position = -1
            found = False
            poses = []
            morph = ''
            ## position search
            ## 특정 단어가 본문의 어디에서 등장하는지 검색
            ## bigram 이면 본문에서 등장하는 index 만 찾는다.
            ## unigram 이면 같은 품사를 갖는 단어의 index 를 찾는다.
            while (position < len(self.content)):
                position = self.content.find(words[0], position + 1)
                if position == -1: break

                left = position - 1
                left_space = 0
                right = position + len(words[0])
                right_space = 0

                # 올바른 품사 tagging 을 위해 공백을 넘어서 앞 단어까지 합쳐서 불완전하지만 문맥을 구성한다.
                while left >=0:
                    if not self.boundaryChecker(self.content[left]):
                        left_space += 1
                    if left_space == 2:
                        break
                    left -= 1
                while right < len(self.content):
                    if not self.boundaryChecker(self.content[right]):
                        right_space += 1
                    if right_space == 2:
                        break
                    right += 1

                # 불완전한 문맥을 가져온다.
                morph = self.content[left+1: right+1]
                poses = self.tagger.pos(morph)

                # bigram 이라면 품사 비교를 하지 않는다.
                # unigram 이라면 품사를 비교하여 본문에서의 올바른 위치인지 판단한다.
                if is_bigram or words[0] in list(map(lambda x: x[0], poses)):
                    found = True
                    break

            if not found:
                continue

            # 경계 검사를 하여 문맥을 구성하느라 불필요하게 확장된 부분을 제거한다.
            while len(poses) and left_space == 2:
                position = morph.find(poses[0][0])
                if position != -1 and (not self.boundaryChecker(morph[position]) or position + len(poses[0][0]) < len(morph) and not self.boundaryChecker(morph[position + len(poses[0][0])])):
                    poses.pop(0)
                    break
                poses.pop(0)

            while len(poses) and right_space == 2:
                position = morph.find(poses[len(poses)-1][0])
                if position != -1 and (not self.boundaryChecker(morph[position]) or position > 0 and not self.boundaryChecker(morph[position-1])):
                    poses.pop()
                    break
                poses.pop()
                
            left = -1
            right = len(poses)

            for idx, pos in enumerate(poses):
                # bigram check
                if idx + 1 <= len(poses)-1:
                    joined1 = poses[idx][0] + poses[idx+1][0]
                    joined2 = poses[idx][0] + ' ' + poses[idx+1][0]
                    if joined1 == words[0] or joined2 == words[0]:
                        left = idx - 1
                        right = idx + 2
                        break

                # unigram check
                if poses[idx][0] == words[0]:
                    left = idx - 1
                    right = idx + 1

            lower = 0
            upper = len(poses)-1

            merged_bigram = words[0]

            # 단어를 확장한다.
            left_partials = []
            while lower <= left:
                if self.mergeNounChecker(poses[lower][1]):
                    left_partials.append(poses[lower][0])
                else:
                    left_partials = []
                lower += 1

            for word in reversed(left_partials):
                merged_bigram = word + merged_bigram

            right_partials = []
            while right <= upper:
                if self.mergeNounChecker(poses[upper][1]):
                    right_partials.append(poses[upper][0])
                else:
                    right_partials = []
                upper -=1

            for word in reversed(right_partials):
                merged_bigram = merged_bigram + word

            res.append((merged_bigram, words[1]))
        return res

    # pagerank apply
    # 문장을 구성.
    # 형태소 분석
    # 명사 추출
    # 영어 명사 추출
    # 수사+명사를 명사로 변형
    # 페이지랭크 그래프 구성
    # 페이지랭크 cost 계산
    def sentence_rank(self):
        self.loadSentence()
        self.loadSentenceGraph()
        self.sentence_ranks = self.pagerank(self.sentenceGraph)

    # get sentences
    # 문장을 추출한다.
    # ratio: 추출된 문장에서 얼마만큼의 비중만 내보낼 것인지
    # ratio: float
    # return: Array(tuple(sentence, cost)) - Reversed by cost
    def sentences(self, ratio = 0.333):

        r = self.sentence_ranks
        ks = sorted(r, key=r.get, reverse=True)[:int(len(r)*ratio)]

        ks = list(map(lambda k:(self.sentenceDictCount[k], r[k]), sorted(ks)))
        return sorted(ks, key=lambda x: x[1], reverse=True)


    # 숫자 + 명사를 한 단어로 인식하게 하는 함수
    # 품사 태깅 후 숫자 + 명사를 명사로 바꾼다.
    # sent: String
    # return: Array(tuple(string, pos))
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
                    if words[idx2][1] == 'SL' or (words[idx2][1][0] == 'N' and words[idx2][1] != 'NNBC' and words[idx2][1] != 'NR'):
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
        self.nng_addition_multiplier = kwargs.get('nng_addition_multiplier', 0)
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
        self.mergeNounChecker = lambda x: x in ('NNG', 'NNP', 'SL', 'NNB')
        # 숫자, 영어, 한글이면 일단 단어 경계에서 제외 (단어를 확장할때 쓴다)
        self.boundaryChecker = lambda x: re.compile('[a-zA-Z0-9가-힣]+').match(x)

        self.taggerTokenizer = lambda x: x[0] not in self.stopwords and self.nounChecker(x[1]) and (len(x[0]) > 1 or x[0] in self.singlewords) and re.compile('[一-龥]+').match(x[0]) is None

        self.englishFilter = lambda x: x[1] != 'SL' or nltk.pos_tag([x[0]])[0][1][0] == 'N'

        # variables for sentence summarization
        self.sentenceDictCount = { }
        self.sentenceDictBiCount = { }
        self.sentenceTokenizer = lambda sent: filter(self.taggerTokenizer, self.numeric(sent))
