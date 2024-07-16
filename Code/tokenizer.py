import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.tokenizer import LTokenizer
import re
import private_data

corpus = DoublespaceLineCorpus(private_data.bpath+"data/corpus.txt")                    # 단어 추출 학습 데이터 준비
word_extractor = WordExtractor()                                                        # 단어 추출 함수 선언
word_extractor.train(corpus)                                                            # 단어 추출 학습
word_score_table = word_extractor.extract()

def processing(a):                                                                      # 문장 전처리
    a = re.sub('[{(<>)}]', repl='', string=a)                                           # 괄호 삭제

    pattern = r'[^a-zA-Z가-힣]'
    a = re.sub(pattern=pattern, repl=' ', string=a)                                     # 특수문자 삭제

    pattern = re.compile("["                                                            # 유니코드 특수문자 삭제
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
                            "]+", flags=re.UNICODE)

    a = re.sub(pattern=pattern, repl='', string=a)
    return a

def main():
    txt = input("input : ")
    txt = processing(txt)
    scores = {word:score.cohesion_forward for word, score in word_score_table.items()}  # 토크나이저 내부 평가 지표 설정

    # l_tokenizer = LTokenizer(scores=scores)
    # print(l_tokenizer.tokenize(txt, flatten=False))
    maxscore_tokenizer = MaxScoreTokenizer(scores=scores)                               # 토크나이저 선언
    a = maxscore_tokenizer.tokenize(txt)                                                # 토큰화
    return a