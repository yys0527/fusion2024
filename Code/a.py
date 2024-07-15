import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.tokenizer import LTokenizer
import re
import private_data

corpus = DoublespaceLineCorpus(private_data.bpath+"data/corpus.txt")
print(len(corpus))
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()

def processing(a):
    a = re.sub('[{(<>)}]', repl='', string=a)

    pattern = r'[^a-zA-Z가-힣]'
    a = re.sub(pattern=pattern, repl=' ', string=a)

    pattern = re.compile("["
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
    scores = {word:score.cohesion_forward for word, score in word_score_table.items()}

    l_tokenizer = LTokenizer(scores=scores)
    print(l_tokenizer.tokenize(txt, flatten=False))
    maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
    print(maxscore_tokenizer.tokenize(txt))

while 1:
    main()
