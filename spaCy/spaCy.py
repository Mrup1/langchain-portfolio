import spacy 

nlp = spacy.load("en_core_web_sm")

with open("football.txt", "r") as f:
    text = f.read()

print("Text length:", len(text))
#print(text)

doc = nlp(text)

print(len(doc))

for sent in doc[0:20]:
    print(sent)
print('-------' * 8 )
#for sen in text[0:10]:
   # print(sen)
print('-------' * 8 )

for sen in text.split()[0:20]:
    print(sen)

for sen in doc.sents:
    print(sen)

sentence1 = list(doc.sents)[5]

print("senteñse 1 : ",sentence1)
