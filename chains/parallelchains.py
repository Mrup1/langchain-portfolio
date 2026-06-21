from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'make me the notes for the given topic \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='make me 5 questions from the given text \n {topic}',
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template='merge the both notes {notes} and the questions {questions} and give me the final output ',
    input_variables=['notes', 'questions']
)

parallelchains = RunnableParallel({
    'notes': prompt1 | model1 | parser ,
    'questions': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallelchains | merge_chain 

chain.get_graph().print_ascii()  # this is to visualize the chain
topic ="""
Tigress" redirects here. For other uses, see Tiger (disambiguation) and Tigress (disambiguation).
Tiger
Temporal range: Early Pleistocene – Present 
PreꞒꞒOSDCPTJKPgN
↓

A Bengal tigress in Kanha Tiger Reserve, India
Conservation status

Endangered  (IUCN 3.1)[1]
CITES Appendix I (CITES)[1]
Scientific classification Edit this classification
Domain:	Eukaryota
Kingdom:	Animalia
Phylum:	Chordata
Class:	Mammalia
Order:	Carnivora
Suborder:	Feliformia
Family:	Felidae
Subfamily:	Pantherinae
Genus:	Panthera
Species:	P. tigris
Binomial name
Panthera tigris
(Linnaeus, 1758)[2]
Subspecies
P. t. tigris
P. t. sondaica
†P. t. acutidens
†P. t. soloensis
†P. t. trinilensis

Tiger distribution as of 2022
Synonyms[3]
Felis tigris Linnaeus, 1758
Tigris striatus Severtzov, 1858
Tigris regalis Gray, 1867
The tiger (Panthera tigris) is a large cat and a member of the genus Panthera native to Asia. It has a powerful, muscular body with a large head and paws, a long tail and orange fur with black, mostly vertical stripes. It is traditionally classified into nine recent subspecies, though some recognise only two subspecies, mainland Asian tigers and the island tigers of the Sunda Islands.

Throughout the tiger's range, it inhabits mainly forests, from coniferous and temperate broadleaf and mixed forests in the Russian Far East and Northeast China to tropical and subtropical moist broadleaf forests on the Indian subcontinent and Southeast Asia. The tiger is an apex predator and preys mainly on ungulates, which it takes by ambush. It lives a mostly solitary life and occupies home ranges, defending these from individuals of the same sex. The range of a male tiger overlaps with that of multiple females with whom he mates. Females give birth to usually two or three cubs that stay with their mother for about two years. When becoming independent, they leave their mother's home range and establish their own.

Since the early 20th century, tiger populations have lost at least 93% of their historic range and are locally extinct in West and Central Asia, in large areas of China and on the islands of Java and Bali. Today, the tiger's range is severely fragmented. It is listed as Endangered on the IUCN Red List of Threatened Species, as its range is thought to have declined by 53% to 68% since the late 1990s. Major threats to tigers are habitat destruction and fragmentation due to deforestation, poaching for fur and the illegal trade of body parts for medicinal purposes. Tigers are also victims of human–wildlife conflict as they attack and prey on livestock in areas where natural prey is scarce. The tiger is legally protected in all range countries. National conservation measures consist of action plans, anti-poaching patrols and schemes for monitoring tiger populations. In several range countries, wildlife corridors have been established and tiger reintroduction is planned.

The tiger is among the most popular of the world's charismatic megafauna. It has been kept in captivity since ancient times and has been trained to perform in circuses and other entertainment shows. The tiger featured prominently in the ancient mythology and folklore of cultures throughout its historic range and has continued to appear in culture worldwide.

Etymology
The Old English tigras derives from Old French tigre, from Latin tigris, which was a borrowing from tigris (Ancient Greek: τίγρις).[4] Since ancient times, the word tigris has been suggested to originate from the Armenian or Persian word for 'arrow', which may also be the origin of the name for the river Tigris.[5][6] However, today, the names are thought to be homonyms, and the connection between the tiger and the river is doubted.[6]"""
resultfinal = chain.invoke({'topic': topic})
print(resultfinal)
