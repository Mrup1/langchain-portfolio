#it first break the text  paragraph wise then line wise then words wise and in the end it goes for the character wise

from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
    Football, known as soccer in some parts of the world, is more than just a game; it's a global language, a cultural phenomenon that unites billions across continents. From the dusty pitches of developing nations to the grand stadiums of international tournaments, the sport evokes unparalleled passion, drama, and collective identity. Its simple premise – two teams attempting to score by getting a ball into the opposing goal, primarily using their feet – belies the intricate strategies, breathtaking athleticism, and profound emotional connections it fosters.

The origins of football can be traced back to various ball games played in ancient civilizations, but the modern codified game emerged in 19th-century England. The establishment of the Football Association (FA) in 1863 and the subsequent standardization of rules laid the groundwork for its rapid expansion. British sailors, merchants, and engineers carried the game to distant lands, where it quickly took root, adapting to local cultures and developing unique styles of play. This organic spread, rather than colonial imposition, contributed significantly to its universal appeal.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print((result[10]))