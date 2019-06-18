# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:25:12 2019

@author: OHyic
"""

from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

#define max token length
max_tokens=60

#input sentences
sentences=["how are you doing","what is your name","can you subscribe to my channel"]

#create a pretrained elmo model (requires internet connection)
elmo = ElmoEmbedder(cuda_device=0)
embeddings=[]

#loop through the input sentences
for index,elmo_embedding in enumerate(elmo.embed_sentences(sentences)):  
    print("elmo:",index)
    # Average the 3 layers returned from Elmo
    avg_elmo_embedding = np.average(elmo_embedding, axis=0)
    padding_length = max_tokens - avg_elmo_embedding.shape[0]
    if(padding_length>0):
        avg_elmo_embedding =np.append(avg_elmo_embedding, np.zeros((padding_length, avg_elmo_embedding.shape[1])), axis=0)
    else:
        avg_elmo_embedding=avg_elmo_embedding[:max_tokens]
    embeddings.append(avg_elmo_embedding) 