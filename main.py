# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:46:28 2020

@author: OHyic
"""
#%%
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import os


def get_elmo_embeddings(sentences,max_tokens):
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
    #return 1024 embeddings per word
    return np.array(embeddings)

def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def get_glove_embeddings(glove_model,sentences,max_tokens):
    glove_embeddings=None
    for indx,sentence in enumerate(sentences):
        temp=[]
        for word in sentence.split(" "):
            try:    
                glove_word=glove_model[word]
                temp.append(glove_word)
  
            except KeyError:
                temp.append(np.zeros(300))
        temp=np.array(temp)
        padding_length = max_tokens - temp.shape[0]
        temp=np.append(temp, np.zeros((padding_length, temp.shape[1])), axis=0)
        if(indx==0):
            print("ran")
            glove_embeddings=np.array([temp])
        else:
            glove_embeddings=np.append(glove_embeddings,np.array([temp]),axis=0)
    return np.array(glove_embeddings)
        
def fuse_embeddings (glove_embeddings,elmo_embeddings):
    fused_embeddings =None
    for indx in range(glove_embeddings.shape[0]):
        temp=np.hstack((glove_embeddings[indx],elmo_embeddings[indx]))
        
        if(indx==0):
            fused_embeddings=np.array([temp])
        else:
            fused_embeddings=np.append(fused_embeddings,np.array([temp]),axis=0)
    return fused_embeddings


#define max token length
max_tokens=3

#input sentences
sentences=["how are you"]

#load glove model, download glove model from https://nlp.stanford.edu/projects/glove/ and place it in the project directory
glove_model=load_glove_model(os.getcwd()+"\\glove.42B.300d.txt")

#%%
elmo_embeddings=get_elmo_embeddings(sentences,max_tokens)
glove_embeddings=get_glove_embeddings(glove_model,sentences,max_tokens)
fused_embeddings=fuse_embeddings (glove_embeddings,elmo_embeddings)
