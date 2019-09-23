import xml.etree.ElementTree as ET
import torch
import pickle
import glob
import argparse
import numpy as np

from allennlp.commands.elmo import ElmoEmbedder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm, trange
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


class ELMO:
    
    def __init__(self):
        
        self.elmo = ElmoEmbedder()
        


class Word_Sense_Model:
    
    def __init__(self):
        
        self.sense_number_map = {'N':1, 'V':2, 'J':3, 'R':4}
        
        self.Elmo_Model = ELMO()
        self.lemmatizer = WordNetLemmatizer()

        
    def open_xml_file(self, file_name):
        
        tree = ET.parse(file_name)
        root = tree.getroot()
        
        return root, tree
    
    def wngt_sent_sense_collect(self, xml_struct):
    
        _sent =[]
        _sent1 = []
        _senses = []
        temp_list_pos = []
            
        _back_sent = []
        _back_sent1 = ""
        _back_senses = []
        
        for idx,j in enumerate(xml_struct.iter('word')):

            _temp_dict = j.attrib
            
            if 'lemma' in _temp_dict:
                _word = _temp_dict['lemma'].lower()
            else:
                _word = _temp_dict['surface_form'].lower()
            
            _back_sent.extend([_word])
            _back_sent1 += _word + " "
            
            if 'wn30_key' in _temp_dict:
                _back_senses.extend( [_temp_dict['wn30_key']]*len([_word]))               
            else:
                _back_senses.extend( [0]*len([_word]))
                
        _temp_dict = xml_struct.attrib
        
        if 'wn30_key' in _temp_dict:
            
            _senses1 = _temp_dict['wn30_key'].split(';')
                   
        for i in _senses1:
            
            _word = [str(i.split('%')[0]), 'is'] 
            
            _temp_sent = []
            _temp_sent1 = ""
            _temp_senses = []
            
            _temp_sent.extend(_word)
            _temp_sent.extend(_back_sent)
            
            _temp_sent1 += ' '.join(_word) + " " + _back_sent1
            
            _temp_senses.extend([i,0])
            _temp_senses.extend(_back_senses)
            
            _sent.append(_temp_sent)
            _sent1.append(_temp_sent1)
            _senses.append(_temp_senses)
            
        return _sent, _sent1, _senses, temp_list_pos
    
    def semcor_sent_sense_collect(self, xml_struct):
        
        _sent =[]
        _sent1 = ""
        _senses = []
        temp_list_pos = []
        
        for idx,j in enumerate(xml_struct.iter('word')):
            
            _temp_dict = j.attrib
            flag = 0
            
            if 'lemma' not in _temp_dict:
                
                words = _temp_dict['surface_form'].lower()
                
                _sent1 += words + " "
                
                words = words.split('_')
                
                words1 = words[0:1]
                words2 = words[1:]
            
            else:
                
                _pos = _temp_dict['pos'].lower()[0] 
                
                if _pos not in ['a', 'v', 'n']:
                    _pos = 'n'
                    
                w2 = _temp_dict['lemma'].lower().split('_')
                words = _temp_dict['surface_form'].lower()
                
                _sent1 += words + " "
                
                words = words.split('_')
                
                l = self.lemmatizer.lemmatize(words[0],pos=_pos)
                if str(l).startswith(w2[0]) or str(w2[0]).startswith(l):
                
                    words1 = words[0:1]
                    words2 = words[1:]
                else:
                    flag = 1
                    
            _sent.extend(words)
            
            if 'wn30_key' in _temp_dict:
                if not flag:
                    
                    _senses.extend([_temp_dict['wn30_key']]*len(words1))
                    _senses.extend([0]*len(words2))
                else:
                    _senses.extend([0]*len(words))

            else:
                _senses.extend([0]*len(words))
            
        return _sent, _sent1, _senses, temp_list_pos
    
        
    def semeval_sent_sense_collect(self, xml_struct):
        
        _sent =[]
        _sent1 = ""
        _senses = []
        pos = []
        
        for idx,j in enumerate(xml_struct.iter('word')):
            
            _temp_dict = j.attrib
            
            if 'lemma' in _temp_dict:
                
                words = _temp_dict['lemma'].lower()

            else:
                
                words = _temp_dict['surface_form'].lower()
            
            if '*' not in words:
                
                _sent1 += words + " "

                _sent.extend([words])
                
                if 'pos' in _temp_dict:
                    pos.extend([_temp_dict['pos']]*len([words]))

                else:
                    pos.extend([0]*len([words]))
                    
                if 'wn30_key' in _temp_dict:

                    _senses.extend([_temp_dict['wn30_key']]*len([words]))

                else:
                    _senses.extend([0]*len([words]))
                
        return _sent, _sent1, _senses, pos
    

        
    def create_word_sense_maps(self, _word_sense_emb):
    
        _sense_emb = {}
        _sentence_maps = {}
        _sense_word_map ={}
        _word_sense_map ={}
    
        for i in _word_sense_emb:
    
            if i not in _word_sense_map:
                _word_sense_map[i] = []

            for j in _word_sense_emb[i]:

                if j not in _sense_word_map:
                    _sense_word_map[j] = []

                _sense_word_map[j].append(i)
                _word_sense_map[i].append(j)

                if j not in _sense_emb:
                    _sense_emb[j] =[]
                    _sentence_maps[j] = []

                _sense_emb[j].extend(_word_sense_emb[i][j]['embs'])
                _sentence_maps[j].extend(_word_sense_emb[i][j]['sentences'])
        
        return _sense_emb, _sentence_maps, _sense_word_map, _word_sense_map
    
        
    def train(self, train_file, training_data_type):
        
        print("Training Embeddings!!")
        
        _word_sense_emb = {}
        
        _train_root, _train_tree = self.open_xml_file(train_file)
        
        for i in tqdm(_train_root.iter('sentence')):
            
            if training_data_type == "SE":
                all_sent, all_sent1, all_senses, _ = self.semeval_sent_sense_collect(i)
                all_sent, all_sent1, all_senses = [all_sent], [all_sent1], [all_senses]
                
            elif training_data_type == "SEM":
                all_sent, all_sent1, all_senses, _ = self.semcor_sent_sense_collect(i)
                all_sent, all_sent1, all_senses = [all_sent], [all_sent1], [all_senses]
            
            elif training_data_type == "WNGT":
                all_sent, all_sent1, all_senses, _ = self.wngt_sent_sense_collect(i)
            
            else:
                print("Argument train_type not specified properly!!")
                quit()
            
            for sent, sent1, senses in zip(all_sent, all_sent1, all_senses):
                
                try:

                    final_layer = self.Elmo_Model.elmo.embed_sentence(sent)[-1]
                    
                    count = 0

                    for idx, j in enumerate(zip(senses, sent)):

                        sense = j[0]
                        word = j[1]

                        if sense != 0:

                            embedding = final_layer[count]

                            if word not in _word_sense_emb:
                                _word_sense_emb[word]={}

                            for s in sense.split(';'):

                                if s not in _word_sense_emb[word]:
                                    _word_sense_emb[word][s]={}
                                    _word_sense_emb[word][s]['embs'] = []
                                    _word_sense_emb[word][s]['sentences'] = []

                                _word_sense_emb[word][s]['embs'].append(embedding)
                                _word_sense_emb[word][s]['sentences'].append(sent1)

                        count += 1

                except Exception as e:
                    print(e)
        
        return _word_sense_emb
   
    def load_embeddings(self, pickle_file_name, train_file, training_data_type):
        
        try:
             
            with open(pickle_file_name, 'rb') as h:
                _x = pickle.load(h)
                
                print("EMBEDDINGS FOUND!")
                return _x
            
        except:
            
            print("Embedding File Not Found!! \n")
            
            word_sense_emb = self.train(train_file, training_data_type)
            
            with open(pickle_file_name, 'wb') as h:
                pickle.dump(word_sense_emb, h)
                
            print("Embeddings Saved to " + pickle_file_name)
            
            return word_sense_emb
        
    def test(self, 
             train_file, 
             test_file, 
             emb_pickle_file,
             training_data_type,
             save_to, 
             k=1, 
             use_euclidean = False,
             reduced_search = True):
        
        
        word_sense_emb = self.load_embeddings(emb_pickle_file, train_file, training_data_type)

        print("Testing!")
        sense_emb, sentence_maps, sense_word_map, word_sense_map = self.create_word_sense_maps(word_sense_emb)
        
        _test_root, _test_tree = self.open_xml_file(test_file)
        
        _correct, _wrong= [], []
            
        open(save_to, "w").close()
        
        for i in tqdm(_test_root.iter('sentence')):
            
            sent, sent1, senses, pos = self.semeval_sent_sense_collect(i)
            
            final_layer = self.Elmo_Model.elmo.embed_sentence(sent)[-1]
              
            count, tag, nn_sentences = 0, [], []
            
            for idx, j in enumerate(zip(senses, sent, pos)):
                
                word = j[1]
                pos_tag = j[2][0]
                
                if j[0] != 0:
                    
                    _temp_tag = 0
                    max_score = -99
                    nearest_sent = 'NONE'
                    
                    embedding = final_layer[count]
                    
                    min_span = 10000
                    
                    if word in word_sense_map:
                        concat_senses = []
                        concat_sentences = []
                        index_maps = {}
                        _reduced_sense_map = []
                        
                        if reduced_search:
                            
                            for sense_id in word_sense_map[word]:

                                if self.sense_number_map[pos_tag] == int(sense_id.split('%')[1][0]):

                                    _reduced_sense_map.append(sense_id)
                        
                        if len(_reduced_sense_map) == 0 :
                            _reduced_sense_map = list(word_sense_map[word])
                        
                        for sense_id in _reduced_sense_map:
                            index_maps[sense_id] = {}
                            index_maps[sense_id]['start'] = len(concat_senses)

                            concat_senses.extend(sense_emb[sense_id])
                            concat_sentences.extend(sentence_maps[sense_id])

                            index_maps[sense_id]['end'] = len(concat_senses) - 1
                            index_maps[sense_id]['count'] = 0

                            if min_span > (index_maps[sense_id]['end']-index_maps[sense_id]['start']+1):

                                min_span = (index_maps[sense_id]['end']-index_maps[sense_id]['start']+1)

                        min_nearest = min(min_span, k)

                        concat_senses = np.array(concat_senses)


                        if use_euclidean:

                            simis = euclidean_distances(embedding.reshape(1,-1), concat_senses)[0]
                            nearest_indexes = simis.argsort()[:min_nearest]

                        else:

                            simis = cosine_similarity(embedding.reshape(1,-1), concat_senses)[0]
                            nearest_indexes = simis.argsort()[-min_nearest:][::-1]

                        for idx1 in nearest_indexes:

                            for sense_id in _reduced_sense_map:

                                if index_maps[sense_id]['start']<= idx1 and index_maps[sense_id]['end']>=idx1:
                                    index_maps[sense_id]['count'] += 1

                                    score = index_maps[sense_id]['count']

                                    if score > max_score:
                                        max_score = score
                                        _temp_tag = sense_id
                                        nearest_sent = concat_sentences[idx1]


                    tag.append(_temp_tag)
                    nn_sentences.append(nearest_sent)

                count += 1
                
            _counter = 0
            
            for j in i.iter('word'):
                
                temp_dict = j.attrib
                
                try:
                    
                    if 'wn30_key' in temp_dict:
                        
                        if tag[_counter] == 0:
                            pass
                        
                        else:
                            j.attrib['WSD'] = str(tag[_counter])
                            
                            if j.attrib['WSD'] in str(temp_dict['wn30_key']).split(';') :
                           
                                _correct.append([temp_dict['wn30_key'], j.attrib['WSD'], (sent1), nn_sentences[_counter]])
                            else:
                                _wrong.append([temp_dict['wn30_key'], j.attrib['WSD'], (sent1), nn_sentences[_counter]])

                        _counter += 1
                        
                except Exception as e:
                    
                    print(e)
            
        with open(save_to, "w") as f:
        
            _test_tree.write(f, encoding="unicode")    
        
        print("OUTPUT STORED TO FILE: " + str(save_to))
        
        return _correct, _wrong
                

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='WSD using ELMo')
    
    parser.add_argument('--train_corpus', type=str, required=True, help='Training Corpus')
    parser.add_argument('--train_type', type=str, required=True, help='SEM/WNGT/SE')
    parser.add_argument('--trained_pickle',type=str,help='Pickle file of Trained ELMo Embeddings/Save Embeddings to this file')
    parser.add_argument('--test_corpus', type=str, required=True, help='Testing Corpus')
    parser.add_argument('--start_k', type=int, default=1, help='Start value of Nearest Neighbour')
    parser.add_argument('--end_k', type=int, default=1, help='End value of Nearest Neighbour')
    parser.add_argument('--save_xml_to', type=str, help='Save the final output to?')
    parser.add_argument('--use_euclidean', type=int, default=0, help='Use Euclidean Distance to Find NNs?')
    parser.add_argument('--reduced_search', type=int, default=0, help='Apply Reduced POS Search?')
    
    args = parser.parse_args()
    
    print("Training Corpus is: " + args.train_corpus)
    print("Testing Corpus is:  " + args.test_corpus)
    print("Nearest Neighbour start: " + str(args.start_k))
    print("Nearest Neighbour end: " + str(args.end_k))
    
    if args.reduced_search:
        print("Using Reduced POS Search!")
    
    else:
        print("Using the Search without POS!")
    
    if args.use_euclidean:
        print("Using Euclidean Distance!")
    
    else:
        print("Using Cosine Similarity!")
    
    print("Loading WSD Model!")
    
    WSD = Word_Sense_Model()
    
    print("Loaded WSD Model!")
    
    for nn in range(args.start_k, args.end_k+1):
        
        correct, wrong = WSD.test(train_file=args.train_corpus, 
                                  test_file = args.test_corpus,
                                  training_data_type = args.train_type,
                                  emb_pickle_file = args.trained_pickle,
                                  save_to = args.save_xml_to[:-4] + "_" + str(nn)+args.save_xml_to[-4:],
                                  k=nn,
                                  use_euclidean = args.use_euclidean, 
                                  reduced_search = args.reduced_search)
                                  
