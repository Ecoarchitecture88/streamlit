# Get Dataset from Mongo

from pymongo import MongoClient


CONNECTION_STRING = "mongodb+srv://eco-platform:gWfXUVDoNvlJr45u@eco-mongo-cluster.mrcy26g.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)

db = client['dir_platform_nlp']
db_documents = db['documents']
db_elements = db['elements']
db_nlp = db['nlp']



def get_sims_legal(embeddings1, list_sentences, min_score, min_text_length=10):
  # Get All Documents (Collection)
  # Get legal corpus label 2000 to 3000
  print(embeddings1)
  print(f'Min Score: {min_score}')
  print(f'Min Length: {min_text_length}')

  legal_docs = db_documents.find({
      '$and': [
          {'label': {'$gte': 1000}},
          {'label': {'$lt': 2000}},
          {'type': 'document'},
         
      ]
  })
  legal_sims = []
  countd = 0
  for d in legal_docs:
    id = d['uuid']
    name = d['name']

    (list_corpus_sentences,embeddings2) = get_embeddings_from(id)
    print(f'#{countd}\t{name}\t#{len(list_corpus_sentences)}')
    countd+=1
    # Compute similarities
    cos_scores = util.cos_sim(embeddings1, embeddings2)
    indexes = np.nonzero(cos_scores > min_score)
    print(indexes.tolist())
    list_index = indexes.tolist()
    for ix in list_index:
      i = ix[0]
      j = ix[1]
      text_original = list_sentences[i]['text']
      id_original = list_sentences[i]['_id']
      if len(text_original)>min_text_length:
        #print(f'{i}, {j}')
        sim_sentence = {
            'id': id_original,
            'original': text_original,
            'similarity': list_corpus_sentences[j]['text'],
            'score': cos_scores[i][j],
            'desc': list_corpus_sentences[j]['desc'],
            'page': list_corpus_sentences[j]['page'],
            'sentence': list_corpus_sentences[j]['sentence'],
            'uuid_sim': list_corpus_sentences[j]['uuid']
        }

        legal_sims.append(sim_sentence)
  print(f'# Legal Sims: {len(legal_sims)}')
  return legal_sims




from sentence_transformers import SentenceTransformer, util
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-mpnet-base-v2', device=device)
