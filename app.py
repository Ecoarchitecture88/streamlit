# Get Dataset from Mongo

from pymongo import MongoClient
import streamlit as st


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


def init_html(title):
  html = ''
  html += '<!DOCTYPE html>\n<html>\n <head>\n</head>\n <body>\n'
  html += f'<h2> {title}</h2>\n'
  return html

def close_html(html):
  html += '</body>\n</html>'
  return html

def get_matches_html_output(sims, title):
  docs = []
  uniques = {}

  html = init_html(title)
  #html += '<!DOCTYPE html>\n<html>\n <head>\n</head>\n <body>\n'
  #html += '<h2> Sustentos Legales</h2>\n'

  for sim in sims:
    id = sim['id']

    if id in uniques:
      item = uniques[id]
    else:
      item = {
          'id': id,
          'text': sim['original'],
          'matches': []
      }

    item['matches'].append(sim)
    uniques[id] = item

  for unique in uniques.keys():
    html += '<section>\n'
    item = uniques[unique]
    text = item["text"]
    docs.append(text)
    #print(f'A: {text}\n')
    html+= f'<p> <strong>A:</strong> {text} </p>\n'

    matches = item['matches']
    matches.sort(key=score, reverse=True)
    count = 1
    html+= f'<dl>\n'
    for m in matches:
      text_sim = m['similarity']
      docs.append(text_sim)


      html+= f'<dt> <strong>B-{count}:</strong> {text_sim}</dt>\n'
      html+= f'<dd> <strong>Score:</strong> {m["score"]}</dd>\n'
      html+= f'<dd> <strong>Name:</strong> {m["desc"]} </dd>\n'
      html+= f'<dd> <strong>Page:</strong> {m["page"]} </dd>\n'
      html+= f'<dd> <strong>Sentence:</strong>{m["sentence"]} </dd>\n'

      #print("B-{}: {}\nScore: {}\nName: {}\nPage: {}\tSentence: {}\t UUID: {}\n\n ".
      #   format(count,
      #           text_sim,
      #           m['score'],
      #           m['desc'],
      #           m['page'],
      #           m['sentence'],
      #         m['uuid_sim'])
      #   )
      count += 1
    #print('----------------------------------\n')
    html+= f'</dl>\n'
    html += '</section>\n'

  html = close_html(html)
  return html

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


# Q&A


prompts = [
    {'text': 'Atribuciones y facultades del Director General mencionadas en el CAPÍTULO CUARTO SECCIÓN PRIMERA DE LA DIRECCIÓN GENERAL','_id': 1},
    {'text': 'Atribuciones y facultades del Director General','_id': 2},
    {'text': 'Presentar anualmente al Consejo Técnico el informe financiero y actuarial','_id': 3}
     

]



embeddings1= []
prompt_embeddings = model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
list_e = prompt_embeddings.tolist()
for i in list_e:
  embeddings1.append(i)
#SIM
answers = get_sims_legal(embeddings1, prompts, 0.80, 1)

html = get_matches_html_output(answers, 'Q&A')
st.markdown(html, unsafe_allow_html=True)