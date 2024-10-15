

from pathlib import Path
import os, sys, time


# path_root = Path(__file__).parents[2]
# sys.path.append(str(path_root))
# print(sys.path)
sys.path.append('.')


from include.get_eurlex_id import  get_lex_id



import AI_Eurlex.sqlitedb as sql
import AI_Eurlex.utils    as utils

# from   pytools.ollamarag import OllamaRAG

"""# Lavoriamo col dataset 'multi_eurlex' IT

Il dataset è contenuto nel database sqlite legalAI.db
che contiene 4 tabelle: \
1. _TEST_ contenente 5000 records
1. _VALIDATION_ anche questa contiene 5000 records
1. _TRAIN_ contenente 55000 records
1. _LABELS_ contenente dati accessori per decodificare la lista _labels_ contenuta in ogni tabella

la struttura dati delle tabelle test,train e validation è la seguente: \
celex_id, text, labels

la struttura della tabella labels è: \
euid, label
"""

# per caricare db eurlex
# scaricare file da 'https://legalai.commonweb.net/shared_datas/legalsqlite.zip'
# ed eseguire unzip per ottenere il file 'legalAI.db'

db_file='legalAI.db'

if not os.path.exists(db_file):
  if not os.path.exists('legalsqlite.zip'):
    utils.download_url('https://legalai.commonweb.net/shared_datas/legalsqlite.zip')
    utils.unzip('legalsqlite.zip')

# effettuare connesion al db (sqlite file)

legalai = sql.SqliteDB(db_file)

# otteniamo dati testing
df_test=legalai.table_as_pd('test')

# df_test=legalai.table_as_pd('train')

df_test.info()

print(df_test.head())

# for row in df_test[:10]:
#   print(row.text[0])

# exit(0)

# esempio utilizzo dati

def pd2data(df):
  for row in range(len(df)):
    x=(df.at[row, 'celex_id'], df.at[row, 'text'][:20].replace('\n',' '), df.at[row, 'labels'])
    print(x)

pd2data(df_test[:10])



# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


# from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
# ef = OllamaEmbeddingFunction(    model_name="nomic-embed-text",    url="http://localhost:11434/api/embeddings", )

model_kwargs = {'device': 'cuda','trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': True, 'clean_up_tokenization_spaces': True, 'weights_only': True}

hf = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5',
    # model_name='all-MiniLM-L6-v1',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print(hf.model_name)

"""## from langchain_core.documents import Document

document = Document(
    page_content="Hello, world!",
    metadata={"source": "https://example.com"}
)
"""

def df_langchain_doc(text, metadati):
  return Document(page_content=text,metadata=metadati)

def df_to_langchain_doc(df):
  docs=[]

  for record in df_test.itertuples():
    # print(record.celex_id)
    docs.append(Document(id=record.celex_id,page_content=record.text,metadata={'labels':record.labels,'source': record.celex_id}))

  return docs

# testiamo
# df_to_langchain_doc(df_test[:1])

# document split
text_splitter=RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
# With splitter

# #########################
# trova stringa in formto xxxxx/yyyy nel testo
# #########################
def find_regolamento(text):
  return get_lex_id(text)


# #########################
# Crea il documento e fa lo split
# #########################
def split_record(record):
  num_doc=find_regolamento(record.text)
  rec_doc=Document(
    id=record.celex_id,
    page_content=record.text,
    metadata={
      'labels':record.labels,
      'source': record.celex_id,
      'number': num_doc
    }
  )
  return text_splitter.split_documents([rec_doc])

# emfunc=OllamaEmbeddings(model="nomic-embed-text")

emfunc=hf # use hugging face all-MiniLM-L6-v1

persist_dir=emfunc.model_name.replace('/','-')

if not os.path.exists(persist_dir):
  # non esiste archivio chroma si deve generare
  # creare un client chroma persistente (se si clicca su icona cartella in radice oltre a sample_data compare legalai)
  pass
else:
  pass
# esiste archivio chroma si deve caricare
chromadb_client=Chroma(collection_name='legalai',
                      embedding_function=emfunc,
                      persist_directory=persist_dir,
                      collection_metadata={"hnsw:space": "cosine"})

  # chromadb_client.add_documents(documents=[to_langchain_doc(i) for i in documents])

x=len(df_test)
i=0

# generiamo db
ttime=time.perf_counter()

for record in df_test.itertuples():
  tm_on=time.perf_counter()
  docs=split_record(record)
  print(len(docs),'documenti...',end='\r')
  chromadb_client.add_documents(documents=docs)

  if(i<1000):
    print(i,x,record.celex_id,f'{len(docs):3d} {time.perf_counter()-tm_on:7.2f} {len(docs):3d}',end='\n')
  else:
    break

  i=i+1

  # break

print(f'\ntempo totale: {time.perf_counter()-ttime:.2f}')
