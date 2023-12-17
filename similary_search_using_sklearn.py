# Imports Intel® Extension for Scikit-learn
from sklearnex import patch_sklearn,config_context

# Imports for regular sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Imports for preprocessing
from haystack.nodes import PreProcessor
from haystack.utils import  convert_files_to_docs

# Fnction to pre process the datasets using Haystack
def covert_docs():

    docs = convert_files_to_docs("new_dataset/datasets", split_paragraphs=True)

    preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_respect_sentence_boundary=True,
    )

    docs = preprocessor.process(docs)

    for doc in docs:
        doc.content = doc.content.replace("\n", " ")
    return docs 


# Initiatng the fuction to preprocess the data
doc = covert_docs()

# Extracting the contents from Haystack Document Object
list_page = [x.content for x in doc]


# Function for Similarity Search
def similarity_search(query,k):

    # Enable Intel Optimization for Scikit-Learn
    patch_sklearn()
    with config_context(target_offload="gpu:0"):
        #Initiating vectorizer
        tfidf_vectorizer = TfidfVectorizer(analyzer="char")

        #Vectorising the data
        sparse_matrix = tfidf_vectorizer.fit_transform([query]+list_page)

        #Performing Cosine Similarity Search
        cosine = cosine_similarity(sparse_matrix[0,:],sparse_matrix[1:,:])

    # Converting the results into Dataframes and sorting it in descending order and considering top k values   
    df = pd.DataFrame({'cosine':cosine[0],'strings':list_page}).sort_values('cosine',ascending=False).iloc[:k]
    print(df["strings"],df["cosine"])

#Defining query
query = "What is the Secretary of States right to recover certain social security benefits."

#Inititiating the similarity search
similarity_search(k=2,query=query)