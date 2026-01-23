import string
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):                
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):        
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty or white space")
        
        input_list = []  #The encode method expects a list of inputs

        #For now, we only have a single term but add it to the list to allow .encode to operate
        input_list.append(text)

        return self.model.encode(input_list)[0]

    


def embed_text(text):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    