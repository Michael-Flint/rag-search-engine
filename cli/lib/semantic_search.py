from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):                
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def msf_encode(self, text):
        #Placeholder for me to remember how the lesson said to call the encode method when we get to that part of the coursework
        return self.model.encode(text)
    


def verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    