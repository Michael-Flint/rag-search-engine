import string

PUNCT_TABLE = str.maketrans("","", string.punctuation)
#1st arg: characters to replace  (not used here)
#2nd arg: characters to replace them with  (not used here)
#3rd arg: characters to delete  (this is what we are using)


def cleaner(text):
    return text.lower().translate(PUNCT_TABLE).strip()