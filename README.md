VectoGPT: AI-Powered Semantic Search & Chatbot  

VectoGPT is an advanced AI-powered query-answering system that integrates a Vector Database (FAISS) with ChatGPT or DeepSeek API for efficient information retrieval and response generation. This project enhances chatbot capabilities using **semantic search**, ensuring more contextually relevant responses based on stored knowledge.  

Features 

✅ **Semantic Search with FAISS** – Fast and scalable retrieval of similar text data.  
✅ **Text Embedding Generation** – Converts text into vector embeddings using **BERT**.  
✅ **AI-Powered Responses** – Uses **GPT-2/GPT-4** to generate human-like answers.  
✅ **Optional DeepSeek API** – Enhances vector search accuracy.  
✅ **Customizable & Scalable** – Easily adaptable for various applications.  

Installation

Clone the repository and install dependencies:  

bash
git clone https://github.com/yourusername/VectoGPT.git  
cd VectoGPT  
pip install -r requirements.txt  
pip install faiss-cpu transformers  


Install DeepSeek API)  

bash
pip install deepseek  

*Usage 

1. Generate Text Embeddings

python
from transformers import BertTokenizer, BertModel  
import torch  

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
model = BertModel.from_pretrained('bert-base-uncased')  

text = "How does AI work?"  
inputs = tokenizer(text, return_tensors='pt')  
with torch.no_grad():  
    outputs = model(**inputs)  

embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  

2. Store and Search with FAISS

python
import faiss  
import numpy as np  

dimension = 512  
index = faiss.IndexFlatL2(dimension)  
embeddings = np.array([embedding], dtype='float32')  
index.add(embeddings)  

query_embedding = generate_embedding("Explain AI")  
D, I = index.search(np.array([query_embedding], dtype='float32'), k=3) 

3. Generate Responses with ChatGPT 

python
from transformers import GPT2LMHeadModel, GPT2Tokenizer  

model = GPT2LMHeadModel.from_pretrained('gpt2')  
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  

input_ids = tokenizer.encode("Explain AI", return_tensors='pt')  
output = model.generate(input_ids, max_length=100)  
print(tokenizer.decode(output[0], skip_special_tokens=True))  

License 

This project is licensed under the MIT License.


