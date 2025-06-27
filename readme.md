<h1>Retrieval Augmented Generation (RAG) with Streamlit, LangChain and Pinecone DB</h1>

<h2>Can Be Utilized For Domain Specific Knowledge... In This Case, EPCI Oil & Gas</h2>


<h2>Prerequisites</h2>
<ul>
  <li>Python 3.13</li>
</ul>

<h2>Installation</h2>

1. Clone the repository:




2. Create a virtual environment



```
python -m venv venv

```

3. Activate the virtual environment


```
venv\Scripts\Activate

```

(or on Mac):

```
source venv/bin/activate

```

4. Install libraries


```
pip install -r requirements.txt

```

5. Create accounts

- Create a free account on Pinecone:

```
  https://www.pinecone.io/

```
- Create an API key for OpenAI:

```
https://platform.openai.com/api-keys

```
6. Add API keys to .env file


- Add the API keys for Pinecone and OpenAI to the .env file

Executing the scripts

1. Open a terminal in VS Code

2. Execute the following command:

```
   python ingestion.py

```

```
   python retrieval.py

```
   
```
   python -m streamlit run chatbot_rag.py 
```
