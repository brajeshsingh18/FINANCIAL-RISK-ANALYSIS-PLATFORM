from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=os.getenv("GROQ_API_KEY"))
# print(llm.invoke("Who is Narendra modi?").content)