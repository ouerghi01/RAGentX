import string 
from google import genai

from fastapi import WebSocket, WebSocketDisconnect
class PredictNextWord:
     def __init__(self):
          self.prompt="""
            Your role is to complete or correct the following sentence:  
            {sentence}  
            {suggestions}
            Provide the full corrected or completed sentence.
            Provide different suggestions separated by a comma.
            
            """
          self.cache={}
        #   translator = str.maketrans('', '', string.punctuation)
        #   df = pd.read_csv("Data_with_explanations.csv")
        #   if "question" not in df.columns:
        #         raise ValueError("Column 'question' not found in CSV!")
        #   questions = df["question"].dropna().tolist()
        #   answers= df["response"].dropna().tolist()
        #   if not questions:
        #         raise ValueError("No questions found in CSV!")
        #   explanations = df["explanation"].dropna().tolist()
        #   if not explanations:
        #         raise ValueError("No explanations found in CSV!")
        #   data_q = [word.translate(translator).lower() for sentence in questions for word in sentence.split() if word not in string.punctuation]
        #   data_answer = [word.translate(translator).lower() for sentence in answers for word in sentence.split() if word not in string.punctuation]
        #   data_ex = [word.translate(translator).lower() for sentence in explanations for word in sentence.split() if word not in string.punctuation]
        #   all_data = data_q + data_answer + data_ex
        #   tokens_all = (ctypes.c_char_p * len(all_data))(*(word.encode("utf-8") for word in all_data))
        #   self.treeServ= TrieService(tokens=tokens_all)
          self.llm=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
     async def predict_next_word(self,websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                if data in self.cache:
                    await websocket.send_text(f" {self.cache[data]}")
                else:
                    suggestions = []
                    completion = self.llm.send_message(self.prompt.format(sentence=data,suggestions=suggestions)).text
                    self.cache[data]=completion
                    await websocket.send_text(f" {completion}")
        except WebSocketDisconnect:
            
            print("Client disconnected")

     def retrieve_suggestions(self, data):
         ll=[]
         for i in range(0,len(data)):
             for j in range(i+1,len(data)):
                 if data[i:j] not in ll:
                     ll.append(data[i:j])
         suggestions=[]
         for term in ll :
             term=term.translate(str.maketrans('', '', string.punctuation))
             term = term.lower().split(" ")
             term = [word for word in term if word not in string.punctuation]
             if(len(term) > 1):
                  for i in range(len(term)):
                     if term[i] not in string.punctuation:
                         term[i]=term[i].translate(str.maketrans('', '', string.punctuation))
                         suggestion = self.treeServ.autocomplete(term[i])
                         suggestions.extend(suggestion)
             else :
                 term = "".join(term)
                 term = term.strip()
                 if len(term) == 0:
                     continue
                 suggestion = self.treeServ.autocomplete(term)
                 suggestions.extend(suggestion)
         suggestions = list(set(suggestions))
         return suggestions[0:20]