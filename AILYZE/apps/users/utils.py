import pandas as pd
import numpy as np
import asyncio
import re
import openai
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import tiktoken
import fitz
import warnings
import os
import json
import docx2txt
openai.api_key = "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"
tokenizer = tiktoken.get_encoding("cl100k_base")


class FileHandler:
    max_tokens = 200
    is_excel = False
    files = []
    data = None
    data_binary = True
    data_demo = False
    data_binary_demo = False
    excel = False
    question = False
    summary = False
    themes = False
    frequency = False
    compare_viewpoints_button = False

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    def __init__(self,files):
        self.files=files

    def _split_into_many(self,text: str, max_tokens: int = 200) -> list[str]:
        sentences = re.split('(?<=[.。!?।]) +', text)
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if token > max_tokens:
                extra_sentences = sentence.splitlines(keepends=True)
                extra_tokens = [len(tokenizer.encode(" " + extra_sentence)) for extra_sentence in extra_sentences]
                del n_tokens[i]
                del sentences[i]
                n_tokens[i:i] = extra_tokens
                sentences[i:i] = extra_sentences

        chunks = []
        tokens_so_far = 0
        chunk = []
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if tokens_so_far + token > max_tokens or (i == (len(n_tokens) - 1)):
                chunks.append(" ".join(chunk))
                chunk = []
                tokens_so_far = 0

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    def _read_document(self, dat,max_words):
        if dat.name.endswith('.docx') or dat.name.endswith('.doc'):
            text = docx2txt.process(dat)
        else:
            doc = fitz.open(stream=dat.read(), filetype="pdf")
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
        text = self.emoji_pattern.sub(r'', text)
        split_text = text.split(' ')
        if len(split_text) > max_words:
            text = " ".join(split_text[:max_words])
            warnings.warn('Only the first '+str(max_words)+' words in the '+str(self.files.name)+' document will be analyzed.')
        if text == '':
            warnings.warn('Warning: The file uploaded contains no content. Please refresh the page and upload another document.')
        return text
    
        
    def process_uploaded_excel(self,dat):
        if dat.name.endswith('.xls') or dat.name.endswith('.xlsx'):
            df = pd.read_excel(dat)
        else:
            df = pd.read_csv(dat)
        self.data=df
    

    def process_uploaded_documents(self, max_documents, max_words):
        filenames = []
        texts = []
        for i, dat in enumerate(self.files):
            if i > (max_documents-1):
                print('Only the first '+str(max_documents)+' documents will be analyzed.')
                break
            filenames.append(dat.name)
            text = self._read_document(dat, max_words)
            texts.append(text + '. ')

        df = pd.DataFrame(zip(filenames, texts), columns = ['filename', 'text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        shortened = []
        filenames = []

        for i, row in df.iterrows():
            if row['text'] is None or row['text'] == '':
                continue
            if row['n_tokens'] > self.max_tokens:
                rows = self._split_into_many(row['text'], self.max_tokens)
                rows = ['**Document ' + row['filename'] + ' :**\n\n'+ row_text for row_text in rows]
                filename_repeated = [row['filename']] * len(rows)
                filenames += filename_repeated
                shortened += rows
            else:
                shortened.append(row['text'])
                filenames.append(row['filename'])
            
        df = pd.DataFrame(data = {'text': shortened, 'filename': filenames})
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        self.data = df
        self.data_binary = True


    def upload_documents(self,max_documents, max_words):
        if self.files :
            for dat in self.files:
                if dat.name.endswith('.xls') or dat.name.endswith('.xlsx') or dat.name.endswith('.csv'):
                    if len(self.files) != 1:
                        raise Exception('Only one Excel file is allowed. Remove all other files except the Excel file.')
                    else:
                        self.process_uploaded_excel(dat)
                        print('Note: This feature to analyze spreadsheets is still in beta.')
                        return
            self.process_uploaded_documents(max_documents, max_words)






class BaseChoiceHandler:
    file_handler = None
    is_excel = False
    def _aggregate_into_few(self,df, join = ' '):
        aggregated_text = []
        current_text = ''
        current_length = 0
        max_index = len(df['n_tokens']) - 1
        token_length = sum(df['n_tokens'])
        if token_length > 3096:
            max_length = round(token_length / np.ceil(token_length / 3096)) + 100
        else:
            max_length = 3096
        if max_index == 0:
            return df['text']
        for i, row in df.iterrows():
            current_length += row['n_tokens']
            if current_length > max_length:
                current_length = 0
                aggregated_text.append(current_text)
                current_text = ''

            current_text = current_text + join + row['text']
            if max_index == i:
                aggregated_text.append(current_text)
        return aggregated_text
  

class SumarrizeClass(BaseChoiceHandler):
    summary_type:str = ""
    individual_summaries:bool = False
    is_demo = False
    summary_instructions:str = ""
    summary:str = ""
    demo_summary: str = ""

    def __init__(self, user, is_demo:bool, file_handler:FileHandler, summary_insctructions:str, summary_type:str, individual_summaries:bool) -> None:
        super().__init__()
        self.user = user
        self.is_demo = is_demo
        self.file_handler = file_handler
        self.is_excel = file_handler.is_excel
        self.summary_type = summary_type
        self.summary_instructions = summary_insctructions
        self.individual_summaries = individual_summaries

  

    def _summarizer(self, contexts, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
        summaries = []
        n_tokens = []

        if len(contexts) == 1 and summary_type == 'Essay':
            response = openai.ChatCompletion.create(model=model,
                                                    temperature=temperature,
                                                    messages=[{"role": "system", "content": "You summarize text."},
                                                            {"role": "user", "content": f"Write an extremely detailed essay summarising the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in essay form):"}]
            )

            summaries.append(response['choices'][0]['message']['content'].strip())
            n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

        elif len(contexts) == 1 and summary_type == 'Bullet points':
            response = openai.ChatCompletion.create(model=model,
                                                    temperature=temperature,
                                                    messages=[{"role": "system", "content": "You summarize text."},
                                                            {"role": "user", "content": f"Using bullet points, write an extremely detailed summary of the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in bullet points):"}]
            )

            summaries.append(response['choices'][0]['message']['content'].strip())
            n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

        else:
            pass
            # filename = dir + "/summaries.jsonl"
            # jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You summarize text."}, {"role": "user", "content": f"You summarize text. Using bullet points and punctuation, summarize the text below. {summary_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nSummary (in bullet points):"}]} for context in contexts]
            # with open(filename, "w") as f:
            #     for job in jobs:
            #         json_string = json.dumps(job)
            #         f.write(json_string + "\n")
            # asyncio.run(process_api_requests_from_file(
            #     requests_filepath=dir+'/summaries.jsonl',
            #     save_filepath=dir+'/summaries_results.jsonl',
            #     request_url='https://api.openai.com/v1/chat/completions',
            #     api_key=os.environ.get("OPENAI_API_KEY"),
            #     max_requests_per_minute=float(3_500 * 0.5),
            #     max_tokens_per_minute=float(90_000 * 0.5),
            #     token_encoding_name="cl100k_base",
            #     max_attempts=int(5),
            #     logging_level=int(logging.INFO)))
            # with open(dir+'/summaries_results.jsonl', 'r') as f:
            #     summaries_details = [json.loads(line) for line in f]
            # if os.path.exists(dir+'/summaries_results.jsonl'):
            #     os.remove(dir+'/summaries_results.jsonl')
            # if os.path.exists(dir+'/summaries.jsonl'):
            #     os.remove(dir+'/summaries.jsonl')
            # for i in range(len(summaries_details)):
            #     summaries.append(summaries_details[i][1]['choices'][0]['message']['content'])
            #     n_tokens.append(summaries_details[i][1]['usage']['completion_tokens'])

        return pd.DataFrame(data={'text': summaries, 'n_tokens': n_tokens})
    

    def summarize(self,df, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
        contexts = self._aggregate_into_few(df)
        df = self._summarizer(contexts, summary_type, summary_instructions)
        if len(df['text']) == 1:
            summary = df['text'][0]
            if summary_type == "Essay" and (summary[-1] != '.' or summary[-1] != '。' or summary[-1] != '।'):
                summary = re.split('(?<=[.。।]) +', summary)
                summary.pop()
                summary = " ".join(summary)
            return summary
        else:
            df = self.summarize(df, summary_type, summary_instructions)
            return df
    
    def call_summarize(self, data, demo=False):
        if summary_instructions == '':
            summary_instructions = 'Include headers (e.g., introduction, conclusion) and specific details.'
        if not self.individual_summaries:
            if demo:
                self.data_demo['summary_type'] = self.summary_type
                self.data_demo['summary_instructions'] = self.summary_instructions
                self.data_demo['summary'] = self.summarize(
                    data,
                    summary_type=str(self.data_demo['summary_type'][0]),
                    summary_instructions=str(self.data_demo['summary_instructions'][0])
                ).strip()   
            else:
                self.data['summary_type'] = self.summary_type
                self.data['summary_instructions'] = self.summary_instructions
                self.data['summary'] = self.summarize(
                    self.data,
                    summary_type=self.summary_type,
                    summary_instructions=self.summary_instructions
                ).strip()
        if demo:
            summary = self.data_demo['summary']
        else:
            summary = self.data['summary']
        # if self.user:
        return ('summary_type' + str(self.summary_type)    + 
                    '\n\n Instructions' + str(self.summary_instructions),summary)
        # else:
        #     new_row = {
        #         'Question': 'Summary Type: ' + str(self.summary_type) +
        #                     '\n\nInstructions: ' + str(self.summary_instructions),
        #         'Answer': summary
            # }
            # st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], pd.DataFrame(new_row, index=[0])], ignore_index=True)



class QuestionClass(BaseChoiceHandler):
    pass

class ThemeAnalysisClass(BaseChoiceHandler):
    pass

class FrequencyHandlerClass(BaseChoiceHandler):
    pass

class CompareViewPointsClass(BaseChoiceHandler):
    pass

# #      process DAta according chocies
# class SummaryProcess:

#     def __init__(self, file_handler:FileHandler) -> None:
#         self.file_handler = file_handler

#     def api_endpoint_from_url(self,request_url):
#         """Extract the API endpoint from the request URL."""
#         return request_url.split("/")[-1]
    
    
#     def task_id_generator_function():
#         """Generate integers 0, 1, 2, and so on."""
#         task_id = 0
#         while True:
#             yield task_id
#             task_id += 1
                                        
#     def aggregate_into_few(self,df, join = ' '):
#             aggregated_text = []
#             current_text = ''
#             current_length = 0
#             max_index = len(df['n_tokens']) - 1
#             token_length = sum(df['n_tokens'])

#             if token_length > 3096:
#                 max_length = round(token_length / np.ceil(token_length / 3096)) + 100
#             else:
#                 max_length = 3096

#             if max_index == 0:
#                 return df['text']

#             for i, row in df.iterrows():
#                 current_length += row['n_tokens']

#                 if current_length > max_length:
#                     current_length = 0
#                     aggregated_text.append(current_text)
#                     current_text = ''

#                 current_text = current_text + join + row['text']

#                 if max_index == i:
#                     aggregated_text.append(current_text)

#             return aggregated_text

#     def summarizer(contexts, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
#         summaries = []
#         n_tokens = []

#         if len(contexts) == 1 and summary_type == 'Essay':
#             response = openai.ChatCompletion.create(model=model,
#                                                     temperature=temperature,
#                                                     messages=[{"role": "system", "content": "You summarize text."},
#                                                             {"role": "user", "content": f"Write an extremely detailed essay summarising the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in essay form):"}]
#             )

#             summaries.append(response['choices'][0]['message']['content'].strip())
#             n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

#         elif len(contexts) == 1 and summary_type == 'Bullet points':
#             response = openai.ChatCompletion.create(model=model,
#                                                     temperature=temperature,
#                                                     messages=[{"role": "system", "content": "You summarize text."},
#                                                             {"role": "user", "content": f"Using bullet points, write an extremely detailed summary of the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in bullet points):"}]
#             )

#             summaries.append(response['choices'][0]['message']['content'].strip())
#             n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

#         else:
#             filename = dir + "/summaries.jsonl"
#             jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You summarize text."}, {"role": "user", "content": f"You summarize text. Using bullet points and punctuation, summarize the text below. {summary_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nSummary (in bullet points):"}]} for context in contexts]
#             with open(filename, "w") as f:
#                 for job in jobs:
#                     json_string = json.dumps(job)
#                     f.write(json_string + "\n")
#             asyncio.run(process_api_requests_from_file(
#                 requests_filepath=dir+'/summaries.jsonl',
#                 save_filepath=dir+'/summaries_results.jsonl',
#                 request_url='https://api.openai.com/v1/chat/completions',
#                 api_key=os.environ.get("OPENAI_API_KEY"),
#                 max_requests_per_minute=float(3_500 * 0.5),
#                 max_tokens_per_minute=float(90_000 * 0.5),
#                 token_encoding_name="cl100k_base",
#                 max_attempts=int(5),
#                 logging_level=int(logging.INFO)))
#             with open(dir+'/summaries_results.jsonl', 'r') as f:
#                 summaries_details = [json.loads(line) for line in f]
#             if os.path.exists(dir+'/summaries_results.jsonl'):
#                 os.remove(dir+'/summaries_results.jsonl')
#             if os.path.exists(dir+'/summaries.jsonl'):
#                 os.remove(dir+'/summaries.jsonl')
#             for i in range(len(summaries_details)):
#                 summaries.append(summaries_details[i][1]['choices'][0]['message']['content'])
#                 n_tokens.append(summaries_details[i][1]['usage']['completion_tokens'])

#         return pd.DataFrame(data={'text': summaries, 'n_tokens': n_tokens})


#     def summarize(self,df, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
#         contexts = self.aggregate_into_few(df)
#         df = self.summarizer(contexts, summary_type, summary_instructions)
        
#         if len(df['text']) == 1:
#             summary = df['text'][0]
#             if summary_type == "Essay" and (summary[-1] != '.' or summary[-1] != '。' or summary[-1] != '।'):
#                 summary = re.split('(?<=[.。।]) +', summary)
#                 summary.pop()
#                 summary = " ".join(summary)
#             return summary
#         else:
#             df = self.summarize(df, summary_type, summary_instructions)
#             return df

    


#     def call_summarize(self,data, summary_type, summary_instructions, get_individual_summaries=False, demo=False):
#         if summary_instructions == '':
#             summary_instructions = 'Include headers (e.g., introduction, conclusion) and specific details.'


#         summary=self.summarize(data,summary_type,summary_instructions).strip()
#         return summary
   




# class SPecificQuestion:

#     def longest_str_intersection(self,a: str, b: str) -> str:
#         """
#         Find the longest common substring between two strings.

#         Args:
#             a (str): The first string.
#             b (str): The second string.

#         Returns:
#             str: The longest common substring.
#         """

#         # Identify all possible character sequences from str a
#         seqs = [a[pos1:pos2 + 1] for pos1 in range(len(a)) for pos2 in range(len(a))]

#         # Remove empty sequences
#         seqs = [seq for seq in seqs if seq != '']

#         # Find segments in str b
#         max_len_match = 0
#         max_match_sequence = ''
#         for seq in seqs:
#             if seq in b and len(seq) > max_len_match:
#                 max_len_match = len(seq)
#                 max_match_sequence = seq


#         return max_match_sequence

#     def create_context(self,keywords, df, max_len=1500, size="ada"):
#         """
#         Create a context for a question by finding the most similar context from the dataframe.
#         """
#         # Get the embeddings for the question
#         q_embeddings = openai.Embedding.create(input=keywords, engine='text-embedding-ada-002')['data'][0]['embedding']

#         # Get the distances from the embeddings
#         df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

#         returns = []
#         cur_len = 0

#         # Sort by distance and add the text to the context until the context is too long
#         for i, row in df.sort_values('distances', ascending=True).iterrows():
            
#             # Add the length of the text to the current length
#             cur_len += row['n_tokens'] + 4
            
#             # If the context is too long, break
#             if cur_len > max_len:
#                 break
            
#             # Else add it to the text that is being returned
#             returns.append(row["text"])

#         # Return the context
#         return "\n\n===\n\n".join(returns)

#     def answer_question(self,df, question = "Does the participant trust doctors and why?", keywords = "doctors, patients, trust, confident, believe", get_quotes = False, model="gpt-3.5-turbo", max_len=3097, size="ada", debug=False, temperature=0):
#         """
#         Answer a question based on the most similar context from the dataframe texts.
#         """
#         context = self.create_context(keywords, df, max_len=max_len, size=size)

#         all_context = ''
#         for line in context.splitlines():
#             all_context = all_context + str(line) + '\n'
        
#         # If debug, print the raw model response
#         if debug:
#             print("Context:\n" + context)
#             print("\n\n")

#         try:
#             # Create a completions using the question and context
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 temperature=temperature,
#                 messages=[
#                     {"role": "user", "content": f"Write an essay to answer the question below, based on a plausible inference of the document extracts below. \n\n###\n\nDocument extracts:\n{context}\n\n###\n\nQuestion:\n{question}\n\n###\n\nEssay:"}]
#             )
            
#             response = response['choices'][0]['message']['content'].strip()
            
#             if not get_quotes:
#                 quotes = 'x'
#                 all_context = 'xy'
            
#             else: 
#                 raw_quotes = openai.ChatCompletion.create(
#                     model=model,
#                     temperature=temperature,
#                     messages=[
#                         {"role": "user", "content": f"You are a quote extractor. Only take quotes from the document extracts below. For each quote, use in-text citations to cite the document source. In bullet points, substantiate the argument below by providing quotes.\n\n###\n\nArgument (don't take quotes from here):\n{response}\n\n###\n\nDocument extracts (take quotes from here):\n{context}\n\n###\n\nQuotes in bullet points (with document source cited for each quote):"}]
#                 )

#                 raw_quotes = raw_quotes['choices'][0]['message']['content'].strip()
#                 raw_quotes = raw_quotes.replace('$', '\$')
#                 all_context = all_context.replace('$', '\$').replace('[', '\[')

#                 quotes = ''

#                 for quote in raw_quotes.split("\n"):
#                     if len(quote) > 50:
#                         raw_quote = self.longest_str_intersection(quote, all_context)
#                         quotes = quotes + '* "' + raw_quote + '"\n'
#                         all_context = all_context.replace(raw_quote, ':green['+raw_quote+']')
                
#             return [response, quotes, all_context]

#         except Exception as e:
#             print(e)
#             return ""

            


# class ThemanticAnalysis:
#     def aggregate_into_few(self,df, join = ' '):
#                 aggregated_text = []
#                 current_text = ''
#                 current_length = 0
#                 max_index = len(df['n_tokens']) - 1
#                 token_length = sum(df['n_tokens'])

#                 if token_length > 3096:
#                     max_length = round(token_length / np.ceil(token_length / 3096)) + 100
#                 else:
#                     max_length = 3096

#                 if max_index == 0:
#                     return df['text']

#                 for i, row in df.iterrows():
#                     current_length += row['n_tokens']

#                     if current_length > max_length:
#                         current_length = 0
#                         aggregated_text.append(current_text)
#                         current_text = ''

#                     current_text = current_text + join + row['text']

#                     if max_index == i:
#                         aggregated_text.append(current_text)

#                 return aggregated_text
    


#     def thematic_analyzer(self,contexts, theme_type, theme_instructions, model="gpt-3.5-turbo", temperature=0):
#         themes = []
#         n_tokens = []

#         if len(contexts) == 1 and theme_type == 'Essay':
#             response = openai.ChatCompletion.create(model=model,
#                                                     temperature=temperature,
#                                                     messages=[{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Write an extremely detailed essay identifying the top 10 themes, based on the context below. For each theme, specify which file(s) contained that theme and include as many quotes and examples as possible. {theme_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nEssay (excluding introduction and conclusion):"}]
#             )

#             themes.append(response['choices'][0]['message']['content'].strip())
#             n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))

#         elif len(contexts) == 1 and theme_type == 'Codebook':
#             response = openai.ChatCompletion.create(model=model,
#                                                     temperature=temperature,
#                                                     messages=[{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Based on the context below, create a codebook in the form of a table with three columns, namely (a) theme; (b) detailed description with as many quotes and examples as possible (with document source cited); and (c) file which contains that theme. {theme_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nCodebook (in table format):"}]
#             )

#             themes.append(response['choices'][0]['message']['content'].strip())
#             n_tokens.append(len(tokenizer.encode(response['choices'][0]['message']['content'].strip())))
            
#         # else:
#         #     filename = dir + "/themes.jsonl"
#         #     jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"You conduct thematic analysis. Write an extremely detailed essay listing the top themes based on the context below. For each theme, identify quotes and examples (with document source cited), as well as which file(s) contained that theme. {theme_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nDetailed List of Themes (in essay form):"}]} for context in contexts]
#         #     with open(filename, "w") as f:
#         #         for job in jobs:
#         #             json_string = json.dumps(job)
#         #             f.write(json_string + "\n")
#         #     asyncio.run(process_api_requests_from_file(
#         #         requests_filepath=dir+'/themes.jsonl',
#         #         save_filepath=dir+'/themes_results.jsonl',
#         #         request_url='https://api.openai.com/v1/chat/completions',
#         #         api_key=os.environ.get("OPENAI_API_KEY"),
#         #         max_requests_per_minute=float(3_500 * 0.5),
#         #         max_tokens_per_minute=float(90_000 * 0.5),
#         #         token_encoding_name="cl100k_base",
#         #         max_attempts=int(5),
#         #         logging_level=int(logging.INFO)))
#         #     with open(dir+'/themes_results.jsonl', 'r') as f:
#         #         themes_details = [json.loads(line) for line in f]
#         #     if os.path.exists(dir+'/themes_results.jsonl'):
#         #         os.remove(dir+'/themes_results.jsonl')
#         #     if os.path.exists(dir+'/themes.jsonl'):
#         #         os.remove(dir+'/themes.jsonl')
#         #     for i in range(len(themes_details)):
#         #         themes.append(themes_details[i][1]['choices'][0]['message']['content'])
#         #         n_tokens.append(themes_details[i][1]['usage']['completion_tokens'])

#         # return pd.DataFrame(data={'text': themes, 'n_tokens': n_tokens})


#     def thematic_analysis(self,df, theme_type, theme_instructions, model="gpt-3.5-turbo", temperature=0):
#         contexts = self.aggregate_into_few(df)
#         df = self.thematic_analyzer(contexts, theme_type, theme_instructions)
        
#         if len(df['text']) == 1:
#             themes = df['text'][0]
#     #        if (themes[-1] != '.' or themes[-1] != '。' or themes[-1] != '।'):
#     #            themes = re.split('(?<=[.。।]) +', themes)
#     #            themes.pop()
#     #            themes = " ".join(themes)
#             return themes
#         else:
#             df = self.thematic_analysis(df, theme_type, theme_instructions)
#             return df  









            
# class IdentifyViewpoints:
#     def aggregate_into_few(self,df, join = ' '):
#         aggregated_text = []
#         current_text = ''
#         current_length = 0
#         max_index = len(df['n_tokens']) - 1
#         token_length = sum(df['n_tokens'])

#         if token_length > 3096:
#             max_length = round(token_length / np.ceil(token_length / 3096)) + 100
#         else:
#             max_length = 3096

#         if max_index == 0:
#             return df['text']

#         for i, row in df.iterrows():
#             current_length += row['n_tokens']

#             if current_length > max_length:
#                 current_length = 0
#                 aggregated_text.append(current_text)
#                 current_text = ''

#             current_text = current_text + join + row['text']

#             if max_index == i:
#                 aggregated_text.append(current_text)

#         return aggregated_text
    

#     def get_frequency(self,context, frequency_viewpoint, model="gpt-3.5-turbo", temperature=0):
#         response = openai.ChatCompletion.create(model=model,
#                                                 temperature=temperature,
#                                                 messages=[{"role": "system", "content": "You are a qualitative researcher."},
#                                                         {"role": "user", "content": f"Based on the text below, answer in the form of a table with 3 columns, namely (a) Document Name; (b) Contains Viewpoint? - Yes/No/Maybe; and (c) Detailed Description. Focus on whether it can be inferred that the document contains the following viewpoint: {frequency_viewpoint}.\n\n###\n\nText:\n{context}\n\n###\n\nTable:"}]
#                                             )
#         response = response['choices'][0]['message']['content'].strip()
#         return response
    

     
#     def create_context(self,keywords, df, max_len=1500, size="ada"):
#         """
#         Create a context for a question by finding the most similar context from the dataframe.
#         """
#         # Get the embeddings for the question
#         q_embeddings = openai.Embedding.create(input=keywords, engine='text-embedding-ada-002')['data'][0]['embedding']

#         # Get the distances from the embeddings
#         df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

#         returns = []
#         cur_len = 0

#         # Sort by distance and add the text to the context until the context is too long
#         for i, row in df.sort_values('distances', ascending=True).iterrows():
            
#             # Add the length of the text to the current length
#             cur_len += row['n_tokens'] + 4
            
#             # If the context is too long, break
#             if cur_len > max_len:
#                 break
            
#             # Else add it to the text that is being returned
#             returns.append(row["text"])

#         # Return the context
#         return "\n\n===\n\n".join(returns)
    
#     def frequency_analyzer(self,df, frequency_viewpoint, model="gpt-3.5-turbo", temperature=0):
#         filenames = df['filename'].unique()
#         jobs = []
#         frequencies = []
#         n_tokens = []
        
#         for filename in filenames:
#             question_frequency_viewpoint = "Based on the context below, write a short essay explaining using examples if the document " + filename + " contains the following viewpoint: " + frequency_viewpoint + "."
#             context = self.create_context(frequency_viewpoint, df.loc[df['filename'] == filename], max_len=3097, size="ada")
#             jobs += [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You are a qualitative researcher."}, {"role": "user", "content": f"{question_frequency_viewpoint}\n\n###\n\nContext:\n{context}\n\n###\n\nShort Essay:"}]}]
            
#         filename = dir + "/frequencies.jsonl"
#         with open(filename, "w") as f:
#             for job in jobs:
#                 json_string = json.dumps(job)
#                 f.write(json_string + "\n")
#         asyncio.run(process_api_requests_from_file(
#             requests_filepath=dir+'/frequencies.jsonl',
#             save_filepath=dir+'/frequencies_results.jsonl',
#             request_url='https://api.openai.com/v1/chat/completions',
#             api_key=os.environ.get("OPENAI_API_KEY"),
#             max_requests_per_minute=float(3_500 * 0.5),
#             max_tokens_per_minute=float(90_000 * 0.5),
#             token_encoding_name="cl100k_base",
#             max_attempts=int(5),
#             logging_level=int(logging.INFO)))
#         with open(dir+'/frequencies_results.jsonl', 'r') as f:
#             frequencies_details = [json.loads(line) for line in f]
#         if os.path.exists(dir+'/frequencies_results.jsonl'):
#             os.remove(dir+'/frequencies_results.jsonl')
#         if os.path.exists(dir+'/frequencies.jsonl'):
#             os.remove(dir+'/frequencies.jsonl')
#         for i in range(len(frequencies_details)):
#             frequencies.append(frequencies_details[i][1]['choices'][0]['message']['content'])
#             n_tokens.append(frequencies_details[i][1]['usage']['completion_tokens'])
        
#         return pd.DataFrame(data={'text': frequencies, 'n_tokens': n_tokens})

        
#     def frequency_analysis(self,df, frequency_viewpoint, model="gpt-3.5-turbo", temperature=0):
#         contexts = self.aggregate_into_few(df, join = '\n\n')

#         if len(contexts) == 1:
#             frequency = self.get_frequency(contexts[0], frequency_viewpoint, model, temperature)
#             return frequency

#         else:
#             df = SummaryProcess.summarizer(contexts, summary_type = 'Essay', summary_instructions = 'Focus on whether the documents contained the following viewpoint: ' + frequency_viewpoint + '.')
#             if len(df['text']) == 1:
#                 frequency = self.get_frequency(df['text'][0], frequency_viewpoint, model, temperature)
#                 return frequency
#             else:
#                 df = self.frequency_analysis(df, theme_type, theme_instructions)
#                 return df




# class CompareViewdetails:
#     def longest_str_intersection(self,a: str, b: str) -> str:
#         """
#         Find the longest common substring between two strings.

#         Args:
#             a (str): The first string.
#             b (str): The second string.

#         Returns:
#             str: The longest common substring.
#         """

#         # Identify all possible character sequences from str a
#         seqs = [a[pos1:pos2 + 1] for pos1 in range(len(a)) for pos2 in range(len(a))]

#         # Remove empty sequences
#         seqs = [seq for seq in seqs if seq != '']

#         # Find segments in str b
#         max_len_match = 0
#         max_match_sequence = ''
#         for seq in seqs:
#             if seq in b and len(seq) > max_len_match:
#                 max_len_match = len(seq)
#                 max_match_sequence = seq


#         return max_match_sequence
    
#     def create_context(self,keywords, df, max_len=1500, size="ada"):
#         """
#         Create a context for a question by finding the most similar context from the dataframe.
#         """
#         # Get the embeddings for the question
#         q_embeddings = openai.Embedding.create(input=keywords, engine='text-embedding-ada-002')['data'][0]['embedding']

#         # Get the distances from the embeddings
#         df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

#         returns = []
#         cur_len = 0

#         # Sort by distance and add the text to the context until the context is too long
#         for i, row in df.sort_values('distances', ascending=True).iterrows():
            
#             # Add the length of the text to the current length
#             cur_len += row['n_tokens'] + 4
            
#             # If the context is too long, break
#             if cur_len > max_len:
#                 break
            
#             # Else add it to the text that is being returned
#             returns.append(row["text"])

#         # Return the context
#         return "\n\n===\n\n".join(returns)
    

#     def answer_question(self,df, question = "Does the participant trust doctors and why?", keywords = "doctors, patients, trust, confident, believe", get_quotes = False, model="gpt-3.5-turbo", max_len=3097, size="ada", debug=False, temperature=0):
#         """
#         Answer a question based on the most similar context from the dataframe texts.
#         """
#         context = self.create_context(keywords, df, max_len=max_len, size=size)

#         all_context = ''
#         for line in context.splitlines():
#             all_context = all_context + str(line) + '\n'
        
#         # If debug, print the raw model response
#         if debug:
#             print("Context:\n" + context)
#             print("\n\n")

#         try:
#             # Create a completions using the question and context
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 temperature=temperature,
#                 messages=[
#                     {"role": "user", "content": f"Write an essay to answer the question below, based on a plausible inference of the document extracts below. \n\n###\n\nDocument extracts:\n{context}\n\n###\n\nQuestion:\n{question}\n\n###\n\nEssay:"}]
#             )
            
#             response = response['choices'][0]['message']['content'].strip()
            
#             if not get_quotes:
#                 quotes = 'x'
#                 all_context = 'xy'
            
#             else: 
#                 raw_quotes = openai.ChatCompletion.create(
#                     model=model,
#                     temperature=temperature,
#                     messages=[
#                         {"role": "user", "content": f"You are a quote extractor. Only take quotes from the document extracts below. For each quote, use in-text citations to cite the document source. In bullet points, substantiate the argument below by providing quotes.\n\n###\n\nArgument (don't take quotes from here):\n{response}\n\n###\n\nDocument extracts (take quotes from here):\n{context}\n\n###\n\nQuotes in bullet points (with document source cited for each quote):"}]
#                 )

#                 raw_quotes = raw_quotes['choices'][0]['message']['content'].strip()
#                 raw_quotes = raw_quotes.replace('$', '\$')
#                 all_context = all_context.replace('$', '\$').replace('[', '\[')

#                 quotes = ''

#                 for quote in raw_quotes.split("\n"):
#                     if len(quote) > 50:
#                         raw_quote = self.longest_str_intersection(quote, all_context)
#                         quotes = quotes + '* "' + raw_quote + '"\n'
#                         all_context = all_context.replace(raw_quote, ':green['+raw_quote+']')
                
#             return [response, quotes, all_context]

#         except Exception as e:
#             print(e)
#             return ""

