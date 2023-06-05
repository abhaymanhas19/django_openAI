import pandas as pd
import numpy as np
import re
import openai
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import tiktoken
import fitz
import warnings
import os
import json



class UploadFiles:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens = 200
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    def __init__(self,files):
        self.files=files
        self.data=[]


    def split_into_many(self,text: str, max_tokens: int = 200) -> list[str]:
        """
        Split the given text into chunks of a maximum number of tokens.

        Args:
            text (str): The text to be split.
            max_tokens (int): The maximum number of tokens allowed in a chunk.

        Returns:
            List[str]: A list of text chunks.
        """

        # Split the text into sentences
        sentences = re.split('(?<=[.。!?।]) +', text)

        # Get the number of tokens for each sentence
        n_tokens = [len(self.tokenizer.encode(" " + sentence)) for sentence in sentences]

        # Handle sentences with tokens greater than max_tokens
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if token > max_tokens:
                extra_sentences = sentence.splitlines(keepends=True)
                extra_tokens = [len(self.tokenizer.encode(" " + extra_sentence)) for extra_sentence in extra_sentences]
                del n_tokens[i]
                del sentences[i]
                n_tokens[i:i] = extra_tokens
                sentences[i:i] = extra_sentences

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for i, (sentence, token) in enumerate(zip(sentences, n_tokens)):
            if tokens_so_far + token > max_tokens or (i == (len(n_tokens) - 1)):
                chunks.append(" ".join(chunk))
                chunk = []
                tokens_so_far = 0

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    def read_document(self, max_words):
        if self.files.name.endswith('.docx') or self.files.name.endswith('.doc'):
            text = self.files.read()
        else:
            doc = fitz.open(stream=self.files.read(), filetype="pdf")
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
        self.data.append(df)
    

    def process_uploaded_documents(self,max_documents, max_words):
        filenames = []
        texts = []

        # Read data
        for i, dat in enumerate(self.files):
            if i > (max_documents-1):
                warnings.warn('Only the first '+str(max_documents)+' documents will be analyzed.')
                break
            filenames.append(dat.name)
            text = self.read_document(dat, max_words)
            texts.append(text + '. ')

        df = pd.DataFrame(zip(filenames, texts), columns = ['filename', 'text'])

        # Calculate number of tokens
        df['n_tokens'] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))

        # Split text into chunks
        shortened = []
        filenames = []

        for i, row in df.iterrows():
            # If the text is None or empty, go to the next row
            if row['text'] is None or row['text'] == '':
                continue
            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row['n_tokens'] > self.max_tokens:
                rows = self.split_into_many(row['text'], self.max_tokens)
                rows = ['**Document ' + row['filename'] + ' :**\n\n'+ row_text for row_text in rows]
                filename_repeated = [row['filename']] * len(rows)
                filenames += filename_repeated
                shortened += rows
            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append(row['text'])
                filenames.append(row['filename'])
            
        df = pd.DataFrame(data = {'text': shortened, 'filename': filenames})

        # Calculate number of tokens
        df['n_tokens'] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))
        self.data.append(df)



    def upload_documents(self,max_documents, max_words):
    # data = st.file_uploader('Upload document', type=['docx', 'pdf', 'xls', 'xlsx', 'csv'], accept_multiple_files=True, label_visibility = 'collapsed')
    # st.caption('_The tool is GDPR and CCPA compliant, and HIPAA compliant if you sign a BAA with us. Your data is encrypted, is not used to train our AI models, and AILYZE cannot view your data. For further security, the page refreshes after 5 minutes of inactivity._')            
    
        # Check if data is uploaded and not processed yet
        if self.files :
            for dat in self.files:
                if dat.name.endswith('.xls') or dat.name.endswith('.xlsx') or dat.name.endswith('.csv'):
                    if len(self.files) != 1:
                        return warnings.warn('Only one Excel file is allowed. Remove all other files except the Excel file.')
                    else:
                        self.process_uploaded_excel(dat)
                        warnings.warn('Note: This feature to analyze spreadsheets is still in beta.')
                        return
            self.process_uploaded_documents(max_documents, max_words)

    








#      process DAta according chocies



# class SummaryProcess:
#     tokenizer = tiktoken.get_encoding("cl100k_base")



#     def api_endpoint_from_url(self,request_url):
#         """Extract the API endpoint from the request URL."""
#         return request_url.split("/")[-1]
    
    
#     def task_id_generator_function():
#         """Generate integers 0, 1, 2, and so on."""
#         task_id = 0
#         while True:
#             yield task_id
#             task_id += 1
        


    
#     async def process_api_requests_from_file(self,
#         requests_filepath: str,
#         save_filepath: str,
#         request_url: str,
#         api_key: str,
#         max_requests_per_minute: float,
#         max_tokens_per_minute: float,
#         token_encoding_name: str,
#         max_attempts: int,
#         logging_level: int,
#     ):

#         # constants
#         seconds_to_pause_after_rate_limit_error = 15
#         seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

#         # infer API endpoint and construct request header
#         api_endpoint = self.api_endpoint_from_url(request_url)
#         request_header = {"Authorization": f"Bearer {api_key}"}

#         # initialize trackers
#         queue_of_requests_to_retry = asyncio.Queue()
#         task_id_generator = self.task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
#         # status_tracker = StatusTracker()  # single instance to track a collection of variables
#         next_request = None  # variable to hold the next request to call

#         # initialize available capacity counts
#         available_request_capacity = max_requests_per_minute
#         available_token_capacity = max_tokens_per_minute
#         last_update_time = time.time()

#         # intialize flags
#         file_not_finished = True  # after file is empty, we'll skip reading it
#         logging.debug(f"Initialization complete.")

#         # initialize file reading
#         with open(requests_filepath) as file:
#             # `requests` will provide requests one at a time
#             requests = file.__iter__()
#             logging.debug(f"File opened. Entering main loop")

#             while True:
#                 # get next request (if one is not already waiting for capacity)
#                 if next_request is None:
#                     if queue_of_requests_to_retry.empty() is False:
#                         next_request = queue_of_requests_to_retry.get_nowait()
#                         logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
#                     elif file_not_finished:
#                         try:
#                             # get new request
#                             request_json = eval(next(requests))
#                             next_request = APIRequest(
#                                 task_id=next(task_id_generator),
#                                 request_json=request_json,
#                                 token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
#                                 attempts_left=max_attempts,
#                             )
#                             status_tracker.num_tasks_started += 1
#                             status_tracker.num_tasks_in_progress += 1
#                             logging.debug(f"Reading request {next_request.task_id}: {next_request}")
#                         except StopIteration:
#                             # if file runs out, set flag to stop reading it
#                             logging.debug("Read file exhausted")
#                             file_not_finished = False

#                 # update available capacity
#                 current_time = time.time()
#                 seconds_since_update = current_time - last_update_time
#                 available_request_capacity = min(
#                     available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
#                     max_requests_per_minute,
#                 )
#                 available_token_capacity = min(
#                     available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
#                     max_tokens_per_minute,
#                 )
#                 last_update_time = current_time

#                 # if enough capacity available, call API
#                 if next_request:
#                     next_request_tokens = next_request.token_consumption
#                     if (
#                         available_request_capacity >= 1
#                         and available_token_capacity >= next_request_tokens
#                     ):
#                         # update counters
#                         available_request_capacity -= 1
#                         available_token_capacity -= next_request_tokens
#                         next_request.attempts_left -= 1

#                         # call API
#                         asyncio.create_task(
#                             next_request.call_API(
#                                 request_url=request_url,
#                                 request_header=request_header,
#                                 retry_queue=queue_of_requests_to_retry,
#                                 save_filepath=save_filepath,
#                                 status_tracker=status_tracker,
#                             )
#                         )
#                         next_request = None  # reset next_request to empty

#                 # if all tasks are finished, break
#                 if status_tracker.num_tasks_in_progress == 0:
#                     break

#                 # main loop sleeps briefly so concurrent tasks can run
#                 await asyncio.sleep(seconds_to_sleep_each_loop)

#                 # if a rate limit error was hit recently, pause to cool down
#                 seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
#                 if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
#                     remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
#                     await asyncio.sleep(remaining_seconds_to_pause)
#                     # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
#                     logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

#             # after finishing, log final status
#             logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
#             if status_tracker.num_tasks_failed > 0:
#                 logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
#             if status_tracker.num_rate_limit_errors > 0:
#                 logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")




















   
   
#    def aggregate_into_few(self,df, join = ' '):
#     aggregated_text = []
#     current_text = ''
#     current_length = 0
#     max_index = len(df['n_tokens']) - 1
#     token_length = sum(df['n_tokens'])

#     if token_length > 3096:
#         max_length = round(token_length / np.ceil(token_length / 3096)) + 100
#     else:
#         max_length = 3096

#     if max_index == 0:
#         return df['text']

#     for i, row in df.iterrows():
#         current_length += row['n_tokens']

#         if current_length > max_length:
#             current_length = 0
#             aggregated_text.append(current_text)
#             current_text = ''

#         current_text = current_text + join + row['text']

#         if max_index == i:
#             aggregated_text.append(current_text)

#     return aggregated_text

#    def summarizer(self,contexts, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
#     summaries = []
#     n_tokens = []

#     if len(contexts) == 1 and summary_type == 'Essay':
#         response = openai.ChatCompletion.create(model=model,
#                                                 temperature=temperature,
#                                                 messages=[{"role": "system", "content": "You summarize text."},
#                                                           {"role": "user", "content": f"Write an extremely detailed essay summarising the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in essay form):"}]
#         )

#         summaries.append(response['choices'][0]['message']['content'].strip())
#         n_tokens.append(len(self.tokenizer.encode(response['choices'][0]['message']['content'].strip())))

#     elif len(contexts) == 1 and summary_type == 'Bullet points':
#         response = openai.ChatCompletion.create(model=model,
#                                                 temperature=temperature,
#                                                 messages=[{"role": "system", "content": "You summarize text."},
#                                                           {"role": "user", "content": f"Using bullet points, write an extremely detailed summary of the context below. {summary_instructions}\n\n###\n\nContext:\n{contexts[0]}\n\n###\n\nDetailed Summary (in bullet points):"}]
#         )

#         summaries.append(response['choices'][0]['message']['content'].strip())
#         n_tokens.append(len(self.tokenizer.encode(response['choices'][0]['message']['content'].strip())))

#     else:
#         filename = dir + "/summaries.jsonl"
#         jobs = [{"model": model, "temperature": temperature, "messages": [{"role": "system", "content": "You summarize text."}, {"role": "user", "content": f"You summarize text. Using bullet points and punctuation, summarize the text below. {summary_instructions}\n\n###\n\nText:\n{context}\n\n###\n\nSummary (in bullet points):"}]} for context in contexts]
#         with open(filename, "w") as f:
#             for job in jobs:
#                 json_string = json.dumps(job)
#                 f.write(json_string + "\n")
#         asyncio.run(process_api_requests_from_file(
#             requests_filepath=dir+'/summaries.jsonl',
#             save_filepath=dir+'/summaries_results.jsonl',
#             request_url='https://api.openai.com/v1/chat/completions',
#             api_key=os.environ.get("OPENAI_API_KEY"),
#             max_requests_per_minute=float(3_500 * 0.5),
#             max_tokens_per_minute=float(90_000 * 0.5),
#             token_encoding_name="cl100k_base",
#             max_attempts=int(5),))
    
#         with open(dir+'/summaries_results.jsonl', 'r') as f:
#             summaries_details = [json.loads(line) for line in f]
#         if os.path.exists(dir+'/summaries_results.jsonl'):
#             os.remove(dir+'/summaries_results.jsonl')
#         if os.path.exists(dir+'/summaries.jsonl'):
#             os.remove(dir+'/summaries.jsonl')
#         for i in range(len(summaries_details)):
#             summaries.append(summaries_details[i][1]['choices'][0]['message']['content'])
#             n_tokens.append(summaries_details[i][1]['usage']['completion_tokens'])

#     return pd.DataFrame(data={'text': summaries, 'n_tokens': n_tokens})
   


#    def summarize(self,df, summary_type, summary_instructions, model="gpt-3.5-turbo", temperature=0):
#     contexts = self.aggregate_into_few(df)
#     df = self.summarizer(contexts, summary_type, summary_instructions)
    
#     if len(df['text']) == 1:
#         summary = df['text'][0]
#         if summary_type == "Essay" and (summary[-1] != '.' or summary[-1] != '。' or summary[-1] != '।'):
#             summary = re.split('(?<=[.。।]) +', summary)
#             summary.pop()
#             summary = " ".join(summary)
#         return summary
#     else:
#         df = self.summarize(df, summary_type, summary_instructions)
#         return df

   


#    def call_summarize(self,data, summary_type, summary_instructions, get_individual_summaries=False, demo=False):
#     if summary_instructions == '':
#         summary_instructions = 'Include headers (e.g., introduction, conclusion) and specific details.'


#     self.summarize(data,summary_type,summary_instructions).strip()
#     st.success('Summary:\n\n' + summary)
         
    