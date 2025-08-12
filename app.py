from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import streamlit as st
import asyncio

import os
from dotenv import load_dotenv
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Lấy id video từ url
def extract_video_id(url):
    # url có dạng https://www.youtube.com/watch?v=...
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    # url có dạng https://youtu.be/...?si=...
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return url.rstrip('/').split('/')[-1]

def get_transcript(video_id):
    try:
        # Lấy transcript với id video
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['vi', 'en', 'en-US', 'en-GB'])
        transcript = [snippet.text for snippet in fetched_transcript]
        transcript = ' '.join(transcript)
        transcript_list = [{'text': snippet.text, 'start': snippet.start, 'duration': snippet.duration} for snippet in fetched_transcript]
        return transcript, transcript_list
    except TranscriptsDisabled:
        raise Exception(f'Không thể trích xuất/Không có transcript cho video này.')
    except Exception as e:
        return e
    
def get_chunks(transcript, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript])
    return chunks

def embeddings_db(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding', google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def main():
    st.set_page_config(page_title='YouTube Q&A', layout='centered')
    st.title('📺 YouTube Transcript Q&A')

    url = st.text_input('Nhập URL video YouTube:', key='video_url')
    if st.button('Lấy transcript', key='get_transcript'):
        if not url:
            st.warning('Vui lòng nhập URL video.')
        else:
            try:
                with st.spinner('Đang lấy transcript...'):
                    video_id = extract_video_id(url)
                    transcript, transcript_list = get_transcript(video_id)
                    # Chunking
                    chunks = get_chunks(transcript)
                    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
                    # Vector db
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 4}, search_type='similarity')
                    
                    st.session_state.transcript = transcript
                    st.session_state.chunks = chunks
                    st.session_state.retriever = retriever

                    st.success('✅ Đã lấy transcript thành công!')
            except Exception as e:
                st.error(f'Lỗi: {e}')

    question = st.text_input('Nhập câu hỏi:', key='question')
    if st.button('Truy vấn', key='ask'):
        if not question:
            st.warning('Vui lòng nhập câu hỏi.')
        else:
            try:
                with st.spinner('Đang truy vấn...'):
                    retriever = st.session_state.retriever
                    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=GOOGLE_API_KEY, temperature=0.2)
                    prompt = PromptTemplate(
                        # template='''
                        # You are a helpful assistant.
                        # Answer ONLY from the provided transcript context.
                        # If the context is insufficient, just say you don't know.

                        # {context}
                        # Question: {question}
                        # ''',
                        template='''
                        Bạn là một trợ lý AI hữu ích.
                        Chỉ trả lời dựa trên nội dung transcript được cung cấp.
                        Nếu thông tin trong transcript không đủ, hãy trả lời "Tôi không biết".
                        Hãy trả lời bằng tiếng Việt.

                        {context}
                        Câu hỏi: {question}
                        ''',
                        input_variables=['context', 'question']
                    )
                    retrieved_docs = retriever.invoke(question)
                    context_text = '\n\n'.join([doc.page_content for doc in retrieved_docs])
                    
                    final_prompt = prompt.invoke({'context': context_text, 'question': question})
                    answer = llm.invoke(final_prompt)
                    st.subheader('💡 Câu trả lời:')
                    st.write(answer.content)
            except Exception as e:
                    st.error(f'Lỗi: {e}')


if __name__ == "__main__":
    main()