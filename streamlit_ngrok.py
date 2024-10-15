

import streamlit as st
import time

# Função para realizar consulta
def do_query(query:str, multiquery=None, sid='1-1-0'):
    tm_on = time.perf_counter()

    stream_model = True
    response = ''

    if stream_model:
        for item in chain.stream(query):
            response += item
            # O Streamlit atualiza dinamicamente a resposta enquanto é processada
            st.write(item, end='')

    else:
        response = chain.invoke(query)

    counter = time.perf_counter() - tm_on

    return {'answer': response, 'time': counter}

# Código da aplicação Streamlit
def main():
    st.title('Chatbot com consulta de documentos')
    st.write('Digite sua pergunta e obtenha uma resposta com base em documentos.')

    user_input = st.text_input('Pergunte algo:')

    if user_input:
        # Aqui você pode integrar sua função de consulta de documentos
        st.write(f"Processando sua pergunta: {user_input}...")

        # Chama a função do_query para obter a resposta
        response = do_query(user_input)

        # Exibe a resposta final e o tempo de processamento
        st.write(f"Resposta: {response['answer']}")
        st.write(f"Tempo de resposta: {response['time']:.2f} segundos")

if __name__ == '__main__':
    main()

