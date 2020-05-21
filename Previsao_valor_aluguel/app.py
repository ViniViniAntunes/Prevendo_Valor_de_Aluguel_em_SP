# Importando as bibliotecas necessárias
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Criando uma função para carregar o dataset
#@st.cache # Notação para ficar em cache
def get_data():
    return pd.read_csv("model/data_deploy.csv")

# Criando uma função para treinar o nosso modelo
def train_model():
    data = get_data()
    X = data.drop(["valor", "bairro"], axis=1)
    y = data["valor"]
    rf_regressor = RandomForestRegressor(n_estimators=100)
    rf_regressor.fit(X, y)
    return rf_regressor

def get_villages_and_id():
    data = get_data()
    names_and_id = dict(zip(data['bairro'], data['bairro_id']))
    return names_and_id

def return_id_village(village):
    return get_villages_and_id()[village]

# Armazenando o dataframe na variável 'data'
data = get_data().drop("bairro_id", axis=1)

# Treinando o modelo
model = train_model()

# Configurando o título do Data App
st.title("Data App - Prevendo Valores de Imóveis")

# Configurando o subtítulo do data app
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning que prevê valores de aluguel de apartamentos na capital de São Paulo.")
st.markdown('Criado por: Vini Antunes')
st.markdown('LinkedIn: https://www.linkedin.com/in/vini-antunes')

# Verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# Selecionando atributos para serem exibidos por padrão
default_cols = ["quartos","bairro","valor"]

# Defindo atributos a partir do multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=default_cols)

# Exibindo os top 10 registros do DataFrame
st.dataframe(data[cols].head(10))

# Configurando outro subtítulo
st.subheader("Distribuição de imóveis por preço do aluguel")

# Definindo a faixa de valores
faixa_valores = st.slider("Faixa de preço", float(data['valor'].min()), float(data['valor'].max()), (1000.0, 2000.0))

# Filtrando os dados
filtered_data = data[data['valor'].between(left=faixa_valores[0], right=faixa_valores[1])]

# Plotando a distribuição dos dados
f = px.histogram(filtered_data, x="valor", nbins=20, title="Distribuição de Preços do Aluguel")
f.update_xaxes(title="valor")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)

# Configurando subtítulo da lateral
st.sidebar.subheader("Defina os atributos do imóvel para predição")

####### Mapeando dados #######
# Armazena os nomes dos bairros e seus respectivos ids
villages = get_villages_and_id().keys()

# Selecionando o bairro
village = st.sidebar.selectbox("Em qual bairro?", sorted(list(villages)))

# Trocando o nome do bairro' pelo seus respectivo id
id_village = return_id_village(village)

# Selecionando a área do apartamento
area = st.sidebar.number_input("Área (em m²)?", min_value=float(data['area'].min()), max_value=float(data['area'].max()), step=1.0, format="%.0f")

# Selecionando a quantidade de quartos
rooms = st.sidebar.number_input("Quantos quartos?", min_value=float(data['quartos'].min()), max_value=float(data['quartos'].max()), step=1.0, format="%.0f")

# Selecionando a quantidade de suites
suites = st.sidebar.number_input("Quantas suítes?", min_value=float(data['suites'].min()), max_value=float(data['suites'].max()), step=1.0, format="%.0f")

# Selecionando a quantidade de vagas de garagem
parking_spaces = st.sidebar.number_input("Quantas vagas de garagem?", min_value=float(data['vagas'].min()), max_value=float(data['vagas'].max()), step=1.0, format="%.0f")

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[area, rooms, suites, parking_spaces, id_village]])
    st.sidebar.subheader("O valor previsto para do aluguel é:")
    st.sidebar.subheader("")
    result = f"R$ {str(round(result[0], 2))}"
    st.sidebar.subheader(result)