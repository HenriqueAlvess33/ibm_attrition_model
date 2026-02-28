import pandas as pd
import streamlit as st
import numpy as np
import joblib
import json
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Modelo de RH - Previsão de Rotatividade",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="./images/snapchat-circle.png",
)

# Inicialização do session state
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {}


# ==================== FUNÇÕES COM CACHE ====================
@st.cache_resource
def load_model():
    """Carrega o modelo treinado (pipeline PyCaret)."""
    return joblib.load("modelo_naive_bayes_02_02_2026.pkl")


@st.cache_data
def load_monthly_rate_avg():
    """Carrega o dicionário com a média de MonthlyRate por cargo."""
    with open("media_monthly_rate_per_job_role.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data(uploaded_file):
    """Carrega dados de um arquivo CSV (funcionalidade futura)."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Não foi possível carregar o arquivo: {e}")
        return None


# ==================== FUNÇÃO AUXILIAR ====================
def build_features_from_session():
    """Coleta todos os valores do session_state e retorna um dicionário de features."""
    return {
        "Age": st.session_state.get("age"),
        "Gender": st.session_state.get("gender"),
        "MaritalStatus": st.session_state.get("marital_status"),
        "DistanceFromHome": st.session_state.get("distance_from_home"),
        "Education": st.session_state.get("education"),
        "EducationField": st.session_state.get("education_field"),
        "Department": st.session_state.get("department"),
        "JobRole": st.session_state.get("job_role"),
        "JobLevel": st.session_state.get("job_level"),
        "OverTime": st.session_state.get("overtime"),
        "BusinessTravel": st.session_state.get("business_travel"),
        "MonthlyIncome": st.session_state.get("monthly_income"),
        "DailyRate": st.session_state.get("daily_rate"),
        "HourlyRate": st.session_state.get("hourly_rate"),
        "PercentSalaryHike": st.session_state.get("percent_salary_hike"),
        "StockOptionLevel": st.session_state.get("stock_option_level"),
        "NumCompaniesWorked": st.session_state.get("num_companies_worked"),
        "TotalWorkingYears": st.session_state.get("total_working_years"),
        "YearsAtCompany": st.session_state.get("years_at_company"),
        "YearsInCurrentRole": st.session_state.get("years_in_current_role"),
        "YearsSinceLastPromotion": st.session_state.get("years_since_last_promotion"),
        "YearsWithCurrManager": st.session_state.get("years_with_curr_manager"),
        "TrainingTimesLastYear": st.session_state.get("training_times_last_year"),
        "EnvironmentSatisfaction": st.session_state.get("environment_satisfaction"),
        "JobSatisfaction": st.session_state.get("job_satisfaction"),
        "RelationshipSatisfaction": st.session_state.get("relationship_satisfaction"),
        "WorkLifeBalance": st.session_state.get("work_life_balance"),
        "JobInvolvement": st.session_state.get("job_involvement"),
        "PerformanceRating": st.session_state.get("performance_rating"),
    }


avatares = {
    "Claudio": {
        "age": 52,
        "gender": "Male",
        "marital_status": "Married",
        "distance_from_home": 2,
        "education": 4,
        "education_field": "Technical Degree",
        "department": "Research & Development",
        "job_role": "Research Director",
        "job_level": 4,
        "overtime": "No",
        "business_travel": "Travel Rarely",
        "monthly_income": 15000,
        "daily_rate": 1200,
        "hourly_rate": 65,
        "percent_salary_hike": 18,
        "stock_option_level": 2,
        "num_companies_worked": 2,
        "total_working_years": 28,
        "years_at_company": 15,
        "years_in_current_role": 8,
        "years_since_last_promotion": 3,
        "years_with_curr_manager": 7,
        "training_times_last_year": 3,
        "environment_satisfaction": 4,
        "job_satisfaction": 4,
        "relationship_satisfaction": 3,
        "work_life_balance": 4,
        "job_involvement": 4,
        "performance_rating": 3,
    },
    "Henrique": {
        "age": 29,
        "gender": "Male",
        "marital_status": "Single",
        "distance_from_home": 15,
        "education": 3,
        "education_field": "Life Sciences",
        "department": "Sales",
        "job_role": "Sales Executive",
        "job_level": 2,
        "overtime": "Yes",
        "business_travel": "Travel Frequently",
        "monthly_income": 4500,
        "daily_rate": 800,
        "hourly_rate": 35,
        "percent_salary_hike": 10,
        "stock_option_level": 0,
        "num_companies_worked": 4,
        "total_working_years": 6,
        "years_at_company": 2,
        "years_in_current_role": 1,
        "years_since_last_promotion": 2,
        "years_with_curr_manager": 1,
        "training_times_last_year": 5,
        "environment_satisfaction": 2,
        "job_satisfaction": 2,
        "relationship_satisfaction": 3,
        "work_life_balance": 2,
        "job_involvement": 2,
        "performance_rating": 2,
    },
    "Leandro": {
        "age": 41,
        "gender": "Male",
        "marital_status": "Divorced",
        "distance_from_home": 8,
        "education": 3,
        "education_field": "Medical",
        "department": "Research & Development",
        "job_role": "Laboratory Technician",
        "job_level": 3,
        "overtime": "Yes",
        "business_travel": "Travel Rarely",
        "monthly_income": 5200,
        "daily_rate": 650,
        "hourly_rate": 28,
        "percent_salary_hike": 12,
        "stock_option_level": 1,
        "num_companies_worked": 3,
        "total_working_years": 18,
        "years_at_company": 10,
        "years_in_current_role": 5,
        "years_since_last_promotion": 4,
        "years_with_curr_manager": 6,
        "training_times_last_year": 2,
        "environment_satisfaction": 3,
        "job_satisfaction": 3,
        "relationship_satisfaction": 2,
        "work_life_balance": 3,
        "job_involvement": 3,
        "performance_rating": 3,
    },
    "Zelia": {
        "age": 35,
        "gender": "Female",
        "marital_status": "Married",
        "distance_from_home": 5,
        "education": 4,
        "education_field": "Marketing",
        "department": "Sales",
        "job_role": "Manager",
        "job_level": 4,
        "overtime": "No",
        "business_travel": "Travel Frequently",
        "monthly_income": 12000,
        "daily_rate": 1100,
        "hourly_rate": 58,
        "percent_salary_hike": 20,
        "stock_option_level": 3,
        "num_companies_worked": 1,
        "total_working_years": 12,
        "years_at_company": 10,
        "years_in_current_role": 5,
        "years_since_last_promotion": 2,
        "years_with_curr_manager": 4,
        "training_times_last_year": 4,
        "environment_satisfaction": 4,
        "job_satisfaction": 4,
        "relationship_satisfaction": 4,
        "work_life_balance": 3,
        "job_involvement": 4,
        "performance_rating": 4,
    },
    "Julia": {
        "age": 26,
        "gender": "Female",
        "marital_status": "Single",
        "distance_from_home": 20,
        "education": 2,
        "education_field": "Human Resources",
        "department": "Human Resources",
        "job_role": "Human Resources",
        "job_level": 1,
        "overtime": "Yes",
        "business_travel": "Non-Travel",
        "monthly_income": 2800,
        "daily_rate": 400,
        "hourly_rate": 20,
        "percent_salary_hike": 8,
        "stock_option_level": 0,
        "num_companies_worked": 2,
        "total_working_years": 3,
        "years_at_company": 1,
        "years_in_current_role": 1,
        "years_since_last_promotion": 1,
        "years_with_curr_manager": 1,
        "training_times_last_year": 6,
        "environment_satisfaction": 2,
        "job_satisfaction": 1,
        "relationship_satisfaction": 2,
        "work_life_balance": 1,
        "job_involvement": 2,
        "performance_rating": 2,
    },
    "Andressa": {
        "age": 45,
        "gender": "Female",
        "marital_status": "Married",
        "distance_from_home": 3,
        "education": 5,
        "education_field": "Medical",
        "department": "Research & Development",
        "job_role": "Healthcare Representative",
        "job_level": 5,
        "overtime": "No",
        "business_travel": "Travel Rarely",
        "monthly_income": 13500,
        "daily_rate": 1300,
        "hourly_rate": 70,
        "percent_salary_hike": 22,
        "stock_option_level": 3,
        "num_companies_worked": 1,
        "total_working_years": 22,
        "years_at_company": 18,
        "years_in_current_role": 12,
        "years_since_last_promotion": 5,
        "years_with_curr_manager": 10,
        "training_times_last_year": 2,
        "environment_satisfaction": 4,
        "job_satisfaction": 4,
        "relationship_satisfaction": 4,
        "work_life_balance": 4,
        "job_involvement": 4,
        "performance_rating": 4,
    },
    "Laura": {
        "age": 32,
        "gender": "Female",
        "marital_status": "Single",
        "distance_from_home": 10,
        "education": 4,
        "education_field": "Life Sciences",
        "department": "Research & Development",
        "job_role": "Research Scientist",
        "job_level": 3,
        "overtime": "No",
        "business_travel": "Travel Frequently",
        "monthly_income": 6200,
        "daily_rate": 850,
        "hourly_rate": 42,
        "percent_salary_hike": 15,
        "stock_option_level": 1,
        "num_companies_worked": 2,
        "total_working_years": 9,
        "years_at_company": 5,
        "years_in_current_role": 3,
        "years_since_last_promotion": 2,
        "years_with_curr_manager": 3,
        "training_times_last_year": 3,
        "environment_satisfaction": 3,
        "job_satisfaction": 3,
        "relationship_satisfaction": 3,
        "work_life_balance": 3,
        "job_involvement": 3,
        "performance_rating": 3,
    },
    "Bruno": {
        "age": 38,
        "gender": "Male",
        "marital_status": "Married",
        "distance_from_home": 25,
        "education": 3,
        "education_field": "Technical Degree",
        "department": "Sales",
        "job_role": "Sales Representative",
        "job_level": 2,
        "overtime": "Yes",
        "business_travel": "Travel Frequently",
        "monthly_income": 4100,
        "daily_rate": 550,
        "hourly_rate": 25,
        "percent_salary_hike": 9,
        "stock_option_level": 0,
        "num_companies_worked": 5,
        "total_working_years": 15,
        "years_at_company": 3,
        "years_in_current_role": 2,
        "years_since_last_promotion": 3,
        "years_with_curr_manager": 2,
        "training_times_last_year": 4,
        "environment_satisfaction": 2,
        "job_satisfaction": 2,
        "relationship_satisfaction": 2,
        "work_life_balance": 2,
        "job_involvement": 2,
        "performance_rating": 2,
    },
    "Gabriela": {
        "age": 29,
        "gender": "Female",
        "marital_status": "Married",
        "distance_from_home": 6,
        "education": 4,
        "education_field": "Marketing",
        "department": "Sales",
        "job_role": "Manager",
        "job_level": 4,
        "overtime": "No",
        "business_travel": "Travel Rarely",
        "monthly_income": 11000,
        "daily_rate": 1050,
        "hourly_rate": 55,
        "percent_salary_hike": 18,
        "stock_option_level": 2,
        "num_companies_worked": 2,
        "total_working_years": 7,
        "years_at_company": 5,
        "years_in_current_role": 3,
        "years_since_last_promotion": 1,
        "years_with_curr_manager": 3,
        "training_times_last_year": 3,
        "environment_satisfaction": 4,
        "job_satisfaction": 4,
        "relationship_satisfaction": 4,
        "work_life_balance": 3,
        "job_involvement": 4,
        "performance_rating": 3,
    },
    "Daniel": {
        "age": 55,
        "gender": "Male",
        "marital_status": "Divorced",
        "distance_from_home": 1,
        "education": 3,
        "education_field": "Other",
        "department": "Research & Development",
        "job_role": "Manufacturing Director",
        "job_level": 5,
        "overtime": "No",
        "business_travel": "Non-Travel",
        "monthly_income": 16000,
        "daily_rate": 1400,
        "hourly_rate": 75,
        "percent_salary_hike": 16,
        "stock_option_level": 3,
        "num_companies_worked": 3,
        "total_working_years": 32,
        "years_at_company": 20,
        "years_in_current_role": 15,
        "years_since_last_promotion": 7,
        "years_with_curr_manager": 12,
        "training_times_last_year": 1,
        "environment_satisfaction": 3,
        "job_satisfaction": 3,
        "relationship_satisfaction": 2,
        "work_life_balance": 3,
        "job_involvement": 3,
        "performance_rating": 3,
    },
}


# ==================== MAIN ====================
def main():
    # ----- Cabeçalho com imagem -----
    try:
        img = Image.open("./images/IBM_image.jpg")
        max_height = 500
        if img.height > max_height:
            new_height = max_height
            new_width = int(img.width * (new_height / img.height))
            img = img.resize((new_width, new_height))

        col1, col2, col3, col4 = st.columns([1, 1, 4, 1])
        with col3:
            st.image(img, use_container_width=False)
    except Exception as e:
        st.error(f"Erro ao carregar a imagem: {e}")

    st.markdown("---")
    st.title("Modelo de Previsão de Rotatividade de Funcionários")

    # ----- Expander com descrição do dataset -----
    with st.expander("📊 Sobre o Dataset"):
        st.write(
            """
            Este dataset contém informações fictícias, porém realistas, de colaboradores da IBM. 
            Ele é amplamente utilizado para explorar fatores que influenciam a **rotatividade de funcionários** (*Attrition*).

            São 35 variáveis que descrevem perfil demográfico (idade, gênero, estado civil), 
            satisfação no trabalho (ambiente, relacionamento, envolvimento), condições contratuais 
            (cargo, departamento, horas extras) e histórico profissional (anos na empresa, promoções, salário).

            O objetivo é permitir a criação de modelos preditivos para identificar quais colaboradores 
            têm maior propensão a deixar a empresa, auxiliando na tomada de decisões estratégicas de RH.
            """
        )

    st.markdown("---")

    # ----- Abas -----
    tab1, tab2 = st.tabs(
        ["📋 Ficha de cadastro para modelo de previsão", "📈 Apresentação de resultado"]
    )

    # -------------------- ABA 1: FORMULÁRIO --------------------
    with tab1:
        st.header("Preencha os dados do funcionário ou escolha um avatar")

        # Seletor de avatar
        avatar_escolhido = st.selectbox(
            "Carregar avatar de exemplo", ["Selecione..."] + list(avatares.keys())
        )
        if avatar_escolhido != "Selecione...":
            dados_avatar = avatares[avatar_escolhido]
            for chave, valor in dados_avatar.items():
                st.session_state[chave.lower()] = valor
            st.success(
                f"Dados do avatar {avatar_escolhido} foram carregados!!! Revise e clique em confirmar"
            )
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Dados pessoais")
            st.number_input(
                "Idade", min_value=18, max_value=65, value=35, step=1, key="age"
            )
            st.selectbox("Gênero", ["Male", "Female"], key="gender")
            st.selectbox(
                "Estado civil", ["Single", "Married", "Divorced"], key="marital_status"
            )
            st.number_input(
                "Distância de casa (km)",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
                key="distance_from_home",
            )

            st.subheader("🎓 Formação")
            st.selectbox(
                "Nível de educação",
                [1, 2, 3, 4, 5],
                format_func=lambda x: [
                    "Below College",
                    "College",
                    "Bachelor",
                    "Master",
                    "Doctor",
                ][x - 1],
                key="education",
            )
            st.selectbox(
                "Área de formação",
                [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other",
                ],
                key="education_field",
            )

        with col2:
            st.subheader("💼 Trabalho")
            st.selectbox(
                "Departamento",
                ["Sales", "Research & Development", "Human Resources"],
                key="department",
            )
            st.selectbox(
                "Cargo",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
                key="job_role",
            )
            st.selectbox("Nível de cargo", [1, 2, 3, 4, 5], key="job_level")
            st.selectbox("Faz horas extras?", ["Yes", "No"], key="overtime")
            st.selectbox(
                "Frequência de viagens",
                ["Travel Rarely", "Travel Frequently", "Non-Travel"],
                key="business_travel",
            )

            st.subheader("💰 Remuneração")
            st.number_input(
                "Renda mensal",
                min_value=1000,
                max_value=20000,
                value=5000,
                step=100,
                key="monthly_income",
            )
            st.number_input(
                "Diária média",
                min_value=100,
                max_value=2000,
                value=800,
                step=10,
                key="daily_rate",
            )
            st.number_input(
                "Salário por hora",
                min_value=10,
                max_value=70,
                value=30,
                step=1,
                key="hourly_rate",
            )
            st.number_input(
                "% de aumento no último ano",
                min_value=0,
                max_value=30,
                value=10,
                step=1,
                key="percent_salary_hike",
            )
            st.selectbox(
                "Nível de stock options", [0, 1, 2, 3], key="stock_option_level"
            )

            st.markdown("---")
            secondary_col1, secondary_col2 = st.columns(2)

            with secondary_col1:
                st.subheader("⏳ Histórico profissional")
                st.number_input(
                    "Nº de empresas trabalhadas",
                    min_value=0,
                    max_value=20,
                    value=3,
                    step=1,
                    key="num_companies_worked",
                )
                st.number_input(
                    "Anos totais de experiência",
                    min_value=0,
                    max_value=50,
                    value=10,
                    step=1,
                    key="total_working_years",
                )
                st.number_input(
                    "Anos na empresa atual",
                    min_value=0,
                    max_value=40,
                    value=5,
                    step=1,
                    key="years_at_company",
                )
                st.number_input(
                    "Anos no cargo atual",
                    min_value=0,
                    max_value=20,
                    value=3,
                    step=1,
                    key="years_in_current_role",
                )
                st.number_input(
                    "Anos desde última promoção",
                    min_value=0,
                    max_value=20,
                    value=1,
                    step=1,
                    key="years_since_last_promotion",
                )
                st.number_input(
                    "Anos com o gestor atual",
                    min_value=0,
                    max_value=20,
                    value=2,
                    step=1,
                    key="years_with_curr_manager",
                )
                st.number_input(
                    "Treinamentos no último ano",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1,
                    key="training_times_last_year",
                )

            with secondary_col2:
                st.subheader("😊 Satisfação e avaliação")
                st.selectbox(
                    "Satisfação com o ambiente",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Baixa", "Média", "Alta", "Muito alta"][
                        x - 1
                    ],
                    key="environment_satisfaction",
                )
                st.selectbox(
                    "Satisfação com o trabalho",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Baixa", "Média", "Alta", "Muito alta"][
                        x - 1
                    ],
                    key="job_satisfaction",
                )
                st.selectbox(
                    "Satisfação com relacionamentos",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Baixa", "Média", "Alta", "Muito alta"][
                        x - 1
                    ],
                    key="relationship_satisfaction",
                )
                st.selectbox(
                    "Equilíbrio trabalho-vida",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Ruim", "Regular", "Bom", "Excelente"][
                        x - 1
                    ],
                    key="work_life_balance",
                )
                st.selectbox(
                    "Envolvimento com o trabalho",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Baixo", "Médio", "Alto", "Muito alto"][
                        x - 1
                    ],
                    key="job_involvement",
                )
                st.selectbox(
                    "Avaliação de desempenho",
                    [1, 2, 3, 4],
                    format_func=lambda x: ["Ruim", "Regular", "Bom", "Excelente"][
                        x - 1
                    ],
                    key="performance_rating",
                )

    # -------------------- ABA 2: RESULTADO --------------------
    with tab2:
        st.header("Resultado da Previsão")
        st.caption("Clique no botão para processar os dados e visualizar a predição.")

        if st.button(
            "🔍 Confirmar e gerar previsão", type="primary", use_container_width=True
        ):
            # --- 1. Carregar médias e modelo ---
            monthly_rate_avg = load_monthly_rate_avg()
            modelo = load_model()

            # --- 2. Calcular fallback para MonthlyRate (média geral) ---
            if monthly_rate_avg:
                media_geral_monthly_rate = np.mean(list(monthly_rate_avg.values()))
            else:
                media_geral_monthly_rate = (
                    800.0  # valor padrão caso o JSON esteja vazio
                )

            # --- 3. Obter cargo e estimar MonthlyRate ---
            job_role = st.session_state.get("job_role")
            monthly_rate_estimado = monthly_rate_avg.get(
                job_role, media_geral_monthly_rate
            )

            # --- 4. Montar dicionário de features ---
            features = build_features_from_session()
            features["MonthlyRate"] = monthly_rate_estimado

            # --- 5. Criar DataFrame e adicionar colunas constantes ---
            df_input = pd.DataFrame([features])

            # Colunas que são fixas no dataset original
            df_input["EmployeeCount"] = 1
            df_input["Over18"] = "Y"
            df_input["StandardHours"] = 80
            # EmployeeNumber: usar 0 em vez de NaN (é apenas um identificador)
            df_input["EmployeeNumber"] = 0

            # --- 6. Garantir a ordem correta das colunas (conforme treinamento) ---
            colunas_esperadas = modelo.feature_names_in_
            # Verifica se todas as colunas esperadas estão presentes
            for col in colunas_esperadas:
                if col not in df_input.columns:
                    st.error(f"Coluna '{col}' não encontrada nos dados de entrada.")
                    st.stop()
            df_input = df_input[colunas_esperadas]

            # --- 7. Verificar se ainda há valores nulos ---
            if df_input.isnull().any().any():
                st.warning(
                    "Ainda existem valores nulos no DataFrame. Verifique as colunas abaixo:"
                )
                st.write(df_input.isnull().sum()[df_input.isnull().sum() > 0])
                st.info(
                    "Isso pode indicar que algum campo do formulário não foi preenchido corretamente. "
                    "Por favor, volte à aba anterior e preencha todos os campos."
                )
                st.stop()

            # --- 8. Realizar predição ---
            pred_num = modelo.predict(df_input)[0]
            proba = modelo.predict_proba(df_input)[0]

            # --- 9. Exibir resultados ---
            st.success(f"### Resultado: **{'Yes' if pred_num == 1 else 'No'}**")
            st.info(f"**Probabilidade de rotatividade:** {proba[1]:.2%}")

            # Opcional: mostrar um resumo dos dados inseridos
            with st.expander("📋 Dados utilizados na previsão"):
                st.dataframe(df_input.T.rename(columns={0: "Valor"}))


# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    main()
