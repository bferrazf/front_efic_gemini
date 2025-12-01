import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
from pypfopt import discrete_allocation

# --- Configuração da Página ---
st.set_page_config(page_title="OptiFolio - Otimizador de Carteiras", layout="wide")

st.title("📈 OptiFolio: Fronteira Eficiente & Otimização de Risco")
st.markdown("""
Esta aplicação utiliza a **Teoria Moderna de Portfólio (MPT)** para encontrar a alocação ótima de ativos.
Utilizamos **Shrinkage de Ledoit-Wolf** para a matriz de covariância e **CAPM** para retornos esperados.
""")

# --- Sidebar: Inputs do Usuário ---
st.sidebar.header("Parâmetros do Portfólio")

tickers_input = st.sidebar.text_area(
    "Insira os Tickers (separados por vírgula)",
    value="PETR4.SA, VALE3.SA, ITUB4.SA, WEGE3.SA, BOVA11.SA",
    height=70
)

start_date = st.sidebar.date_input("Data de Início", value=pd.to_datetime("2020-01-01"))
risk_free_rate = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=11.75, step=0.25) / 100
amount_to_invest = st.sidebar.number_input("Valor Total para Investir (R$)", value=10000.00)

submit_btn = st.sidebar.button("Otimizar Portfólio")

# --- Funções Auxiliares ---
def get_data(tickers, start):
    """
    Baixa dados do Yahoo Finance.
    Usa auto_adjust=True para já receber os preços ajustados (dividendos/splits).
    """
    data = yf.download(tickers, start=start, auto_adjust=True)
    
    # Tratamento para garantir que pegamos apenas os preços de fechamento
    if 'Close' in data.columns:
        return data['Close']
    else:
        # Fallback caso a estrutura venha diferente
        return data

def plot_correlation_matrix(df):
    """Gera heatmap de correlação"""
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Matriz de Correlação dos Ativos")
    return fig

# --- Lógica Principal ---
if submit_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Baixando dados e calculando estatísticas...'):
        try:
            # 1. Obtenção de Dados
            prices = get_data(tickers, start_date)
            
            # Checagem se veio vazio
            if prices.empty:
                st.error("Não foi possível baixar dados. Verifique os tickers.")
                st.stop()
            
            # Limpeza de dados (remove colunas ou linhas vazias)
            prices = prices.dropna(axis=1, how='all').dropna() 
            
            if prices.shape[1] < 2:
                st.error("É necessário pelo menos 2 ativos válidos para otimizar um portfólio.")
                st.stop()

            # 2. Motor Estatístico (Otimizações Teóricas)
            # Retornos Esperados via CAPM
            mu = expected_returns.capm_return(prices, risk_free_rate=risk_free_rate)
            
            # Matriz de Covariância via Ledoit-Wolf
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

            # 3. Otimização (Fronteira Eficiente)
            ef = EfficientFrontier(mu, S)
            
            # Adiciona regularização L2 (Gamma)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            # Otimizar para Máximo Sharpe Ratio
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights = ef.clean_weights()
            
            # Performance Esperada
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            exp_return, volatility, sharpe = perf

            # --- Visualização dos Resultados ---
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🏆 Alocação Ótima (Max Sharpe)")
                
                st.metric(label="Retorno Esperado (Anual)", value=f"{exp_return:.2%}")
                st.metric(label="Volatilidade (Risco)", value=f"{volatilidade:.2%}")
                st.metric(label="Índice de Sharpe", value=f"{sharpe:.2f}")

                # Tabela de pesos
                df_weights = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Peso'])
                df_weights = df_weights[df_weights['Peso'] > 0].sort_values(by='Peso', ascending=False)
                st.dataframe(df_weights.style.format("{:.2%}"))
                
                # Alocação Discreta
                latest_prices = prices.iloc[-1]
                da = discrete_allocation.DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount_to_invest)
                allocation, leftover = da.greedy_portfolio()
                
                st.info(f"Com R$ {amount_to_invest:,.2f}, compre aproximadamente:")
                if allocation:
                    st.json(allocation)
                else:
                    st.write("O valor investido é muito baixo para comprar uma ação inteira destes ativos.")
                st.write(f"Troco estimado: R$ {leftover:.2f}")

            with col2:
                st.subheader("📊 Fronteira Eficiente & Carteiras Aleatórias")
                
                # Simulação Monte Carlo
                n_samples = 1000
                w_samples = np.random.dirichlet(np.ones(len(mu)), n_samples)
                rets = w_samples.dot(mu)
                stds = np.sqrt(np.diag(w_samples @ S @ w_samples.T))
                sharpes = (rets - risk_free_rate) / stds

                sim_df = pd.DataFrame({'Volatilidade': stds, 'Retorno': rets, 'Sharpe': sharpes})
                
                fig_ef = px.scatter(sim_df, x='Volatilidade', y='Retorno', color='Sharpe',
                                    color_continuous_scale='Viridis', hover_data={'Sharpe':':.2f'})
                
                # Ponto Ótimo
                fig_ef.add_trace(go.Scatter(x=[volatilidade], y=[exp_return], mode='markers',
                                            marker=dict(color='red', size=15, symbol='star'),
                                            name='Máximo Sharpe'))
                
                fig_ef.update_layout(title="Risco vs Retorno (Simulação)", xaxis_title="Volatilidade (Risco)", yaxis_title="Retorno Esperado")
                st.plotly_chart(fig_ef, use_container_width=True)

            # Matriz de Correlação
            st.markdown("---")
            st.subheader("🔗 Matriz de Correlação e Risco")
            fig_corr = plot_correlation_matrix(prices)
            st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.warning("Dica: Se o erro persistir, tente reduzir o período de tempo ou trocar os tickers.")

else:
    st.info("Insira os tickers e clique em 'Otimizar Portfólio' na barra lateral para começar.")
