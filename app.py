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
    """Baixa dados ajustados do Yahoo Finance"""
    data = yf.download(tickers, start=start)['Adj Close']
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
            
            # Checagem de integridade
            if prices.empty:
                st.error("Não foi possível baixar dados. Verifique os tickers.")
                st.stop()
            
            # Remover ativos com muitos NaNs (limpeza)
            prices = prices.dropna(axis=1, how='all').dropna() 
            
            # 2. Motor Estatístico (Otimizações Teóricas)
            # Retornos Esperados via CAPM (Melhor prática que média histórica)
            mu = expected_returns.capm_return(prices, risk_free_rate=risk_free_rate)
            
            # Matriz de Covariância via Ledoit-Wolf (Reduz erros extremos)
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

            # 3. Otimização (Fronteira Eficiente)
            ef = EfficientFrontier(mu, S)
            
            # Adiciona regularização L2 (evita pesos insignificantes como 0.0001%)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            # Otimizar para Máximo Sharpe Ratio
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights = ef.clean_weights()
            
            # Performance Esperada
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            exp_return, volatility, sharpe = perf

            # --- Visualização dos Resultados ---
            
            # Layout em Colunas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🏆 Alocação Ótima (Max Sharpe)")
                
                # Exibir métricas principais
                st.metric(label="Retorno Esperado (Anual)", value=f"{exp_return:.2%}")
                st.metric(label="Volatilidade (Risco)", value=f"{volatilidade:.2%}")
                st.metric(label="Índice de Sharpe", value=f"{sharpe:.2f}")

                # Tabela de pesos
                df_weights = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Peso'])
                df_weights = df_weights[df_weights['Peso'] > 0].sort_values(by='Peso', ascending=False)
                st.dataframe(df_weights.style.format("{:.2%}"))
                
                # Alocação Discreta (Quantidade de ações)
                latest_prices = prices.iloc[-1]
                da = discrete_allocation.DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount_to_invest)
                allocation, leftover = da.greedy_portfolio()
                
                st.info(f"Com R$ {amount_to_invest:,.2f}, compre aproximadamente:")
                st.json(allocation)
                st.write(f"Troco estimado: R$ {leftover:.2f}")

            with col2:
                # Gráfico de Fronteira Eficiente (Simulação de Monte Carlo para visualização)
                st.subheader("📊 Fronteira Eficiente & Carteiras Aleatórias")
                
                # Simular 1000 portfolios para desenhar a "nuvem"
                n_samples = 1000
                w_samples = np.random.dirichlet(np.ones(len(mu)), n_samples)
                rets = w_samples.dot(mu)
                stds = np.sqrt(np.diag(w_samples @ S @ w_samples.T))
                sharpes = (rets - risk_free_rate) / stds

                # Criar DataFrame da Simulação
                sim_df = pd.DataFrame({'Volatilidade': stds, 'Retorno': rets, 'Sharpe': sharpes})
                
                # Plotar Scatter Plot
                fig_ef = px.scatter(sim_df, x='Volatilidade', y='Retorno', color='Sharpe',
                                    color_continuous_scale='Viridis', hover_data={'Sharpe':':.2f'})
                
                # Adicionar o ponto ótimo (Estrela Vermelha)
                fig_ef.add_trace(go.Scatter(x=[volatilidade], y=[exp_return], mode='markers',
                                            marker=dict(color='red', size=15, symbol='star'),
                                            name='Máximo Sharpe'))
                
                fig_ef.update_layout(title="Risco vs Retorno (Simulação)", xaxis_title="Volatilidade (Risco)", yaxis_title="Retorno Esperado")
                st.plotly_chart(fig_ef, use_container_width=True)

            # Matriz de Correlação
            st.markdown("---")
            st.subheader("🔗 Matriz de Correlação e Risco")
            st.markdown("Ativos com **correlação baixa ou negativa** (cores azuis/escuras) aumentam a segurança do portfólio.")
            fig_corr = plot_correlation_matrix(prices)
            st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"Ocorreu um erro durante o cálculo: {e}")
            st.warning("Dica: Verifique se os tickers são válidos no Yahoo Finance (Ex: use '.SA' para ações brasileiras).")

else:
    st.info("Insira os tickers e clique em 'Otimizar Portfólio' na barra lateral para começar.")