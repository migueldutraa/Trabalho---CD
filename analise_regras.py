import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# 🔹 Carregamento do CSV
nome_arquivo = "vendas_limpo.csv"
df = pd.read_csv(nome_arquivo)

# 🔹 Agrupamento por id_da_compra
agrupado = df.groupby("id_da_compra")["produto"].apply(list)

# 🔹 Transações com mais de um produto
transacoes_multiplos = agrupado[agrupado.apply(lambda x: len(set(x)) > 1)]
print(f"\n🔍 Transações com múltiplos produtos: {len(transacoes_multiplos)}")

# 🔹 Simula caso não existam transações suficientes
if len(transacoes_multiplos) == 0:
    print("⚙️ Simulando transações com múltiplos produtos...")
    df_simulado = df.sample(frac=1).copy()
    df_simulado["id_simulado"] = (df_simulado.index // 5) + 1
    transacoes_multiplos = df_simulado.groupby("id_simulado")["produto"].apply(list)
    print(f"Transações simuladas: {len(transacoes_multiplos)}")

# 🔄 One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transacoes_multiplos).transform(transacoes_multiplos)
df_binario = pd.DataFrame(te_ary, columns=te.columns_)

# 🧠 Apriori
frequent_itemsets = apriori(df_binario, min_support=0.0003, use_colnames=True)

# ❌ Sem itens frequentes
if frequent_itemsets.empty:
    print("❌ Nenhum item frequente encontrado.")
else:
    # ⚖️ Regras de associação
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

    # 🎯 Apenas suporte, confiança e lift, com filtro de qualidade
    regras_qualidade = regras[["antecedents", "consequents", "support", "confidence", "lift"]]
    regras_qualidade = regras_qualidade[
        (regras_qualidade["lift"] > 1) & (regras_qualidade["confidence"] >= 0.2)
    ]
    regras_qualidade = regras_qualidade.sort_values(by="lift", ascending=False)

    # ✅ Resultado
    print(f"\n✅ Regras de associação encontradas: {len(regras_qualidade)}")
    print(regras_qualidade.head(10))

    # 💾 Exporta CSV
    regras_qualidade.to_csv("regras_associacao.csv", index=False)
    print("\n📁 Arquivo 'regras_associacao.csv' salvo com sucesso!")

# 📦 Ranking dos produtos mais vendidos
if 'produto' in df.columns:
    ranking_produtos = df['produto'].value_counts().reset_index()
    ranking_produtos.columns = ['produto', 'quantidade_vendida']

    print("\n📦 Top 10 Produtos Mais Vendidos:")
    print(ranking_produtos.head(10))

    ranking_produtos.to_csv("ranking_produtos.csv", index=False)
    print("📁 Arquivo 'ranking_produtos.csv' salvo com sucesso!")

# 🌎 Ranking de estados com mais vendas
if 'estado' in df.columns:
    ranking_estados = df['estado'].value_counts().reset_index()
    ranking_estados.columns = ['estado', 'quantidade_vendas']

    print("\n🌎 Top Estados com Mais Compras:")
    print(ranking_estados.head(10))

    ranking_estados.to_csv("ranking_estados.csv", index=False)
    print("📁 Arquivo 'ranking_estados.csv' salvo com sucesso!")