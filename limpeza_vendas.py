import pandas as pd
import numpy as np

# 1. Carrega o CSV original bruto
df = pd.read_csv("vendas_modificado.csv")

# 2. Limpa espaços invisíveis
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Converte datas e horas
df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
df["hora"] = pd.to_datetime(df["hora"], errors="coerce").dt.time

# 4. Converte valores monetários
df["valor"] = df["valor"].astype(str).str.replace("R$", "", regex=False).str.replace(",", ".", regex=False)
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
df["valor"] = df.groupby("produto")["valor"].transform(lambda x: x.fillna(x.mean()))
df["valor"] = df["valor"].fillna(df["valor"].median())

# 5. Padroniza os nomes dos produtos
correcoes_produto = {
    "acucar": "Açúcar", "amaciante": "Amaciante", "amaciayte": "Amaciante", "arroc": "Arroz",
    "arroz": "Arroz", "azeite": "Azeite", "biscoito recheado": "Biscoito Recheado",
    "biscoitq recheado": "Biscoito Recheado", "cafc": "Café", "cafe": "Café", "caff": "Café",
    "caft": "Café", "carvao": "Carvão", "cerveja": "Cerveja", "clfé": "Café", "cnfé": "Café",
    "condibionador": "Condicionador", "condicioiador": "Condicionador", "condicionador": "Condicionador",
    "deqergente": "Detergente", "deterwente": "Detergente", "detergente": "Detergente",
    "desinfekante": "Desinfetante", "desinfetanue": "Desinfetante", "desinfetante": "Desinfetante",
    "farinha de trigo": "Farinha De Trigo", "farinha de tripo": "Farinha De Trigo",
    "feijao": "Feijão", "ieijao": "Feijão", "leite integral": "Leite Integral",
    "macarrao": "Macarrão", "macarrao espaguete": "Macarrão", "macarrao parafuso": "Macarrão",
    "macirrão": "Macarrão", "macawração": "Macarrão", "majarrão": "Macarrão", "manteiga": "Manteiga",
    "manteigt": "Manteiga", "molho de tomate": "Molho De Tomate", "molmo de tomate": "Molho De Tomate",
    "mopho de tomate": "Molho De Tomate", "mqcarrão": "Macarrão", "oleo de soja": "Óleo De Soja",
    "oleo soja": "Óleo De Soja", "papel higiênico": "Papel Higiênico", "papel qoalha": "Papel Toalha",
    "papel toalha": "Papel Toalha", "papel twalha": "Papel Toalha", "pasta de dente": "Pasta De Dente",
    "presuntd": "Presunto", "presunto": "Presunto", "pao de forma": "Pão De Forma",
    "qbeijo mussarela": "Queijo Mussarela", "queijo mussarela": "Queijo Mussarela",
    "queijo mussarelz": "Queijo Mussarela", "refrigkrante": "Refrigerante", "refrigerante": "Refrigerante",
    "sabao em po": "Sabão Em Pó", "sabonepe": "Sabonete", "sabonete": "Sabonete", "sal": "Sal",
    "scl": "Sal", "shampoo": "Shampoo", "suco de laranja": "Suco De Laranja",
    "sucoyde laranja": "Suco De Laranja", "tal": "Produto Diverso", "vinho": "Vinho",
    "zabonete": "Sabonete", "agua mineral": "Água Mineral", "agua mineras": "Água Mineral",
    "agua mineual": "Água Mineral"
}

df["produto"] = df["produto"].str.lower().str.strip()
df["produto"] = df["produto"].str.replace(r"[^a-zçáéíóúâêôãõà ]", "", regex=True)
df["produto"] = df["produto"].str.replace(r"\s{2,}", " ", regex=True)
df["produto"] = df["produto"].replace(correcoes_produto)
df["produto"] = df["produto"].fillna("Produto Desconhecido").replace("", "Produto Desconhecido")
df["produto"] = df["produto"].str.title()

# 6. Padronização dos estados (UF)
mapa_uf = {
    "minas gerais": "MG", "mg": "MG", "Minas Gerais": "MG",
    "são paulo": "SP", "sp": "SP", "São Paulo": "SP",
    "rio de janeiro": "RJ", "rj": "RJ", "Rio de Janeiro": "RJ",
    "bahia": "BA", "ba": "BA", "Bahia": "BA",
    "paraná": "PR", "pr": "PR", "Paraná": "PR",
    "rs": "RS", "rio grande do sul": "RS",
    "psc": "SC", "santa catarina": "SC",
    "mtsa": "MT", "mato grosso": "MT"
}

df["estado"] = df["estado"].astype(str).str.strip().str.lower()
df["estado"] = df["estado"].replace(mapa_uf)
df["estado"] = df["estado"].fillna("UF Desconhecida")
df["estado"] = df["estado"].str.upper()

# 7. Frete: preenche nulos com a mediana por cidade
df["frete"] = df.groupby("cidade")["frete"].transform(lambda x: x.fillna(x.median()))
df["frete"] = df["frete"].fillna(df["frete"].median())
df.loc[df["frete"] < 0, "frete"] = df["frete"].median()

# 8. Quantidade
df["quantidade"] = df["quantidade"].fillna(1)
df.loc[df["quantidade"] <= 0, "quantidade"] = 1

# 9. Vendedor
df["vendedor"] = df["vendedor"].fillna("Vendedor Desconhecido")

# 10. CEP
df["cep"] = df["cep"].astype(str).str.extract(r"(\d{5}-\d{3})")[0]
df["cep"] = df["cep"].fillna("00000-000")

# 11. Total: corrige valores inconsistentes
total_calc = df["valor"] * df["quantidade"] + df["frete"]
df["total"] = df.apply(
    lambda row: total_calc[row.name]
    if pd.isna(row["total"]) or abs(row["total"] - total_calc[row.name]) > 0.01
    else row["total"],
    axis=1
)

# 12. Remove duplicatas
df = df.drop_duplicates()

# 13. Remove registros completamente inválidos
df = df[~((df["valor"] == 0) & (df["frete"] == 0) & (df["quantidade"] == 1) & (df["produto"] == "Produto Desconhecido"))]

# 14. Exporta o CSV limpo
df.to_csv("vendas_limpo.csv", index=False)
print("✅ Arquivo 'vendas_limpo.csv' salvo com sucesso!")