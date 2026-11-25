import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import pandas as pd

# Stopwords comunes en español (conservando 'no', 'sí', 'si', 'sin', 'más', 'todos'
# cuando aportan significado semántico en el análisis de sentimiento/postura,
# y definiendo una lista estándar de palabras vacías gramaticales)
STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais",
    "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "estaba", "estabais", "estaban", "estabas", "estad", "estada",
    "estadas", "estado", "estados", "estamos", "estan", "estando", "estar",
    "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis",
    "estaría", "estaríais", "estaríamos", "estarían", "estarías", "estas",
    "este", "estemos", "esto", "estos", "estoy", "estuve", "estuviera",
    "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese",
    "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste",
    "estuvisteis", "estuviéramos", "estuviésemos", "estuvo", "está",
    "estábamos", "estáis", "están", "estás", "esté", "estéis", "estén",
    "estés", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron",
    "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste",
    "fuisteis", "fuéramos", "fuésemos", "ha", "habida", "habidas", "habido",
    "habidos", "habiendo", "habremos", "habrá", "habrán", "habrás", "habré",
    "habréis", "habría", "habríais", "habríamos", "habrían", "habrías",
    "habéis", "había", "habíais", "habíamos", "habían", "habías", "han",
    "has", "hasta", "hay", "haya", "hayamos", "hayan", "hayas", "hayáis",
    "he", "hemos", "hube", "hubiera", "hubierais", "hubieran", "hubieras",
    "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses", "hubimos",
    "hubiste", "hubisteis", "hubiéramos", "hubiésemos", "hubo", "la", "las",
    "le", "les", "lo", "los", "me", "mi", "mis", "mucho", "muchos", "muy",
    "mí", "mía", "mías", "mío", "míos", "nada", "ni", "nos", "nosotras",
    "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os",
    "otra", "otras", "otro", "otros", "para", "pero", "poco", "por",
    "porque", "que", "quien", "quienes", "qué", "se", "sea", "seamos",
    "sean", "seas", "seremos", "será", "serán", "serás", "seré", "seréis",
    "sería", "seríais", "seríamos", "serían", "serías", "seáis", "sido",
    "siendo", "sobre", "sois", "somos", "son", "soy", "su", "sus", "suya",
    "suyas", "suyo", "suyos", "también", "tanto", "te", "tendremos",
    "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría",
    "tendríais", "tendríamos", "tendrían", "tendrías", "tened", "tenemos",
    "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis", "tenida",
    "tenidas", "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais",
    "teníamos", "tenían", "tenías", "ti", "tiene", "tienen", "tienes",
    "todo", "tu", "tus", "tuve", "tuviera", "tuvierais", "tuvieran",
    "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses",
    "tuvimos", "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo",
    "tuya", "tuyas", "tuyo", "tuyos", "tú", "un", "una", "uno", "unos",
    "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
    "y", "ya", "yo", "él", "éramos"
}


def quitar_acentos(texto: str) -> str:
    """Elimina tildes conservando la letra ñ."""
    texto_norm = unicodedata.normalize("NFD", texto)
    resultado = []
    for char in texto_norm:
        if unicodedata.category(char) == "Mn":
            continue
        resultado.append(char)
    # Restaurar ñ si se descompuso
    return unicodedata.normalize("NFC", "".join(texto)).translate(
        str.maketrans("áéíóúÁÉÍÓÚüÜ", "aeiouAEIOUuU")
    )


def extraer_hashtags(texto: str) -> list[str]:
    """Extrae hashtags como #voto o #encuesta."""
    return re.findall(r"#\w+", texto.lower())


def limpiar_comentario(texto: str, quitar_ruido_redes: bool = True) -> str:
    """
    Normaliza el comentario:
    - Pasa a minúsculas
    - Opcionalmente elimina hashtags (#voto, #encuesta) y muletillas ('pls')
    - Elimina signos de puntuación (?, !, ., :)
    - Colapsa espacios múltiples
    """
    t = texto.lower().strip()
    if quitar_ruido_redes:
        t = re.sub(r"#\w+", "", t)
        t = re.sub(r"\bpls\b", "", t)
    t = re.sub(r"[^\w\sáéíóúñü]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenizar(texto_limpio: str, eliminar_stopwords: bool = False) -> list[str]:
    """Tokeniza por palabras y opcionalmente filtra stopwords."""
    tokens = texto_limpio.split()
    if eliminar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    return tokens


def construir_corpus(ruta_csv: str, carpeta_salida: str = "corpus_votacion"):
    ruta_entrada = Path(ruta_csv)
    out_dir = Path(carpeta_salida)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ruta_entrada)

    # 1. Procesamiento de texto para cada documento del corpus
    hashtags_lista = []
    texto_limpio_lista = []
    texto_base_sin_tildes = []
    tokens_completos_lista = []
    tokens_sin_stopwords_lista = []
    texto_sin_stopwords_lista = []

    contador_vocab_total = Counter()
    contador_doc_freq = Counter()
    contador_vocab_sin_sw = Counter()
    contador_bigramas = Counter()

    documentos_json = []

    for _, row in df.iterrows():
        comentario_orig = str(row["comentario"])
        hashtags = extraer_hashtags(comentario_orig)
        limpio = limpiar_comentario(comentario_orig, quitar_ruido_redes=False)
        base = limpiar_comentario(comentario_orig, quitar_ruido_redes=True)
        base_norm = quitar_acentos(base)

        tokens = tokenizar(base, eliminar_stopwords=False)
        tokens_sw = tokenizar(base, eliminar_stopwords=True)

        hashtags_lista.append(", ".join(hashtags))
        texto_limpio_lista.append(limpio)
        texto_base_sin_tildes.append(base_norm)
        tokens_completos_lista.append(tokens)
        tokens_sin_stopwords_lista.append(tokens_sw)
        texto_sin_stopwords_lista.append(" ".join(tokens_sw))

        # Actualizar conteos de vocabulario y bigramas
        contador_vocab_total.update(tokens)
        contador_doc_freq.update(set(tokens))
        contador_vocab_sin_sw.update(tokens_sw)

        bigramas_doc = [f"{tokens_sw[i]} {tokens_sw[i+1]}" for i in range(len(tokens_sw) - 1)]
        contador_bigramas.update(bigramas_doc)

        documentos_json.append({
            "id": int(row["id"]),
            "usuario": row["usuario"],
            "texto_original": comentario_orig,
            "texto_limpio": limpio,
            "texto_base": base,
            "texto_normalizado_sin_tildes": base_norm,
            "texto_sin_stopwords": " ".join(tokens_sw),
            "tokens": tokens,
            "tokens_sin_stopwords": tokens_sw,
            "hashtags": hashtags,
            "etiquetas": {
                "categoria": row["categoria"],
                "sentimiento": row["sentimiento"],
                "puede_votar": int(row["puede_votar"]),
                "cuenta_verificada": int(row["cuenta_verificada"]),
                "dias_cuenta": int(row["dias_cuenta"]),
                "voto": row["voto"]
            }
        })

    # Añadir columnas procesadas al DataFrame
    df["comentario_limpio"] = texto_limpio_lista
    df["comentario_base"] = [limpiar_comentario(str(c), True) for c in df["comentario"]]
    df["comentario_sin_tildes"] = texto_base_sin_tildes
    df["comentario_sin_stopwords"] = texto_sin_stopwords_lista
    df["hashtags"] = hashtags_lista
    df["num_tokens"] = [len(t) for t in tokens_completos_lista]

    # 2. Guardar CSV procesado
    df.to_csv(out_dir / "corpus_procesado.csv", index=False, encoding="utf-8")

    # 3. Guardar Corpus en texto plano (1500 líneas y 44 frases base únicas)
    with open(out_dir / "corpus_completo.txt", "w", encoding="utf-8") as f:
        for linea in df["comentario_base"]:
            f.write(linea + "\n")

    frases_unicas = sorted(df["comentario_base"].unique())
    with open(out_dir / "corpus_frases_unicas.txt", "w", encoding="utf-8") as f:
        for linea in frases_unicas:
            f.write(linea + "\n")

    # 4. Construir tabla de Vocabulario
    filas_vocab = []
    for idx, (termino, freq) in enumerate(contador_vocab_total.most_common(), start=1):
        filas_vocab.append({
            "id_token": idx,
            "termino": termino,
            "frecuencia_total": freq,
            "frecuencia_documentos": contador_doc_freq[termino],
            "es_stopword": int(termino in STOPWORDS_ES)
        })
    df_vocab = pd.DataFrame(filas_vocab)
    df_vocab.to_csv(out_dir / "vocabulario.csv", index=False, encoding="utf-8")

    # 5. Guardar Corpus Estructurado en JSON (con resumen estadístico y vocabulario)
    corpus_POR_categoria = {
        cat: sorted(df[df["categoria"] == cat]["comentario_base"].unique().tolist())
        for cat in df["categoria"].unique()
    }
    corpus_POR_sentimiento = {
        sent: sorted(df[df["sentimiento"] == sent]["comentario_base"].unique().tolist())
        for sent in df["sentimiento"].unique()
    }

    resumen_corpus = {
        "nombre_corpus": "Corpus de Comentarios: No Todos Pueden Votar",
        "total_documentos": len(df),
        "comentarios_superficiales_unicos": int(df["comentario"].nunique()),
        "oraciones_base_unicas": len(frases_unicas),
        "total_tokens_corpus": int(sum(contador_vocab_total.values())),
        "tamano_vocabulario": len(contador_vocab_total),
        "tamano_vocabulario_sin_stopwords": len(contador_vocab_sin_sw),
        "distribucion_categorias": df["categoria"].value_counts().to_dict(),
        "distribucion_sentimientos": df["sentimiento"].value_counts().to_dict(),
        "distribucion_votos": df["voto"].value_counts().to_dict(),
        "top_15_palabras_clave": dict(contador_vocab_sin_sw.most_common(15)),
        "top_15_bigramas": dict(contador_bigramas.most_common(15)),
        "oraciones_base_por_categoria": corpus_POR_categoria,
        "oraciones_base_por_sentimiento": corpus_POR_sentimiento,
        "documentos": documentos_json
    }

    with open(out_dir / "corpus_estructurado.json", "w", encoding="utf-8") as f:
        json.dump(resumen_corpus, f, ensure_ascii=False, indent=2)

    # 6. Generar matriz Bag of Words (BoW) sobre las palabras clave (sin stopwords)
    vocab_clave = [palabra for palabra, _ in contador_vocab_sin_sw.most_common()]
    bow_rows = []
    for doc_id, tokens_sw, cat, sent in zip(
        df["id"], tokens_sin_stopwords_lista, df["categoria"], df["sentimiento"]
    ):
        conteo_doc = Counter(tokens_sw)
        fila = {"id": doc_id, "categoria": cat, "sentimiento": sent}
        for palabra in vocab_clave:
            fila[palabra] = conteo_doc[palabra]
        bow_rows.append(fila)
    df_bow = pd.DataFrame(bow_rows)
    df_bow.to_csv(out_dir / "matriz_bow.csv", index=False, encoding="utf-8")

    print("=== RESUMEN DEL CORPUS GENERADO ===")
    print(f"Total de documentos (filas): {resumen_corpus['total_documentos']}")
    print(f"Comentarios únicos (con variantes): {resumen_corpus['comentarios_superficiales_unicos']}")
    print(f"Oraciones base únicas (sin ruido): {resumen_corpus['oraciones_base_unicas']}")
    print(f"Total de tokens en el corpus: {resumen_corpus['total_tokens_corpus']}")
    print(f"Tamaño del vocabulario total: {resumen_corpus['tamano_vocabulario']} palabras únicas")
    print(f"Tamaño del vocabulario (sin stopwords): {resumen_corpus['tamano_vocabulario_sin_stopwords']} palabras clave")
    print("\nTop 10 palabras clave (sin stopwords):")
    for pal, c in contador_vocab_sin_sw.most_common(10):
        print(f"  - {pal}: {c}")
    print("\nTop 10 bigramas (sin stopwords):")
    for bg, c in contador_bigramas.most_common(10):
        print(f"  - {bg}: {c}")


if __name__ == "__main__":
    construir_corpus("comentarios_no_todos_pueden_votar.csv")

