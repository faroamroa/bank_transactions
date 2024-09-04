# Importar librerías

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Función para análisis univariado de variables numéricas con histogramas y KDE
def univariate_numeric_hist(df, columns):
    for col in columns:
        plt.figure(figsize=(6, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()

# Función adicional para análisis univariado de variables numéricas con boxplots
def univariate_numeric_boxplot(df, columns):
    for col in columns:
        plt.figure(figsize=(6, 6))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)
        plt.show()

# Función para análisis univariado de variables categóricas con gráficos de barras
def univariate_categorical(df, columns):
    for col in columns:
        plt.figure(figsize=(6, 6))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Frecuencia de categorías en {col}')
        plt.xlabel(col)
        plt.ylabel('Conteo')
        plt.xticks(rotation=45)
        plt.show()

# Función para análisis multivariado de variables numéricas con pairplot
def multivariate_pairplot(df, columns):
    sns.pairplot(df[columns])
    # plt.suptitle('Relaciones entre variables numéricas', y=1.02)
    plt.show()

# Función para análisis multivariado con violinplot de variables numéricas según una variable categórica
def multivariate_violinplot(df, categorical_col, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(6, 6))
        sns.violinplot(data=df, x=categorical_col, y=col)
        plt.title(f'Violinplot de {col} por {categorical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.show()

# Función para calcular y mostrar la matriz de correlación de variables numéricas
def correlation_matrix(df, columns):
    plt.figure(figsize=(6, 6))
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación')
    plt.show()
    return corr_matrix

def generate_wordcloud(df, text_column, max_words=100, background_color='white'):
    """
    Genera y muestra una nube de palabras basada en el texto de una columna de un DataFrame.

    :param df: DataFrame que contiene los datos.
    :param text_column: Nombre de la columna que contiene texto para generar la nube de palabras.
    :param max_words: Número máximo de palabras a mostrar en la nube de palabras (por defecto 100).
    :param background_color: Color de fondo de la nube de palabras (por defecto 'white').
    """
    # Unir todos los textos de la columna en un solo string
    text = ' '.join(df[text_column].dropna().astype(str))
    text = text.upper()

    # Generar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color=background_color).generate(text)

    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nube de Palabras de la Columna: {text_column}')
    plt.show()
