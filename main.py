import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import copy

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        print(self.df.head(n=5))
        for coluna in self.df:
            nzeros = self.df[coluna].isnull().sum()
            print(f'{nzeros} elementos nulos na {coluna}')
        """Como não tem elementos nulos no banco de dados, nenhuma linha precisou ser excluída"""
        cores = ['red' if s == 'Iris-setosa' else 'blue' if s == 'Iris-versicolor' else 'orange' for s in self.df['Species']]
        fig1, ax1 = plt.subplots()
        X1 = self.df['SepalLengthCm']
        Y1 = self.df['SepalWidthCm']
        ax1.scatter(X1, Y1, c=cores)
        fig2, ax2 = plt.subplots()
        X2 = self.df['PetalLengthCm']
        Y2 = self.df['PetalWidthCm']
        ax2.scatter(X2, Y2, c=cores)
        plt.show(block=True)
        del self.df['SepalLengthCm']
        del self.df['SepalWidthCm']
        """analisando o gráfico gerado, é possível perceber que a separação das classes pode ser
        feita facilmente mesmo sem essas duas colunas. Além disso, o gráfico gerado por essas
        duas colunas mostrava que as três classes não tinham uma separação muito simples através dessas 
        variáveis, por isso elas foram deletadas"""
        pass

    def Treinamento(self, modelo):
        self.modelo = modelo
        self.clinear = SVC(kernel=modelo)
        x = self.df[['PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']
        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
        self.clinear.fit(x_train, y_train)
        pass

    def Teste(self):
        taxa = self.clinear.score(self.x_test, self.y_test)
        print(f'Taxa de acerto do modelo {self.modelo}: {round(100*taxa, 2)}%')
        pass

    def Train(self, path):
        self.CarregarDataset(path)
        self.TratamentoDeDados()
        M2 = copy.deepcopy(self)
        self.Treinamento('linear')
        """para o SVC, quando o kernel= 'linear', na prática o algoritmo é o mesmo da regressão linear,
        apenas com algumas diferenças de implementação em relação ao linear_regression"""
        M2.Treinamento('poly')
        return self, M2

M = Modelo()
p = r'C:\Users\Joao Victor\OneDrive\Área de Trabalho\Faculdade\Curso DATA ICMC\iris.DATA'
M, M2 = M.Train(p)
M.Teste()
M2.Teste()