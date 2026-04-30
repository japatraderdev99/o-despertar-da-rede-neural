<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/5/55/FIAP_Logo.png" alt="FIAP" width="160"/>

# O Despertar da Rede Neural
## Fase 6 · Capítulo 1 · PBL
### Sistema de Visão Computacional com YOLOv5

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/yolov5)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Sobre o Projeto

Este repositório contém a entrega do projeto prático da **Fase 6 — Capítulo 1** do curso de **Inteligência Artificial e Machine Learning** da **FIAP**, turma 1TIAO.

O projeto foi desenvolvido no contexto da **FarmTech Solutions**, empresa fictícia que expandiu seus serviços de IA para novas áreas, incluindo **saúde animal**. O objetivo é demonstrar o potencial de um sistema de visão computacional para um cliente veterinário, treinando um modelo capaz de **detectar e classificar gatos e cachorros** em tempo real.

---

## 🎬 Demonstração

> 🎥 **[Clique aqui para assistir ao vídeo de demonstração no YouTube](https://youtu.be/3fGEKbsE86U)**
> *(Vídeo demonstrando o funcionamento do sistema — não listado)*

---

## 👨‍🎓 Informações do Aluno

| Campo | Informação |
|---|---|
| **Nome** | Guilherme Yamada Dantas |
| **RM** | 568506 |
| **Curso** | Inteligência Artificial e Machine Learning |
| **Turma** | 1TIAO |
| **Instituição** | FIAP |

---

## 📁 Estrutura do Repositório

```
o-despertar-da-rede-neural/
│
├── 📓 GuilhermeYamadaDantas_rm568506_pbl_fase6.ipynb
│       └── Entrega 1: YOLOv5 customizado — treinamento, validação e testes
│
├── 📓 GuilhermeYamadaDantas_rm568506_pbl_fase6_entrega2.ipynb
│       └── Entrega 2: Comparação — YOLO Tradicional vs YOLO Customizado vs CNN do Zero
│
└── 📄 README.md
        └── Este arquivo — documentação introdutória do projeto
```

---

## 🗺️ Navegação pelos Notebooks

Este projeto está documentado **integralmente nos notebooks Jupyter/Colab** abaixo. O README serve como ponto de entrada; toda a implementação técnica, análises, gráficos e conclusões estão nos notebooks.

### 📓 Entrega 1 — YOLOv5 Customizado

**[`GuilhermeYamadaDantas_rm568506_pbl_fase6.ipynb`](./GuilhermeYamadaDantas_rm568506_pbl_fase6.ipynb)**

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/japatraderdev99/o-despertar-da-rede-neural/blob/main/GuilhermeYamadaDantas_rm568506_pbl_fase6.ipynb)

O notebook da Entrega 1 contém o **pipeline completo** de visão computacional:

| Seção | Conteúdo |
|---|---|
| **Introdução** | Contexto FarmTech Solutions, objetivos e tecnologias |
| **Configuração** | Setup do ambiente Colab com GPU, instalação do YOLOv5 |
| **Dataset** | Download Oxford-IIIT Pet, separação de 40 gatos + 40 cachorros |
| **Rotulação** | Processo com Make Sense AI; conversão de máscaras em bboxes YOLO |
| **Visualização** | Amostras com bounding boxes, distribuição de classes |
| **Configuração YOLO** | Criação do `data.yaml` |
| **Experimento 1** | Treinamento com **30 épocas** + análise das curvas |
| **Experimento 2** | Treinamento com **60 épocas** + análise das curvas |
| **Comparação** | Tabela e gráficos: mAP, Precisão, Recall, Loss, tempo |
| **Validação** | Curvas P-R, matriz de confusão |
| **Testes** | Inferência nas 8 imagens nunca vistas, visualização das detecções |
| **Conclusões** | Análise crítica, lições aprendidas, recomendações ao cliente |

---

### 📓 Entrega 2 — Comparação de Abordagens

**[`GuilhermeYamadaDantas_rm568506_pbl_fase6_entrega2.ipynb`](./GuilhermeYamadaDantas_rm568506_pbl_fase6_entrega2.ipynb)**

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/japatraderdev99/o-despertar-da-rede-neural/blob/main/GuilhermeYamadaDantas_rm568506_pbl_fase6_entrega2.ipynb)

Comparação crítica das três abordagens usando o mesmo dataset:

| Abordagem | Descrição |
|---|---|
| **YOLO Customizado** | Referência da Entrega 1 — fine-tuning 60 épocas |
| **YOLO Tradicional** | YOLOv5s pré-treinado no COCO, sem fine-tuning |
| **CNN do Zero** | Rede convolucional construída e treinada do zero no Keras |

Critérios avaliados: facilidade de uso, precisão, tempo de treino e tempo de inferência.

---

## 🧪 Dataset

| Parâmetro | Valor |
|---|---|
| **Classes** | Gato (0) · Cachorro (1) |
| **Total de imagens** | 80 (40 por classe) |
| **Split por classe** | 32 treino · 4 validação · 4 teste |
| **Fonte** | [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) via TensorFlow Datasets |
| **Rotulação** | Bounding boxes derivadas de máscaras de segmentação + Make Sense AI |
| **Resolução** | 640×640 px (redimensionado para YOLOv5) |

**Estrutura das pastas (formato YOLO):**

```
dataset/
├── images/
│   ├── train/   (64 imagens: 32 gatos + 32 cachorros)
│   ├── val/     (8 imagens)
│   └── test/    (8 imagens)
└── labels/
    ├── train/   (64 arquivos .txt)
    ├── val/     (8 arquivos .txt)
    └── test/    (8 arquivos .txt)
```

---

## 🔬 Resultados Principais

### Entrega 1 — Comparação 30 vs 60 Épocas

| Métrica | 30 Épocas | 60 Épocas |
|---|---|---|
| mAP@0.5 | Ver notebook | Ver notebook |
| Precisão | Ver notebook | Ver notebook |
| Recall | Ver notebook | Ver notebook |
| Tempo treino (GPU T4) | ~12 min | ~25 min |

> Os valores exatos das métricas são gerados durante a execução do notebook e variam conforme os dados de treino. Execute o notebook para obter os resultados completos.

### Entrega 2 — Comparação das Abordagens

| Abordagem | Tarefa | Treino | Destaque |
|---|---|---|---|
| YOLO Customizado | Detecção | Fine-tuning 60e | Melhor precisão no domínio |
| YOLO Tradicional | Detecção | Zero (COCO) | Mais rápido de implantar |
| CNN do Zero | Classificação | ~100e (early stop) | Controle total da arquitetura |

---

## 🚀 Como Executar

### Opção 1 — Google Colab (Recomendado)

1. Acesse o notebook via badge [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/japatraderdev99/o-despertar-da-rede-neural/blob/main/GuilhermeYamadaDantas_rm568506_pbl_fase6.ipynb)
2. Ative a GPU: `Ambiente de execução > Alterar tipo > T4 GPU`
3. Execute todas as células em ordem (`Ctrl+F9`)
4. Para a **Entrega 2**, execute depois de completar a Entrega 1 na mesma sessão

### Opção 2 — Execução Local

```bash
# Clonar o repositório
git clone https://github.com/japatraderdev99/o-despertar-da-rede-neural.git
cd o-despertar-da-rede-neural

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Instalar dependências
pip install tensorflow tensorflow-datasets ultralytics opencv-python \
            matplotlib seaborn scikit-learn pillow pyyaml jupyter

# Clonar YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt && cd ..

# Iniciar Jupyter
jupyter notebook
```

---

## 🛠️ Tecnologias e Dependências

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | 3.10+ | Linguagem base |
| TensorFlow | 2.13+ | CNN do zero e carregamento do dataset |
| tensorflow-datasets | 4.9+ | Oxford-IIIT Pet Dataset |
| YOLOv5 | latest | Detecção de objetos customizada e tradicional |
| OpenCV | 4.8+ | Processamento e visualização de imagens |
| Matplotlib / Seaborn | 3.7+ / 0.12+ | Visualizações e gráficos |
| Scikit-learn | 1.3+ | Métricas de classificação |
| NumPy | 1.24+ | Operações numéricas |
| Pandas | 2.0+ | Análise de resultados |
| PyYAML | 6.0+ | Configuração do YOLOv5 |
| Pillow | 10.0+ | Manipulação de imagens |

---

## 📚 Referências

- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Make Sense AI — Rotulação de Imagens](https://www.makesense.ai/)
- [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/)
- FIAP — Material didático Fase 6: Redes Neurais e Deep Learning

---

<div align="center">

**Guilherme Yamada Dantas** · RM 568506 · FIAP 1TIAO · 2025

*Este repositório faz parte do programa acadêmico da FIAP. Mantenha o repositório **público** para acesso da equipe avaliadora.*

</div>
