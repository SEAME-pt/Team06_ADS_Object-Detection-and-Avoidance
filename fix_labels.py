import os

def alterar_classes_labels(diretorio_labels, mapeamento_classes):
    arquivos_processados = 0
    for nome_arquivo in os.listdir(diretorio_labels):
        if nome_arquivo.endswith('.txt'):
            caminho_arquivo = os.path.join(diretorio_labels, nome_arquivo)
            prefixo = None
            for chave in mapeamento_classes.keys():
                if nome_arquivo.lower().startswith(chave.lower()):
                    prefixo = chave
                    break
            if prefixo is None:
                continue  
            novo_valor_classe = mapeamento_classes[prefixo]
            try:
                with open(caminho_arquivo, 'r') as f:
                    linhas = f.readlines()
                linhas_alteradas = []
                for linha in linhas:
                    partes = linha.strip().split()
                    if len(partes) > 0:
                        partes[0] = str(novo_valor_classe)
                        linhas_alteradas.append(' '.join(partes) + '\n')
                    else:
                        linhas_alteradas.append(linha)
                with open(caminho_arquivo, 'w') as f:
                    f.writelines(linhas_alteradas)
                arquivos_processados += 1
            except Exception as e:
                print(f"Erro ao processar {caminho_arquivo}: {e}")
    return arquivos_processados

diretorio_labels = './dataset_split/train/labels'
mapeamento_classes = {'passadeira': 80, 'proibido': 81}

arquivos_alterados = alterar_classes_labels(diretorio_labels, mapeamento_classes)
print(f"Arquivos alterados: {arquivos_alterados}")

