import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForPreTraining, AdamW
from torch.optim import AdamW

import glob

# Diretório onde estão os arquivos de texto
diretorio = 'D:/Users/ter95063/Documents/Ferramentas/notebooks/classification/textos_do_machado/'

padrao_arquivos = '*.txt'

caminhos_arquivos = glob.glob(f"{diretorio}/{padrao_arquivos}")

# Carregar os textos dos arquivos em uma lista
textos = []
for caminho_arquivo in caminhos_arquivos:
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        texto = arquivo.read()
        textos.append(texto)

# Carregar o tokenizer do BERT para o modelo 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

# Tokenizar os textos e criar input_ids e attention_mask
preprocessed_texts = []
max_length = 128  # Valor de max_length desejado, ajuste conforme necessário

for text in textos:  # Corrigido para usar a variável 'textos'
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=max_length, return_tensors='pt')
    
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    preprocessed_texts.append({
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })

# Verificar os tamanhos dos input_ids e attention_mask
for idx, text in enumerate(preprocessed_texts):
    input_ids_shape = text['input_ids'].shape
    attention_mask_shape = text['attention_mask'].shape
    print(f"Texto {idx + 1}:")
    print("Input IDs Shape:", input_ids_shape)
    print("Attention Mask Shape:", attention_mask_shape)
    print()



input_ids = torch.cat([text['input_ids'] for text in preprocessed_texts], dim=0)
attention_mask = torch.cat([text['attention_mask'] for text in preprocessed_texts], dim=0)

dataset = TensorDataset(input_ids, attention_mask)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
model.to(device)

epochs = 3
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Aqui você pode adicionar códigos para verificar o desempenho no conjunto de validação

    print(f'Epoch {epoch + 1}/{epochs} - Avg. Loss: {avg_loss}')

print('Treinamento concluído!')

