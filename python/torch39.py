import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

text = "나는 파이토치를 이용한 딥러닝 합습중이다."
marked_text = "[CLS]" + text + "[SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)

text ="과수원에 사과가많았다." \
    "친구가 나에게 사과했다." \
        "백설공주는 독이 든 사과를 먹었다."
marked_text = "[CLS] " + text + " [SEP]" 
tokenized_text = tokenizer.tokenize(marked_text) 
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
for tup in zip(tokenized_text, indexed_tokens): 
    print('[:<12) (:>6,)'.format(tup[o], tup[1]))
    
segments_ids = [1] * len(tokenized_text)
print(segments_ids)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)

model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    
print("계층 개수:", len(hidden_states), " (initial embeddings + 12 BERT layers )")
layer_i = 0
print("배치 개수:", len(hidden_states[layer_i]))
batch_i =0
8
print("토큰 개수:", len(hidden_states[layer_i][batch_i]))
token_i =0
print("은닉층의 유닛 개수: ", len(hidden_states[layer_i][batch_i][token_i]))

print('은익 상태의 유형: ', type(hidden_states))
print('각 계층에서의 텐서 형태: ', hidden_states[0].size())

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()

token_embeddings = torch.squeeze(token_embeddings, dim=1)
token_embeddings.size()

token_embeddings = token_embeddings.permute(1, 0, 2)
token_embeddings.size()

token_vecscat = [] 
for token in token_embeddings: 
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    token_vecs_cat.append(cat_vec)
print('형태는: %d x %d' % (len( token_vecs_cat), len(token_vecs_cat[0])))

token_vecs_sum = [] 
for token in token_embeddings: 
    sum_vec = torch. sum( token[-4:], dim=0) 
    token_vecs_sum. append(sum_vec) 
print('형태는: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

token_vecs = hidden_states[-2][0]
sentence_embedding = torch.mean(token_vecs, dim=0)
print("최종 임베딩 벡터의 형태:", sentence_embedding.size())

for i, token_str in enumerate(tokenized_text):
    print(i, token_str)
    
print("사과가 많았다", str(token_vecs_sum[6][:5]))
print("나에게 사과했다", str(token_vecs_sum[10][:5]))
print("사과를 먹었다", str(token_vecs_sum[19][:5]))

from scipy.spatial.distance import cosine
diff_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[27])
same_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[16])
print('*유사한* 의미에 대한 벡터 유사성: %.2f' % same_apple)
print('*다른* 의미에 대한 벡터 유사성: %.2f' % diff_apple)