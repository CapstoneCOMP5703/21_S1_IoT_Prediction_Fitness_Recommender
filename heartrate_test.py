from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder

import torch
###test sample###
use_cuda = torch.cuda.is_available()
print("Is CUDA available? %s.", use_cuda)
learning_rate = 0.005
batch_size = 290
hidden_size = 64
T=10
model = da_rnn(parallel = False, T = T, encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, learning_rate = learning_rate, batch_size=batch_size)

model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
output = model.predict()
print(output)
# print(len(output[0][0]))
# print(len(output[0][1]))