from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder
import torch
###test sample###
model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
use_cuda = torch.cuda.is_available()
output = model.predict()

print(output)
# print(len(output[0][0]))
# print(print(output[0][1]))