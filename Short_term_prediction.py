import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import datetime, time
import os
import pickle
from collections import defaultdict
import os
import multiprocessing
from multiprocessing import Pool

from heartrate_predict import dataInterpreter_predict

use_cuda = torch.cuda.is_available()

class dataInterpreter(object):
    def __init__(self, T, inputAtts, includeUser, includeSport, includeTemporal, targetAtts, fn="endomondoHR_proper.json", scaleVals=True, trimmed_workout_len=450, scaleTargets="scaleVals", trainValidTestSplit=[.8,.1,.1], zMultiple=5, trainValidTestFN=None):
        self.T = T
        self.filename = fn
        self.data_path = "."
        self.metaDataFn = fn.split(".")[0] + "_metaData.pkl"

        self.scaleVals = scaleVals
        self.trimmed_workout_len = trimmed_workout_len
        if scaleTargets == "scaleVals":
            scaleTargets = scaleVals
        self.scale_targets = scaleTargets # set to false when scale only inputs
        self.smooth_window = 1 # window size = 1 means no smoothing
        self.perform_target_smoothing = False

        self.isNominal = ['gender', 'sport']
        self.isDerived = ['time_elapsed', 'distance', 'derived_speed', 'since_begin', 'since_last']
        self.isSequence = ['altitude', 'heart_rate'] + self.isDerived

        self.inputAtts = inputAtts
        self.includeUser = includeUser
        self.includeSport = includeSport
        self.includeTemporal = includeTemporal

        self.targetAtts = ["tar_" + tAtt for tAtt in targetAtts]

        print("input attributes: ", self.inputAtts)
        print("target attributes: ", self.targetAtts)

        self.trainValidTestSplit = trainValidTestSplit
        self.trainValidTestFN = "./" + trainValidTestFN
        self.zMultiple = zMultiple

    def preprocess_data(self):
 
        # self.processed_path = self.data_path + "/processed_endomondoHR_proper_interpolate.csv"   
        self.processed_path = self.data_path + "/processed_endomondoHR_proper_interpolate_5k.csv"    
 
        self.loadTrainValidTest()      
        
        print("{} exists".format(self.processed_path))
        self.original_data = pd.read_csv(self.processed_path)
        self.map_workout_id()
        
        self.load_meta()
        
        self.input_dim = len(self.inputAtts)
        self.output_dim = len(self.targetAtts) # each continuous target has dimension 1, so total length = total dimension
     
    def map_workout_id(self):
        # convert workout id to original data id
        self.idxMap = defaultdict(int)

        # for idx, d in enumerate(self.original_data):
        for idx, d in self.original_data.iterrows():
            self.idxMap[d['id']] = idx

        self.trainingSet = [self.idxMap[wid] for wid in self.trainingSet]
        self.validationSet = [self.idxMap[wid] for wid in self.validationSet]
        self.testSet = [self.idxMap[wid] for wid in self.testSet]

        # update workout id to index in original_data
        contextMap2 = {}
        for wid in self.contextMap:
            context = self.contextMap[wid]
            contextMap2[self.idxMap[wid]] = (context[0], context[1], [self.idxMap[wid] for wid in context[2]])
        self.contextMap = contextMap2
    
    def load_meta(self): 
        self.buildMetaData() 
        
    # yield input and target data
    def dataIteratorSupervised(self, trainValidTest):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim
        # run on train, valid or test?
        if trainValidTest == 'train':
            indices = self.trainingSet
        elif trainValidTest == 'valid':
            indices = self.validationSet
        elif trainValidTest == 'test':
            indices = self.testSet
        else:
            raise (Exception("invalid dataset type: must be 'train', 'valid', or 'test'"))
        # loop each data point
        for idx in indices:
            # current_input = self.original_data[idx] 
            current_input = self.original_data.loc[idx] 
            workoutid = current_input['id']    
 
            # for real time prediction, data is smoothed
            num_steps = len(current_input['distance'].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(','))
            inputs = np.zeros([inputDataDim, num_steps])
            outputs = np.zeros([targetDataDim, num_steps])
            for att_idx, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    #inputs[att_idx, :] = np.ones([1, num_steps]) * current_input[att][-1] # given the total workout length
                    inputs[att_idx, :] = current_input[att][:num_steps] # given the total workout length
                else:
                    inputs[att_idx, :] = current_input[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')[:num_steps]
            for att in targetAtts:
                outputs[0, :] = current_input[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')[:num_steps]
            inputs = np.transpose(inputs)
            outputs = np.transpose(outputs)

            if self.includeUser:
                user_inputs = np.ones([num_steps]) * self.oneHotMap['userId'][current_input['userId']]
            if self.includeSport:
                sport_inputs = np.ones([num_steps]) * self.oneHotMap['sport'][current_input['sport']]
  
            # build context input    
            if self.includeTemporal:
                context_idx = self.contextMap[idx][2][-1] # index of previous workouts
                context_input = self.original_data.loc[context_idx]

                context_since_last = np.ones([1, num_steps]) * self.contextMap[idx][0]
                # consider what context?
                context_inputs = np.zeros([inputDataDim, num_steps])
                context_outputs = np.zeros([targetDataDim, num_steps])
                for att_idx, att in enumerate(inputAtts):
                    if att == 'time_elapsed':
                        #context_inputs[att_idx, :] = np.ones([1, num_steps]) * context_input[att][num_steps-1]
                        context_inputs[att_idx, :] = context_input[att][:num_steps]
                    else:
                        context_inputs[att_idx, :] = context_input[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')[:num_steps]
                for att in targetAtts:
                    context_outputs[0, :] = context_input[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')[:num_steps]
                context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
                context_input_2 = np.transpose(context_outputs)

            inputs_dict = {'input':inputs}
            if self.includeUser:       
                inputs_dict['user_input'] = user_inputs
            if self.includeSport:       
                inputs_dict['sport_input'] = sport_inputs
            if self.includeTemporal:
                inputs_dict['context_input_1'] = context_input_1
                inputs_dict['context_input_2'] = context_input_2

            # [T,D] -> many [10,D] windows   
            for t in range(num_steps - self.T): # total time - T segments with window size T
                inputs_dict_t = {}
                for k in inputs_dict:
                    inputs_dict_t[k] = inputs_dict[k][t : t + self.T]
                outputs_t = outputs[t : t + self.T]    
                    
                # yield one batch of window size time steps
                yield (inputs_dict_t, outputs_t, [workoutid, t])


    # feed into Keras' fit_generator (automatically resets)
    def generator_for_autotrain(self, batch_size, num_steps, trainValidTest):
        print("batch size = {}, num steps = {}".format(batch_size, num_steps))
        print("start new generator epoch: " + trainValidTest)

        # get the batch generator based on mode: train/valid/test
        # if trainValidTest=="train":
        #     data_len = sum([ (len(self.original_data[idx]['distance']) - self.T) for idx in self.trainingSet])
        # elif trainValidTest=="valid":
        #     data_len = sum([ (len(self.original_data[idx]['distance']) - self.T) for idx in self.validationSet])
        # elif trainValidTest=="test":
        #     data_len = sum([ (len(self.original_data[idx]['distance']) - self.T) for idx in self.testSet])
        data_len = 0
        if trainValidTest=="train":
            for idx in self.trainingSet:
                if(type(self.original_data.loc[idx]['distance'])==str):
                    data_len = data_len + (len(self.original_data.loc[idx]['distance'].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')) - self.T)
                else:
                    data_len = data_len + 1
        elif trainValidTest=="valid":
            for idx in self.validationSet:
                if(type(self.original_data.loc[idx]['distance'])==str):
                    data_len = data_len + (len(self.original_data.loc[idx]['distance'].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')) - self.T)
                else:
                    data_len = data_len + 1
        elif trainValidTest=="test":
            for idx in self.testSet: 
                if(type(self.original_data.loc[idx]['distance'])==str):
                    data_len = data_len + (len(self.original_data.loc[idx]['distance'].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')) - self.T)
                else:
                    data_len = data_len + 1
        else:
            raise(ValueError("trainValidTest is not a valid value"))
        batchGen = self.dataIteratorSupervised(trainValidTest)
        epoch_size = int(data_len / batch_size)
        print(epoch_size, data_len, batch_size)

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        for i in range(epoch_size):

            inputs = np.zeros([batch_size, self.T, inputDataDim])
            outputs = np.zeros([batch_size, self.T, targetDataDim])
            workoutids = np.zeros([batch_size, 2])

            if self.includeUser:
                user_inputs = np.zeros([batch_size, self.T])
                #user_inputs = np.zeros([batch_size, 1])
            if self.includeSport:
                sport_inputs = np.zeros([batch_size, self.T])
                #sport_inputs = np.zeros([batch_size, 1])
            if self.includeTemporal:
                context_input_1 = np.zeros([batch_size, self.T, inputDataDim + 1])
                context_input_2 = np.zeros([batch_size, self.T, targetDataDim])

            inputs_dict = {'input':inputs}
            for j in range(batch_size):
                current = next(batchGen)
                inputs[j,:,:] = current[0]['input']
                outputs[j,:,:] = current[1]
                workoutids[j] = current[2]
                if self.includeUser:
                    user_inputs[j,:] = current[0]['user_input']
                    #user_inputs[j] = current[0]['user_input']
                    inputs_dict['user_input'] = user_inputs
                if self.includeSport:
                    sport_inputs[j,:] = current[0]['sport_input']
                    #sport_inputs[j] = current[0]['sport_input']
                    inputs_dict['sport_input'] = sport_inputs
                if self.includeTemporal:
                    context_input_1[j,:,:] = current[0]['context_input_1']
                    context_input_2[j,:,:] = current[0]['context_input_2']
                    inputs_dict['context_input_1'] = context_input_1
                    inputs_dict['context_input_2'] = context_input_2
                inputs_dict['workoutid'] = workoutids

            # yield batch
            yield(inputs_dict, outputs)


    def loadTrainValidTest(self):
        with open(self.trainValidTestFN, "rb") as f:
            self.trainingSet, self.validationSet, self.testSet, self.contextMap = pickle.load(f)
            print("train/valid/test set size = {}/{}/{}".format(len(self.trainingSet), len(self.validationSet), len(self.testSet)))
            print("dataset split loaded")       


    # derive 'time_elapsed', 'distance', 'new_workout', 'derived_speed'
    def deriveData(self, att, currentDataPoint, idx):
        if att == 'time_elapsed':
            # Derive the time elapsed from the start
            timestamps = currentDataPoint['timestamp']
            initialTime = timestamps[0]
            return [x - initialTime for x in timestamps]
        elif att == 'distance':
            # Derive the distance
            lats = currentDataPoint['latitude']
            longs = currentDataPoint['longitude']
            indices = range(1, len(lats)) 
            distances = [0]
            # Gets distance traveled since last time point in kilometers
            distances.extend([haversine([lats[i-1],longs[i-1]], [lats[i],longs[i]]) for i in indices]) 
            return distances
        # derive the new_workout list
        elif att == 'new_workout': 
            workoutLength = self.trimmed_workout_len
            newWorkout = np.zeros(workoutLength)
            # Add the signal at start
            newWorkout[0] = 1 
            return newWorkout
        elif att == 'derived_speed':
            distances = self.deriveData('distance', currentDataPoint, idx)
            timestamps = currentDataPoint['timestamp']
            indices = range(1, len(timestamps))
            times = [0]
            times.extend([timestamps[i] - timestamps[i-1] for i in indices])
            derivedSpeeds = [0]
            for i in indices:
                try:
                    curr_speed = 3600 * distances[i] / times[i]
                    derivedSpeeds.append(curr_speed)
                except:
                    derivedSpeeds.append(derivedSpeeds[i-1])
            return derivedSpeeds
        elif att == 'since_last':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][0]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len) * total_time
        elif att == 'since_begin':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][1]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len) * total_time
        else:
            raise(Exception("No such derived data attribute"))

        
    # computing z-scores and multiplying them based on a scaling paramater
    # produces zero-centered data, which is important for the drop-in procedure
    def scaleData(self, data, att, zMultiple=2):
        mean, std = self.variableMeans[att], self.variableStds[att]
        diff = [d - mean for d in data]
        zScore = [d / std for d in diff] 
        return [x * zMultiple for x in zScore]

    # perform fixed-window median smoothing on a sequence
    def median_smoothing(self, seq, context_size):
        # seq is a list
        if context_size == 1: # if the window is 1, no smoothing should be applied
            return seq
        seq_len = len(seq)
        if context_size % 2 == f0:
            raise(exception("Context size must be odd for median smoothing"))

        smoothed_seq = []
        # loop through sequence and smooth each time step
        for i in range(seq_len): 
            cont_diff = (context_size - 1) / 2
            context_min = int(max(0, i - cont_diff))
            context_max = int(min(seq_len, i + cont_diff))
            median_val = np.median(seq[context_min:context_max])
            smoothed_seq.append(median_val)

        return smoothed_seq
    
    def buildEncoder(self, classLabels):
        # Constructs a dictionary that maps each class label to a list 
        # where one entry in the list is 1 and the remainder are 0
        encodingLength = len(classLabels)
        encoder = {}
        mapper = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
            mapper[label] = i
        return encoder, mapper
    
    
    def writeSummaryFile(self):
        metaDataForWriting=metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders,  
                                             self.oneHotMap, self.isSequence, self.isNominal, self.isDerived, 
                                             self.variableMeans, self.variableStds)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)
        print("Summary file written")
        
    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)
        except:
            raise(IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints = metaData.numDataPoints
        self.encodingLengths = metaData.encodingLengths
        self.oneHotEncoders = metaData.oneHotEncoders
        self.oneHotMap = metaData.oneHotMap
        self.isSequence = metaData.isSequence 
        self.isNominal = metaData.isNominal
        self.variableMeans = metaData.variableMeans
        self.variableStds = metaData.variableStds
        print("Metadata loaded")

        
    def derive_data(self):
        print("derive data")
        # derive based on original data
        for idx, d in enumerate(self.original_data):
            for att in self.isDerived:
                self.original_data[idx][att] = self.deriveData(att, d, idx) # add derived attribute
            
        
    # Generate meta information about data
    def buildMetaData(self):
        if os.path.isfile(self.metaDataFn):
            self.loadSummaryFile()
        else:
            print("Building data schema")
            # other than categoriacl, all are continuous
            # categorical to one-hot: gender, sport
            # categorical to embedding: userId  
            
            # continuous attributes
            print("is sequence: {}".format(self.isSequence))  
            # sum of variables? 
            variableSums = defaultdict(float)
            
            # number of categories for each categorical variable
            classLabels = defaultdict(set)
        
            # consider all data to first get the max, min, etc...      
            # for currData in self.original_data:
            for index, currData in self.original_data.iterrows():
                # update number of users
                att = 'userId'
                user = currData[att]
                classLabels[att].add(user)
                
                # update categorical attribute
                for att in self.isNominal:
                    val  = currData[att]
                    classLabels[att].add(val)
                    
                # update continuous attribute
                for att in self.isSequence:
                    sum_num = 0
                    if(type(currData[att])==str):
                      for num in currData[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(','):
                        sum_num = sum_num + float(num)
                    # variableSums[att] += sum(currData[att])
                      variableSums[att] += sum_num
                    else:
                      variableSums[att] += currData[att]


            oneHotEncoders = {}
            oneHotMap = {}
            encodingLengths = {}
            for att in self.isNominal:
                oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
                encodingLengths[att] = len(classLabels[att])
            
            att = 'userId'
            oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
            encodingLengths[att] = 1
            
            for att in self.isSequence:
                encodingLengths[att] = 1
            
            # summary information
            self.numDataPoints=len(self.original_data)
            
            # normalize continuous: altitude, heart_rate, latitude, longitude, speed and all derives            
            self.computeMeanStd(variableSums, self.numDataPoints, self.isSequence)
    
            self.oneHotEncoders=oneHotEncoders
            self.oneHotMap = oneHotMap
            self.encodingLengths = encodingLengths
            #Save that summary file so that it can be used next time
            self.writeSummaryFile()

 
    def computeMeanStd(self, varSums, numDataPoints, attributes):
        print("Computing variable means and standard deviations")
        
        # assume each data point has 500 time step?! is it correct?
        numSequencePoints = numDataPoints * 500 
        
        variableMeans = {}
        for att in varSums:
            variableMeans[att] = varSums[att] / numSequencePoints
        
        varResidualSums = defaultdict(float)
        
        # for numDataPoints, currData in enumerate(self.original_data):
        for numDataPoints, currData in self.original_data.iterrows():
            # loop each continuous attribute
            for att in attributes:
                # dataPointArray = np.array(currData[att])
                if(type(currData[att])==str):
                    dataArray = []
                    for item in currData[att].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(','):
                        dataArray.append(float(item))
                    dataPointArray = np.array(dataArray)
                else:
                    dataPointArray = np.array(currData[att])
                # add to the variable running sum of squared residuals
                diff = np.subtract(dataPointArray, variableMeans[att])
                sq = np.square(diff)
                varResidualSums[att] += np.sum(sq)

        variableStds = {}
        for att in varResidualSums:
            variableStds[att] = np.sqrt(varResidualSums[att] / numSequencePoints)
            
        self.variableMeans = variableMeans
        self.variableStds = variableStds
        
        
    # scale continuous data
    def scale_data(self, scaling=True): 
        print("scale data")
        targetAtts = ['heart_rate', 'derived_speed']

        for idx, currentDataPoint in enumerate(self.original_data):
            # target attribute, add to dict 
            for tAtt in targetAtts:         
                if self.perform_target_smoothing:
                    tar_data = self.median_smoothing(currentDataPoint[tAtt], self.smooth_window)
                else:
                    tar_data = currentDataPoint[tAtt]
                if self.scale_targets:
                    tar_data = self.scaleData(tar_data, tAtt, self.zMultiple) 
                self.original_data[idx]["tar_" + tAtt] = tar_data
                    
            # continuous input attribute, update dict
            for att in self.isSequence: 
                if scaling:
                    in_data = currentDataPoint[att]
                    self.original_data[idx][att] = self.scaleData(in_data, att, self.zMultiple) 
        for d in self.original_data:
            key = 'url'
            del d[key]
            key = 'speed'
            if key in d:
                del d[key]
        
        # write to disk
        with open(self.processed_path, 'w') as f:
            for l in self.original_data:
                f.write(str(l) + '\n')    
class metaDataEndomondo(object):
    def __init__(self, numDataPoints, encodingLengths, oneHotEncoders, oneHotMap, isSequence, isNominal, isDerived,
                 variableMeans, variableStds):
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.isSequence = isSequence
        self.isNominal = isNominal
        self.isDerived = isDerived
        self.variableMeans = variableMeans
        self.variableStds = variableStds

class contextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(contextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_dim = self.output_size
        self.context_layer_1 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_dim, batch_first=True)
        self.context_layer_2 = nn.LSTM(input_size = 1, hidden_size = self.hidden_dim, batch_first=True)
        # self.context_layer_1 = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_dim, batch_first=True)
        # self.context_layer_2 = nn.RNN(input_size = 1, hidden_size = self.hidden_dim, batch_first=True)
        # self.context_layer_1 = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_dim, batch_first=True)
        # self.context_layer_2 = nn.GRU(input_size = 1, hidden_size = self.hidden_dim, batch_first=True)
        self.dropout_rate = 0.2
        print("context encoder dropout: {}".format(self.dropout_rate))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.project = nn.Linear(self.hidden_dim * 2, self.context_dim)

    def forward(self, context_input_1, context_input_2):
        context_input_1 = self.dropout(context_input_1)
        context_input_2 = self.dropout(context_input_2)
         
        hidden_1 = self.init_hidden(context_input_1) # 1 * batch_size * hidden_size
        cell_1 = self.init_hidden(context_input_1)
        hidden_2 = self.init_hidden(context_input_2) # 1 * batch_size * hidden_size
        cell_2 = self.init_hidden(context_input_2)

        #print("context_input_1: ", context_input_1.shape)
        #print("context_input_2: ", context_input_2.shape)

        self.context_layer_1.flatten_parameters()
        outputs_1, lstm_states_1 = self.context_layer_1(context_input_1, (hidden_1, cell_1))
        # outputs_1, lstm_states_1 = self.context_layer_1(context_input_1, hidden_1)
        context_embedding_1 = outputs_1
        self.context_layer_2.flatten_parameters()
        outputs_2, lstm_states_2 = self.context_layer_2(context_input_2, (hidden_2, cell_2))
        # outputs_2, lstm_states_2 = self.context_layer_2(context_input_2, hidden_2)
        context_embedding_2 = outputs_2

        #context_embedding_1 = self.dropout(context_embedding_1)
        #context_embedding_2 = self.dropout(context_embedding_2)
        context_embedding = self.project(torch.cat([context_embedding_1, context_embedding_2], dim=-1))

        '''print(context_embedding_1.shape)
        print(context_embedding_2.shape)
        print(context_embedding.shape)'''

        return context_embedding

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T, attr_embeddings, dropout=0.1):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.user_embedding = attr_embeddings[0]
        self.sport_embedding = attr_embeddings[1]
        # self.logger = logger
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        print("encoder dropout: {}".format(self.dropout_rate))

        #self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first=True)
        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        # self.lstm_layer = nn.RNN(input_size = input_size, hidden_size = hidden_size)
        # self.lstm_layer = nn.GRU(input_size = input_size, hidden_size = hidden_size)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T, out_features = 1)

    def forward(self, attr_inputs, context_embedding, input_variable):
        for attr in attr_inputs:
            attr_input = attr_inputs[attr]
            if attr == "user_input":
                attr_embed = self.user_embedding(attr_input)
            if attr == "sport_input":
                attr_embed = self.sport_embedding(attr_input)
            input_variable = torch.cat([attr_embed, input_variable], dim=-1)

        input_variable = torch.cat([context_embedding, input_variable], dim=-1)

        input_data = input_variable
        # input_data: batch_size * T * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
  
        for t in range(self.T):
            #print("time step {}".format(t))
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T)
            # Eqn. 9: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T)) # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size), dim = -1) # batch_size * input_size, attn weights with values sum up to 1.
            # attn_weights = torch.ones(290,39).cuda()
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282

            # print("attn_weights: ", attn_weights.shape)
            # print("weighted_input: ", weighted_input.shape)
            # print("hidden: ", hidden.shape)
            # print("cell: ", cell.shape)

            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            # _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), hidden)
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # hidden = lstm_states
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # self.logger = logger

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        #self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size, batch_first=True)
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        # self.lstm_layer = nn.RNN(input_size = 1, hidden_size = decoder_hidden_size)
        # self.lstm_layer = nn.GRU(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        #print("input_encoded: ", input_encoded.shape)

        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T), dim = -1) # batch_size * T, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size

            #print("time step {}".format(t))

            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                #print("y_tilde: ", y_tilde.shape)
                #print("hidden: ", hidden.shape)
                #print("cell: ", cell.shape)

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                # _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), hidden)
                hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
                # hidden = lstm_output

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        # self.logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred.view(y_pred.size(0))

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())

# Train the model
class da_rnn:
    def __init__(self, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10,
                 learning_rate = 0.01, batch_size = 5120, parallel = True, debug = False, test_model_path = None):
        super(da_rnn, self).__init__()
        self.T = T
        # self.logger = logger
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        path = "."
        self.data_path = "endomondoHR_proper.json"
        self.patience = 3 # [3,5,10]
        self.max_epochs = 50
        self.zMultiple = 5

        self.pretrain, self.includeUser, self.includeSport, self.includeTemporal = False, True, False, True

        print("include pretrain/user/sport/temporal = {}/{}/{}/{}".format(self.pretrain,self.includeUser,self.includeSport,self.includeTemporal))

        self.model_file_name = []
        if self.includeUser:
            self.model_file_name.append("userId")
        if self.includeSport:
            self.model_file_name.append("sport")
        if self.includeTemporal:
            self.model_file_name.append("context")
        print(self.model_file_name)

        self.user_dim = 5
        self.sport_dim = 5
        # self.user_dim = 10
        # self.sport_dim = 10
        # self.user_dim = 1
        # self.sport_dim = 1

        self.trainValidTestSplit = [0.8, 0.1, 0.1]
        self.targetAtts = ['heart_rate']
        self.inputAtts = ['derived_speed', 'altitude']

        self.trimmed_workout_len = 300
        self.num_steps = self.trimmed_workout_len

        # Should the data values be scaled to their z-scores with the z-multiple?
        self.scale_toggle = True
        self.scaleTargets = False 

        # self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset_updated.pkl"
        self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset_updated_5k.pkl"
        # self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset_updated_2w.pkl"

        self.endo_reader = dataInterpreter(self.T, self.inputAtts, self.includeUser, self.includeSport, 
                                           self.includeTemporal, self.targetAtts, fn=self.data_path,
                                           scaleVals=self.scale_toggle, trimmed_workout_len=self.trimmed_workout_len, 
                                           scaleTargets=self.scaleTargets, trainValidTestSplit=self.trainValidTestSplit, 
                                           zMultiple = self.zMultiple, trainValidTestFN=self.trainValidTestFN)

        self.endo_reader.preprocess_data()

        self.input_dim = self.endo_reader.input_dim 
        self.output_dim = self.endo_reader.output_dim 

        self.train_size = len(self.endo_reader.trainingSet)
        self.valid_size = len(self.endo_reader.validationSet)
        self.test_size = len(self.endo_reader.testSet)

        modelRunIdentifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name.append(modelRunIdentifier) # Applend a unique identifier to the filenames
        self.model_file_name = "_".join(self.model_file_name)
        
        # build model
        # model
        self.num_users = len(self.endo_reader.oneHotMap['userId'])
        self.num_sports = len(self.endo_reader.oneHotMap['sport'])
        self.num_genders = len(self.endo_reader.oneHotMap['gender'])

        self.input_size = self.input_dim
        self.attr_num = 0
        self.attr_embeddings = []
        user_embedding = nn.Embedding(self.num_users, self.user_dim)
        torch.nn.init.xavier_uniform(user_embedding.weight.data)
        self.attr_embeddings.append(user_embedding)
        sport_embedding = nn.Embedding(self.num_sports, self.sport_dim)
        self.attr_embeddings.append(sport_embedding) 

        if self.includeUser:
            self.attr_num += 1
            self.input_size += self.user_dim
        if self.includeSport:
            self.attr_num += 1
            self.input_size += self.sport_dim
       
        if self.includeTemporal:
            # self.context_dim = self.user_dim
            # self.context_dim = encoder_hidden_size
            self.context_dim = int(encoder_hidden_size / 2)
            self.input_size += self.context_dim
            self.context_encoder = contextEncoder(input_size = self.input_dim + 1, hidden_size = encoder_hidden_size, output_size = self.context_dim)

        if use_cuda:
            for attr_embedding in self.attr_embeddings:
                attr_embedding = attr_embedding.cuda()       
 
        self.encoder = encoder(input_size = self.input_size, hidden_size = encoder_hidden_size, T = T, 
                               attr_embeddings = self.attr_embeddings)
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T)

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.context_encoder = nn.DataParallel(self.context_encoder)
            self.decoder = nn.DataParallel(self.decoder)
 
        wd1 = 0.002
        #wd1 = 0.003
        wd2 = 0.005
        if self.includeUser:
            print("user weight decay: {}".format(wd1))
        if self.includeSport:
            print("sport weight decay: {}".format(wd2))
        #self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
        #                                   lr = learning_rate, weight_decay=wd)

        self.encoder_optimizer = optim.Adam([
                {'params': [param for name, param in self.encoder.named_parameters() if 'user_embedding' in name], 'weight_decay':wd1},
                {'params': [param for name, param in self.encoder.named_parameters() if 'sport_embedding' in name], 'weight_decay':wd2},
                {'params': [param for name, param in self.encoder.named_parameters() if 'embedding' not in name]}
            ], lr=learning_rate)

        self.context_encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.context_encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        self.loss_func = nn.MSELoss(size_average=True)
        
        if test_model_path:
            checkpoint = torch.load(test_model_path)
            self.encoder.load_state_dict(checkpoint['en'])
            self.context_encoder.load_state_dict(checkpoint['context_en'])
            self.decoder.load_state_dict(checkpoint['de'])
            print("test model: {}".format(test_model_path))


    def get_batch(self, batch):
        
        attr_inputs = {}
        if self.includeUser:
            user_input = batch[0]['user_input']
            attr_inputs['user_input'] = user_input
        if self.includeSport:
            sport_input = batch[0]['sport_input']
            attr_inputs['sport_input'] = sport_input
        
        for attr in attr_inputs:
            attr_input = attr_inputs[attr]     
            attr_input = Variable(torch.from_numpy(attr_input).long())
            if use_cuda:
                attr_input = attr_input.cuda()
            attr_inputs[attr] = attr_input
        
        context_input_1 = batch[0]['context_input_1']
        context_input_2 = batch[0]['context_input_2']
        context_input_1 = Variable(torch.from_numpy(context_input_1).float())
        context_input_2 = Variable(torch.from_numpy(context_input_2).float())

        input_variable = batch[0]['input'] 
        target_variable = batch[1]
        input_variable = Variable(torch.from_numpy(input_variable).float())
        target_variable = Variable(torch.from_numpy(target_variable).float())
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
            context_input_1 = context_input_1.cuda()
            context_input_2 = context_input_2.cuda()
        
        y_history = target_variable[:, :self.T - 1, :].squeeze(-1)
        y_target = target_variable[:, -1, :].squeeze(-1)      
        
        return attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target 

    def train(self, n_epochs = 5, print_every=2000):
        
        # initialize
        print('Initializing ...')
        start_epoch = 0
        best_val_loss = None
        best_epoch_path = None
        best_valid_score = 9999999999
        best_epoch = 0

        for iteration in range(n_epochs):

            print()
            print('-' * 50)
            print('Iteration', iteration)

            epoch_start_time = time.time()
            start_time = time.time()
            
            # train
            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "train")
            
            print_loss = 0
            for batch, training_batch in enumerate(trainDataGen):
              
                 
                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(training_batch) 
                loss = self.train_iteration(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
                
                print_loss += loss
                if batch % print_every == 0 and batch > 0:
                    cur_loss = print_loss / print_every
                    elapsed = time.time() - start_time

                    print('| epoch {:3d} | {:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                          'loss {:5.3f}'.format(
                          iteration, batch, self.learning_rate,
                          elapsed * 1000 / print_every, cur_loss))

                    print_loss = 0
                    start_time = time.time()
            
            # evaluate    
            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "valid")
            val_loss = 0
            val_batch_num = 0
            for val_batch in validDataGen:
                val_batch_num += 1

                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(val_batch) 
                loss = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
                
                val_loss += loss
            val_loss /= val_batch_num
            print(val_batch_num)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f}'.format(iteration, (time.time() - epoch_start_time),
                                               val_loss))
            print('-' * 89)  
                
        # test
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "test")
        test_loss = 0
        test_batch_num = 0
        
        for test_batch in testDataGen:
            test_batch_num += 1
             
            attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(test_batch) 
            loss = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target)
            # print("y_history: ", y_history)
            # print("y_target: ", y_target)
            test_loss += loss
        print(test_batch_num)
        test_loss /= test_batch_num
        print('-' * 89)
        print('| test loss {:5.3f}'.format(test_loss))
        print('-' * 89)                
                
                
    def train_iteration(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target):
        self.encoder.train()
        self.context_encoder.train()
        self.decoder.train()
        
        self.encoder_optimizer.zero_grad()
        self.context_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        
        loss = self.loss_func(y_pred, y_target)
        loss.backward()

        self.encoder_optimizer.step()
        self.context_encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.item()

    def evaluate(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target):
        self.encoder.eval()
        self.context_encoder.eval()
        self.decoder.eval()

        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        # print("context_input_1: ", context_input_1)
        # print("context_input_2: ", context_input_2)
        # print("attr_inputs: ", len(attr_inputs), attr_inputs)
        # print("input_variable: ", input_variable.shape, input_variable)
        # print("y_history: ", y_history)
        # print("y_pred: ", y_pred)
        # print("y_target: ", y_target)
        loss = self.loss_func(y_pred, y_target)

        return loss.data.item()

    def predict(self, id):
        predict_reader = dataInterpreter_predict(self.T, self.inputAtts, self.includeUser, self.includeSport,
                                                 self.includeTemporal, self.targetAtts,
                                                 trimmed_workout_len=self.trimmed_workout_len,
                                                 predictFN='./predict_v2.pkl')

        predict_reader.preprocess_data(id)

        testDataGen = predict_reader.generator_for_autotrain(self.batch_size, self.num_steps)
        test_loss = 0
        test_batch_num = 0
        result = []

        for test_batch in testDataGen:
            result_temp = []
            self.encoder.eval()
            self.context_encoder.eval()
            self.decoder.eval()

            attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target = self.get_batch(
                test_batch)
            context_embedding = self.context_encoder(context_input_1, context_input_2)
            input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
            y_pred = self.decoder(input_encoded, y_history)
            result_temp.append(y_pred.detach().cpu().numpy().tolist())
            result_temp.append(y_target.detach().cpu().numpy().tolist())
            result.append(result_temp)
        return result

###test sample###
# model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
# use_cuda = torch.cuda.is_available()
# output = model.predict()

# print(output)
# print(len(output[0][0]))
# print(print(output[0][1]))