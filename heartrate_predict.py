class dataInterpreter_predict(object):
    def __init__(self, T, inputAtts, includeUser, includeSport, includeTemporal, targetAtts, trimmed_workout_len=450, predictFN=None):
        self.T = T

        self.trimmed_workout_len = trimmed_workout_len
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

        self.predictFN = "./" + predictFN

    def preprocess_data(self, id):
   
        self.processed_path =  "./processed_endomondoHR_proper_interpolate_2w.csv"    
        with open(self.predictFN, "rb") as f:
            self.predictSet, self.contextMap = pickle.load(f)

        self.original_data = pd.read_csv(self.processed_path)
        self.map_workout_id(id)
        
        self.load_meta()
        
        self.input_dim = len(self.inputAtts)
        self.output_dim = len(self.targetAtts) # each continuous target has dimension 1, so total length = total dimension
     
    def map_workout_id(self, id):
        # convert workout id to original data id
        self.idxMap = defaultdict(int)

        # for idx, d in enumerate(self.original_data):
        for idx, d in self.original_data.iterrows():
            self.idxMap[d['id']] = idx
        # self.predictSet = [self.idxMap[wid] for wid in self.predictSet]

        # for wid in self.predictSet:
        #     if wid == id:
        #         self.predictSet = [self.idxMap[wid]]
        #         break
        self.predictSet = [self.idxMap[id]]
        
        # update workout id to index in original_data
        contextMap2 = {}
        for wid in self.contextMap:
            context = self.contextMap[wid]
            contextMap2[self.idxMap[wid]] = (context[0], context[1], [self.idxMap[wid] for wid in context[2]])
        self.contextMap = contextMap2
    
    def load_meta(self): 
        # self.buildMetaData()
          # other than categoriacl, all are continuous
          # categorical to one-hot: gender, sport
          # categorical to embedding: userId  
          
          # continuous attributes 
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
        
    # yield input and target data
    def dataIteratorSupervised(self):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim
        indices = self.predictSet
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

    def generator_for_autotrain(self, batch_size, num_steps):
        data_len = 0
        for idx in self.predictSet: 
            if(type(self.original_data.loc[idx]['distance'])==str):
                data_len = data_len + (len(self.original_data.loc[idx]['distance'].replace('[','').replace(']','').replace('Decimal(\'', '').replace('\')', '').split(',')) - self.T)
            else:
                data_len = data_len + 1
        batchGen = self.dataIteratorSupervised()
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
            if self.includeSport:
                sport_inputs = np.zeros([batch_size, self.T])
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
                    inputs_dict['user_input'] = user_inputs
                if self.includeSport:
                    sport_inputs[j,:] = current[0]['sport_input']
                    inputs_dict['sport_input'] = sport_inputs
                if self.includeTemporal:
                    context_input_1[j,:,:] = current[0]['context_input_1']
                    context_input_2[j,:,:] = current[0]['context_input_2']
                    inputs_dict['context_input_1'] = context_input_1
                    inputs_dict['context_input_2'] = context_input_2
                inputs_dict['workoutid'] = workoutids

            # yield batch
            yield(inputs_dict, outputs)      
    
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
 
    def computeMeanStd(self, varSums, numDataPoints, attributes):
        
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