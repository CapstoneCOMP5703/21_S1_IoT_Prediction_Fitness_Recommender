FROM fitastic/fitasticbase
 
RUN pip install --upgrade pip \
   && pip install requests \
   && pip install flask \
   && pip install xgboost==0.90 \
   && pip install jinja2\
   && pip install numpy \
   && pip install pandas \
   && pip install pyecharts \
   && pip install scipy \
   && pip install sklearn \     
   && pip install pymysql \
   && pip install matplotlib
 
RUN git clone --depth=1 https://github.com/CapstoneCOMP5703/CS25-2
 
WORKDIR /workspace/CS25-2
 
RUN mv ../model_epoch_04.pt model_epoch_04.pt
RUN mv ../mock_dataset.csv dataset/mock_dataset.csv
RUN mv ../processed_endomondoHR_proper_interpolate_1k.csv dataset/processed_endomondoHR_proper_interpolate_1k.csv
RUN mv ../processed_endomondoHR_proper_interpolate_5k.csv dataset/processed_endomondoHR_proper_interpolate_5k.csv
RUN mv ../recipes.csv dataset/recipes.csv
CMD python app.py
