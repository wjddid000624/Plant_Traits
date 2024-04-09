import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class NumericPreprocessor():
    def __init__(self, root, train_data, test_data, mode):

        """
        NumericPreprocessor 오브젝트 초기화.

        Parameters:
        - root: 루트 디렉토리.
        - train_data: 훈련 데이터 데이터프레임.
        - test_data: 테스트 데이터 데이터프레임.
        - mode: "train" / "test"
        
        Variables:
        - label: 라벨 정보 저장.
        - image: 이미지 패스 저장.
        - pca: 훈련 데이터 PCA 정보 저장.
        - scaler: 칼럼 별로 데이터 정규화

        """

        self.root = root
        self.train_data = train_data
        self.test_data = test_data
        self.mode = mode
        self.label = None
        self.image = None
        self.pca = PCA(random_state=42, n_components=2)
        self.scaler = StandardScaler()

    def group_separator(self, data, index):
         
        """
        데이터 칼럼을 index번째 언더스코어 앞 단어를 기준으로 분할. 두고 두고 쓰이는 효자 함수.

        Parameters:
        - data: 데이터프레임.
        - index: 분할 기준으로 삼고 싶은 단어의 인덱스. (근본없게 1부터 시작.)

        Returns:
        - 분할 기준이 되는 단어를 키로 삼고 해당 단어를 포함하는 칼럼들을 밸류로 가지는 딕셔너리.
        """
        
        columns = list(data.columns)
        grouped_columns = {}
        for column in columns:
            prefix = column.split('_')[index-1]
            if prefix not in grouped_columns:
                grouped_columns[prefix] = [column]
            else:
                grouped_columns[prefix].append(column)
        return grouped_columns

    def WORLDCLIM_processor(self, data):

        """
        WORLDCLIM 데이터에서 "seasonality" 제외하고 전부 제외.

        Parameters:
        - data: WORLDCLIM 데이터를 포함한 원본 데이터프레임.

        Returns:
        - 기존의 WORLDCLIM 칼럼 중 seaonality만 포함한 데이터프레임.
        """

        data = data.drop(['WORLDCLIM_BIO1_annual_mean_temperature',
                          'WORLDCLIM_BIO12_annual_precipitation',
                          'WORLDCLIM_BIO13.BIO14_delta_precipitation_of_wettest_and_dryest_month',
                          'WORLDCLIM_BIO7_temperature_annual_range'], axis=1)
        return data

    def SOIL_processor(self, data):

        """
        SOIL 데이터 카테고리 기준으로 전부 평균값 처리.

        Parameters:
        - data: SOIL 데이터를 포함한 원본 데이터프레임.

        Returns:
        - 기존의 SOIL 칼럼 전부 삭제 후 뒤에 카테고리 별 평균값만 알려주는 칼럼들 붙인 데이터프레임.
        """

        soil_columns = self.group_separator(data, 1)["SOIL"]
        soil_data = data[soil_columns]
        data = data.drop(soil_columns, axis=1)

        soil_category = self.group_separator(soil_data, 2)
        for name in soil_category:
            new_name = 'SOIL_' + name
            old_data = soil_data[soil_category[name]]
            new_data = old_data.sum(axis=1) / len(old_data.columns)
            data[new_name] = new_data.astype(int)
        return data

    def MODIS_processor(self, data):

        """
        MODIS 데이터의 Band 01, 03, 04를 평균내서 Type 01으로 통일.
        Band 02는 Type 02, Band 05는 Type 03으로 이름 통일.
        Type 별로 분기별 평균값을 산출 후 새로운 칼럼으로 저장, 기존의 MODIS 칼럼 전부 삭제.

        Parameters:
        - data: MODIS 데이터를 포함한 원본 데이터프레임.

        Returns:
        - Processed DataFrame.
        """

        modis_columns = self.group_separator(data, 1)["MODIS"]
        modis_data = data[modis_columns]
        data = data.drop(modis_columns, axis=1)

        band_category = {}
        for key, value in self.group_separator(modis_data, 8).items():
            band_category[key] = sorted(value, key=lambda x: int(x.split('_month_m')[-1]))

        band_01 = modis_data[band_category["01"]]
        band_02 = modis_data[band_category["02"]]
        band_03 = modis_data[band_category["03"]]
        band_04 = modis_data[band_category["04"]]
        band_05 = modis_data[band_category["05"]]

        type_01 = ((band_01.values + band_03.values + band_04.values) / 3).astype(int)
        type_02 = band_02.values
        type_03 = band_05.values
        type_dict = {"type_01" : type_01, "type_02" : type_02, "type_03" : type_03}

        for type, array in type_dict.items():
            for i in range(array.shape[1] // 3):
                start = i*3
                end = (i+1)*3
                mean = np.mean(array[:, start:end], axis=1)
                name = "MODIS_" + str(type) + "_Q" + str(i+1)
                data[name] = mean.astype(int)
        return data

    def VOD_processor(self, train_data, test_data):

        """
        VOD 데이터 전체에 대해서 PCA 처리해서 2개의 칼럼으로 압축.
        단, 훈련 데이터에 적용한 PCA를 테스트 데이터에도 똑같이 적용해야 한다.
        따라서 테스트 데이터를 받을 때엔 훈련 데이터를 같이 받아서
        훈련 데이터에 PCA를 피팅하고, 가중치만 테스트 데이터에 적용한다.
        훈련 데이터 및 random_state에 변동이 없으니 안전할 것.
        연산 속도에 안좋은 영향을 미치긴 하나, 4초대라 알바 아님.

        Parameters:
        - train_data: VOD 데이터를 포함한 훈련 데이터프레임.
        - test_data: VOD 데이터를 포함한 테스트 데이터프레임.

        Returns:
        - Processed DataFrame for train or test data.
        """

        vod_columns = self.group_separator(train_data, 1)["VOD"]
        vod_train_data = train_data[vod_columns]
        vod_test_data = test_data[vod_columns]

        train_data = train_data.drop(vod_columns, axis=1)
        test_data = test_data.drop(vod_columns, axis=1)

        train_pca = self.pca.fit_transform(vod_train_data)
        if self.mode == "train":
            train_data["VOD_pca_01"] = train_pca[:,0]
            train_data["VOD_pca_02"] = train_pca[:,1]
            return train_data
        
        test_pca = self.pca.transform(vod_test_data)
        test_data["VOD_pca_01"] = test_pca[:,0]
        test_data["VOD_pca_02"] = test_pca[:,1]
        return test_data

    def TARGET_processor(self, data):

        """
        원본 데이터프레임에서 타겟 값 포함된 칼럼들 전부 제거.
        타켓 중 mean에 대한 데이터만 따로 추려서 self.labels에 저장.
        테스트 데이터를 받을 시 해당 메소드는 무시.

        Parameters:
        - data: TARGET 데이터를 포함한 원본 데이터프레임.

        Returns:
        - Processed DataFrame.
        """

        if self.mode == "train":
            self.label = data[['X4_mean','X11_mean','X18_mean','X26_mean','X50_mean','X3112_mean']]
            data = data.drop(['X4_mean','X11_mean','X18_mean',
                            'X26_mean','X50_mean','X3112_mean',
                            'X4_sd','X11_sd','X18_sd','X26_sd',
                            'X50_sd','X3112_sd'], axis=1)
        return data
    
    def ID_processor(self, data):

        """
        원본 데이터프레임에선 ID 칼럼을 제거.
        따로 가져온 ID 칼럼을 재가공해서 이미지 패스를 담은 데이터프레임으로 변한 후 self.image에 저장.

        Parameters:
        - data: DataFrame containing ID data.

        Returns:
        - Processed DataFrame.
        """

        id = data["id"]
        data = data.drop(["id"], axis=1)
        image_path = os.path.join(self.root, self.mode, "image")
        self.image = [os.path.join(image_path, str(x) + ".jpeg") for x in id]
        return data
    
    def Standardize(self, data):

        """
        칼럼 별로 데이터 정규화.

        Parameters:
        - data: DataFrame containing data to be standardized.

        Returns:
        - Standardized DataFrame.
        """

        self.scaler.fit(data)
        data_standardized = self.scaler.transform(data)
        data_standardized = pd.DataFrame(data_standardized, columns=data.columns)
        return data_standardized
    
    def preprocess(self):

        """
        순차적으로 전처리 실행.
        테스트 데이터의 경우엔 label에 None이 저장된다.

        Returns:
        - Dictionary containing image paths, data, and labels.
        """

        if self.mode == "train":
            self.train_data = self.TARGET_processor(self.train_data)
            print("Processing Target...")
            self.train_data = self.WORLDCLIM_processor(self.train_data)
            print("Processing WORLDCLIM...")
            self.train_data = self.SOIL_processor(self.train_data)
            print("Processing SOIL...")
            self.train_data = self.MODIS_processor(self.train_data)
            print("Processing MODIS...")
            self.train_data = self.VOD_processor(self.train_data, self.test_data)
            print("Processing VOD...")
            self.train_data = self.ID_processor(self.train_data)
            print("Processing ID...")
            self.train_data = self.Standardize(self.train_data)
            print("Standardizing...")
            return {"image" : self.image, "data" : self.train_data, "label" : self.label}

        else:
            self.test_data = self.TARGET_processor(self.test_data)
            print("Processing Target...")
            self.test_data = self.WORLDCLIM_processor(self.test_data)
            print("Processing WORLDCLIM...")
            self.test_data = self.SOIL_processor(self.test_data)
            print("Processing SOIL...")
            self.test_data = self.MODIS_processor(self.test_data)
            print("Processing MODIS...")
            self.test_data = self.VOD_processor(self.train_data, self.test_data)
            print("Processing VOD...")
            self.test_data = self.ID_processor(self.test_data)
            print("Processing ID...")
            self.test_data = self.Standardize(self.test_data)
            print("Standardizing...")
            return {"image" : self.image, "data" : self.test_data, "label" : self.label}
