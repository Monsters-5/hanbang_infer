# hanbang_infer
## YOLOv8 모델 fine tuning에 사용할 데이터 구성
- tent dataset 구성
    1. [ImageNet dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview)내 tent class(n03792972: mountain tent) 1300장 사용
    2. [Open Images V6 dataset](https://storage.googleapis.com/openimages/web/download.html#download_manually)내 tent class(/m/01j61q: tent) 3508장 사용
- Image preprocessing
    - **Resize:** Stretch to 640x640
    - augmentation진행  
        **Grayscale:** Apply to 25% of images  
        **Brightness:** Between -25% and +25%  
        **Blur:** Up to 2.5px  
        **Noise:** Up to 5% of pixels  
        **Cutout:** 3 boxes with 10% size each  
    - dataset split
        - train:val = 80:20으로 진행
        - 이후 텐트를 포함한 동영상을 이용해 test진행

## YOLOv8 모델 fine tuning결과
- 학습 과정
    - 이후 모델 성능 향상을 위해 15 - 20 epoch에서 학습 중단할 필요 있음  
  <img src="https://github.com/Monsters-5/hanbang_infer/assets/76683835/db8769c6-df5e-4dec-8323-bd68ecd94a5f.png" width="400" height="300"/>
- validation set  
  <img src="https://github.com/Monsters-5/hanbang_infer/assets/76683835/05c726b9-bf2f-4296-9b36-89613a991fa8.png" width="300" height="300"/>
- test video  
  <img src="https://github.com/Monsters-5/hanbang_infer/assets/76683835/08b38569-1639-419f-ad4a-e932f25b234f.png" width="300" height="200"/>

## Inference server
- 추론 과정
    1. 요청받은 영상의 뒤에서 90프레임 각각에 추론을 진행
        - 영상은 CCTV를 통해 계속 업데이트 된다고 가정
        - 요청을 받은 순간 구역 내 텐트 수는 가장 최근 영상을 가지고 추론하면 알 수 있음
    2. 모델이 동영상 특정 구역의 텐트를 각 인스턴스로 인식
    3. 추론 결과를 반환
        - 추론 결과
            - id: POST시 body에 넣는 값, 추론할 공원 위치 id
            - tent_cnt: 90 프레임에서 추론된 텐트 개수 값들의 중앙값
            - timestamp: 추론이 시작된 datetime 로그  
              ![Untitled2](https://github.com/Monsters-5/hanbang_infer/assets/76683835/0ead1fb5-37f3-4187-b523-b59a002ed1fb)

