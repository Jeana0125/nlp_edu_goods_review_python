"""
리뷰 쓰기
parameter config file
"""
class Config(object):
    #model
    weight_file = "review_model.h5"
    # 단어 수
    max_len = 20
    # 매 epoch 트레이닝 횟수
    batch_size = 512
    #학습 율
    learning_rate = 0.001