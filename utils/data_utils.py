from string import punctuation

"""
문구의 구두점 제거 & 소문자화
"""
def pre_processing(s):
    pre = s.encode("utf8").decode("ascii",'ignore')
    return ''.join(c for c in pre if c not in punctuation).lower()