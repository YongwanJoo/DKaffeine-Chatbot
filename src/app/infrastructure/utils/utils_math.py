"""수학 유틸리티 함수

FAQ 검색에서 사용하는 벡터 유사도 계산 함수를 제공합니다.
scipy의 표준 함수를 사용합니다.
"""
import numpy as np
from scipy.spatial.distance import cosine


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간의 cosine similarity 계산
    
    scipy.spatial.distance.cosine을 사용하여 계산합니다.
    (거리를 유사도로 변환: similarity = 1 - distance)
    
    벡터 간의 각도를 측정하여 유사도를 계산합니다.
    방향이 같으면 1.0, 직교하면 0.0, 반대 방향이면 -1.0을 반환합니다.
    
    Example:
        ```python
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)  # 1.0
        ```
    
    Args:
        vec1: 첫 번째 벡터 (numpy array)
        vec2: 두 번째 벡터 (numpy array)
    
    Returns:
        Cosine similarity 값 (-1.0 ~ 1.0)
        - 1.0: 완전히 같은 방향
        - 0.0: 직교 또는 영벡터
        - -1.0: 완전히 반대 방향
    
    Note:
        scipy.spatial.distance.cosine은 거리를 반환하므로
        1에서 빼서 유사도로 변환합니다.
        영벡터가 포함된 경우 0.0을 반환합니다.
    """
    # scipy의 cosine은 거리를 반환 (0.0 ~ 2.0)
    # 유사도로 변환: similarity = 1 - distance
    distance = cosine(vec1, vec2)
    
    # NaN 체크 (영벡터인 경우)
    if np.isnan(distance):
        return 0.0
    
    return 1.0 - distance

