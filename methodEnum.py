from enum import Enum
MethodEnum = Enum('Method', ('TSNET', 'TSNETS', 'TNKK', 'TNFR',
                             'TNNL', 'TNEL', 'HOR', 'HORTS'))
# 对应的方法分别是: tsNet, tsNet*, tNem-KK, tNem-FR, tNem-nodeLinlog, tNem-edgeLinlog, High-Order-Repulsive, Hor+tsNet
