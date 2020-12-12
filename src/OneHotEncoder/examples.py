#!/usr/bin/env python3

from sklearn.preprocessing import OneHotEncoder

print("BASIC ONE HOT ENCODER:")
enc = OneHotEncoder(handle_unknown='ignore')

X = [['Male', 1], ['Female', 3], ['Female', 2]]

data = enc.fit(X)
print(data)

data = enc.categories_
print(data)

data = enc.transform([['Female', 1], ['Male', 4]]).toarray()
print(data)

data = enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
print(data)

data = enc.get_feature_names(['gender', 'group'])
print(data)


print("DROP ENCODER:")
drop_enc = OneHotEncoder(drop='first').fit(X)

data = drop_enc.categories_
print(data)

data = drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
print(data)

print("DROP IF BINARY ENCODER:")
drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
data = drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
print(data)