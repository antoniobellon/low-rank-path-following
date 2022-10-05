import pickle

# dic={}

# f = open(f"experiments.pkl", "wb")   
# pickle.dump(dic, f)
# f.close() 


# dic_1 = pickle.load(open("experiments.pkl","rb") )
# print(dic_1)
 

# # An arbitrary collection of objects supported by pickle.
# data = {
#     'a': [1, 2.0, 3+4j],
#     'b': ("character string", b"byte string"),
#     'c': {None, True, False}
# }

# f = open('data.pickle', 'wb')
# pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# f = open('data.pickle', 'rb') 
# data = pickle.load(f)
# print(data)
# data['c'] = {True}

# f = open('data.pickle', 'wb')
# pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# f = open('data.pickle', 'rb')  
# data = pickle.load(f)
# print(data)
# f = open('experiments.pkl', 'wb')
# dict={}
# pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)
# f.close

# f = open('experiments_10&50.pkl','rb')
# pickled_exp = pickle.load(f) 
# f.close()
 

f = open('experiments.pkl', 'wb')
pickle.dump({}, f, pickle.HIGHEST_PROTOCOL)
f.close


f = open('experiments.pkl', 'rb')
pickle.dump({}, f, pickle.HIGHEST_PROTOCOL)
f.close