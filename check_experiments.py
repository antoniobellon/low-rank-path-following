import pickle


f = open('MaxCut experiments old/exp_100.pkl', 'rb')
pickled_exp = pickle.load(f) 
f.close

print(pickled_exp['exp_info']['exp_data']['C_init'][0])
print("dfgsDSFGs")
print(pickled_exp['exp_info']['exp_data']['C_pert'][0])