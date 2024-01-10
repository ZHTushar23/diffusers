import csv
import numpy as np

# import data file
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
og_file = dataset_dir+"data_files.csv"

# 1. create train val and test split
input_file_path = og_file
train_file_path =  dataset_dir+'train.csv'
val_file_path =  dataset_dir+'val.csv'
test_file_path =  dataset_dir+'test.csv'


train_list = np.arange(1,31,1)
train_list = np.concatenate((train_list,np.arange(61,103,1)))
# print(train_list)
# print(len(train_list))

val_list = np.arange(51,61,1)
# print(val_list)
# print(len(val_list))

test_list = np.arange(31,51,1)


# print(test_list)
# print(len(test_list))
# Function to read CSV file with specified value range filter
def check_csv(input_path):
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        i=0
        count=0
        listt = []
        for row in reader:
            if i==0:
                print('Skip')
            else:
                listt.append(int(row[2]))
                count+=1

            i+=1
    return count,listt

# verify train list
t_c, r_train_list = check_csv(train_file_path)
p_train = np.unique(r_train_list)
print("total train samples: ", t_c)
# Check if array1 is equal to array2
are_arrays_equal = np.array_equal(train_list, p_train)
print(f"Train Verfied : {are_arrays_equal}")



# verify val list
v_c, r_val_list = check_csv(val_file_path)
p_val = np.unique(r_val_list)
print("total train samples: ", v_c)
# Check if array1 is equal to array2
are_arrays_equal = np.array_equal(val_list, p_val)
print(f"Val Verfied : {are_arrays_equal}")

# verify test list
t_c, r_test_list = check_csv(test_file_path)
p_test = np.unique(r_test_list)
print("total train samples: ", t_c)
# Check if array1 is equal to array2
are_arrays_equal = np.array_equal(test_list, p_test)
print(f"Test Verfied : {are_arrays_equal}")