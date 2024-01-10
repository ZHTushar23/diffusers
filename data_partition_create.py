#imprt libraries
import csv
import os

# import data file
dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
og_file = dataset_dir+"data_files.csv"

# 1. create train val and test split
input_file_path = og_file
train_file_path =  dataset_dir+'train.csv'
val_file_path =  dataset_dir+'val.csv'
test_file_path =  dataset_dir+'test.csv'

# Function to create a new CSV file with specified column exclusion and value range filter
def create_csv(input_path, output_path, value_range,value_range2=None):
    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        i=0
        for row in reader:
            if i==0:
                print('Skip')
                writer.writerow(row)
            else:
                # print("Hello")
                # Check if the value in the 4th column is within the specified range
                if value_range[0] <= int(row[2]) <= value_range[1]:
                    writer.writerow(row)
                elif value_range2 and value_range2[0] <= int(row[2]) <= value_range2[1]:
                    writer.writerow(row)

            i+=1

# Create train.csv excluding the specified range (31 to 60)
create_csv(input_file_path, train_file_path, value_range=(1, 30), value_range2=(61, 102))
# Create val.csv with values in the range 31 to 50
create_csv(input_file_path, val_file_path, value_range=(51, 60))

# Create test.csv with values in the range 51 to 60
create_csv(input_file_path, test_file_path, value_range=(31, 50))








# Function to create a new CSV file with specified column exclusion
def create_csv(input_path, output_path, exclude_column):
    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        for row in reader:
            # Exclude the specified column
            new_row = row[:exclude_column] + row[exclude_column+1:]
            writer.writerow(new_row)

# 1.
trainA_file_path = dataset_dir+'trainA.csv'
trainB_file_path = dataset_dir+'trainB.csv'

# Create trainA.csv excluding the 2nd column (index 1)
create_csv(train_file_path, trainA_file_path, exclude_column=1)

# Create trainB.csv excluding the 1st column (index 0)
create_csv(train_file_path, trainB_file_path, exclude_column=0)


# 2.
testA_file_path = dataset_dir+'testA.csv'
testB_file_path = dataset_dir+'testB.csv'

# Create testA.csv excluding the 2nd column (index 1)
create_csv(test_file_path, testA_file_path, exclude_column=1)

# Create testB.csv excluding the 1st column (index 0)
create_csv(test_file_path, testB_file_path, exclude_column=0)


# 3.
valA_file_path = dataset_dir+'valA.csv'
valB_file_path = dataset_dir+'valB.csv'

# Create testA.csv excluding the 2nd column (index 1)
create_csv(val_file_path, valA_file_path, exclude_column=1)

# Create testB.csv excluding the 1st column (index 0)
create_csv(val_file_path, valB_file_path, exclude_column=0)