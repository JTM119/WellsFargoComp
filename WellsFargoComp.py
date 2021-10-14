import pandas as pd
import numpy as np
import sklearn
import pickle
import torch.nn as nn
import torch

"""
    Model Definition:
       1.) Linear input layer : 12 -> 32 nodes
       2.) Tanh
       3.) Dropout layer
       4.) Linear layer :   32->64 Nodes
       5.) Another dropout layer
       6.) Another tanh layer
       7.) Next a Linear layer which injects the output from the random forest model
       8.) Finally, a softmax layer
"""
class MLP_with_RF_preds(nn.Module):
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(p=.2)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(64 + 2, output_dim)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x, rf):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = torch.cat((x, rf),1 )
        x = self.out(x)
        if self.output_dim != 1:
            
            x = self.softmax(x)
        return x


"""
    This function selects and modifies the data in the following ways:
    Numerical : Select 'TRAN_AMT', 'CUST_AGE', 'OPEN_ACCT_CT', 'WF_dvc_age'

    String : Select 'CARR_NAME', 'RGN_NAME', 'DVC_TYPE_TXT', 'AUTHC_PRIM_TYPE_CD', 'AUTHC_SCNDRY_STAT_TXT'
             Categorize them 

    Customer/Transaction State: Compare them to see if they are the same. This is a simple binary value

    Date Time : Takes the number of days between the transaction date and each of the following last password update, the last phone number update
                and the custoer since date. In the event that a date is missing time passed is -1
    
    @param file_name : The file to read from
    @param id_string : The name of the column holding the id value

    @return the modified dataframe, and a dataframe holding the ids
"""
def refine_data(file_name, id_string):
    load_test = pd.read_csv(file_name)
    test_df = pd.DataFrame()
    
    #Numerical Data
    num_list = ['TRAN_AMT', 'CUST_AGE', 'OPEN_ACCT_CT', 'WF_dvc_age']
    for x in num_list:
        test_df[x] = load_test[x].copy()


    #String Data
    str_list = ['CARR_NAME', 'RGN_NAME', 'DVC_TYPE_TXT', 'AUTHC_PRIM_TYPE_CD', 'AUTHC_SCNDRY_STAT_TXT']
    str_df = pd.DataFrame()
    for x in str_list:
        str_df[x] = load_test[x].copy()
        str_df[x] = str_df[x].astype('category').cat.codes    

    for x in str_df.columns:
        test_df[x] = str_df[x]

    #State Match

    df_state_match_test = load_test[['STATE_PRVNC_TXT',  'CUST_STATE']]
    states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
    
    state_match_test = df_state_match_test.to_numpy()

    matches = []
    for row in state_match_test:
        try:
            cust_state = row[1]
            trans_loc = row[0].lower()
            cust_state = states[cust_state.upper()].lower()
            matches.append(int(trans_loc == cust_state))
        except:
            matches.append(False)

    test_df['matches'] = matches

    test_df['matches'] = test_df['matches'].astype('category').cat.codes

    #Date Time

    df_date_times = load_test[['PWD_UPDT_TS','TRAN_TS', 'PH_NUM_UPDT_TS', 'CUST_SINCE_DT']].copy()
    for x in df_date_times.columns:
        
        df_date_times[x] = pd.to_datetime(df_date_times[x], errors='coerce')

    df_test_diffs = pd.DataFrame()
    for x in df_date_times.columns:
        if x != "TRAN_TS":
            df_test_diffs[x] = df_date_times['TRAN_TS'] - df_date_times[x]
            df_test_diffs[x] = df_test_diffs[x].astype('timedelta64[D]')
            df_test_diffs[x] = df_test_diffs[x].fillna(value = -1)

    #df_test_diffs['GOAL'] = df['GOAL']


    test_df['PWD_UPDT_TS'] = df_test_diffs['PWD_UPDT_TS'].copy()
    test_df['PH_NUM_UPDT_TS'] = df_test_diffs['PH_NUM_UPDT_TS'].copy()
    return test_df, load_test[id_string]
    
def main():
    #Load the data and put it into the necessary format with the refine data function
    
    df, ids = refine_data('testset_for_participants.csv', 'dataset_id')
    ids = ids.to_numpy()
    #Load the rf model that feeds into the mlp and run it on the df data
    rf_model = pickle.load(open('mlp_feeder', 'rb'))
    rf_preds = rf_model.predict_proba(df)
    
    #Build the mlp data with these predictions
    new_df = pd.DataFrame()
    temp = []
    temp_val = []
    for row in rf_preds:
        max_val = max(row)
        temp_val.append(max_val)
        temp.append(list(row).index(max_val))
    new_df['Class_RF_Pred'] = temp
    new_df['Class_Prob'] = temp_val
    rf_to_transfer = new_df.to_numpy()

    #Load the network   
    mlp_model = MLP_with_RF_preds(input_dim = len(df.columns), output_dim = 2)

    mlp_model.load_state_dict(torch.load('best-model.pt'))
    mlp_model = mlp_model.double()
    mlp_model.eval()
    df = df.to_numpy()

    #The order should be preserved
    output = []
    with torch.no_grad():
        for i in range(df.shape[0]):
            class_probs = mlp_model(torch.from_numpy(df[i,:]).view(1,df.shape[1]), torch.from_numpy(rf_to_transfer[i, :]).view(1,2))
            _, fraud_nonfraud = torch.max(class_probs,1)
            #It was trained as fraud = 1, this needs to be reversed
            output.append(fraud_nonfraud.item())
    
    #create the output dataframe
    output_df = pd.DataFrame()
    output_df['dataset_id'] = ids
    output_df['FRAUD_NONFRAUD'] = output
    #Output the dataframe
    output_df.to_csv('output.csv', index = False)

if __name__=='__main__':
    main()